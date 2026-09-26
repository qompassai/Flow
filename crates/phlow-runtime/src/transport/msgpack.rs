//! Hand-rolled msgpack-RPC transport for the Neovim private socket.
//!
//! Implements [`EditorTransport`] with exactly one msgpack-RPC method:
//! `nvim_exec_lua`, called only with [`SCHEMAS_LUA`] or [`CALL_LUA`] (the
//! bridge asserts this before the transport ever sees the expression).
//!
//! Concurrency model (mirrors the Python threaded bridge):
//!
//! - One current-thread tokio runtime, owned by the transport.
//! - One worker task owns the `UnixStream` for its whole life: no shared
//!   mutable session, no hidden thread ownership. Requests cross a bounded
//!   (`1`) mpsc channel; `exec` takes `&mut self`, so at most one request is
//!   ever in flight and the channel cannot back up.
//! - The connection is lazy (like Python's `pynvim.socket_session`): the
//!   first request connects, after the bridge validated the socket.
//! - Each request carries a `u32` msgpack-RPC id and a per-request oneshot.
//!   A reply is deliverable only to its own oneshot, so a stale reply can
//!   never surface to a later call — even after a timeout.
//! - A call that exceeds its deadline reports [`TransportError::Timeout`]
//!   and the worker drops the stream, forcing a fresh connection next time
//!   (a half-read frame must never desynchronize the next call).
//!
//! [`EditorTransport`]: phlow_editor::EditorTransport
//! [`TransportError`]: phlow_editor::TransportError
//! [`SCHEMAS_LUA`]: phlow_editor::SCHEMAS_LUA
//! [`CALL_LUA`]: phlow_editor::CALL_LUA

use std::time::Duration;

use phlow_editor::{CALL_LUA, SCHEMAS_LUA, TransportError, WORKER_GRACE};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// Bound on one msgpack-RPC response frame: 8 MiB. Generous (file tool
/// results are capped at 1 MiB upstream), but a malicious or desynchronized
/// peer cannot make the worker buffer without end.
const FRAME_BYTES_MAX: usize = 8 * 1024 * 1024;
/// Channel capacity: one in-flight request; `exec` takes `&mut self`.
const WORKER_QUEUE_CAPACITY: usize = 1;
/// Detail for `TransportError::Failed` when a response carries the wrong
/// msgpack-RPC id: the peer desynchronized, so the stream (and its buffer)
/// must be dropped and the next request reconnects.
const ID_MISMATCH_DETAIL: &str = "msgpack-RPC reply id mismatch";
/// Scratch read size for response frames.
const READ_CHUNK: usize = 4096;

/// Most notifications skipped while waiting for one response; bounds the
/// work a chatty peer can force into a single request (the outer timeout
/// still applies).
const NOTIFICATIONS_MAX: usize = 16;

/// One request handed to the owning worker task.
struct WorkerRequest {
    /// msgpack-RPC id, echoed by the peer.
    id: u32,
    /// `nvim_exec_lua` params: `[expression, args]`. The args travel as one
    /// array parameter, exactly like pynvim's `exec_lua(code, *args)`; Neovim
    /// rejects spliced extra params ("expecting 2").
    params: Vec<rmpv::Value>,
    /// Per-call deadline for connect + write + read.
    timeout: Duration,
    /// Delivers exactly one reply; dropped when the caller times out.
    reply: oneshot::Sender<Result<rmpv::Value, TransportError>>,
}

/// Hand-rolled msgpack-RPC transport over a Neovim private socket.
pub struct MsgpackTransport {
    runtime: tokio::runtime::Runtime,
    tx: Option<mpsc::Sender<WorkerRequest>>,
    worker: Option<JoinHandle<()>>,
    next_id: u32,
}

impl MsgpackTransport {
    /// Build the transport for `socket_path`. Nothing connects yet: the
    /// worker dials lazily on the first request, after the bridge validated
    /// the socket path.
    pub fn new(socket_path: &str) -> Self {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime builds");
        let (tx, rx) = mpsc::channel(WORKER_QUEUE_CAPACITY);
        let path = socket_path.to_owned();
        let worker = runtime.spawn(async move { worker_loop(&path, rx).await });
        MsgpackTransport {
            runtime,
            tx: Some(tx),
            worker: Some(worker),
            next_id: 0,
        }
    }
}

impl phlow_editor::EditorTransport for MsgpackTransport {
    fn exec(
        &mut self,
        expression: &str,
        args: &[Value],
        timeout: Duration,
    ) -> Result<Value, TransportError> {
        assert!(
            expression == SCHEMAS_LUA || expression == CALL_LUA,
            "only the two audited Lua expressions may cross the socket"
        );
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| TransportError::Failed("editor transport is closed".to_owned()))?;
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let params = request_params(expression, args)?;
        let (reply_tx, reply_rx) = oneshot::channel();
        let request = WorkerRequest {
            id,
            params,
            timeout,
            reply: reply_tx,
        };
        // Single owner and capacity 1: the send only fails when the worker died.
        self.runtime
            .block_on(async { tx.send(request).await })
            .map_err(|_| TransportError::Failed("editor worker is gone".to_owned()))?;
        // Outer deadline: the call timeout plus WORKER_GRACE for the worker to
        // wind down, mirroring the bridge's grace before it declares the
        // worker dead. The timeout future is built inside `block_on`: tokio's
        // `timeout()` eagerly grabs the runtime handle at construction.
        let reply = self
            .runtime
            .block_on(async { tokio::time::timeout(timeout + WORKER_GRACE, reply_rx).await });
        match reply {
            Err(_) => Err(TransportError::Timeout),
            Ok(Err(_)) => Err(TransportError::Failed("editor worker is gone".to_owned())),
            Ok(Ok(outcome)) => outcome.and_then(|value| rmpv_to_json(&value)),
        }
    }

    fn close(&mut self) {
        // Dropping the last sender closes the channel; the worker exits.
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            // Built inside `block_on` for the same eager-handle reason as
            // `exec` above.
            let _ = self
                .runtime
                .block_on(async { tokio::time::timeout(WORKER_GRACE, worker).await });
        }
    }
}

/// The single owning worker: serializes requests over one `UnixStream`.
async fn worker_loop(socket_path: &str, mut rx: mpsc::Receiver<WorkerRequest>) {
    let mut stream: Option<WorkerStream> = None;
    while let Some(request) = rx.recv().await {
        let outcome = service_request(&mut stream, socket_path, &request).await;
        // The caller may have timed out; its oneshot is gone and the stale
        // reply is dropped here, never delivered to a later call.
        let _ = request.reply.send(outcome);
    }
}

/// Serve one request: lazy connect, write `[0, id, "nvim_exec_lua", params]`,
/// read and validate `[1, id, error, result]`.
async fn service_request(
    stream: &mut Option<WorkerStream>,
    socket_path: &str,
    request: &WorkerRequest,
) -> Result<rmpv::Value, TransportError> {
    let outcome = tokio::time::timeout(request.timeout, async {
        if stream.is_none() {
            let connected = UnixStream::connect(socket_path).await.map_err(|error| {
                TransportError::Failed(format!("Neovim socket connect failed: {error}"))
            })?;
            *stream = Some(WorkerStream::new(connected));
        }
        let stream = stream.as_mut().expect("connected above");
        let frame = encode_request(request.id, &request.params)?;
        stream.stream.write_all(&frame).await.map_err(|error| {
            TransportError::Failed(format!("Neovim socket write failed: {error}"))
        })?;
        let response = read_response(stream).await?;
        decode_response(request.id, response)
    })
    .await;
    match outcome {
        Err(_) => {
            // The stream may hold a half-read frame; drop it (buffer and
            // all) so the next request reconnects instead of desynchronizing.
            *stream = None;
            Err(TransportError::Timeout)
        }
        Ok(Err(TransportError::Failed(detail))) if detail == ID_MISMATCH_DETAIL => {
            // A mismatched reply id means the peer desynchronized: the bytes
            // waiting in the buffer belong to no request, so drop the stream
            // and let the next request reconnect (as `read_response`
            // documents).
            *stream = None;
            Err(TransportError::Failed(detail))
        }
        Ok(result) => result,
    }
}

/// Build the `nvim_exec_lua` params `[expression, args]`: the args travel as
/// one array parameter, exactly like pynvim's `exec_lua(code, *args)`.
/// Splicing args as extra params makes Neovim reject the call with
/// "Wrong number of arguments: expecting 2".
fn request_params(expression: &str, args: &[Value]) -> Result<Vec<rmpv::Value>, TransportError> {
    let mut encoded_args = Vec::with_capacity(args.len());
    for arg in args {
        encoded_args.push(json_to_rmpv(arg)?);
    }
    Ok(vec![
        rmpv::Value::from(expression),
        rmpv::Value::Array(encoded_args),
    ])
}

/// Encode the msgpack-RPC request `[0, id, "nvim_exec_lua", params]`.
fn encode_request(id: u32, params: &[rmpv::Value]) -> Result<Vec<u8>, TransportError> {
    let request = rmpv::Value::Array(vec![
        rmpv::Value::from(0),
        rmpv::Value::from(id),
        rmpv::Value::from("nvim_exec_lua"),
        rmpv::Value::Array(params.to_vec()),
    ]);
    let mut frame = Vec::new();
    rmpv::encode::write_value(&mut frame, &request)
        .map_err(|error| TransportError::Failed(format!("msgpack encode failed: {error}")))?;
    Ok(frame)
}

/// The worker's stream plus its persistent read buffer. The buffer MUST
/// outlive one `read_frame`: a single socket read can deliver our response
/// followed by a notification (or part of the next frame), and discarding
/// those bytes would desynchronize the stream.
struct WorkerStream {
    stream: UnixStream,
    buffer: Vec<u8>,
}

impl WorkerStream {
    fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            buffer: Vec::new(),
        }
    }
}

/// Read one self-delimiting msgpack-RPC frame, preserving any trailing
/// bytes in the stream's buffer for the next read.
async fn read_frame(stream: &mut WorkerStream) -> Result<rmpv::Value, TransportError> {
    let mut chunk = [0u8; READ_CHUNK];
    loop {
        {
            let mut slice = stream.buffer.as_slice();
            if let Ok(value) = rmpv::decode::read_value(&mut slice) {
                let consumed = stream.buffer.len() - slice.len();
                // The cap applies to complete frames too: a peer could send
                // one oversized frame that decodes before the accumulation
                // check below ever fires. Drain first so the buffer stays
                // consistent for the next read.
                stream.buffer.drain(..consumed);
                if consumed > FRAME_BYTES_MAX {
                    return Err(TransportError::Failed(
                        "Neovim response frame exceeded 8 MiB".to_owned(),
                    ));
                }
                return Ok(value);
            }
        }
        if stream.buffer.len() > FRAME_BYTES_MAX {
            return Err(TransportError::Failed(
                "Neovim response frame exceeded 8 MiB".to_owned(),
            ));
        }
        let read = stream.stream.read(&mut chunk).await.map_err(|error| {
            TransportError::Failed(format!("Neovim socket read failed: {error}"))
        })?;
        if read == 0 {
            return Err(TransportError::Failed(
                "Neovim closed the socket mid-response".to_owned(),
            ));
        }
        stream.buffer.extend_from_slice(&chunk[..read]);
    }
}

/// Read frames until our response arrives, skipping msgpack-RPC
/// notifications `[2, method, params]` Neovim may interleave. A response
/// carrying the wrong id means the peer desynchronized: the caller drops
/// the stream so the next request reconnects.
async fn read_response(stream: &mut WorkerStream) -> Result<rmpv::Value, TransportError> {
    for _ in 0..NOTIFICATIONS_MAX {
        let frame = read_frame(stream).await?;
        let kind = frame.as_array().and_then(|frame| frame.first());
        if kind == Some(&rmpv::Value::from(2)) {
            continue;
        }
        if kind == Some(&rmpv::Value::from(1)) {
            return Ok(frame);
        }
        return Err(TransportError::Failed(
            "malformed msgpack-RPC response".to_owned(),
        ));
    }
    Err(TransportError::Failed(
        "Neovim sent too many notifications without a response".to_owned(),
    ))
}

/// Validate `[1, id, error, result]` and return the result.
fn decode_response(id: u32, response: rmpv::Value) -> Result<rmpv::Value, TransportError> {
    let malformed = || TransportError::Failed("malformed msgpack-RPC response".to_owned());
    let frame = match response {
        rmpv::Value::Array(frame) if frame.len() == 4 => frame,
        _ => return Err(malformed()),
    };
    if frame[0] != rmpv::Value::from(1) {
        return Err(malformed());
    }
    let replied = match &frame[1] {
        rmpv::Value::Integer(id) => id.as_u64(),
        _ => None,
    };
    if replied != Some(u64::from(id)) {
        return Err(TransportError::Failed(ID_MISMATCH_DETAIL.to_owned()));
    }
    if !frame[2].is_nil() {
        let detail = match rmpv_to_json(&frame[2]) {
            Ok(Value::String(text)) => text,
            Ok(other) => other.to_string(),
            Err(_) => "<undecodable>".to_owned(),
        };
        return Err(TransportError::Failed(format!(
            "nvim_exec_lua failed: {detail}"
        )));
    }
    Ok(frame[3].clone())
}

/// Convert JSON tool arguments to msgpack values.
fn json_to_rmpv(value: &Value) -> Result<rmpv::Value, TransportError> {
    let out_of_range = || TransportError::Failed("numeric argument out of range".to_owned());
    match value {
        Value::Null => Ok(rmpv::Value::Nil),
        Value::Bool(flag) => Ok(rmpv::Value::Boolean(*flag)),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                Ok(rmpv::Value::from(int))
            } else if let Some(uint) = number.as_u64() {
                Ok(rmpv::Value::from(uint))
            } else if let Some(float) = number.as_f64() {
                Ok(rmpv::Value::F64(float))
            } else {
                Err(out_of_range())
            }
        }
        Value::String(text) => Ok(rmpv::Value::from(text.as_str())),
        Value::Array(items) => items
            .iter()
            .map(json_to_rmpv)
            .collect::<Result<Vec<_>, _>>()
            .map(rmpv::Value::Array),
        Value::Object(fields) => fields
            .iter()
            .map(|(key, item)| Ok((rmpv::Value::from(key.as_str()), json_to_rmpv(item)?)))
            .collect::<Result<Vec<_>, _>>()
            .map(rmpv::Value::Map),
    }
}

/// Convert a msgpack response value back to JSON.
fn rmpv_to_json(value: &rmpv::Value) -> Result<Value, TransportError> {
    match value {
        rmpv::Value::Nil => Ok(Value::Null),
        rmpv::Value::Boolean(flag) => Ok(Value::Bool(*flag)),
        rmpv::Value::Integer(int) => {
            if let Some(signed) = int.as_i64() {
                Ok(Value::from(signed))
            } else if let Some(unsigned) = int.as_u64() {
                Ok(Value::from(unsigned))
            } else {
                Err(TransportError::Failed("integer out of range".to_owned()))
            }
        }
        rmpv::Value::F32(float) => Ok(Value::from(f64::from(*float))),
        rmpv::Value::F64(float) => Ok(Value::from(*float)),
        rmpv::Value::String(text) => text
            .as_str()
            .map(|text| Value::String(text.to_owned()))
            .ok_or_else(|| TransportError::Failed("non-UTF8 string in response".to_owned())),
        rmpv::Value::Binary(_) => Err(TransportError::Failed(
            // Python's bridge would raise `TypeError` serializing `bytes` to
            // JSON here; failing closed at the boundary keeps that behavior.
            "binary data in response".to_owned(),
        )),
        rmpv::Value::Array(items) => items
            .iter()
            .map(rmpv_to_json)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        rmpv::Value::Map(fields) => {
            let mut object = serde_json::Map::new();
            for (key, item) in fields {
                let key = match key {
                    rmpv::Value::String(text) => text.as_str().ok_or_else(|| {
                        TransportError::Failed("non-UTF8 map key in response".to_owned())
                    })?,
                    _ => {
                        return Err(TransportError::Failed(
                            "non-string map key in response".to_owned(),
                        ));
                    }
                };
                object.insert(key.to_owned(), rmpv_to_json(item)?);
            }
            Ok(Value::Object(object))
        }
        rmpv::Value::Ext(_, _) => Err(TransportError::Failed(
            "extension type in response".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Decode one `encode_request` frame back to `(id, method, params)`.
    fn decode_frame(frame: &[u8]) -> (u32, String, Vec<rmpv::Value>) {
        let value = rmpv::decode::read_value(&mut &frame[..]).expect("test frame must decode");
        let parts = value.as_array().expect("request must be an array");
        assert_eq!(parts.len(), 4, "msgpack-RPC request has 4 parts");
        assert_eq!(parts[0], rmpv::Value::from(0), "type 0 means request");
        let id = parts[1].as_u64().expect("id must be a u64") as u32;
        let method = parts[2]
            .as_str()
            .expect("method must be a string")
            .to_owned();
        let params = parts[3]
            .as_array()
            .expect("params must be an array")
            .clone();
        (id, method, params)
    }

    #[test]
    fn nvim_exec_lua_params_are_expression_plus_one_args_array() {
        // The schemas call passes no args, but Neovim's `nvim_exec_lua`
        // still requires the second parameter: the wire params must be
        // `[expr, []]`, never `[expr]`. A live Neovim rejected the spliced
        // form with "Wrong number of arguments: expecting 2 but got 1".
        let params = request_params(SCHEMAS_LUA, &[]).expect("no args cannot fail");
        let frame = encode_request(7, &params).expect("encoding cannot fail");
        let (id, method, wire_params) = decode_frame(&frame);
        assert_eq!(id, 7);
        assert_eq!(method, "nvim_exec_lua");
        assert_eq!(
            wire_params,
            vec![rmpv::Value::from(SCHEMAS_LUA), rmpv::Value::Array(vec![]),],
            "nvim_exec_lua takes exactly [expression, args]"
        );
    }

    #[test]
    fn call_args_are_wrapped_in_one_array_not_spliced() {
        // `editor_debug` is called as `CALL_LUA` with `[name, args]`; on the
        // wire those must arrive as the single second parameter, matching
        // pynvim's `exec_lua(code, *args)`.
        let args = vec![json!("editor_debug"), json!({"action": "status"})];
        let params = request_params(CALL_LUA, &args).expect("args must encode");
        let frame = encode_request(8, &params).expect("encoding cannot fail");
        let (_, method, wire_params) = decode_frame(&frame);
        assert_eq!(method, "nvim_exec_lua");
        assert_eq!(wire_params.len(), 2, "exactly [expression, args]");
        assert_eq!(wire_params[0], rmpv::Value::from(CALL_LUA));
        let expected_args: Vec<rmpv::Value> =
            args.iter().map(|arg| json_to_rmpv(arg).unwrap()).collect();
        assert_eq!(wire_params[1], rmpv::Value::Array(expected_args));
    }
}
