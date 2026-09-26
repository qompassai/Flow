//! The JSON-RPC dispatch loop and the synchronous runtime trait.
//!
//! [`McpServer`] is a byte-faithful port of `flow/mcp.py::MCPServer`. The
//! framing rules are the security boundary: every frame is capped at
//! [`MAX_FRAME_BYTES`](crate::protocol::MAX_FRAME_BYTES) (1 MiB), oversized
//! input is a terminal `-32700` error, and stdout carries only protocol
//! JSON — never diagnostics.
//!
//! The agent runtime sits behind [`McpRuntime`]. Phase 3 ships the
//! synchronous trait; Phase 4 provides the async runtime and reuses this
//! dispatch unchanged.

use crate::error::RuntimeError;
use crate::json_ascii::{DumpsError, dumps};
use crate::protocol::{
    self, INVALID_PARAMS, INVALID_REQUEST, MAX_FRAME_BYTES, METHOD_NOT_FOUND, NOT_INITIALIZED,
    PARSE_ERROR, PROTOCOL_VERSION, RequestId, SUPPORTED_PROTOCOL_VERSIONS,
};
use crate::schema::validate_arguments;
use serde_json::{Map, Value};
use std::io::{self, BufRead, ErrorKind, Write};

/// The agent runtime behind the MCP server.
///
/// # stdout contract
///
/// Implementations must never write to stdout: the MCP transport owns it
/// and a single stray byte corrupts the client's frame stream. Diagnostics
/// go to stderr or a log. (`flow/mcp.py` enforces this by redirecting the
/// runtime's stdout to stderr around every tool call.)
pub trait McpRuntime {
    /// Execute the bounded planner/coder/reviewer pipeline for `task`.
    fn run(&mut self, task: &str) -> Result<Value, RuntimeError>;
    /// Report workspace status.
    fn status(&mut self) -> Result<Value, RuntimeError>;
    /// Run one approved check by name, or all checks when `name` is `None`.
    fn check(&mut self, name: Option<&str>) -> Result<Value, RuntimeError>;
    /// Release runtime resources. Called exactly once when the serve loop
    /// ends, mirroring the `finally: self.runtime.close()` in `serve`.
    fn close(&mut self);
}

/// How the [`McpServer::serve`] loop terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeEnd {
    /// The client closed stdin (clean EOF).
    Eof,
    /// An oversized frame arrived; the `-32700` error was sent and the
    /// connection closed, exactly like the Python server.
    OversizeFrame,
    /// The output stream broke (broken pipe / connection reset); the
    /// server exited quietly.
    OutputClosed,
}

/// One raw read from the input stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameRead {
    /// Clean EOF: no bytes at all.
    Eof,
    /// A line that is blank after ASCII-whitespace stripping; skipped.
    Blank,
    /// A candidate frame (at most [`MAX_FRAME_BYTES`] bytes); may still be
    /// oversized when it is exactly `MAX_FRAME_BYTES + 1` bytes.
    Bytes(Vec<u8>),
}

/// The MCP server: initialize/ready lifecycle plus tool dispatch.
pub struct McpServer<R: McpRuntime> {
    runtime: R,
    initialized: bool,
    ready: bool,
    frames_read: u64,
}

impl<R: McpRuntime> McpServer<R> {
    /// Create a server over `runtime`.
    pub fn new(runtime: R) -> McpServer<R> {
        McpServer {
            runtime,
            initialized: false,
            ready: false,
            frames_read: 0,
        }
    }

    /// Frames read so far (counts every `readline`, including blank and
    /// oversized lines, like the Python server).
    pub fn frames_read(&self) -> u64 {
        self.frames_read
    }

    /// Whether the initialize/notifications handshake completed.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Access the wrapped runtime (for tests).
    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    /// Access the wrapped runtime mutably (for tests).
    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }

    /// Dispatch one validated request object.
    ///
    /// Returns `None` for notifications, which never get a reply.
    fn handle(&mut self, request: &Map<String, Value>) -> Option<Value> {
        let method = request.get("method").and_then(|method| method.as_str());
        let jsonrpc_ok = request.get("jsonrpc").and_then(|version| version.as_str()) == Some("2.0");
        let raw_id = request.get("id");
        // Absent id: notification. Present-but-invalid id (null, bool,
        // float, array, object, over-i64 integer): Invalid Request with a
        // null id, exactly like the Python server.
        if !jsonrpc_ok || method.is_none() || !RequestId::id_is_valid(request) {
            return Some(protocol::error_value(
                &RequestId::Absent,
                INVALID_REQUEST,
                "Invalid JSON-RPC request",
            ));
        }
        let id = RequestId::extract(request).expect("the id was just validated");
        // Python validates params-is-object before any dispatch: missing
        // params default to `{}`, a present non-object is -32602 for
        // requests and is silently dropped for notifications.
        if let Some(params) = request.get("params")
            && !params.is_object()
        {
            raw_id?;
            return Some(protocol::error_value(
                &id,
                INVALID_PARAMS,
                "Params must be an object",
            ));
        }
        // From here the envelope is valid; notifications never execute tools.
        if raw_id.is_none() {
            if method == Some("notifications/initialized") && self.initialized {
                self.ready = true;
            }
            return None;
        }
        match method {
            Some("ping") => Some(protocol::result_value(&id, Value::Object(Map::new()))),
            Some("initialize") => {
                let params = request.get("params");
                Some(self.initialize(&id, params))
            }
            Some(_) if !self.ready => Some(protocol::error_value(
                &id,
                NOT_INITIALIZED,
                "Initialize and send notifications/initialized first",
            )),
            Some("tools/list") => Some(protocol::result_value(&id, protocol::tools_list_value())),
            Some("tools/call") => {
                let params = request.get("params");
                Some(self.tools_call(&id, params))
            }
            Some(other) => Some(protocol::error_value(
                &id,
                METHOD_NOT_FOUND,
                &format!("Method not found: {other}"),
            )),
            None => None,
        }
    }

    /// Handle `initialize`: version negotiation and capability exchange.
    fn initialize(&mut self, id: &RequestId, params: Option<&Value>) -> Value {
        if self.initialized {
            return protocol::error_value(id, INVALID_REQUEST, "Already initialized");
        }
        let params_obj = params.and_then(|params| params.as_object());
        let version = params_obj
            .and_then(|params| params.get("protocolVersion"))
            .and_then(|version| version.as_str())
            .unwrap_or("");
        if version.is_empty() {
            return protocol::error_value(
                id,
                INVALID_PARAMS,
                "protocolVersion must be a nonempty string",
            );
        }
        let capabilities_ok = params_obj
            .and_then(|params| params.get("capabilities"))
            .is_some_and(|capabilities| capabilities.is_object());
        let client_info_ok = params_obj
            .and_then(|params| params.get("clientInfo"))
            .and_then(|info| info.as_object())
            .is_some_and(|info| {
                info.get("name").and_then(|name| name.as_str()).is_some()
                    && info
                        .get("version")
                        .and_then(|version| version.as_str())
                        .is_some()
            });
        if !capabilities_ok || !client_info_ok {
            return protocol::error_value(
                id,
                INVALID_PARAMS,
                "Initialize requires capabilities and clientInfo",
            );
        }
        // Unknown versions negotiate to the newest supported version.
        let negotiated = if SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
            version
        } else {
            PROTOCOL_VERSION
        };
        self.initialized = true;
        protocol::result_value(id, protocol::initialize_value(negotiated))
    }

    /// Handle `tools/call`: strict params, schema-validated arguments, then dispatch.
    fn tools_call(&mut self, id: &RequestId, params: Option<&Value>) -> Value {
        debug_assert!(self.ready, "tools/call is only dispatched when ready");
        let params_obj = match params.map(phlow_json::object_map) {
            Some(Ok(params_obj)) => params_obj,
            _ => {
                return protocol::error_value(id, INVALID_PARAMS, "Params must be an object");
            }
        };
        // Strict: unknown tools/call parameters are rejected even though the
        // schema would ignore them. `_meta` is the MCP extension point.
        if params_obj
            .keys()
            .any(|key| !matches!(key.as_str(), "name" | "arguments" | "_meta"))
        {
            return protocol::error_value(id, INVALID_PARAMS, "Unknown tools/call parameters");
        }
        let name = params_obj
            .get("name")
            .and_then(|name| name.as_str())
            .unwrap_or("");
        let Some(spec) = protocol::tool_spec(name) else {
            return protocol::error_value(id, INVALID_PARAMS, "Unknown Flow tool");
        };
        let args = params_obj
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        if let Err(reason) = validate_arguments(&args, &spec.input_schema) {
            return protocol::error_value(id, INVALID_PARAMS, &reason.to_string());
        }
        let args_obj = match phlow_json::object_map(&args) {
            Ok(args_obj) => args_obj,
            Err(_) => {
                return protocol::error_value(id, INVALID_PARAMS, "Arguments must be an object");
            }
        };
        self.execute_tool(id, name, args_obj)
    }

    /// Dispatch one validated tool call to the runtime and wrap the result.
    /// `name` is a known tool and `args` passed schema validation.
    fn execute_tool(&mut self, id: &RequestId, name: &str, args: &Map<String, Value>) -> Value {
        // Defense in depth: the schema already enforces these shapes, but
        // the explicit checks preserve the Python error precedence.
        let outcome = match name {
            "flow_run" => match args.get("task").and_then(|task| task.as_str()) {
                Some(task) => self.runtime.run(task),
                None => {
                    return protocol::error_value(id, INVALID_PARAMS, "Task must be a string");
                }
            },
            "flow_check" => {
                let check_name = args.get("name").and_then(|name| name.as_str());
                if args.get("name").is_some() && check_name.is_none() {
                    return protocol::error_value(
                        id,
                        INVALID_PARAMS,
                        "Check name must be a string",
                    );
                }
                self.runtime.check(check_name)
            }
            "flow_status" => self.runtime.status(),
            _ => {
                return protocol::error_value(id, INVALID_PARAMS, "Unknown Flow tool");
            }
        };
        match outcome {
            Err(reason) => {
                // A runtime failure is a failed tool, not a protocol error:
                // isError with {status: "error", error: text}.
                let failed = serde_json::json!({"status": "error", "error": reason.to_string()});
                self.tool_result(id, &failed, true)
            }
            Ok(result) => {
                if !result.is_object() {
                    return protocol::error_value(
                        id,
                        crate::protocol::INTERNAL_ERROR,
                        "Runtime returned a non-object result",
                    );
                }
                let is_error =
                    result.get("status").and_then(|status| status.as_str()) != Some("ok");
                self.tool_result(id, &result, is_error)
            }
        }
    }

    /// Wrap a tool result in the MCP content envelope.
    fn tool_result(&self, id: &RequestId, result: &Value, is_error: bool) -> Value {
        match protocol::tool_result_value(id, result, is_error) {
            Ok(envelope) => envelope,
            Err(DumpsError::NonFiniteFloat) => protocol::error_value(
                id,
                crate::protocol::INTERNAL_ERROR,
                "Runtime returned non-JSON data",
            ),
        }
    }

    /// Parse one frame and dispatch it, returning the reply frame bytes
    /// (with trailing newline) or `None` for notifications.
    ///
    /// `frame` must be non-empty after blank-stripping and at most
    /// [`MAX_FRAME_BYTES`] bytes — the [`McpServer::serve`] loop upholds
    /// this; the assert mirrors the Python server's. The frame is parsed
    /// with plain `serde_json`, whose only depth limit is its own recursion
    /// bound (past ~128 levels): Python's only limit is `RecursionError`
    /// near ~1000, so a strict cap here would reject wire frames Python
    /// accepts. Depth limits belong to schema validation, not the parser.
    pub fn reply(&mut self, frame: &[u8]) -> Option<Vec<u8>> {
        assert!(
            !frame.is_empty() && frame.len() <= MAX_FRAME_BYTES,
            "reply() takes a stripped frame of at most MAX_FRAME_BYTES bytes"
        );
        let request: Value = match serde_json::from_slice(frame) {
            Ok(request) => request,
            Err(_) => {
                // Malformed UTF-8, malformed JSON, non-finite constants, or
                // nesting past serde_json's recursion bound: all parse errors.
                return Some(frame_bytes(&protocol::error_value(
                    &RequestId::Absent,
                    PARSE_ERROR,
                    "Parse error",
                )));
            }
        };
        let request_obj = match phlow_json::object_map(&request) {
            Ok(map) => map,
            Err(_) => {
                return Some(frame_bytes(&protocol::error_value(
                    &RequestId::Absent,
                    INVALID_REQUEST,
                    "Invalid Request: expected one JSON-RPC object, not a batch",
                )));
            }
        };
        // A non-object request can never be a notification, so handle()
        // always replies here; the defensive id extraction mirrors the
        // Python except-path for truly unexpected failures.
        let response = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.handle(request_obj)
        })) {
            Ok(response) => response,
            Err(_) => {
                let id = match RequestId::extract(request_obj) {
                    Some(id) => id,
                    None => RequestId::Absent,
                };
                Some(protocol::error_value(
                    &id,
                    crate::protocol::INTERNAL_ERROR,
                    "Internal error",
                ))
            }
        };
        response.map(|payload| frame_bytes(&payload))
    }

    /// Read one raw line (at most `MAX_FRAME_BYTES + 1` bytes) from `source`.
    fn read_frame(source: &mut impl BufRead) -> io::Result<FrameRead> {
        let mut line: Vec<u8> = Vec::new();
        // `take` caps the read at MAX_FRAME_BYTES + 1: a longer line arrives
        // truncated without its newline, which is exactly the oversize case.
        let mut limited = io::Read::take(&mut *source, (MAX_FRAME_BYTES + 1) as u64);
        limited.read_until(b'\n', &mut line)?;
        if line.is_empty() {
            return Ok(FrameRead::Eof);
        }
        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
            return Ok(FrameRead::Blank);
        }
        Ok(FrameRead::Bytes(line))
    }

    /// Run the transport event loop: read frames from `source`, write reply
    /// frames to `target`, until EOF, an oversized frame, or a broken output.
    ///
    /// Only protocol JSON is ever written to `target`. The runtime is closed
    /// exactly once on every exit path, mirroring the Python `finally`.
    pub fn serve(
        &mut self,
        source: &mut impl BufRead,
        target: &mut impl Write,
    ) -> io::Result<ServeEnd> {
        let outcome = self.serve_inner(source, target);
        self.runtime.close();
        outcome
    }

    fn serve_inner(
        &mut self,
        source: &mut impl BufRead,
        target: &mut impl Write,
    ) -> io::Result<ServeEnd> {
        loop {
            let frame = match Self::read_frame(source)? {
                FrameRead::Eof => return Ok(ServeEnd::Eof),
                // Blank lines are skipped but counted, like the Python server.
                FrameRead::Blank => {
                    self.frames_read += 1;
                    continue;
                }
                FrameRead::Bytes(line) => line,
            };
            self.frames_read += 1;
            assert!(frame.len() <= MAX_FRAME_BYTES + 1);
            if frame.len() > MAX_FRAME_BYTES {
                // Terminal framing error: report, flush, and close. The rest
                // of the over-long line is deliberately left unread.
                let mut error = frame_bytes(&protocol::error_value(
                    &RequestId::Absent,
                    PARSE_ERROR,
                    "Frame exceeds 1 MiB; connection closing",
                ));
                error.push(b'\n');
                // Only a broken pipe / reset connection exits quietly here;
                // any other I/O error propagates, like Python (which catches
                // only BrokenPipeError and ConnectionResetError).
                match write_frame(target, &error) {
                    Ok(()) => return Ok(ServeEnd::OversizeFrame),
                    Err(reason) if is_output_closed(&reason) => {
                        return Ok(ServeEnd::OutputClosed);
                    }
                    Err(reason) => return Err(reason),
                }
            }
            // Strip ASCII whitespace (the trailing newline at least) exactly
            // like Python's bytes.strip(); an all-whitespace line was already
            // classified Blank above, so this is non-empty.
            let stripped = strip_ascii_whitespace(&frame);
            let Some(mut reply) = self.reply(&stripped) else {
                continue;
            };
            reply.push(b'\n');
            match write_frame(target, &reply) {
                Ok(()) => {}
                Err(reason) if is_output_closed(&reason) => {
                    return Ok(ServeEnd::OutputClosed);
                }
                Err(reason) => return Err(reason),
            }
        }
    }
}

/// Serialize a reply payload to wire bytes (no trailing newline; the serve
/// loop appends it, exactly like the Python server).
fn frame_bytes(payload: &Value) -> Vec<u8> {
    dumps(payload)
        .expect("reply payloads are built from finite JSON values")
        .into_bytes()
}

/// Write bytes and flush. Broken pipe / connection reset is the caller's
/// quiet-exit signal (see [`ServeEnd::OutputClosed`]); every other I/O
/// error propagates to the caller, like the Python server.
fn write_frame(target: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    target.write_all(bytes).and_then(|()| target.flush())
}

/// True for the two quiet-exit I/O errors: broken pipe and reset connection.
fn is_output_closed(reason: &io::Error) -> bool {
    matches!(
        reason.kind(),
        ErrorKind::BrokenPipe | ErrorKind::ConnectionReset
    )
}

/// Strip ASCII whitespace from both ends, like Python's `bytes.strip()`.
fn strip_ascii_whitespace(frame: &[u8]) -> Vec<u8> {
    let start = frame
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(frame.len());
    let end = frame
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|pos| pos + 1)
        .unwrap_or(0);
    frame[start..end.max(start)].to_vec()
}

/// A scripted [`McpRuntime`] for tests: canned results, recorded calls, and
/// an optional injected failure.
#[derive(Debug, Default)]
pub struct FakeRuntime {
    /// Result returned by `run`.
    pub run_result: Value,
    /// Result returned by `status`.
    pub status_result: Value,
    /// Result returned by `check`.
    pub check_result: Value,
    /// When set, every runtime call fails with this message.
    pub fail_with: Option<String>,
    /// `check` calls received (each `Option<String>` is the name argument).
    pub check_calls: Vec<Option<String>>,
    /// `run` calls received.
    pub run_calls: Vec<String>,
    /// Whether `close` was called.
    pub closed: bool,
}

impl FakeRuntime {
    /// A fake whose `run`/`status`/`check` all report
    /// `{"status": "ok", ...}`, matching the Python contract-test stub.
    pub fn ok() -> FakeRuntime {
        FakeRuntime {
            run_result: serde_json::json!({"status": "ok"}),
            status_result: serde_json::json!({"status": "ok"}),
            check_result: serde_json::json!({"status": "ok", "checks": []}),
            ..FakeRuntime::default()
        }
    }
}

impl McpRuntime for FakeRuntime {
    fn run(&mut self, task: &str) -> Result<Value, RuntimeError> {
        self.run_calls.push(task.to_owned());
        match &self.fail_with {
            Some(message) => Err(RuntimeError(message.clone())),
            None => Ok(self.run_result.clone()),
        }
    }

    fn status(&mut self) -> Result<Value, RuntimeError> {
        match &self.fail_with {
            Some(message) => Err(RuntimeError(message.clone())),
            None => Ok(self.status_result.clone()),
        }
    }

    fn check(&mut self, name: Option<&str>) -> Result<Value, RuntimeError> {
        self.check_calls.push(name.map(str::to_owned));
        match &self.fail_with {
            Some(message) => Err(RuntimeError(message.clone())),
            None => Ok(self.check_result.clone()),
        }
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

/// A [`BufRead`] that yields its input in tiny fragments, to prove the
/// framing loop reassembles split writes.
#[cfg(test)]
use std::collections::VecDeque;

#[cfg(test)]
pub struct FragmentedReader {
    chunks: VecDeque<Vec<u8>>,
}

#[cfg(test)]
impl FragmentedReader {
    fn new(data: &[u8], fragment: usize) -> FragmentedReader {
        let chunks = data
            .chunks(fragment.max(1))
            .map(|chunk| chunk.to_vec())
            .collect();
        FragmentedReader { chunks }
    }
}

#[cfg(test)]
impl io::Read for FragmentedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let Some(chunk) = self.chunks.front_mut() else {
            return Ok(0);
        };
        let take = chunk.len().min(buf.len());
        buf[..take].copy_from_slice(&chunk[..take]);
        chunk.drain(..take);
        if chunk.is_empty() {
            self.chunks.pop_front();
        }
        Ok(take)
    }
}

#[cfg(test)]
impl BufRead for FragmentedReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self.chunks.front() {
            Some(chunk) => Ok(chunk),
            None => Ok(&[]),
        }
    }

    fn consume(&mut self, amount: usize) {
        if let Some(chunk) = self.chunks.front_mut() {
            chunk.drain(..amount.min(chunk.len()));
            if chunk.is_empty() {
                self.chunks.pop_front();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initialized_server() -> McpServer<FakeRuntime> {
        let mut server = McpServer::new(FakeRuntime::ok());
        let init = br#"{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}"#;
        assert!(server.reply(init).is_some());
        assert!(
            server
                .reply(br#"{"jsonrpc": "2.0", "method": "notifications/initialized"}"#)
                .is_none()
        );
        assert!(server.is_ready());
        server
    }

    /// A writer that fails every write with the given [`ErrorKind`].
    struct FailingWriter {
        kind: ErrorKind,
    }

    impl Write for FailingWriter {
        fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(self.kind, "injected"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn broken_pipe_exits_quietly() {
        let mut server = initialized_server();
        let ping = br#"{"jsonrpc": "2.0", "id": 9, "method": "ping"}"#.as_slice();
        let mut source = io::BufReader::new(ping);
        let mut target = FailingWriter {
            kind: ErrorKind::BrokenPipe,
        };
        let outcome = server.serve(&mut source, &mut target).unwrap();
        assert_eq!(outcome, ServeEnd::OutputClosed);
    }

    #[test]
    fn other_io_errors_propagate() {
        let mut server = initialized_server();
        let ping = br#"{"jsonrpc": "2.0", "id": 9, "method": "ping"}"#.as_slice();
        let mut source = io::BufReader::new(ping);
        let mut target = FailingWriter {
            kind: ErrorKind::PermissionDenied,
        };
        let err = server.serve(&mut source, &mut target).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::PermissionDenied);
    }

    #[test]
    fn version_negotiation_matrix() {
        for (requested, negotiated) in [
            ("2025-11-25", "2025-11-25"),
            ("2025-06-18", "2025-06-18"),
            ("2025-03-26", "2025-03-26"),
            ("1999-01-01", "2025-11-25"),
            ("", "2025-11-25"),
        ] {
            let mut server = McpServer::new(FakeRuntime::ok());
            let frame = format!(
                "{{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"initialize\", \
                  \"params\": {{\"protocolVersion\": \"{requested}\", \"capabilities\": {{}}, \
                  \"clientInfo\": {{\"name\": \"t\", \"version\": \"1\"}}}}}}"
            );
            let reply = server.reply(frame.as_bytes()).unwrap();
            let text = String::from_utf8(reply).unwrap();
            if requested.is_empty() {
                assert!(text.contains("-32602"), "empty version must be rejected");
            } else {
                assert!(
                    text.contains(&format!("\"protocolVersion\": \"{negotiated}\"")),
                    "for {requested}: {text}"
                );
            }
        }
    }

    #[test]
    fn initialize_requires_capabilities_and_client_info() {
        let mut server = McpServer::new(FakeRuntime::ok());
        let frame = br#"{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}"#;
        let reply = String::from_utf8(server.reply(frame).unwrap()).unwrap();
        assert!(reply.contains("-32602"));
        assert!(reply.contains("capabilities and clientInfo"));
        assert!(
            !server.initialized,
            "failed initialize must not flip the flag"
        );
    }

    #[test]
    fn frame_cap_boundary() {
        // A wire line of exactly MAX_FRAME_BYTES bytes (a JSON string of
        // spaces: MAX-1 content bytes plus the newline) passes framing and
        // is dispatched: a string is not a request object, so -32600, and
        // crucially not the oversize error.
        let mut server = initialized_server();
        let mut line = vec![b'"'];
        line.extend(std::iter::repeat_n(b' ', MAX_FRAME_BYTES - 3));
        line.push(b'"');
        line.push(b'\n');
        assert_eq!(line.len(), MAX_FRAME_BYTES);
        let mut source: &[u8] = &line;
        let mut target: Vec<u8> = Vec::new();
        let end = server.serve(&mut source, &mut target).unwrap();
        assert_eq!(end, ServeEnd::Eof);
        assert_eq!(server.frames_read(), 1);
        let text = String::from_utf8(target).unwrap();
        assert!(text.contains("-32600"), "{text}");
        assert!(!text.contains("exceeds"), "{text}");

        // MAX content bytes plus the newline (MAX+1 on the wire) trips the
        // oversize path: -32700 with the framing message, then close.
        let mut server = initialized_server();
        let oversize: Vec<u8> = vec![b'x'; MAX_FRAME_BYTES + 1];
        let mut source: &[u8] = &oversize;
        let mut target: Vec<u8> = Vec::new();
        let end = server.serve(&mut source, &mut target).unwrap();
        assert_eq!(end, ServeEnd::OversizeFrame);
        let text = String::from_utf8(target).unwrap();
        assert!(text.contains("-32700"), "{text}");
        assert!(
            text.contains("Frame exceeds 1 MiB; connection closing"),
            "{text}"
        );
        assert!(text.ends_with('\n'));
        assert!(
            server.runtime().closed,
            "runtime must close on the oversize path"
        );
    }

    #[test]
    fn fragmented_writes_reassemble() {
        let mut server = initialized_server();
        let frame = b"{\"jsonrpc\": \"2.0\", \"id\": 42, \"method\": \"ping\"}\n";
        let mut source = FragmentedReader::new(frame, 3);
        let mut target: Vec<u8> = Vec::new();
        let end = server.serve(&mut source, &mut target).unwrap();
        assert_eq!(end, ServeEnd::Eof);
        let text = String::from_utf8(target).unwrap();
        assert_eq!(text, "{\"jsonrpc\": \"2.0\", \"id\": 42, \"result\": {}}\n");
        assert!(server.runtime().closed);
    }

    #[test]
    fn blank_lines_are_skipped_but_counted() {
        let mut server = McpServer::new(FakeRuntime::ok());
        let input = b"\n   \n{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"ping\"}\n";
        let mut source: &[u8] = input;
        let mut target: Vec<u8> = Vec::new();
        server.serve(&mut source, &mut target).unwrap();
        assert_eq!(server.frames_read(), 3);
        assert_eq!(
            String::from_utf8(target).unwrap(),
            "{\"jsonrpc\": \"2.0\", \"id\": 1, \"result\": {}}\n"
        );
    }

    #[test]
    fn malformed_frames_report_parse_error() {
        let mut server = McpServer::new(FakeRuntime::ok());
        // Not JSON at all, or not UTF-8: -32700.
        for frame in [b"{oops".as_slice(), b"\xff\xfe".as_slice()] {
            let reply = String::from_utf8(server.reply(frame).unwrap()).unwrap();
            assert!(reply.contains("-32700"), "for {frame:?}: {reply}");
            assert!(reply.contains("Parse error"), "for {frame:?}: {reply}");
        }
        // Valid JSON but not an object (a batch): -32600, not -32700.
        let reply = String::from_utf8(server.reply(b"[1, 2]").unwrap()).unwrap();
        assert!(reply.contains("-32600"), "{reply}");
        assert!(reply.contains("not a batch"), "{reply}");
    }

    #[test]
    fn deep_nesting_follows_the_wire_not_a_strict_cap() {
        // 100-deep nesting is accepted: Python's only limit is
        // RecursionError near ~1000, so the wire parser must not impose a
        // stricter cap (schema validation keeps its own depth bound).
        let mut server = McpServer::new(FakeRuntime::ok());
        let deep = format!(
            "{{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"ping\", \
              \"params\": {{\"deep\": {}}}}}",
            "[".repeat(100) + &"]".repeat(100),
        );
        let reply = String::from_utf8(server.reply(deep.as_bytes()).unwrap()).unwrap();
        assert!(reply.contains("\"result\": {}"), "{reply}");
        // Past serde_json's own recursion bound it is still a parse error.
        let deeper = format!(
            "{{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"ping\", \
              \"params\": {{\"deep\": {}}}}}",
            "[".repeat(500) + &"]".repeat(500),
        );
        let reply = String::from_utf8(server.reply(deeper.as_bytes()).unwrap()).unwrap();
        assert!(reply.contains("-32700"), "{reply}");
    }

    #[test]
    fn non_object_params_rejected_before_dispatch() {
        // Python validates params-is-object before any method dispatch.
        let mut server = McpServer::new(FakeRuntime::ok());
        // ping with array params: -32602, not success.
        let reply = String::from_utf8(
            server
                .reply(br#"{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1]}"#)
                .unwrap(),
        )
        .unwrap();
        assert!(reply.contains("-32602"), "{reply}");
        assert!(reply.contains("Params must be an object"), "{reply}");
        // initialize with array params: -32602, not "Already initialized"
        // or the version error.
        let reply = String::from_utf8(
            server
                .reply(br#"{"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": [1]}"#)
                .unwrap(),
        )
        .unwrap();
        assert!(reply.contains("-32602"), "{reply}");
        assert!(reply.contains("Params must be an object"), "{reply}");
        assert!(
            !server.initialized,
            "failed initialize must not flip the flag"
        );
        // Request-form notifications/initialized with bad params: -32602,
        // not -32601.
        let reply = String::from_utf8(
            server
                .reply(
                    br#"{"jsonrpc": "2.0", "id": 3, "method": "notifications/initialized", "params": null}"#,
                )
                .unwrap(),
        )
        .unwrap();
        assert!(reply.contains("-32602"), "{reply}");
        // Notification (no id) with bad params: silently dropped, no reply.
        assert!(
            server
                .reply(br#"{"jsonrpc": "2.0", "method": "ping", "params": [1]}"#)
                .is_none()
        );
        // Missing params still default to {}: ping succeeds.
        let reply = String::from_utf8(
            server
                .reply(br#"{"jsonrpc": "2.0", "id": 4, "method": "ping"}"#)
                .unwrap(),
        )
        .unwrap();
        assert!(reply.contains("\"result\": {}"), "{reply}");
    }

    #[test]
    fn big_int_id_echoes_verbatim_on_the_wire() {
        let mut server = McpServer::new(FakeRuntime::ok());
        let reply = String::from_utf8(
            server
                .reply(br#"{"jsonrpc": "2.0", "id": 1180591620717411303424, "method": "ping"}"#)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            reply,
            r#"{"jsonrpc": "2.0", "id": 1180591620717411303424, "result": {}}"#
        );
    }

    #[test]
    fn check_name_is_optional_and_verified() {
        let mut server = initialized_server();
        // Named passing check verifies; named failing check fails the call.
        for (args, expect_name) in [
            (r#"{"name": "python_syntax"}"#, Some("python_syntax")),
            (r#"{}"#, None),
        ] {
            let frame = format!(
                "{{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"tools/call\", \
                  \"params\": {{\"name\": \"flow_check\", \"arguments\": {args}}}}}"
            );
            let reply = String::from_utf8(server.reply(frame.as_bytes()).unwrap()).unwrap();
            assert!(reply.contains("\"isError\": false"), "{reply}");
            assert_eq!(
                server.runtime().check_calls.last().unwrap(),
                &expect_name.map(str::to_owned)
            );
        }
        // A failing check result is an error envelope, not a protocol error.
        server.runtime_mut().check_result = serde_json::json!({"status": "failed"});
        let frame = br#"{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "flow_check", "arguments": {}}}"#;
        let reply = String::from_utf8(server.reply(frame).unwrap()).unwrap();
        assert!(reply.contains("\"isError\": true"), "{reply}");
    }

    #[test]
    fn non_string_check_name_is_expected_string_arguments() {
        // Schema validation runs before dispatch on both sides (Python's
        // `validate_arguments` raises `ValueError("Expected string
        // arguments")` for `{"name": 42}`), so a non-string `name` is
        // -32602 "Expected string arguments" — never "Check name must be
        // a string", which is unreachable defense-in-depth in both
        // implementations.
        let mut server = initialized_server();
        let frame = br#"{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "flow_check", "arguments": {"name": 42}}}"#;
        let reply = String::from_utf8(server.reply(frame).unwrap()).unwrap();
        assert!(reply.contains("\"code\": -32602"), "{reply}");
        assert!(reply.contains("Expected string arguments"), "{reply}");
        assert!(!reply.contains("Check name must be a string"), "{reply}");
    }

    #[test]
    fn runtime_failure_is_an_error_tool_result() {
        let mut server = initialized_server();
        server.runtime_mut().fail_with = Some("boom".to_owned());
        let frame = br#"{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "flow_run", "arguments": {"task": "x"}}}"#;
        let reply = String::from_utf8(server.reply(frame).unwrap()).unwrap();
        assert!(reply.contains("\"isError\": true"), "{reply}");
        assert!(reply.contains("\"status\": \"error\""), "{reply}");
        assert!(reply.contains("boom"), "{reply}");
    }
}
