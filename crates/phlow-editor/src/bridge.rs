//! The editor bridge: allowlisted tools, cached schemas, poison-on-failure.
//!
//! Mirrors `flow/editor.py::EditorBridge`. The `pynvim` session is replaced
//! by the [`EditorTransport`] trait; everything else — lazy socket
//! validation, the schema cache, the poison latch, the exact envelope texts —
//! is preserved.
//!
//! State machine: a timeout or transport failure sets the poison latch
//! (`failure`), after which every `schemas`/`call` short-circuits with
//! `"Neovim bridge unavailable: ..."`. Only dropping the bridge and building
//! a new one clears it. `close()` marks the bridge closed; calls then report
//! `"Editor bridge closed"`.

use std::time::Duration;

use serde_json::{Map, Value};

use crate::contract::{
    CALL_LUA, SCHEMAS_LUA, TIMEOUT_MAX, TIMEOUT_MIN, filter_schemas, is_allowed_tool,
};
use crate::error::{BridgeError, TransportError};
use crate::socket::validate_socket;

/// The MsgPack-RPC session seam. Phase 3 is synchronous; Phase 4 provides a
/// real `rmp`/`rmpv` implementation behind this trait.
///
/// Contract for implementors:
/// - evaluate `expression` with `args` inside `timeout`; on expiry return
///   [`TransportError::Timeout`] (the bridge applies the canonical message);
/// - allow the bridge an additional [`WORKER_GRACE`] past `timeout` for the
///   worker to wind down before treating it as dead;
/// - only the two expressions [`SCHEMAS_LUA`] and [`CALL_LUA`] ever arrive.
pub trait EditorTransport {
    /// Evaluate `expression` with `args`, returning the decoded Lua result.
    fn exec(
        &mut self,
        expression: &str,
        args: &[Value],
        timeout: Duration,
    ) -> Result<Value, TransportError>;

    /// Release the session. Called at most once, from [`EditorBridge::close`].
    fn close(&mut self);
}

/// The private Neovim socket bridge.
pub struct EditorBridge<T: EditorTransport> {
    socket: Option<String>,
    timeout: Duration,
    transport: T,
    /// Whether the socket has passed [`validate_socket`] yet (lazy, like
    /// Python, which validates on the first request, not in `__init__`).
    validated: bool,
    /// `None` until the first `schemas()` call; then the filtered cache.
    cached_schemas: Option<Vec<Value>>,
    /// The poison latch: set once, never cleared.
    failure: Option<String>,
    closed: bool,
}

impl<T: EditorTransport> EditorBridge<T> {
    /// Build a bridge. The timeout must be within
    /// [`TIMEOUT_MIN`]..=[`TIMEOUT_MAX`]; the socket is validated lazily on
    /// first use, exactly like Python.
    pub fn new(
        socket: Option<String>,
        timeout: Duration,
        transport: T,
    ) -> Result<Self, BridgeError> {
        if timeout < TIMEOUT_MIN || timeout > TIMEOUT_MAX {
            return Err(BridgeError::BadTimeout(timeout));
        }
        Ok(EditorBridge {
            socket,
            timeout,
            transport,
            validated: false,
            cached_schemas: None,
            failure: None,
            closed: false,
        })
    }

    /// The request timeout this bridge was built with.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Replace the per-request timeout, e.g. from the CLI's
    /// `--editor-timeout` flag. The new value must lie within
    /// [`TIMEOUT_MIN`]..=[`TIMEOUT_MAX`]; out-of-range values are a
    /// [`BridgeError`], never a silent clamp. Mirrors constructing the
    /// Python `EditorBridge` with a different `timeout_s`.
    pub fn set_timeout(&mut self, timeout: Duration) -> Result<(), BridgeError> {
        if timeout < TIMEOUT_MIN || timeout > TIMEOUT_MAX {
            return Err(BridgeError::BadTimeout(timeout));
        }
        self.timeout = timeout;
        Ok(())
    }

    /// True once a timeout or transport failure has poisoned the bridge.
    pub fn is_poisoned(&self) -> bool {
        self.failure.is_some()
    }

    /// Fetch and cache the editor tool schemas.
    ///
    /// With no socket configured this returns `[]` without touching the
    /// transport, like Python. Otherwise the first call evaluates
    /// [`SCHEMAS_LUA`], filters to well-formed editor tools, and caches the
    /// (cloned) result; later calls return the cache.
    pub fn schemas(&mut self) -> Vec<Value> {
        if self.socket.is_none() {
            return Vec::new();
        }
        if self.cached_schemas.is_none() {
            let fetched = match self.submit(SCHEMAS_LUA, &[]) {
                Ok(result) => filter_schemas(&result),
                Err(message) => Err(message),
            };
            match fetched {
                Ok(schemas) => self.cached_schemas = Some(schemas),
                Err(message) => {
                    self.poison(message);
                    self.cached_schemas = Some(Vec::new());
                }
            }
        }
        self.cached_schemas.clone().unwrap_or_default()
    }

    /// Call one allowlisted editor tool.
    ///
    /// Unknown tool names and non-object arguments are refused before
    /// anything crosses the socket. Transport failures become
    /// `{"status": "unavailable", "error": ...}`.
    pub fn call(&mut self, name: &str, args: &Value) -> Value {
        if !is_allowed_tool(name) {
            return refused("Editor tool is not allowlisted");
        }
        if !args.is_object() {
            return refused("Editor arguments must be an object");
        }
        if self.socket.is_none() {
            return unavailable("No --nvim socket configured");
        }
        match self.submit(CALL_LUA, &[Value::from(name), args.clone()]) {
            Ok(result) if phlow_json::is_object(&result) => result,
            Ok(_) => refused("Rose returned a non-object tool result"),
            Err(message) => unavailable(&message),
        }
    }

    /// Bridge health, mirroring `EditorBridge.status`: `"ok"` only when the
    /// schema cache is non-empty, with the cached tool names and the poison
    /// message (or `"No editor bridge configured"` when there is no socket).
    pub fn status(&mut self) -> Value {
        let schemas = self.schemas();
        let tools: Vec<Value> = schemas
            .iter()
            .filter_map(|schema| {
                schema
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .cloned()
            })
            .collect();
        let error = match (&self.failure, schemas.is_empty()) {
            (Some(failure), _) => Value::from(failure.clone()),
            (None, false) => Value::Null,
            (None, true) => Value::from("No editor bridge configured"),
        };
        let mut map = Map::new();
        map.insert(
            "status".to_owned(),
            Value::from(if schemas.is_empty() {
                "unavailable"
            } else {
                "ok"
            }),
        );
        map.insert("tools".to_owned(), Value::Array(tools));
        map.insert("error".to_owned(), error);
        Value::Object(map)
    }

    /// Close the bridge. Later `schemas`/`call` requests report
    /// `"Editor bridge closed"`; the poison latch is untouched.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.transport.close();
    }

    /// Set the poison latch. Public so a future Phase 4 owner can poison on
    /// events the synchronous core cannot see (e.g. a dead worker thread);
    /// ordinary failures go through [`EditorBridge::submit`].
    pub fn poison(&mut self, message: String) {
        if self.failure.is_none() {
            self.failure = Some(message);
        }
    }

    /// One guarded request: closed/poison checks, lazy socket validation,
    /// then the transport call. Any failure poisons the bridge and returns
    /// the canonical message.
    fn submit(&mut self, expression: &str, args: &[Value]) -> Result<Value, String> {
        if self.closed || self.failure.is_some() {
            return Err(self
                .failure
                .clone()
                .unwrap_or_else(|| "Editor bridge closed".to_owned()));
        }
        if !self.validated {
            let socket = self.socket.clone().unwrap_or_default();
            if let Err(error) = validate_socket(&socket) {
                let message = format!("Neovim bridge unavailable: {error}");
                self.poison(message.clone());
                return Err(message);
            }
            self.validated = true;
        }
        match self.transport.exec(expression, args, self.timeout) {
            Ok(result) => Ok(result),
            Err(TransportError::Timeout) => {
                // Python's worker raises TimeoutError("Neovim request timed
                // out"); _submit wraps it as "Neovim bridge unavailable: ...".
                let message = "Neovim bridge unavailable: Neovim request timed out".to_owned();
                self.poison(message.clone());
                Err(message)
            }
            Err(TransportError::Failed(detail)) => {
                let message = format!("Neovim bridge unavailable: {detail}");
                self.poison(message.clone());
                Err(message)
            }
        }
    }
}

/// Build the `{"status": "unavailable", "error": ...}` envelope for a bridge
/// that cannot serve the request.
fn unavailable(error: &str) -> Value {
    envelope("unavailable", error)
}

/// Build the `{"status": "error", "error": ...}` envelope for a request that
/// was refused before touching the socket.
fn refused(error: &str) -> Value {
    envelope("error", error)
}

fn envelope(status: &str, error: &str) -> Value {
    let mut map = Map::new();
    map.insert("status".to_owned(), Value::from(status));
    map.insert("error".to_owned(), Value::from(error));
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::FakeTransport;
    use serde_json::json;

    fn bridge_with(transport: FakeTransport) -> EditorBridge<FakeTransport> {
        EditorBridge::new(
            Some("/tmp/phlow-editor-bridge-test.sock".to_owned()),
            Duration::from_secs(120),
            transport,
        )
        .unwrap()
    }

    #[test]
    fn bad_timeout_is_rejected() {
        let transport = FakeTransport::new();
        assert!(matches!(
            EditorBridge::new(None, Duration::from_millis(99), transport,),
            Err(BridgeError::BadTimeout(_))
        ));
    }

    #[test]
    fn timeout_edges_are_accepted() {
        for timeout in [TIMEOUT_MIN, TIMEOUT_MAX] {
            EditorBridge::new(None, timeout, FakeTransport::new()).unwrap();
        }
    }

    #[test]
    fn no_socket_returns_empty_schemas_without_transport() {
        let mut bridge =
            EditorBridge::new(None, Duration::from_secs(120), FakeTransport::new()).unwrap();
        assert!(bridge.schemas().is_empty());
        assert!(!bridge.is_poisoned());
    }

    #[test]
    fn socket_validated_lazily_on_first_schemas() {
        // An invalid socket poisons on first use, not at construction.
        let mut bridge = bridge_with(FakeTransport::new());
        assert!(!bridge.is_poisoned());
        assert!(bridge.schemas().is_empty());
        assert!(bridge.is_poisoned());
        let status = bridge.status();
        assert_eq!(status["status"], "unavailable");
        assert!(
            status["error"]
                .as_str()
                .unwrap()
                .starts_with("Neovim bridge unavailable:")
        );
    }

    #[test]
    fn poison_is_sticky() {
        let mut transport = FakeTransport::new();
        transport.fail_next(TransportError::Timeout);
        let mut bridge = bridge_with(transport);
        // Skip socket validation: mark validated via a first successful
        // path is impossible with a bad socket, so validate the latch via
        // poison() and check every entry point short-circuits.
        bridge.poison("boom".to_owned());
        assert!(bridge.schemas().is_empty());
        let reply = bridge.call("editor_lint", &json!({}));
        assert_eq!(reply["status"], "unavailable");
        assert_eq!(reply["error"], "boom");
        assert!(bridge.is_poisoned());
    }

    #[test]
    fn unknown_tool_is_refused_before_transport() {
        let mut bridge = bridge_with(FakeTransport::new());
        let reply = bridge.call("editor_pwn", &json!({}));
        assert_eq!(reply["status"], "error");
        assert_eq!(reply["error"], "Editor tool is not allowlisted");
        assert!(!bridge.is_poisoned());
    }

    #[test]
    fn non_object_args_are_refused() {
        let mut bridge = bridge_with(FakeTransport::new());
        let reply = bridge.call("editor_lint", &json!([1, 2]));
        assert_eq!(reply["status"], "error");
        assert_eq!(reply["error"], "Editor arguments must be an object");
    }

    #[test]
    fn close_reports_bridge_closed() {
        let mut bridge = bridge_with(FakeTransport::new());
        bridge.close();
        bridge.close(); // idempotent
        let reply = bridge.call("editor_lint", &json!({}));
        assert_eq!(reply["status"], "unavailable");
        assert_eq!(reply["error"], "Editor bridge closed");
    }

    /// A real Unix socket lets the whole request path run: schema caching,
    /// tool calls, and the timeout poison latch.
    mod live_socket {
        use super::*;
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        fn private_socket(tag: &str) -> (std::path::PathBuf, UnixListener) {
            let dir = std::env::temp_dir()
                .join(format!("phlow-editor-live-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("nvim.sock");
            let _ = std::fs::remove_file(&path);
            let listener = UnixListener::bind(&path).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            (path, listener)
        }

        fn function_schema(name: &str) -> Value {
            json!({
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            })
        }

        #[test]
        fn schemas_are_fetched_once_and_cached() {
            let (path, _listener) = private_socket("cache");
            let mut transport = FakeTransport::new();
            transport.queue_reply(Ok(json!([
                function_schema("editor_lint"),
                function_schema("editor_debug"),
            ])));
            let mut bridge = EditorBridge::new(
                Some(path.to_str().unwrap().to_owned()),
                Duration::from_secs(120),
                transport,
            )
            .unwrap();
            let first = bridge.schemas();
            let second = bridge.schemas();
            assert_eq!(first.len(), 2);
            assert_eq!(first, second);
            // One transport call total: the second schemas() hit the cache.
            // (The bridge owns the transport; count via a fresh check on the
            // cached names instead.)
            assert_eq!(first[0]["function"]["name"], "editor_lint");
            let status = bridge.status();
            assert_eq!(status["status"], "ok");
            assert_eq!(status["tools"], json!(["editor_lint", "editor_debug"]));
            assert_eq!(status["error"], Value::Null);
        }

        #[test]
        fn call_sends_name_and_args() {
            let (path, _listener) = private_socket("call");
            let mut transport = FakeTransport::new();
            transport.queue_reply(Ok(json!([function_schema("editor_lint")])));
            transport.queue_reply(Ok(json!({"status": "ok", "diagnostics": []})));
            let mut bridge = EditorBridge::new(
                Some(path.to_str().unwrap().to_owned()),
                Duration::from_secs(120),
                transport,
            )
            .unwrap();
            assert_eq!(bridge.schemas().len(), 1);
            let reply = bridge.call("editor_lint", &json!({"path": "a.rs"}));
            assert_eq!(reply["status"], "ok");
        }

        #[test]
        fn timeout_poisons_until_rebuilt() {
            let (path, _listener) = private_socket("timeout");
            let mut transport = FakeTransport::new();
            transport.queue_reply(Ok(json!([function_schema("editor_lint")])));
            transport.fail_next(TransportError::Timeout);
            let mut bridge = EditorBridge::new(
                Some(path.to_str().unwrap().to_owned()),
                Duration::from_secs(120),
                transport,
            )
            .unwrap();
            assert_eq!(bridge.schemas().len(), 1);
            // The tool call times out: canonical poison message.
            let reply = bridge.call("editor_lint", &json!({}));
            assert_eq!(reply["status"], "unavailable");
            assert_eq!(
                reply["error"],
                "Neovim bridge unavailable: Neovim request timed out"
            );
            assert!(bridge.is_poisoned());
            // Everything after short-circuits with the same message. Note the
            // schema cache survives: Python's schemas() returns the cache once
            // fetched, and status() stays "ok" with the poison in "error".
            let again = bridge.call("editor_lint", &json!({}));
            assert_eq!(
                again["error"],
                "Neovim bridge unavailable: Neovim request timed out"
            );
            assert_eq!(bridge.schemas().len(), 1);
            let status = bridge.status();
            assert_eq!(status["status"], "ok");
            assert_eq!(
                status["error"],
                "Neovim bridge unavailable: Neovim request timed out"
            );
        }

        #[test]
        fn transport_failure_poisons_with_detail() {
            let (path, _listener) = private_socket("failure");
            let mut transport = FakeTransport::new();
            transport.fail_next(TransportError::Failed("connection reset".to_owned()));
            let mut bridge = EditorBridge::new(
                Some(path.to_str().unwrap().to_owned()),
                Duration::from_secs(120),
                transport,
            )
            .unwrap();
            assert!(bridge.schemas().is_empty());
            assert_eq!(
                bridge.status()["error"],
                "Neovim bridge unavailable: connection reset"
            );
        }
    }
}
