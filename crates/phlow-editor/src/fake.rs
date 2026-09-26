//! In-memory [`EditorTransport`] for tests.
//!
//! The fake records every request (so tests can assert the exact Lua and
//! argument shape) and replays scripted replies or failures. It asserts that
//! only the two contract expressions cross the wire.

use std::collections::VecDeque;
use std::time::Duration;

use serde_json::Value;

use crate::bridge::EditorTransport;
use crate::contract::{CALL_LUA, SCHEMAS_LUA};
use crate::error::TransportError;

/// One request the bridge issued.
#[derive(Debug, Clone)]
pub struct RecordedCall {
    /// The Lua expression (one of [`SCHEMAS_LUA`], [`CALL_LUA`]).
    pub expression: String,
    /// The argument list (`[]` for schemas, `[name, args]` for calls).
    pub args: Vec<Value>,
    /// The timeout the bridge passed.
    pub timeout: Duration,
}

/// Scriptable in-memory transport.
#[derive(Debug, Default)]
pub struct FakeTransport {
    calls: Vec<RecordedCall>,
    replies: VecDeque<Result<Value, TransportError>>,
    closed: bool,
}

impl FakeTransport {
    /// A fake with no scripted replies: `exec` returns `Value::Null`.
    pub fn new() -> Self {
        FakeTransport::default()
    }

    /// Queue one reply (or failure) for the next `exec`.
    pub fn queue_reply(&mut self, reply: Result<Value, TransportError>) {
        self.replies.push_back(reply);
    }

    /// Queue a transport failure for the next `exec`.
    pub fn fail_next(&mut self, error: TransportError) {
        self.queue_reply(Err(error));
    }

    /// Every request the bridge has issued, in order.
    pub fn calls(&self) -> &[RecordedCall] {
        &self.calls
    }

    /// True once the bridge called [`EditorTransport::close`].
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

impl EditorTransport for FakeTransport {
    fn exec(
        &mut self,
        expression: &str,
        args: &[Value],
        timeout: Duration,
    ) -> Result<Value, TransportError> {
        assert!(
            expression == SCHEMAS_LUA || expression == CALL_LUA,
            "unexpected Lua crossed the bridge: {expression}"
        );
        if expression == SCHEMAS_LUA {
            assert!(args.is_empty(), "schemas takes no arguments");
        } else {
            assert!(
                args.len() == 2 && args[0].is_string() && args[1].is_object(),
                "call takes [name, args]"
            );
        }
        self.calls.push(RecordedCall {
            expression: expression.to_owned(),
            args: args.to_vec(),
            timeout,
        });
        match self.replies.pop_front() {
            Some(reply) => reply,
            None => Ok(Value::Null),
        }
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn records_schemas_call_shape() {
        let mut fake = FakeTransport::new();
        fake.exec(SCHEMAS_LUA, &[], Duration::from_secs(120))
            .unwrap();
        assert_eq!(fake.calls().len(), 1);
        assert_eq!(fake.calls()[0].expression, SCHEMAS_LUA);
        assert!(fake.calls()[0].args.is_empty());
    }

    #[test]
    fn records_tool_call_shape() {
        let mut fake = FakeTransport::new();
        let args = json!({"path": "a.rs"});
        fake.exec(
            CALL_LUA,
            &[json!("editor_lint"), args],
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(fake.calls()[0].args[0], "editor_lint");
        assert_eq!(fake.calls()[0].timeout, Duration::from_secs(5));
    }

    #[test]
    #[should_panic(expected = "unexpected Lua")]
    fn rejects_foreign_lua() {
        let mut fake = FakeTransport::new();
        let _ = fake.exec("os.execute('rm -rf /')", &[], Duration::from_secs(1));
    }
}
