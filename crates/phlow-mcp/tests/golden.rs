//! Byte-exact golden tests: replay Python-produced MCP frames through the
//! Rust server and compare every reply byte.
//!
//! The fixture `fixtures/mcp_golden.jsonl` was generated from the real
//! `flow/mcp.py` by `/tmp/gen_mcp_fixtures.py` (kept out of the repo; the
//! fixture is the checked-in artifact). Each line is
//! `{"request": <frame>, "reply": <frame|null>, "fresh"?: true,
//! "failing"?: true}`. Lines run through one server in order unless
//! `"fresh"` starts a new session; `"failing"` uses a runtime whose `run`
//! raises, exercising the `isError` tool-result path.

use phlow_mcp::{FakeRuntime, McpServer, RuntimeError};
use serde_json::Value;

struct GoldenPair {
    request: Option<String>,
    reply: Option<String>,
    fresh: bool,
    failing: bool,
}

fn load_pairs() -> Vec<GoldenPair> {
    let text = include_str!("fixtures/mcp_golden.jsonl");
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let pair: Value = serde_json::from_str(line).expect("fixture line is JSON");
            let obj = pair.as_object().expect("fixture line is an object");
            GoldenPair {
                request: obj
                    .get("request")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
                reply: obj
                    .get("reply")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
                fresh: obj
                    .get("fresh")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
                failing: obj
                    .get("failing")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
            }
        })
        .collect()
}

fn fake_runtime(failing: bool) -> FakeRuntime {
    let mut runtime = FakeRuntime::ok();
    if failing {
        runtime.fail_with = Some("boom".to_owned());
    }
    runtime
}

/// The contract-test stub's `run` returns exactly `{"status": "ok"}`.
fn contract_stub_runtime() -> FakeRuntime {
    fake_runtime(false)
}

#[test]
fn golden_frames_match_python_byte_for_byte() {
    let pairs = load_pairs();
    assert!(!pairs.is_empty(), "fixture must not be empty");
    let mut server: Option<McpServer<FakeRuntime>> = None;
    let mut compared = 0;
    for (index, pair) in pairs.iter().enumerate() {
        if pair.fresh || server.is_none() {
            let mut fresh = McpServer::new(contract_stub_runtime());
            if pair.failing {
                // The Python session completed the handshake before the
                // failing call; replay it so `ready` matches.
                handshake(&mut fresh);
                fresh.runtime_mut().fail_with = Some("boom".to_owned());
            }
            server = Some(fresh);
        }
        let server = server.as_mut().expect("server exists");
        let Some(request) = &pair.request else {
            // Synthetic server-side frames (the oversize error) are covered
            // by unit tests; there is no client request to replay.
            continue;
        };
        let reply = server.reply(request.as_bytes());
        // `reply` returns the frame without the trailing newline; the serve
        // loop appends it (exactly like the Python server), so compare with
        // the newline the fixture captured.
        let actual = reply.map(|mut bytes| {
            bytes.push(b'\n');
            String::from_utf8(bytes).expect("reply frames are ASCII")
        });
        assert_eq!(
            actual, pair.reply,
            "pair {index} diverged from Python;\nrequest:  {request}\nexpected: {:?}\nactual:   {actual:?}",
            pair.reply,
        );
        compared += 1;
    }
    assert!(
        compared >= 20,
        "expected most pairs to replay, got {compared}"
    );
}

/// The one contract-test stub behavior the golden file does not cover:
/// a runtime result that is not JSON-serializable. In Rust every `Value`
/// serializes, so the equivalent failure is a runtime whose result the
/// server must still wrap with the request id intact.
const HANDSHAKE_INIT: &str = r#"{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}"#;
const HANDSHAKE_NOTIF: &str = r#"{"jsonrpc": "2.0", "method": "notifications/initialized"}"#;

fn handshake(server: &mut McpServer<FakeRuntime>) {
    server.reply(HANDSHAKE_INIT.as_bytes());
    server.reply(HANDSHAKE_NOTIF.as_bytes());
    assert!(server.is_ready());
}
#[test]
fn runtime_nonjson_result_preserves_request_id() {
    // `serde_json::Value` cannot hold a non-serializable value, so the
    // closest reachable failure is a runtime error: the tool result keeps
    // the request id and reports isError.
    let mut server = McpServer::new({
        let mut runtime = FakeRuntime::ok();
        runtime.fail_with = Some("object() is not JSON serializable".to_owned());
        runtime
    });
    handshake(&mut server);
    let reply = server
        .reply(br#"{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "flow_run", "arguments": {"task": "x"}}}"#)
        .expect("a reply");
    let text = String::from_utf8(reply).expect("ASCII frame");
    assert!(text.contains(r#""id": 7"#), "{text}");
    assert!(text.contains(r#""isError": true"#), "{text}");
    assert!(text.contains("not JSON serializable"), "{text}");
}

/// `RuntimeError::new` carries the failure text the tool envelope reports.
#[test]
fn runtime_error_text() {
    let error = RuntimeError::new("boom");
    assert_eq!(error.to_string(), "boom");
}
