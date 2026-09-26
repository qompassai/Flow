//! Phase 4 runtime tests.
//!
//! Covers, against scripted fakes (no network, no Neovim):
//!
//! - prompt fidelity: vendored template bytes, production resolution,
//!   tool-call normalization, reviewer verdicts, diagnostic detection;
//! - the run loop: golden planner → coder → verification → reviewer,
//!   busy/closed/error reports, `status`, `check`;
//! - the tool authorization matrix via [`Runtime::call_tool`];
//! - editor freshness gates (dirty buffers, workspace mismatch);
//! - argument depth/node boundaries through the runtime's validator;
//! - both real transports ([`MsgpackTransport`] against a scripted socket
//!   peer, [`ReqwestTransport`] against a local HTTP server).

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use phlow_config::{FlowConfig, LoadOptions, load_config};
use phlow_editor::{CALL_LUA, EditorTransport, SCHEMAS_LUA, TransportError};
use phlow_llm::{LlmError, LlmTransport, load_system_prompt};
use phlow_runtime::transport::{MsgpackTransport, ReqwestTransport};
use phlow_runtime::{
    Runtime, has_error_diagnostics, normalize_tool_calls, reviewer_verdict, system_prompt_for_role,
};
use serde_json::{Map, Value, json};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Unique scratch directory (canonicalized: `/tmp` may be a symlink).
fn testdir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "phlow-runtime-test-{}-{}-{}",
        std::process::id(),
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("testdir");
    std::fs::canonicalize(&dir).expect("canonical testdir")
}

fn load_test_config(dir: &std::path::Path, toml: &str, trusted: bool) -> FlowConfig {
    std::fs::write(dir.join("config.toml"), toml).expect("config.toml");
    load_config(&LoadOptions {
        config_path: Some(dir.join("config.toml")),
        workspace: Some(dir.to_path_buf()),
        trusted,
        model: None,
    })
    .expect("test config loads")
}

/// Scripted model transport. Pops chat responses in order, records every
/// payload (so tests can inspect the tool messages the runtime sent), and
/// optionally blocks inside the first `post_chat` for the busy test.
/// Clone shares the same underlying script and log.
#[derive(Clone)]
struct ScriptLlm {
    shared: std::sync::Arc<std::sync::Mutex<ScriptState>>,
}

struct ScriptState {
    script: VecDeque<Value>,
    fallback: Value,
    payloads: Vec<Map<String, Value>>,
    gate: Option<Gate>,
    entered: bool,
}

struct Gate {
    entered: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}

impl ScriptLlm {
    fn new(script: Vec<Value>) -> Self {
        ScriptLlm {
            shared: std::sync::Arc::new(std::sync::Mutex::new(ScriptState {
                script: script.into(),
                fallback: chat_text("Fallback. Done."),
                payloads: Vec::new(),
                gate: None,
                entered: false,
            })),
        }
    }

    /// Block inside the first `post_chat` until the test releases it.
    fn gated() -> (Self, mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let llm = ScriptLlm::new(vec![chat_text("done")]);
        llm.shared.lock().unwrap().gate = Some(Gate {
            entered: entered_tx,
            release: release_rx,
        });
        (llm, entered_rx, release_tx)
    }

    /// The tool messages the runtime sent to the model, in order.
    fn tool_messages(&self) -> Vec<Value> {
        self.shared
            .lock()
            .unwrap()
            .payloads
            .iter()
            .flat_map(|payload| {
                payload
                    .get("messages")
                    .and_then(|messages| messages.as_array())
                    .cloned()
                    .unwrap_or_default()
            })
            .filter(|message| message.get("role") == Some(&Value::String("tool".to_owned())))
            .collect()
    }

    fn post_count(&self) -> usize {
        self.shared.lock().unwrap().payloads.len()
    }
}

impl LlmTransport for ScriptLlm {
    fn post_chat(
        &mut self,
        _base_url: &str,
        payload: &Map<String, Value>,
        _timeout: Duration,
    ) -> Result<Value, LlmError> {
        // Take the gate out of the lock before blocking on it: the test
        // thread needs no lock to release us.
        let (gate, response) = {
            let mut state = self.shared.lock().unwrap();
            state.payloads.push(payload.clone());
            let response = state
                .script
                .pop_front()
                .unwrap_or_else(|| state.fallback.clone());
            let gate = if state.entered {
                None
            } else {
                state.entered = true;
                state.gate.take()
            };
            (gate, response)
        };
        if let Some(gate) = gate {
            gate.entered.send(()).expect("entered signal");
            gate.release.recv().expect("release signal");
        }
        Ok(response)
    }

    fn get_tags(&mut self, _base_url: &str, _timeout: Duration) -> Result<Value, LlmError> {
        Ok(json!({"models": []}))
    }

    fn close(&mut self) {}
}

/// A chat completion with plain text content.
fn chat_text(content: &str) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": content}}]})
}

/// A chat completion whose message carries tool calls.
fn chat_tools(calls: Value) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": calls}}]})
}

fn tool_call(id: &str, name: &str, arguments: Value) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    })
}

/// Scripted editor transport: answers `SCHEMAS_LUA` / `CALL_LUA` from tables
/// and counts every crossing. Clone shares the count.
#[derive(Clone)]
struct FakeEditor {
    shared: std::sync::Arc<std::sync::Mutex<FakeState>>,
}

struct FakeState {
    schemas: Value,
    calls: HashMap<String, Value>,
    exec_count: usize,
}

impl FakeEditor {
    fn new() -> Self {
        FakeEditor {
            shared: std::sync::Arc::new(std::sync::Mutex::new(FakeState {
                schemas: json!([]),
                calls: HashMap::new(),
                exec_count: 0,
            })),
        }
    }

    /// Serve `editor_context` from `context`; any other tool gets `{"status": "ok"}`.
    fn with_context(context: Value) -> Self {
        let editor = FakeEditor::new();
        {
            let mut state = editor.shared.lock().unwrap();
            state.schemas = json!([
                {"type": "function",
                 "function": {"name": "editor_context",
                              "parameters": {"type": "object", "properties": {}}}},
                {"type": "function",
                 "function": {"name": "editor_debug",
                              "parameters": {"type": "object",
                                             "properties": {"action": {"type": "string"}}}}},
            ]);
            state.calls.insert("editor_context".to_owned(), context);
        }
        editor
    }

    fn exec_count(&self) -> usize {
        self.shared.lock().unwrap().exec_count
    }
}

impl EditorTransport for FakeEditor {
    fn exec(
        &mut self,
        expression: &str,
        args: &[Value],
        _timeout: Duration,
    ) -> Result<Value, TransportError> {
        let mut state = self.shared.lock().unwrap();
        state.exec_count += 1;
        if expression == SCHEMAS_LUA {
            Ok(state.schemas.clone())
        } else if expression == CALL_LUA {
            let name = args.first().and_then(|arg| arg.as_str()).unwrap_or("");
            Ok(state
                .calls
                .get(name)
                .cloned()
                .unwrap_or(json!({"status": "ok"})))
        } else {
            Err(TransportError::Failed("unexpected expression".to_owned()))
        }
    }

    fn close(&mut self) {}
}

/// A real Unix socket the bridge's validator accepts (absolute path, owned
/// socket, 0700 parent). The fake transport answers every call; the socket
/// just has to pass validation. Holds the listener so the path stays live.
struct SocketGuard {
    path: String,
    _listener: std::os::unix::net::UnixListener,
}

impl SocketGuard {
    fn bind(dir: &std::path::Path) -> Self {
        let sockdir = dir.join("sock");
        std::fs::create_dir_all(&sockdir).expect("sock dir");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&sockdir, std::fs::Permissions::from_mode(0o700))
            .expect("sock dir 0700");
        let path = sockdir.join("nvim.sock");
        let listener = std::os::unix::net::UnixListener::bind(&path).expect("bind test socket");
        SocketGuard {
            path: path.to_string_lossy().into_owned(),
            _listener: listener,
        }
    }
}

fn ok_context(dir: &std::path::Path) -> Value {
    json!({
        "status": "ok",
        "workspace": dir.to_string_lossy(),
        "workspace_snapshot_version": 1,
        "workspace_snapshot": {"files": {}},
        "dirty_buffers": [],
    })
}

type TestRuntime = Runtime<ScriptLlm, FakeEditor>;

fn runtime(dir: &std::path::Path, toml: &str, trusted: bool, llm: ScriptLlm) -> TestRuntime {
    let config = load_test_config(dir, toml, trusted);
    Runtime::new(config, llm, FakeEditor::new(), None).expect("runtime builds")
}

// ---------------------------------------------------------------------------
// Prompt fidelity
// ---------------------------------------------------------------------------

#[test]
fn vendored_prompt_is_byte_identical_to_flow_skills() {
    // Path derived from the crate location, not the developer's home: the
    // repo root is three levels above this crate's manifest dir.
    let shipped = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("flow/skills/system_prompt.md"),
    )
    .expect("flow skills prompt");
    assert_eq!(
        phlow_runtime::prompt::VENDORED_SYSTEM_PROMPT.as_bytes(),
        shipped.as_slice()
    );
}

#[test]
fn resolved_prompt_substitutes_tools_and_profile() {
    let prompt = load_system_prompt(
        Some(phlow_runtime::prompt::VENDORED_SYSTEM_PROMPT),
        "file_read: Read a file.",
        "rust",
    );
    assert!(
        prompt.starts_with("You are Flow"),
        "template preamble survives"
    );
    assert!(
        prompt.contains("file_read: Read a file."),
        "tool descriptions are substituted"
    );
    assert!(prompt.contains("rust"), "language profile is substituted");
    assert!(
        !prompt.contains("{tool_descriptions}") && !prompt.contains("{language_profile}"),
        "no placeholder leaks"
    );
}

#[test]
fn planner_prompt_is_read_only_and_names_role() {
    let prompt = system_prompt_for_role("planner");
    assert!(prompt.contains("You are read-only."));
    assert!(prompt.contains("planner"));
    assert!(
        prompt.contains("None loaded yet."),
        "empty tools render honestly"
    );
    assert!(
        !prompt.contains("{tool_descriptions}"),
        "no placeholder leaks"
    );
}

#[test]
fn missing_template_falls_back_to_default_prompt() {
    // `None` selects the built-in default template; the production path
    // passes `Some(VENDORED_SYSTEM_PROMPT)` so the vendored copy is used.
    let prompt = load_system_prompt(None, "", "");
    assert!(
        prompt.contains("None loaded yet."),
        "default template renders empty tools honestly"
    );
    assert!(
        !prompt.contains("Use only explicitly supplied tools within"),
        "default template, not the vendored production copy"
    );
}

#[test]
fn normalize_tool_calls_rejects_empty_id() {
    let calls = vec![json!({"id": "", "function": {"name": "file_read"}})];
    assert_eq!(
        normalize_tool_calls(&calls, "coder-1-0").unwrap_err(),
        "Invalid tool call id"
    );
}

#[test]
fn normalize_tool_calls_rejects_non_string_id() {
    let calls = vec![json!({"id": 42, "function": {"name": "file_read"}})];
    assert_eq!(
        normalize_tool_calls(&calls, "coder-1-0").unwrap_err(),
        "Invalid tool call id"
    );
}

#[test]
fn normalize_tool_calls_generates_missing_id() {
    let calls = vec![json!({"function": {"name": "file_read"}})];
    let normalized = normalize_tool_calls(&calls, "coder-1-0").expect("normalizes");
    assert_eq!(normalized[0]["id"], Value::String("coder-1-0-0".to_owned()));
    assert_eq!(normalized[0]["type"], Value::String("function".to_owned()));
}

#[test]
fn normalize_tool_calls_keeps_explicit_id() {
    let calls = vec![json!({"id": "call_9", "function": {"name": "file_read"}})];
    let normalized = normalize_tool_calls(&calls, "coder-1-0").expect("normalizes");
    assert_eq!(normalized[0]["id"], Value::String("call_9".to_owned()));
}

#[test]
fn normalize_tool_calls_rejects_duplicates() {
    let calls = vec![
        json!({"id": "dup", "function": {"name": "file_read"}}),
        json!({"id": "dup", "function": {"name": "file_list"}}),
    ];
    assert_eq!(
        normalize_tool_calls(&calls, "coder-1-0").unwrap_err(),
        "Duplicate tool call ids"
    );
}

#[test]
fn normalize_tool_calls_rejects_malformed_shape() {
    let calls = vec![json!({"id": "x"})];
    assert_eq!(
        normalize_tool_calls(&calls, "coder-1-0").unwrap_err(),
        "Malformed function call"
    );
}

#[test]
fn reviewer_verdict_accepts_structured_approval() {
    let verdict = reviewer_verdict(r#"{"approved": true, "summary": "Good.", "issues": []}"#);
    assert_eq!(verdict.get("approved"), Some(&Value::Bool(true)));
    assert_eq!(
        verdict.get("summary"),
        Some(&Value::String("Good.".to_owned()))
    );
    assert!(
        verdict.get("status").is_none(),
        "valid verdict sets no status"
    );
}

#[test]
fn reviewer_verdict_defaults_missing_issues() {
    let verdict = reviewer_verdict(r#"{"approved": false, "summary": "Bad."}"#);
    assert_eq!(verdict.get("approved"), Some(&Value::Bool(false)));
    assert_eq!(verdict.get("issues"), Some(&Value::Array(vec![])));
}

#[test]
fn reviewer_verdict_rejects_prose() {
    let verdict = reviewer_verdict("Looks good to me");
    assert_eq!(
        verdict.get("status"),
        Some(&Value::String("unverified".to_owned()))
    );
    assert_eq!(verdict.get("approved"), Some(&Value::Bool(false)));
    assert_eq!(
        verdict.get("error"),
        Some(&Value::String(
            "Reviewer must return a structured JSON verdict".to_owned()
        ))
    );
}

#[test]
fn reviewer_verdict_rejects_non_bool_approved() {
    let verdict = reviewer_verdict(r#"{"approved": 1, "issues": []}"#);
    assert_eq!(verdict.get("approved"), Some(&Value::Bool(false)));
}

#[test]
fn reviewer_verdict_stringifies_non_string_summary() {
    let verdict = reviewer_verdict(r#"{"approved": true, "summary": 42, "issues": []}"#);
    assert_eq!(
        verdict.get("summary"),
        Some(&Value::String("42".to_owned()))
    );
}

#[test]
fn reviewer_verdict_missing_summary_is_empty_string() {
    // Python: `str(verdict.get("summary", ""))` — missing is `""`, while an
    // explicit null degrades to `"None"`.
    let missing = reviewer_verdict(r#"{"approved": true, "issues": []}"#);
    assert_eq!(missing.get("summary"), Some(&Value::String(String::new())));
    let explicit_null = reviewer_verdict(r#"{"approved": true, "summary": null, "issues": []}"#);
    assert_eq!(
        explicit_null.get("summary"),
        Some(&Value::String("None".to_owned()))
    );
}

#[test]
fn has_error_diagnostics_flags_severity_one() {
    let evidence = vec![json!({"diagnostics": [{"severity": 1, "message": "boom"}]})];
    assert!(has_error_diagnostics(&evidence));
    let clean = vec![json!({"diagnostics": [{"severity": 2, "message": "warn"}]})];
    assert!(!has_error_diagnostics(&clean));
    assert!(!has_error_diagnostics(&[json!({"status": "ok"})]));
}

// ---------------------------------------------------------------------------
// Argument depth / node boundaries through the runtime's validator
// ---------------------------------------------------------------------------

/// Build the file_write schema's shape the runtime validates against, with a
/// nestable property for boundary probing.
fn nestable_schema(depth: u32, width: usize) -> (Value, Value) {
    let mut schema = json!({"type": "string"});
    for _ in 0..depth {
        schema = json!({"type": "object", "properties": {"a": schema}});
    }
    let mut args = Value::String("leaf".to_owned());
    for _ in 0..depth {
        let mut map = Map::new();
        map.insert("a".to_owned(), args);
        args = Value::Object(map);
    }
    let _ = width;
    (args, schema)
}

#[test]
fn argument_depth_16_ok_17_rejected() {
    let (args, schema) = nestable_schema(16, 0);
    assert!(phlow_mcp::validate_arguments(&args, &schema).is_ok());
    let (args, schema) = nestable_schema(17, 0);
    let error = phlow_mcp::validate_arguments(&args, &schema).unwrap_err();
    assert_eq!(error.to_string(), "Arguments exceed nesting depth 16");
}

#[test]
fn argument_nodes_boundary() {
    // Flat object: 1 root + N keys. 4096 nodes pass, 4097 fail.
    for (keys, ok) in [(4094, true), (4095, true), (4096, false)] {
        let mut properties = Map::new();
        let mut args = Map::new();
        for index in 0..keys {
            let key = format!("k{index}");
            properties.insert(key.clone(), json!({"type": "string"}));
            args.insert(key, Value::String("v".to_owned()));
        }
        let schema = json!({"type": "object", "properties": properties});
        let result = phlow_mcp::validate_arguments(&Value::Object(args), &schema);
        assert_eq!(result.is_ok(), ok, "keys={keys}");
        if !ok {
            assert_eq!(
                result.unwrap_err().to_string(),
                "Arguments exceed 4096 schema nodes"
            );
        }
    }
}

#[test]
fn runtime_surfaces_validator_errors_without_verified_key() {
    // The built-in tool schemas are flat, so depth itself is covered by the
    // boundary tests above; here the runtime must surface a schema violation
    // as `{"status": "error", "error": ...}` with no `"verified"` key.
    let dir = testdir("validator-shape");
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(vec![]));
    let result = rt.call_tool("file_list", &json!({"path": 42}), "coder");
    assert_eq!(
        result,
        json!({"status": "error", "error": "Expected string arguments"})
    );
}

// ---------------------------------------------------------------------------
// Tool authorization matrix (direct, via the public call_tool)
// ---------------------------------------------------------------------------

#[test]
fn planner_cannot_write_files() {
    let dir = testdir("matrix-planner");
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(vec![]));
    let result = rt.call_tool(
        "file_write",
        &json!({"path": "x.txt", "content": "hi"}),
        "planner",
    );
    assert_eq!(
        result,
        json!({"status": "error", "error": "Tool 'file_write' is unavailable for role planner"})
    );
    assert!(!dir.join("x.txt").exists());
}

#[test]
fn reviewer_cannot_write_files() {
    let dir = testdir("matrix-reviewer");
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    let result = rt.call_tool(
        "file_write",
        &json!({"path": "x.txt", "content": "hi"}),
        "reviewer",
    );
    assert_eq!(result["status"], json!("error"));
    assert!(
        result["error"]
            .as_str()
            .unwrap()
            .contains("unavailable for role reviewer")
    );
}

/// Golden script extended with a second coder/reviewer pass: the first
/// reviewer's turn is raced, so cycle 1 goes stale and the loop retries.
fn raced_script() -> Vec<Value> {
    let mut script = golden_script();
    script.push(chat_text("Done. No further changes needed."));
    script.push(chat_text(
        r#"{"approved": true, "summary": "Still good.", "issues": []}"#,
    ));
    script
}

#[test]
fn mutation_during_review_marks_verification_stale() {
    // The reviewer must run BEFORE the staleness gates: evidence that was
    // valid when the reviewer read it is marked stale afterwards, the run
    // retries, and the stale mark is synced into the history entry.
    let dir = testdir("review_race");
    let config = load_test_config(&dir, SMOKE_CHECK_TOML, true);
    let sink: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let llm = RacingLlm::new(&dir, raced_script(), Arc::clone(&sink));
    let mut rt = Runtime::new(config, llm, FakeEditor::new(), None).expect("runtime builds");
    let report = rt.run("Write hello.txt");

    assert_eq!(report["cycles"], json!(2));
    let first = &report["verification_history"][0];
    assert_eq!(first["status"], json!("stale"));
    assert_eq!(first["verified"], json!(false));
    assert!(
        first["reason"]
            .as_str()
            .unwrap_or("")
            .contains("modified during verification/review"),
        "unexpected reason: {first}"
    );
    // Cycle 2 saw a stable workspace and verified cleanly.
    assert_eq!(report["verification_history"][1]["status"], json!("ok"));
    assert_eq!(report["verified"], json!(true));
    // But the first reviewer saw valid evidence: its user message carried
    // the pre-stale `"status":"ok"` verification.
    let seen = sink.lock().expect("sink lock");
    assert_eq!(seen.len(), 2, "expected two reviewer turns");
    let first_context: Value =
        serde_json::from_str(&seen[0]).expect("reviewer user content is JSON");
    assert_eq!(
        first_context["context"]["verification"]["status"],
        json!("ok"),
        "first reviewer did not see pre-stale evidence"
    );
}

#[test]
fn untrusted_coder_cannot_write_files() {
    let dir = testdir("matrix-untrusted");
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(vec![]));
    let result = rt.call_tool(
        "file_write",
        &json!({"path": "x.txt", "content": "hi"}),
        "coder",
    );
    assert_eq!(result["status"], json!("error"));
    assert!(!dir.join("x.txt").exists());
}

#[test]
fn trusted_coder_writes_and_tracks() {
    let dir = testdir("matrix-trusted");
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    let result = rt.call_tool(
        "file_write",
        &json!({"path": "x.txt", "content": "hello\n"}),
        "coder",
    );
    assert_eq!(result["status"], json!("ok"));
    assert_eq!(
        std::fs::read_to_string(dir.join("x.txt")).unwrap(),
        "hello\n"
    );
}

#[test]
fn file_write_overwrites_like_python() {
    // Neither implementation versions writes: file_write replaces the file
    // and the write is tracked for the run's changed_files.
    let dir = testdir("overwrite");
    std::fs::write(dir.join("victim.txt"), "original\n").unwrap();
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    let result = rt.call_tool(
        "file_write",
        &json!({"path": "victim.txt", "content": "replaced\n"}),
        "coder",
    );
    assert_eq!(result["status"], json!("ok"));
    assert_eq!(
        std::fs::read_to_string(dir.join("victim.txt")).unwrap(),
        "replaced\n"
    );
}

#[test]
fn untrusted_coder_cannot_flow_check() {
    // flow_check is coder+trusted only, mirroring file_write's gate.
    let dir = testdir("nocheck");
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, false, ScriptLlm::new(vec![]));
    let result = rt.call_tool("flow_check", &json!({}), "coder");
    assert_eq!(
        result,
        json!({"status": "error", "error": "Tool 'flow_check' is unavailable for role coder"})
    );
}

#[test]
fn trusted_coder_flow_check_runs() {
    let dir = testdir("yescheck");
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, true, ScriptLlm::new(vec![]));
    let result = rt.call_tool("flow_check", &json!({}), "coder");
    assert_eq!(result["status"], json!("ok"));
    assert_eq!(result["verified"], json!(true));
}

#[test]
fn unknown_tool_reports_exact_shape() {
    let dir = testdir("unknown-tool");
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    let result = rt.call_tool("frobnicate", &json!({}), "coder");
    assert_eq!(
        result,
        json!({"status": "error", "error": "Tool 'frobnicate' is unavailable for role coder"})
    );
}

#[test]
fn non_object_arguments_rejected() {
    let dir = testdir("non-object-args");
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    let result = rt.call_tool("file_list", &json!([1, 2]), "coder");
    assert_eq!(
        result,
        json!({"status": "error", "error": "Tool arguments must be an object"})
    );
}

#[test]
fn call_tool_on_closed_runtime() {
    let dir = testdir("closed-tool");
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    rt.close();
    assert!(rt.is_closed());
    let result = rt.call_tool("file_list", &json!({}), "coder");
    assert_eq!(
        result,
        json!({"status": "error", "error": "Runtime closed"})
    );
}

// ---------------------------------------------------------------------------
// The run loop
// ---------------------------------------------------------------------------

/// An [`LlmTransport`] that tampers with the workspace on the reviewer's
/// turn, racing the review. Captured reviewer payloads go into the shared
/// sink so the test can prove the reviewer saw pre-stale evidence.
struct RacingLlm {
    inner: ScriptLlm,
    dir: PathBuf,
    raced: bool,
    sink: Arc<Mutex<Vec<String>>>,
}

impl RacingLlm {
    fn new(dir: &Path, script: Vec<Value>, sink: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            inner: ScriptLlm::new(script),
            dir: dir.to_path_buf(),
            raced: false,
            sink,
        }
    }

    /// True when this payload is addressed to the reviewer role.
    fn is_reviewer_turn(payload: &Map<String, Value>) -> bool {
        payload
            .get("messages")
            .and_then(|m| m.as_array())
            .and_then(|m| m.first())
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .is_some_and(|content| content.contains("You are Flow's reviewer"))
    }
}

impl LlmTransport for RacingLlm {
    fn post_chat(
        &mut self,
        base_url: &str,
        payload: &Map<String, Value>,
        timeout: Duration,
    ) -> Result<Value, LlmError> {
        if Self::is_reviewer_turn(payload) {
            if !self.raced {
                self.raced = true;
                // External edit racing the first reviewer: the post-review
                // fingerprint must observe this drift.
                fs::write(self.dir.join("hello.txt"), "tampered by another process\n")
                    .expect("race write works");
            }
            if let Some(content) = payload
                .get("messages")
                .and_then(|m| m.as_array())
                .and_then(|m| m.get(1))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                self.sink
                    .lock()
                    .expect("sink lock")
                    .push(content.to_owned());
            }
        }
        self.inner.post_chat(base_url, payload, timeout)
    }

    fn get_tags(&mut self, base_url: &str, timeout: Duration) -> Result<Value, LlmError> {
        self.inner.get_tags(base_url, timeout)
    }

    fn close(&mut self) {
        self.inner.close();
    }
}

/// Script for the golden run: planner plans, coder writes `hello.txt` via a
/// tool call, reviewer approves with a structured verdict.
fn golden_script() -> Vec<Value> {
    vec![
        chat_text("Plan: write a greeting file.\nI will create hello.txt."),
        chat_tools(json!([tool_call(
            "call_1",
            "file_write",
            json!({"path": "hello.txt", "content": "hello, world\n"})
        )])),
        chat_text("Done. Wrote hello.txt."),
        chat_text(r#"{"approved": true, "summary": "Looks good.", "issues": []}"#),
        // Second cycle (unreached when the first approves, but the script
        // must survive max_cycles if verification ever fails).
        chat_text("Done. Wrote hello.txt."),
        chat_text(r#"{"approved": true, "summary": "Looks good.", "issues": []}"#),
    ]
}

const SMOKE_CHECK_TOML: &str =
    "[checks.smoke]\ncmd = [\"true\"]\nkind = \"lint\"\nrequired = true\nfiletypes = [\"*\"]\n";

#[test]
fn run_golden_planner_coder_reviewer() {
    let dir = testdir("golden");
    let llm = ScriptLlm::new(golden_script());
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, true, llm.clone());
    let report = rt.run("write a greeting file");

    assert_eq!(report["status"], json!("ok"));
    assert_eq!(report["verified"], json!(true));
    assert_eq!(report["cycles"], json!(1));
    assert_eq!(report["changed_files"], json!(["hello.txt"]));
    assert_eq!(report["summary"], json!("Done. Wrote hello.txt."));

    let roles = report["roles"].as_array().expect("roles");
    assert_eq!(roles.len(), 3);
    assert_eq!(roles[0]["role"], json!("planner"));
    assert_eq!(roles[1]["role"], json!("coder"));
    assert_eq!(roles[2]["role"], json!("reviewer"));
    assert_eq!(roles[2]["approved"], json!(true));

    let verification = &report["verification"];
    assert_eq!(verification["verified"], json!(true));
    assert_eq!(verification["status"], json!("ok"));

    assert_eq!(
        std::fs::read_to_string(dir.join("hello.txt")).unwrap(),
        "hello, world\n"
    );

    // The tool result the model saw carries the exact success shape.
    let tools = llm.tool_messages();
    assert_eq!(tools.len(), 1);
    let content: Value =
        serde_json::from_str(tools[0]["content"].as_str().unwrap()).expect("tool content JSON");
    assert_eq!(content["status"], json!("ok"));

    // The reviewer saw the verification evidence in its context.
    assert!(llm.post_count() >= 4, "planner + coder x2 + reviewer");
}

/// After a run, `status()` exposes the last report without model contact.
#[test]
fn status_exposes_last_report() {
    let dir = testdir("last-report");
    let llm = ScriptLlm::new(golden_script());
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, true, llm.clone());
    let before = rt.status();
    assert_eq!(before["last_report"], json!(null));
    assert_eq!(before["busy"], json!(false));
    rt.run("write a greeting file");
    let after = rt.status();
    assert_eq!(after["last_report"]["status"], json!("ok"));
    assert_eq!(after["last_report"]["cycles"], json!(1));
}

// ---------------------------------------------------------------------------
// Run-loop edges: invalid/busy/closed/error, status, check
// ---------------------------------------------------------------------------

#[test]
fn run_rejects_empty_and_oversized_tasks() {
    let dir = testdir("bad-tasks");
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(vec![]));
    for task in ["", "   "] {
        let report = rt.run(task);
        assert_eq!(
            report,
            json!({"status": "error", "verified": false, "error": "Invalid or oversized task"})
        );
    }
    let oversized = "x".repeat(16_001);
    let report = rt.run(&oversized);
    assert_eq!(report["error"], json!("Invalid or oversized task"));
    // Exactly at the cap is accepted (and then fails at the backend, which
    // is the honest signal the task was attempted).
    let at_cap = "x".repeat(16_000);
    let report = rt.run(&at_cap);
    assert_ne!(report["error"], json!("Invalid or oversized task"));
}

#[test]
fn run_on_closed_runtime_reports_error() {
    let dir = testdir("run-closed");
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(vec![]));
    rt.close();
    let report = rt.run("do things");
    assert_eq!(report["status"], json!("error"));
    assert_eq!(report["verified"], json!(false));
    assert_eq!(report["error"], json!("Runtime closed"));
    assert_eq!(
        report["model_calls"],
        json!(0),
        "no model contact after close"
    );
    // Closing twice is idempotent.
    rt.close();
}

#[test]
fn run_reports_backend_failure_as_error() {
    struct FailingLlm;
    impl LlmTransport for FailingLlm {
        fn post_chat(
            &mut self,
            _base_url: &str,
            _payload: &Map<String, Value>,
            _timeout: Duration,
        ) -> Result<Value, LlmError> {
            Err(LlmError::Transport("connection refused".to_owned()))
        }
        fn get_tags(&mut self, _b: &str, _t: Duration) -> Result<Value, LlmError> {
            Err(LlmError::Transport("connection refused".to_owned()))
        }
        fn close(&mut self) {}
    }
    let dir = testdir("backend-fail");
    let config = load_test_config(&dir, "", false);
    let mut rt = Runtime::new(config, FailingLlm, FakeEditor::new(), None).expect("runtime builds");
    let report = rt.run("do things");
    assert_eq!(report["status"], json!("error"));
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("Backend/protocol failure"),
        "unexpected: {report}"
    );
    let roles = report["roles"].as_array().unwrap();
    assert_eq!(roles.len(), 1);
    assert_eq!(roles[0]["status"], json!("error"));
}

#[test]
fn single_writer_releases_between_runs() {
    // `run`/`check` take `&mut self`, so the borrow checker — not just the
    // runtime lock — prevents two concurrent runs through the public API;
    // the "Runtime busy" reports are fail-safes for reentrancy. What is
    // observable: `is_busy()` is false before and after, and a run executing
    // on a worker thread completes and releases the writer so the next run
    // proceeds.
    let dir = testdir("busy");
    let (llm, entered, release) = ScriptLlm::gated();
    let config = load_test_config(&dir, "", false);
    let rt = Runtime::new(config, llm.clone(), FakeEditor::new(), None).expect("runtime builds");
    assert!(!rt.is_busy());
    let worker = thread::scope(|scope| {
        let handle = scope.spawn(|| {
            // Move the runtime into the worker: the main thread cannot
            // borrow it concurrently, which is exactly the single-writer
            // guarantee the busy error defends.
            let mut rt = rt;
            let report = rt.run("long task");
            (rt, report)
        });
        entered
            .recv_timeout(Duration::from_secs(10))
            .expect("worker entered run");
        // While the worker is inside run(), the main thread has no access:
        // attempting a second run would require a second `&mut`, which does
        // not exist. Release and rejoin instead.
        release.send(()).expect("release worker");
        handle.join().expect("worker done")
    });
    let (rt, report) = worker;
    assert!(!rt.is_busy(), "writer released after run");
    assert_ne!(
        report["status"],
        json!("error"),
        "gated run completed: {report}"
    );
    // A subsequent run proceeds normally (no checks configured, so the
    // reviewer verdict falls back to prose and the run is "unverified",
    // never a protocol error).
    let llm2 = ScriptLlm::new(vec![chat_text("second done")]);
    let config2 = load_test_config(&dir, "", false);
    let mut rt2 = Runtime::new(config2, llm2, FakeEditor::new(), None).expect("runtime builds");
    let second = rt2.run("follow-up");
    assert!(
        ["ok", "unverified", "failed"].contains(&second["status"].as_str().unwrap_or("?")),
        "second run proceeds: {second}"
    );
    let _ = rt;
}

#[test]
fn chat_accepts_missing_content_and_tool_calls() {
    // The model may omit both fields; both default ("" and []).
    let dir = testdir("chat-lenient");
    let script = vec![
        chat_text("plan"),
        json!({"choices": [{"message": {"role": "assistant"}}]}),
        chat_text(r#"{"approved": true, "summary": "ok", "issues": []}"#),
        chat_text("done"),
        chat_text(r#"{"approved": true, "summary": "ok", "issues": []}"#),
    ];
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(script));
    let report = rt.run("lenient task");
    // No checks configured: verification can never pass, but the loop must
    // complete without a protocol failure.
    assert!(
        !report["error"]
            .as_str()
            .unwrap_or("")
            .contains("Backend/protocol failure"),
        "unexpected protocol failure: {report}"
    );
}

#[test]
fn chat_rejects_wrong_shaped_message() {
    let dir = testdir("chat-shape");
    let script = vec![json!({"choices": [{"message": "not-an-object"}]})];
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(script));
    let report = rt.run("shapely task");
    assert_eq!(report["status"], json!("error"));
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("Model message must be an object"),
        "unexpected: {report}"
    );
}

#[test]
fn chat_rejects_truthy_wrong_types() {
    let dir = testdir("chat-types");
    // Truthy non-string content is an error; falsy content defaults to "".
    let script = vec![json!({"choices": [{"message": {"role": "assistant", "content": 42}}]})];
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(script));
    let report = rt.run("typed task");
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("Invalid model content/tool_calls shape"),
        "unexpected: {report}"
    );
}

#[test]
fn status_shape_without_editor() {
    let dir = testdir("status");
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, false, ScriptLlm::new(vec![]));
    let status = rt.status();
    assert_eq!(status["status"], json!("ok"));
    assert_eq!(status["workspace"], json!(dir.to_string_lossy()));
    assert_eq!(status["trusted"], json!(false));
    assert_eq!(status["backend"]["type"], json!("ollama"));
    assert_eq!(status["capabilities"]["single_writer"], json!(true));
    assert_eq!(status["capabilities"]["arbitrary_commands"], json!(false));
    assert_eq!(status["capabilities"]["plugins"], json!(false));
    assert_eq!(status["editor"]["attached"], json!(false));
    assert_eq!(status["busy"], json!(false));
    assert!(status["models"]["planner"].as_str().is_some());
    assert!(status["models"]["coder"].as_str().is_some());
    assert!(status["models"]["reviewer"].as_str().is_some());
    let checks = status["checks"].as_array().unwrap();
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0]["name"], json!("smoke"));
}

#[test]
fn check_runs_configured_check() {
    let dir = testdir("check-smoke");
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, true, ScriptLlm::new(vec![]));
    let report = rt.check(None);
    assert_eq!(report["status"], json!("ok"));
    assert_eq!(report["verified"], json!(true));
    let report = rt.check(Some("smoke"));
    assert_eq!(report["status"], json!("ok"));
}

#[test]
fn check_unknown_name_is_unverified() {
    let dir = testdir("check-unknown");
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    let report = rt.check(Some("nope"));
    assert_eq!(report["status"], json!("unverified"));
}

#[test]
fn check_untrusted_never_verifies() {
    let dir = testdir("check-untrusted");
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, false, ScriptLlm::new(vec![]));
    let report = rt.check(None);
    assert_eq!(report["verified"], json!(false));
}

#[test]
fn check_on_closed_runtime() {
    let dir = testdir("check-closed");
    let mut rt = runtime(&dir, "", false, ScriptLlm::new(vec![]));
    rt.close();
    let report = rt.check(None);
    assert_eq!(
        report,
        json!({"status": "error", "verified": false, "error": "Runtime closed"})
    );
}

/// A scripted run that writes `second.txt` instead of `hello.txt`.
fn second_file_script() -> Vec<Value> {
    vec![
        chat_text("Plan: write a different file.\nI will create second.txt."),
        chat_tools(json!([tool_call(
            "call_1",
            "file_write",
            json!({"path": "second.txt", "content": "second run\n"})
        )])),
        chat_text("Done. Wrote second.txt."),
        chat_text(r#"{"approved": true, "summary": "Looks good.", "issues": []}"#),
        chat_text("Done. Wrote second.txt."),
        chat_text(r#"{"approved": true, "summary": "Looks good.", "issues": []}"#),
    ]
}

#[test]
fn second_run_sees_only_its_own_files() {
    // Python clears the workspace ledger per run; the report must not
    // accumulate the previous run's files. The runs write DIFFERENT files:
    // `changed_files` is a set, so identical names could not prove the
    // first run's entry was cleared.
    let dir = testdir("two-runs");
    let llm = ScriptLlm::new([golden_script(), second_file_script()].concat());
    let mut rt = runtime(&dir, SMOKE_CHECK_TOML, true, llm.clone());
    let first = rt.run("first");
    assert_eq!(first["changed_files"], json!(["hello.txt"]));
    let second = rt.run("second");
    assert_eq!(
        second["changed_files"],
        json!(["second.txt"]),
        "second run must not accumulate run-one files"
    );
}

// ---------------------------------------------------------------------------
// Editor freshness gates and the editor_debug action matrix
// ---------------------------------------------------------------------------

fn runtime_with_editor(
    dir: &std::path::Path,
    toml: &str,
    trusted: bool,
    llm: ScriptLlm,
    editor: FakeEditor,
) -> (TestRuntime, SocketGuard) {
    let guard = SocketGuard::bind(dir);
    let config = load_test_config(dir, toml, trusted);
    let runtime =
        Runtime::new(config, llm, editor, Some(guard.path.clone())).expect("runtime builds");
    (runtime, guard)
}

#[test]
fn editor_attached_status() {
    let dir = testdir("editor-status");
    let editor = FakeEditor::with_context(ok_context(&dir));
    let (mut rt, _guard) =
        runtime_with_editor(&dir, "", false, ScriptLlm::new(vec![]), editor.clone());
    let status = rt.status();
    assert_eq!(status["editor"]["attached"], json!(true));
    assert_eq!(
        status["editor"]["context"]["status"],
        json!("ok"),
        "status surfaces the validated editor context"
    );
    assert!(
        editor.exec_count() > 0,
        "status actually consulted the editor"
    );
}

#[test]
fn editor_workspace_mismatch_blocks_tools() {
    let dir = testdir("editor-mismatch");
    let other = testdir("editor-mismatch-other");
    let editor = FakeEditor::with_context(ok_context(&other));
    let (mut rt, _guard) = runtime_with_editor(&dir, "", true, ScriptLlm::new(vec![]), editor);
    let result = rt.call_tool("editor_context", &json!({}), "coder");
    assert_eq!(result["status"], json!("error"));
    assert!(
        result["error"]
            .as_str()
            .unwrap()
            .contains("workspace mismatch"),
        "unexpected: {result}"
    );
}

#[test]
fn editor_dirty_buffers_make_checks_stale() {
    // Python gates verification — not file tools — on dirty buffers:
    // unsaved editor state would not be checked.
    let dir = testdir("editor-dirty");
    let mut context = ok_context(&dir);
    context["dirty_buffers"] = json!(["main.rs"]);
    let editor = FakeEditor::with_context(context);
    let (mut rt, _guard) =
        runtime_with_editor(&dir, SMOKE_CHECK_TOML, true, ScriptLlm::new(vec![]), editor);
    let report = rt.check(None);
    assert_eq!(report["status"], json!("stale"));
    assert_eq!(report["verified"], json!(false));
    assert_eq!(
        report["reason"],
        json!("Unsaved editor buffers would not be checked; save manually")
    );
}

#[test]
fn editor_context_fetch_failure_fails_closed() {
    let dir = testdir("editor-fail");
    let editor = FakeEditor::with_context(json!({"status": "error", "error": "nvim exploded"}));
    let (mut rt, _guard) = runtime_with_editor(&dir, "", true, ScriptLlm::new(vec![]), editor);
    let result = rt.call_tool("editor_context", &json!({}), "coder");
    assert_eq!(result["status"], json!("error"));
    assert!(
        result["error"]
            .as_str()
            .unwrap()
            .contains("Editor context unavailable"),
        "unexpected: {result}"
    );
}

#[test]
fn editor_debug_schema_admits_status_only() {
    // The model-facing schema narrows `action` to `status`, mirroring
    // `runtime.py`'s `schemas()`; validation runs before the dispatch gate,
    // so every other action is rejected with the exact Python message and
    // `launch`/`run` stay manual-only.
    let dir = testdir("editor-debug-matrix");
    let editor = FakeEditor::with_context(ok_context(&dir));
    let (mut rt, _guard) = runtime_with_editor(&dir, "", true, ScriptLlm::new(vec![]), editor);
    let schemas = rt.schemas("coder");
    let debug = schemas
        .iter()
        .find(|schema| schema["function"]["name"] == json!("editor_debug"))
        .expect("editor_debug schema");
    assert_eq!(
        debug["function"]["parameters"]["properties"]["action"],
        json!({"type": "string", "enum": ["status"]})
    );
    for args in [json!({"action": "status"}), json!({})] {
        let result = rt.call_tool("editor_debug", &args, "coder");
        assert_eq!(result["status"], json!("ok"), "args {args}");
    }
    for action in [
        "list",
        "config",
        "discover",
        "launch",
        "run",
        "STATUS",
        "dap_state",
    ] {
        let result = rt.call_tool("editor_debug", &json!({"action": action}), "coder");
        assert_eq!(result["status"], json!("error"), "action {action}");
        assert_eq!(
            result["error"],
            json!("Argument is not one of the allowed values"),
            "action {action}"
        );
    }
}

#[test]
fn editor_tools_unavailable_without_socket() {
    let dir = testdir("editor-nosocket");
    let mut rt = runtime(&dir, "", true, ScriptLlm::new(vec![]));
    let result = rt.call_tool("editor_context", &json!({}), "coder");
    assert_eq!(
        result,
        json!({"status": "error", "error": "Tool 'editor_context' is unavailable for role coder"})
    );
}

#[test]
fn planner_and_reviewer_get_read_only_editor_tools() {
    // READ_ONLY_EDITOR_NAMES (editor_context, editor_debug, ...) are visible
    // to every role; only file_write/flow_check are coder+trusted. This
    // mirrors Python's `role == "coder" and trusted or name in
    // READ_ONLY_EDITOR_NAMES` gate.
    let dir = testdir("editor-roles");
    let editor = FakeEditor::with_context(ok_context(&dir));
    let (mut rt, _guard) = runtime_with_editor(&dir, "", true, ScriptLlm::new(vec![]), editor);
    for role in ["planner", "reviewer"] {
        let context = rt.call_tool("editor_context", &json!({}), role);
        assert_eq!(context["status"], json!("ok"), "role {role}");
        let write = rt.call_tool(
            "file_write",
            &json!({"path": "x.txt", "content": "hi"}),
            role,
        );
        assert_eq!(write["status"], json!("error"), "role {role}");
    }
}

// ---------------------------------------------------------------------------
// Transport tests: MsgpackTransport against a scripted socket peer
// ---------------------------------------------------------------------------

/// A minimal msgpack-RPC peer speaking the exact framing the runtime uses:
/// raw self-delimiting msgpack over the stream (no length prefix, like
/// Neovim). Reads `[0, id, "nvim_exec_lua", params]` requests, writes
/// `[1, id, error, result]` responses.
struct SocketPeer {
    listener: std::os::unix::net::UnixListener,
    path: String,
}

struct PeerConn {
    stream: std::os::unix::net::UnixStream,
}

impl SocketPeer {
    fn pair(dir: &std::path::Path) -> Self {
        let path = dir.join("peer.sock");
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        SocketPeer {
            listener,
            path: path.to_string_lossy().into_owned(),
        }
    }

    fn accept(&self) -> PeerConn {
        let (stream, _) = self.listener.accept().unwrap();
        PeerConn { stream }
    }
}

impl PeerConn {
    /// Read one self-delimiting msgpack value from the stream.
    fn read_value(&mut self) -> rmpv::Value {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            {
                let mut slice = buffer.as_slice();
                if let Ok(value) = rmpv::decode::read_value(&mut slice) {
                    return value;
                }
            }
            let read = self.stream.read(&mut chunk).unwrap();
            assert!(read > 0, "peer: socket closed mid-request");
            buffer.extend_from_slice(&chunk[..read]);
        }
    }

    fn write_value(&mut self, value: &rmpv::Value) {
        let mut frame = Vec::new();
        rmpv::encode::write_value(&mut frame, value).unwrap();
        self.stream.write_all(&frame).unwrap();
    }

    /// Write raw bytes in one syscall, so the peer's reader observes them
    /// as a single chunk (pipelined frames, notifications).
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.stream.write_all(bytes).unwrap();
    }

    /// Read the next request, check its id/method, and answer `result`.
    fn answer_ok(&mut self, id: u64, result: rmpv::Value) {
        let request = self.read_value();
        assert_eq!(request[0], rmpv::Value::from(0), "request type");
        assert_eq!(request[1], rmpv::Value::from(id), "request id");
        assert_eq!(request[2], rmpv::Value::from("nvim_exec_lua"), "method");
        self.write_value(&rmpv::Value::Array(vec![
            1.into(),
            id.into(),
            rmpv::Value::Nil,
            result,
        ]));
    }

    /// Read the next request and answer with a remote (Lua-side) error.
    fn answer_remote_error(&mut self, id: u64, detail: &str) {
        let request = self.read_value();
        assert_eq!(request[1], rmpv::Value::from(id), "request id");
        self.write_value(&rmpv::Value::Array(vec![
            1.into(),
            id.into(),
            rmpv::Value::from(detail),
            rmpv::Value::Nil,
        ]));
    }
}

#[test]
fn msgpack_transport_round_trip() {
    let dir = testdir("msgpack");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut conn = peer.accept();
        conn.answer_ok(0, rmpv::Value::from("pong"));
        // Second call on the same connection: ids stay ordered.
        conn.answer_ok(1, rmpv::Value::from(42u64));
    });
    let first = transport
        .exec(SCHEMAS_LUA, &[json!("a")], Duration::from_secs(5))
        .expect("first call");
    assert_eq!(first, json!("pong"));
    let second = transport
        .exec(CALL_LUA, &[], Duration::from_secs(5))
        .expect("second call");
    assert_eq!(second, json!(42u64));
    worker.join().unwrap();
    transport.close();
}

#[test]
fn msgpack_transport_rejects_binary() {
    let dir = testdir("msgpack-binary");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut conn = peer.accept();
        conn.answer_ok(0, rmpv::Value::Binary(vec![0xff, 0xfe]));
    });
    let error = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .unwrap_err();
    assert_eq!(
        error,
        TransportError::Failed("binary data in response".to_owned())
    );
    worker.join().unwrap();
}

#[test]
fn msgpack_transport_surfaces_remote_error() {
    let dir = testdir("msgpack-remote-error");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut conn = peer.accept();
        conn.answer_remote_error(0, "E5108: boom");
    });
    let error = transport
        .exec(CALL_LUA, &[], Duration::from_secs(5))
        .unwrap_err();
    assert_eq!(
        error,
        TransportError::Failed("nvim_exec_lua failed: E5108: boom".to_owned())
    );
    worker.join().unwrap();
}

#[test]
fn msgpack_transport_rejects_id_mismatch() {
    let dir = testdir("msgpack-id");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut conn = peer.accept();
        let request = conn.read_value();
        assert_eq!(request[1], rmpv::Value::from(0u64));
        // Answer with the wrong id: the transport must fail closed.
        conn.write_value(&rmpv::Value::Array(vec![
            1.into(),
            999u64.into(),
            rmpv::Value::Nil,
            rmpv::Value::from("stale"),
        ]));
    });
    let error = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .unwrap_err();
    assert_eq!(
        error,
        TransportError::Failed("msgpack-RPC reply id mismatch".to_owned())
    );
    worker.join().unwrap();
}

#[test]
fn msgpack_transport_reconnects_after_id_mismatch() {
    // A wrong-id reply desynchronizes the stream: the transport must drop
    // it so the next request dials a fresh connection instead of eating
    // the stale reply.
    let dir = testdir("msgpack-id-reconnect");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut first = peer.accept();
        let request = first.read_value();
        assert_eq!(request[1], rmpv::Value::from(0u64));
        first.write_value(&rmpv::Value::Array(vec![
            1.into(),
            999u64.into(),
            rmpv::Value::Nil,
            rmpv::Value::from("stale"),
        ]));
        let mut second = peer.accept();
        second.answer_ok(1, rmpv::Value::from("recovered"));
    });
    let error = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .unwrap_err();
    assert_eq!(
        error,
        TransportError::Failed("msgpack-RPC reply id mismatch".to_owned())
    );
    let recovered = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .expect("reconnect after id mismatch");
    assert_eq!(recovered, json!("recovered"));
    worker.join().unwrap();
}

#[test]
fn msgpack_transport_rejects_frame_just_over_cap() {
    // A complete response frame larger than 8 MiB: the old decode-first
    // path returned it without ever consulting the frame cap.
    let dir = testdir("msgpack-frame-cap");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut conn = peer.accept();
        let request = conn.read_value();
        assert_eq!(request[1], rmpv::Value::from(0u64));
        let payload = "x".repeat(8 * 1024 * 1024);
        conn.write_value(&rmpv::Value::Array(vec![
            1.into(),
            0u64.into(),
            rmpv::Value::Nil,
            rmpv::Value::from(payload),
        ]));
    });
    let error = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(30))
        .unwrap_err();
    assert_eq!(
        error,
        TransportError::Failed("Neovim response frame exceeded 8 MiB".to_owned())
    );
    worker.join().unwrap();
}

#[test]
fn msgpack_transport_skips_interleaved_notification() {
    // Neovim may send `[2, method, params]` notifications before our
    // response; the transport skips them instead of failing.
    let dir = testdir("msgpack-notify");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut conn = peer.accept();
        let _ = conn.read_value();
        // Notification and response in a single write: the reader must
        // consume both frames and return the response.
        let mut bytes = Vec::new();
        rmpv::encode::write_value(
            &mut bytes,
            &rmpv::Value::Array(vec![
                2.into(),
                rmpv::Value::from("rose_event"),
                rmpv::Value::Array(vec![]),
            ]),
        )
        .expect("notification encodes");
        rmpv::encode::write_value(
            &mut bytes,
            &rmpv::Value::Array(vec![
                1.into(),
                0u64.into(),
                rmpv::Value::Nil,
                rmpv::Value::from("fine"),
            ]),
        )
        .expect("response encodes");
        conn.write_bytes(&bytes);
    });
    let result = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .expect("notification skipped");
    assert_eq!(result, json!("fine"));
    worker.join().unwrap();
}

#[test]
fn msgpack_transport_preserves_pipelined_bytes() {
    // Bytes arriving after our response (a trailing notification) must not
    // be discarded: the next request still works on the same stream.
    let dir = testdir("msgpack-pipeline");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        let mut conn = peer.accept();
        let _ = conn.read_value();
        let mut bytes = Vec::new();
        rmpv::encode::write_value(
            &mut bytes,
            &rmpv::Value::Array(vec![
                1.into(),
                0u64.into(),
                rmpv::Value::Nil,
                rmpv::Value::from("first"),
            ]),
        )
        .expect("response encodes");
        // Trailing notification pipelined with the response.
        rmpv::encode::write_value(
            &mut bytes,
            &rmpv::Value::Array(vec![
                2.into(),
                rmpv::Value::from("rose_event"),
                rmpv::Value::Array(vec![]),
            ]),
        )
        .expect("notification encodes");
        conn.write_bytes(&bytes);
        // Second request on the same connection (`answer_ok` reads it).
        conn.answer_ok(1, rmpv::Value::from("second"));
    });
    let first = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .expect("first call works");
    assert_eq!(first, json!("first"));
    let second = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .expect("second call works");
    assert_eq!(second, json!("second"));
    worker.join().unwrap();
}

#[test]
fn msgpack_transport_timeout_recovers_next_call() {
    let dir = testdir("msgpack-timeout");
    let peer = SocketPeer::pair(&dir);
    let mut transport = MsgpackTransport::new(&peer.path);
    let worker = thread::spawn(move || {
        // First connection: read the request, then stay silent WITHOUT
        // closing — the client must hit its deadline, drop the stream, and
        // reconnect. (Closing the socket would surface EOF before the
        // timeout; that path is covered by the bridge's own tests.)
        let mut first = peer.accept();
        let _ = first.read_value();
        let mut second = peer.accept();
        second.answer_ok(1, rmpv::Value::from("recovered"));
        // `first` stays open until here so the client's first call hits its
        // deadline instead of seeing EOF; dropping it now is harmless — the
        // client already abandoned that stream at timeout.
    });
    let error = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_millis(200))
        .unwrap_err();
    assert_eq!(error, TransportError::Timeout);
    let second = transport
        .exec(SCHEMAS_LUA, &[], Duration::from_secs(5))
        .expect("recovered call");
    assert_eq!(second, json!("recovered"));
    worker.join().unwrap();
    transport.close();
}

// ---------------------------------------------------------------------------
// Transport tests: ReqwestTransport against a hand-rolled local HTTP server
// ---------------------------------------------------------------------------

/// A one-shot HTTP/1.1 server: reads the request (headers + Content-Length
/// body), hands it to `respond`, writes the raw response bytes.
struct HttpPeer {
    listener: std::net::TcpListener,
    addr: std::net::SocketAddr,
}

impl HttpPeer {
    fn bind() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        HttpPeer { listener, addr }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Serve exactly one request on a background thread.
    fn serve_once(
        self,
        respond: impl FnOnce(&str, &str, &[u8]) -> Vec<u8> + Send + 'static,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let (mut stream, _) = self.listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            let response = respond(&request.method, &request.path, &request.body);
            stream.write_all(&response).unwrap();
            stream.flush().unwrap();
        })
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut std::net::TcpStream) -> HttpRequest {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        head.push(byte[0]);
        assert!(head.len() < 1 << 20, "request head too large");
    }
    let head = String::from_utf8(head).unwrap();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap().to_owned();
    let path = parts.next().unwrap().to_owned();
    let mut content_length = 0usize;
    for line in lines {
        // Hyper writes header names in lowercase.
        if line.to_ascii_lowercase().starts_with("content-length:") {
            content_length = line.split(':').nth(1).unwrap_or("").trim().parse().unwrap();
        }
    }
    let mut body = vec![0u8; content_length];
    stream.read_exact(&mut body).unwrap();
    HttpRequest { method, path, body }
}

fn http_response(status: u16, reason: &str, body: &[u8]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

#[test]
fn http_transport_round_trip() {
    let peer = HttpPeer::bind();
    let base_url = peer.base_url();
    let server = peer.serve_once(|method, path, body| {
        assert_eq!(method, "POST");
        assert_eq!(path, "/v1/chat/completions");
        let payload: Value = serde_json::from_slice(body).unwrap();
        assert_eq!(payload["model"], json!("test-model"));
        http_response(200, "OK", br#"{"choices": []}"#)
    });
    let mut transport = ReqwestTransport::new().expect("transport builds");
    let mut payload = Map::new();
    payload.insert("model".to_owned(), json!("test-model"));
    let result = transport
        .post_chat(&base_url, &payload, Duration::from_secs(5))
        .expect("round trip");
    assert_eq!(result, json!({"choices": []}));
    server.join().unwrap();
}

#[test]
fn http_transport_surfaces_http_error_status() {
    let peer = HttpPeer::bind();
    let base_url = peer.base_url();
    let server = peer.serve_once(|_, _, _| http_response(500, "Internal Server Error", b"boom"));
    let mut transport = ReqwestTransport::new().expect("transport builds");
    let error = transport
        .post_chat(&base_url, &Map::new(), Duration::from_secs(5))
        .unwrap_err();
    match error {
        LlmError::Transport(detail) => {
            assert!(
                detail.contains("Ollama request failed"),
                "unexpected: {detail}"
            )
        }
        other => panic!("unexpected error: {other:?}"),
    }
    server.join().unwrap();
}

#[test]
fn http_transport_rejects_oversized_body() {
    let peer = HttpPeer::bind();
    let base_url = peer.base_url();
    // One byte over the 2 MiB cap.
    let big = vec![b'x'; 2 * 1024 * 1024 + 1];
    let server = peer.serve_once(move |_, _, _| http_response(200, "OK", &big));
    let mut transport = ReqwestTransport::new().expect("transport builds");
    let error = transport
        .post_chat(&base_url, &Map::new(), Duration::from_secs(10))
        .unwrap_err();
    assert!(
        matches!(error, LlmError::ResponseTooLarge { limit } if limit == 2 * 1024 * 1024),
        "unexpected: {error:?}"
    );
    server.join().unwrap();
}

#[test]
fn http_transport_rejects_invalid_json() {
    let peer = HttpPeer::bind();
    let base_url = peer.base_url();
    let server = peer.serve_once(|_, _, _| http_response(200, "OK", b"not json"));
    let mut transport = ReqwestTransport::new().expect("transport builds");
    let error = transport
        .post_chat(&base_url, &Map::new(), Duration::from_secs(5))
        .unwrap_err();
    match error {
        LlmError::BadJson(detail) => {
            assert!(
                detail.starts_with("Ollama returned invalid JSON: "),
                "unexpected: {detail}"
            );
            assert_eq!(
                detail.matches("Ollama returned invalid JSON").count(),
                1,
                "prefix must not double"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
    server.join().unwrap();
}

#[test]
fn http_transport_get_tags_hits_api_tags() {
    let peer = HttpPeer::bind();
    let base_url = peer.base_url();
    let server = peer.serve_once(|method, path, _| {
        assert_eq!(method, "GET");
        assert_eq!(path, "/api/tags");
        http_response(200, "OK", br#"{"models": []}"#)
    });
    let mut transport = ReqwestTransport::new().expect("transport builds");
    let result = transport
        .get_tags(&base_url, Duration::from_secs(5))
        .expect("tags");
    assert_eq!(result, json!({"models": []}));
    server.join().unwrap();
}
