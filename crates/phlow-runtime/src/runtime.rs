//! The safe agent runtime. Mirrors `flow/runtime.py`.
//!
//! [`Runtime`] is generic over the model transport (`L`) and the editor
//! transport (`E`) so tests can script both without a network or Neovim.
//! Production wiring (Phase 6) uses [`ReqwestTransport`] and
//! [`MsgpackTransport`].
//!
//! Change-ledger note: Python's `Workspace` exposes `changed_files` as a
//! mutable attribute that `run()` clears per run. The Rust [`Workspace`]
//! owns its ledger behind a lock with no external mutation API, so the
//! runtime keeps its own per-run ledger (`run_changed`), cleared at the
//! start of every `run`, and records every write made through the run's
//! tools. Reports, fingerprints, and coverage read only this ledger, so a
//! second run never sees the first run's files — exactly like Python.
//!
//! [`ReqwestTransport`]: crate::transport::ReqwestTransport
//! [`MsgpackTransport`]: crate::transport::MsgpackTransport
//! [`Workspace`]: phlow_workspace::Workspace

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use phlow_checks::CheckRunner;
use phlow_config::{FlowConfig, ModelRole};
use phlow_editor::{EditorBridge, EditorTransport, TIMEOUT_DEFAULT};
use phlow_llm::LlmTransport;
use phlow_llm::transport::OllamaBackend;
use phlow_workspace::Workspace;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::RuntimeError;
use crate::prompt::{ROLES, normalize_tool_calls, reviewer_verdict, system_prompt_for_role};
use crate::report::{new_report, run_all_report_to_value};
use crate::tools::{schemas_for_role, tool_function};

/// Model summary cap, in characters.
pub const SUMMARY_CHARS_MAX: usize = 16_000;
/// Tool result cap, in characters of Python-compatible JSON.
pub const TOOL_RESULT_CHARS_MAX: usize = 40_000;
/// Preview cap inside a truncated tool result, in characters.
pub const TOOL_RESULT_PREVIEW_CHARS: usize = 38_000;
/// Statuses that count as verification failures.
const FAILURE_STATUSES: [&str; 4] = ["failed", "stale", "error", "timeout"];
/// Check kinds that count as static-analysis evidence.
const STATIC_CHECK_KINDS: [&str; 3] = ["lint", "typecheck", "diagnostics"];
/// File extensions that never need static-analysis coverage.
const NON_CODE_EXTENSIONS: [&str; 9] = [
    ".md", ".txt", ".rst", ".json", ".toml", ".yaml", ".yml", ".lock", ".csv",
];
/// `editor_debug` actions the runtime lets the model invoke. Debug
/// launch/run is manual, never a model tool: any other action is rejected
/// here, before it reaches the bridge (mirroring `runtime.py`).
const DEBUG_TOOL_ACTIONS: [&str; 4] = ["status", "list", "config", "discover"];

const _: () = assert!(
    TOOL_RESULT_PREVIEW_CHARS < TOOL_RESULT_CHARS_MAX,
    "preview must fit inside the tool result cap"
);
const _: () = assert!(SUMMARY_CHARS_MAX > 0, "summary cap must be positive");

/// Language name for a lowercase dotted extension, mirroring `FILETYPES`.
fn language_for_extension(extension: &str) -> &'static str {
    match extension {
        ".py" => "python",
        ".rs" => "rust",
        ".go" => "go",
        ".ts" | ".tsx" => "typescript",
        ".js" | ".jsx" => "javascript",
        ".lua" => "lua",
        ".sh" | ".bash" => "bash",
        ".c" | ".h" => "c",
        ".cc" | ".cpp" | ".hpp" => "cpp",
        ".nix" => "nix",
        ".java" => "java",
        ".kt" => "kotlin",
        ".rb" => "ruby",
        ".hs" => "haskell",
        ".ex" | ".exs" => "elixir",
        ".zig" => "zig",
        ".swift" => "swift",
        _ => "unknown",
    }
}

/// Lowercase dotted suffix of `path` (`""` when there is none), mirroring
/// `Path(path).suffix.lower()`.
fn path_extension(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!(".{}", extension.to_lowercase()))
        .unwrap_or_default()
}

/// Python truthiness for JSON values (`modified`/`dirty_buffers` flags).
fn is_truthy(value: Option<&Value>) -> bool {
    match value {
        None => false,
        Some(value) => is_truthy_value(value),
    }
}

/// Python truthiness for a JSON value: null, false, zero (int or float),
/// empty string/array/object are falsy; everything else is truthy.
fn is_truthy_value(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => {
            number.as_i64().is_some_and(|int| int != 0)
                || number.as_u64().is_some_and(|uint| uint != 0)
                || number.as_f64().is_some_and(|float| float != 0.0)
        }
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

/// `{"status": "error", "verified": false, "error": message}`.
fn error_report(message: impl Into<String>) -> Value {
    let mut report = Map::new();
    report.insert("status".to_owned(), Value::String("error".to_owned()));
    report.insert("verified".to_owned(), Value::Bool(false));
    report.insert("error".to_owned(), Value::String(message.into()));
    Value::Object(report)
}

/// `{"status": "error", "error": message}`: the `call_tool` failure shape.
/// Unlike [`error_report`], this carries no `"verified"` key, mirroring
/// Python's `call_tool` (`{"status": "error", "error": str(exc)}`).
fn tool_error(message: impl Into<String>) -> Value {
    let mut result = Map::new();
    result.insert("status".to_owned(), Value::String("error".to_owned()));
    result.insert("error".to_owned(), Value::String(message.into()));
    Value::Object(result)
}

/// `{"status", "verified", "checks": [], "reason", "source"}` gate report.
fn gate_report(status: &str, verified: bool, reason: String, source: &str) -> Value {
    let mut report = Map::new();
    report.insert("status".to_owned(), Value::String(status.to_owned()));
    report.insert("verified".to_owned(), Value::Bool(verified));
    report.insert("checks".to_owned(), Value::Array(Vec::new()));
    report.insert("reason".to_owned(), Value::String(reason));
    report.insert("source".to_owned(), Value::String(source.to_owned()));
    Value::Object(report)
}

/// The runtime capability advertisement: single writer, no arbitrary
/// commands, no plugins, POSIX no-follow file I/O on Unix.
fn capabilities_status() -> Map<String, Value> {
    let mut capabilities = Map::new();
    capabilities.insert("single_writer".to_owned(), Value::Bool(true));
    capabilities.insert("read_only_concurrency".to_owned(), Value::Number(1.into()));
    capabilities.insert("arbitrary_commands".to_owned(), Value::Bool(false));
    capabilities.insert("plugins".to_owned(), Value::Bool(false));
    capabilities.insert(
        "file_io".to_owned(),
        Value::String(
            if cfg!(unix) {
                "posix-no-follow"
            } else {
                "unavailable"
            }
            .to_owned(),
        ),
    );
    capabilities
}

/// Attach the coverage list and revision to a verification report,
/// downgrading it when static analysis is missing or failing.
fn apply_coverage_gate(verification: &mut Value, coverage: &[Value], revision: u64) {
    let all_ok = coverage
        .iter()
        .all(|item| item.get("status") == Some(&Value::String("ok".to_owned())));
    let object = verification
        .as_object_mut()
        .expect("check report is an object");
    object.insert("coverage".to_owned(), Value::Array(coverage.to_vec()));
    object.insert("revision".to_owned(), Value::Number(revision.into()));
    if !all_ok {
        let status = object
            .get("status")
            .and_then(|status| status.as_str())
            .unwrap_or("");
        let status = if FAILURE_STATUSES.contains(&status) {
            status.to_owned()
        } else {
            "unverified".to_owned()
        };
        object.insert("status".to_owned(), Value::String(status));
        object.insert("verified".to_owned(), Value::Bool(false));
        object.insert(
            "reason".to_owned(),
            Value::String("Missing or failing static analysis for changed source".to_owned()),
        );
    }
}

/// Names of required, passing static checks covering a path's language.
fn check_evidence(checks: &[Value], language: &str, extension: &str) -> Vec<String> {
    checks
        .iter()
        .filter(|check| {
            check.get("required") == Some(&Value::Bool(true))
                && check.get("status") == Some(&Value::String("ok".to_owned()))
                && check
                    .get("kind")
                    .and_then(|kind| kind.as_str())
                    .is_some_and(|kind| STATIC_CHECK_KINDS.contains(&kind))
                && check
                    .get("filetypes")
                    .and_then(|filetypes| filetypes.as_array())
                    .is_some_and(|filetypes| {
                        filetypes.iter().any(|filetype| {
                            filetype.as_str().is_some_and(|filetype| {
                                filetype == language || filetype == extension || filetype == "*"
                            })
                        })
                    })
        })
        .filter_map(|check| {
            check
                .get("name")
                .and_then(|name| name.as_str())
                .map(str::to_owned)
        })
        .collect()
}

/// Mark a verification object `"stale"`: drift was observed after the
/// reviewer ran, so the evidence no longer describes the workspace.
fn mark_stale(verification: &mut Value, reason: &str) {
    let object = verification
        .as_object_mut()
        .expect("verification is an object");
    object.insert("status".to_owned(), Value::String("stale".to_owned()));
    object.insert("verified".to_owned(), Value::Bool(false));
    object.insert("reason".to_owned(), Value::String(reason.to_owned()));
}

/// Lowercase hex of bytes (SHA-256 digests for fingerprints).
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// The safe agent runtime. See the module docs for the safety contract.
pub struct Runtime<L: LlmTransport, E: EditorTransport> {
    config: FlowConfig,
    workspace: Workspace,
    backend: Option<OllamaBackend<L>>,
    editor: EditorBridge<E>,
    editor_configured: bool,
    /// Single-writer lock. Shared ownership (not a plain field) so the
    /// guard never borrows `self`: `run`/`check` hold it across `&mut self`
    /// calls, which a field borrow would forbid.
    writer: Arc<Mutex<()>>,
    last_report: Option<Value>,
    closed: bool,
    checked_editor_snapshot: Option<Value>,
    run_changed: BTreeSet<String>,
}

impl<L: LlmTransport, E: EditorTransport> Runtime<L, E> {
    /// Build the runtime from a validated config and the two transports.
    ///
    /// `nvim_socket` is the Neovim private-socket path (or `None`); the
    /// editor transport is built separately (it owns its own copy for the
    /// lazy connection). `FlowConfig` can only exist validated, so there is
    /// no separate validation step.
    pub fn new(
        config: FlowConfig,
        llm_transport: L,
        editor_transport: E,
        nvim_socket: Option<String>,
    ) -> Result<Self, RuntimeError> {
        let protected: Vec<PathBuf> = config
            .config_path()
            .map(|path| vec![path.clone()])
            .unwrap_or_default();
        let workspace = Workspace::open(config.workspace_dir(), config.trusted(), &protected)?;
        let backend = OllamaBackend::new(config.ollama().clone(), llm_transport);
        let editor = EditorBridge::new(nvim_socket.clone(), TIMEOUT_DEFAULT, editor_transport)?;
        Ok(Runtime {
            config,
            workspace,
            backend: Some(backend),
            editor,
            editor_configured: nvim_socket.is_some(),
            writer: Arc::new(Mutex::new(())),
            last_report: None,
            closed: false,
            checked_editor_snapshot: None,
            run_changed: BTreeSet::new(),
        })
    }

    /// True while a `run`/`check` holds the single-writer lock.
    pub fn is_busy(&self) -> bool {
        self.writer.try_lock().is_err()
    }

    /// True after [`Runtime::close`].
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The tool schemas visible to `role`, mirroring `Runtime.schemas`.
    pub fn schemas(&mut self, role: &str) -> Vec<Value> {
        schemas_for_role(role, self.config.trusted(), &self.editor.schemas())
    }

    /// This run's changed files, sorted. Mirrors Python's
    /// `workspace.changed_files`, which `run()` clears at the start: only
    /// writes made through this run's tools are listed, never files from
    /// earlier runs lingering in the workspace ledger.
    fn changed_files_sorted(&self) -> Vec<String> {
        self.run_changed.iter().cloned().collect()
    }

    /// Note a write in the run ledger, normalizing like Python's
    /// `str(Path(path))`.
    fn note_write(&mut self, path: &str) {
        let normalized: String = Path::new(path)
            .components()
            .collect::<PathBuf>()
            .to_string_lossy()
            .into_owned();
        self.run_changed.insert(normalized);
    }
}

/// Report-mutation helpers. The report is a plain JSON object (Python dict);
/// these keep key access total so model-shaped data cannot panic the loop.
mod report_mut {
    use serde_json::{Map, Value};

    pub fn object_mut(report: &mut Value) -> &mut Map<String, Value> {
        report.as_object_mut().expect("report is an object")
    }

    pub fn set(report: &mut Value, key: &str, value: Value) {
        object_mut(report).insert(key.to_owned(), value);
    }

    pub fn counter(report: &mut Value, key: &str) -> u64 {
        object_mut(report)
            .get(key)
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
    }

    pub fn add(report: &mut Value, key: &str, delta: u64) {
        let next = counter(report, key).saturating_add(delta);
        set(report, key, Value::Number(next.into()));
    }

    pub fn push(report: &mut Value, key: &str, value: Value) {
        let array = object_mut(report)
            .entry(key.to_owned())
            .or_insert_with(|| Value::Array(Vec::new()));
        array
            .as_array_mut()
            .expect("report list field is an array")
            .push(value);
    }
}

impl<L: LlmTransport, E: EditorTransport> Runtime<L, E> {
    /// Execute one tool call for `role`. Never fails: every failure becomes
    /// `{"status": "error", "error": ...}`, mirroring `call_tool`.
    pub fn call_tool(&mut self, name: &str, args: &Value, role: &str) -> Value {
        assert!(ROLES.contains(&role), "unknown role: {role}");
        if self.closed {
            return tool_error("Runtime closed");
        }
        let schemas = self.schemas(role);
        let parameters = match tool_function(&schemas, name) {
            Some(function) => function.get("parameters").cloned().unwrap_or(Value::Null),
            None => {
                return tool_error(format!("Tool '{name}' is unavailable for role {role}"));
            }
        };
        if !args.is_object() {
            return tool_error("Tool arguments must be an object");
        }
        if let Err(error) = phlow_mcp::validate_arguments(args, &parameters) {
            return tool_error(error.to_string());
        }
        match self.dispatch_tool(name, args) {
            Ok(value) => value,
            Err(error) => tool_error(error.to_string()),
        }
    }

    /// The tool dispatch behind [`Runtime::call_tool`]; fallible so the
    /// caller can render one error shape.
    fn dispatch_tool(&mut self, name: &str, args: &Value) -> Result<Value, RuntimeError> {
        if self.editor_configured && (name == "file_read" || name == "file_write") {
            return self.editor_file_tool(name, args);
        }
        match name {
            "file_read" => {
                let path = tool_path(args)?;
                Ok(crate::report::read_result_to_value(
                    &self.workspace.read(&path)?,
                ))
            }
            "file_list" => {
                let path = args
                    .get("path")
                    .and_then(|path| path.as_str())
                    .unwrap_or(".");
                Ok(crate::report::list_result_to_value(
                    path,
                    &self.workspace.list(path)?,
                ))
            }
            "file_write" => {
                let path = tool_path(args)?;
                let content = args
                    .get("content")
                    .and_then(|content| content.as_str())
                    .ok_or_else(|| {
                        RuntimeError::Invalid("Tool arguments must be an object".to_owned())
                    })?;
                let result = self.workspace.write(&path, content)?;
                self.note_write(&path);
                Ok(crate::report::write_result_to_value(&result))
            }
            "flow_check" => Ok(self.run_checks(args.get("name").and_then(|name| name.as_str()))),
            _ if phlow_editor::contract::EDITOR_TOOLS.contains(&name) => {
                self.editor_tool(name, args)
            }
            _ => Err(RuntimeError::Invalid("Unknown tool".to_owned())),
        }
    }

    /// `file_read`/`file_write` when the editor bridge is configured: the
    /// editor performs the I/O so open buffers stay synchronized.
    fn editor_file_tool(&mut self, name: &str, args: &Value) -> Result<Value, RuntimeError> {
        self.editor_context()?;
        let path = tool_path(args)?;
        self.workspace.path(&path, name == "file_write")?;
        if name == "file_write" {
            let content = args
                .get("content")
                .and_then(|content| content.as_str())
                .ok_or_else(|| {
                    RuntimeError::Invalid("Tool arguments must be an object".to_owned())
                })?;
            if content.len() > phlow_workspace::FILE_BYTES_MAX as usize {
                return Err(RuntimeError::Invalid(
                    "File content exceeds limit".to_owned(),
                ));
            }
            let result = self.editor.call(name, args);
            if result.get("status") == Some(&Value::String("ok".to_owned())) {
                self.note_write(&path);
            }
            return Ok(result);
        }
        let result = self.editor.call(name, args);
        let content_len = result
            .get("content")
            .and_then(|content| content.as_str())
            .unwrap_or("")
            .len();
        if content_len > phlow_workspace::FILE_BYTES_MAX as usize {
            return Err(RuntimeError::Invalid(
                "Editor file content exceeds limit".to_owned(),
            ));
        }
        Ok(result)
    }

    /// A rose.nvim editor tool: freshness gate, containment check for `path`
    /// args, the `editor_debug` action gate, then the bridge call.
    fn editor_tool(&mut self, name: &str, args: &Value) -> Result<Value, RuntimeError> {
        if self.editor_configured {
            self.editor_context()?;
        }
        if let Some(path) = args.get("path").and_then(|path| path.as_str()) {
            self.workspace.path(path, false)?;
        }
        if name == "editor_debug" {
            let action = args
                .get("action")
                .and_then(|action| action.as_str())
                .unwrap_or("status");
            if !DEBUG_TOOL_ACTIONS.contains(&action) {
                return Err(RuntimeError::Invalid(
                    "editor_debug launch/run is manual, not an agent tool".to_owned(),
                ));
            }
        }
        Ok(self.editor.call(name, args))
    }

    /// Execute one normalized tool call and build its tool message.
    ///
    /// Mirrors `_tool_turn`: the global budget is incremented and asserted,
    /// failures increment the role state's `tool_errors`, every call appends
    /// one event, and the result is JSON-encoded with the 40k truncation cap.
    fn tool_turn(
        &mut self,
        role: &str,
        call: &Value,
        report: &mut Value,
        state: &mut Map<String, Value>,
    ) -> Value {
        assert_eq!(
            call.get("type"),
            Some(&Value::String("function".to_owned()))
        );
        let id = call
            .get("id")
            .and_then(|id| id.as_str())
            .expect("normalized calls carry string ids")
            .to_owned();
        report_mut::add(report, "tool_calls", 1);
        assert!(
            report_mut::counter(report, "tool_calls")
                <= self.config.agent().max_tool_calls() as u64,
            "tool call budget exceeded"
        );
        let function = call
            .get("function")
            .and_then(|function| function.as_object())
            .expect("normalized calls carry function objects");
        let name = function.get("name").cloned().unwrap_or(Value::Null);
        let result = self.run_tool_call(role, &name, function);
        record_tool_event(report, state, role, &name, &id, &result);
        let encoded = truncate_tool_result(&result);
        let mut message = Map::new();
        message.insert("role".to_owned(), Value::String("tool".to_owned()));
        message.insert("tool_call_id".to_owned(), Value::String(id));
        message.insert("name".to_owned(), name);
        message.insert("content".to_owned(), Value::String(encoded));
        Value::Object(message)
    }

    /// Parse a normalized call's arguments and dispatch through
    /// [`Runtime::call_tool`], which enforces role authorization and schema
    /// validation. Any failure renders as a tool error object.
    fn run_tool_call(&mut self, role: &str, name: &Value, function: &Map<String, Value>) -> Value {
        let parsed = match parse_tool_arguments(function.get("arguments")) {
            Ok(parsed) => parsed,
            Err(message) => return tool_error(message),
        };
        let Some(tool) = name.as_str() else {
            // Unreachable: normalization rejects non-string names. Kept
            // total so a malformed call cannot panic the loop.
            return tool_error(format!("Tool {name} is unavailable for role {role}"));
        };
        self.call_tool(tool, &parsed, role)
    }

    /// One model turn; returns the validated `(content, tool_calls)` shape.
    ///
    /// Mirrors `_chat`: the response's `choices[0].message` must be an object
    /// with a string `content` and a list `tool_calls`.
    fn chat(
        &mut self,
        messages: &[Value],
        tools: &[Value],
        model: &str,
    ) -> Result<(String, Vec<Value>), RuntimeError> {
        let backend = self.backend.as_mut().ok_or(RuntimeError::Closed)?;
        let response = backend.chat(messages, tools, Some(model))?;
        let message = response
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .filter(|message| message.is_object())
            .ok_or_else(|| RuntimeError::Invalid("Model message must be an object".to_owned()))?;
        let content = match message.get("content") {
            // Python: `message.get("content") or ""` — missing, null, and
            // falsy values become `""`; only a truthy non-string is an error.
            None => String::new(),
            Some(value) if !is_truthy_value(value) => String::new(),
            Some(Value::String(text)) => text.clone(),
            Some(_) => {
                return Err(RuntimeError::Invalid(
                    "Invalid model content/tool_calls shape".to_owned(),
                ));
            }
        };
        let calls = match message.get("tool_calls") {
            // Python: `message.get("tool_calls") or []` — same falsy rule.
            None => Vec::new(),
            Some(value) if !is_truthy_value(value) => Vec::new(),
            Some(Value::Array(calls)) => calls.clone(),
            Some(_) => {
                return Err(RuntimeError::Invalid(
                    "Invalid model content/tool_calls shape".to_owned(),
                ));
            }
        };
        Ok((content, calls))
    }

    /// Run one role's model loop, mirroring `_role`.
    fn role(
        &mut self,
        role: &str,
        task: &str,
        context: &Value,
        report: &mut Value,
    ) -> Result<Value, RuntimeError> {
        assert!(ROLES.contains(&role), "unknown role: {role}");
        let model = self
            .config
            .model_for(match role {
                "planner" => ModelRole::Planner,
                "reviewer" => ModelRole::Reviewer,
                _ => ModelRole::Coder,
            })
            .to_owned();
        let user_content = must_dumps(&serde_json::json!({"task": task, "context": context}));
        let mut messages = vec![
            serde_json::json!({"role": "system", "content": system_prompt_for_role(role)}),
            serde_json::json!({"role": "user", "content": user_content}),
        ];
        let mut state = Self::role_state(role, &model);
        let roles_len = roles_len(report);
        let tools = self.schemas(role);
        let budgets = RoleBudgets::from_config(self.config.agent());
        for turn in 0..budgets.max_iterations {
            if let Some(result) = self.role_turn(
                role,
                turn,
                roles_len,
                &mut messages,
                &mut state,
                &tools,
                &model,
                report,
                &budgets,
            ) {
                return Ok(result);
            }
        }
        let turns = state
            .get("turns")
            .and_then(|turns| turns.as_u64())
            .unwrap_or(0);
        assert_eq!(
            turns, budgets.max_iterations as u64,
            "role loop must run to budget"
        );
        state.insert("status".to_owned(), Value::String("unverified".to_owned()));
        state.insert(
            "error".to_owned(),
            Value::String("Role iteration budget exhausted".to_owned()),
        );
        Ok(finish_role(report, state, roles_len))
    }

    /// The initial per-role state object.
    fn role_state(role: &str, model: &str) -> Map<String, Value> {
        let mut state = Map::new();
        state.insert("role".to_owned(), Value::String(role.to_owned()));
        state.insert("model".to_owned(), Value::String(model.to_owned()));
        state.insert("status".to_owned(), Value::String("unverified".to_owned()));
        state.insert("turns".to_owned(), Value::Number(0.into()));
        state.insert("summary".to_owned(), Value::String(String::new()));
        state.insert("tool_errors".to_owned(), Value::Number(0.into()));
        state
    }

    /// One model turn. Returns `Some(result)` when the role finishes this
    /// turn (context/tool budget hit, protocol failure, or a text-only
    /// reply); `None` means the loop continues with tool results appended.
    #[allow(clippy::too_many_arguments)]
    fn role_turn(
        &mut self,
        role: &str,
        turn: usize,
        roles_len: usize,
        messages: &mut Vec<Value>,
        state: &mut Map<String, Value>,
        tools: &[Value],
        model: &str,
        report: &mut Value,
        budgets: &RoleBudgets,
    ) -> Option<Value> {
        if must_dumps(&Value::Array(messages.clone())).chars().count() > budgets.max_context_chars {
            role_terminal(state, "Context budget exhausted");
            return Some(finish_role(report, state.clone(), roles_len));
        }
        let turns = state
            .get("turns")
            .and_then(|turns| turns.as_u64())
            .unwrap_or(0);
        state.insert("turns".to_owned(), Value::Number((turns + 1).into()));
        report_mut::add(report, "model_calls", 1);
        let (content, calls) = match self.chat(messages, tools, model) {
            Ok(outcome) => outcome,
            Err(error) => {
                return Some(role_error(
                    report,
                    state.clone(),
                    roles_len,
                    format!("Backend/protocol failure: {error}"),
                ));
            }
        };
        if calls.len() as u64
            > budgets
                .max_tool_calls
                .saturating_sub(report_mut::counter(report, "tool_calls"))
        {
            role_terminal(state, "Tool call budget exhausted");
            return Some(finish_role(report, state.clone(), roles_len));
        }
        let id_prefix = format!("{role}-{}-{turn}", roles_len + 1);
        let normalized = match normalize_tool_calls(&calls, &id_prefix) {
            Ok(normalized) => normalized,
            Err(error) => {
                return Some(role_error(
                    report,
                    state.clone(),
                    roles_len,
                    format!("Backend/protocol failure: {error}"),
                ));
            }
        };
        messages.push(assistant_message(&content, &normalized));
        if normalized.is_empty() {
            finish_text_turn(role, &content, state);
            return Some(finish_role(report, state.clone(), roles_len));
        }
        for call in &normalized {
            let tool_message = self.tool_turn(role, call, report, state);
            messages.push(tool_message);
        }
        None
    }
}

/// Count a failed tool call and record its event in the report.
fn record_tool_event(
    report: &mut Value,
    state: &mut Map<String, Value>,
    role: &str,
    name: &Value,
    id: &str,
    result: &Value,
) {
    if result.get("status") != Some(&Value::String("ok".to_owned())) {
        let errors = state
            .get("tool_errors")
            .and_then(|count| count.as_u64())
            .unwrap_or(0);
        state.insert("tool_errors".to_owned(), Value::Number((errors + 1).into()));
    }
    let mut event = Map::new();
    event.insert("role".to_owned(), Value::String(role.to_owned()));
    event.insert("tool".to_owned(), name.clone());
    event.insert("tool_call_id".to_owned(), Value::String(id.to_owned()));
    event.insert(
        "status".to_owned(),
        result
            .get("status")
            .cloned()
            .unwrap_or(Value::String("unverified".to_owned())),
    );
    if let Some(error) = result.get("error") {
        event.insert("error".to_owned(), error.clone());
    }
    report_mut::push(report, "events", Value::Object(event));
}

/// Encode a tool result, replacing oversized payloads with a bounded
/// preview so one tool cannot blow the model's context budget.
fn truncate_tool_result(result: &Value) -> String {
    let mut encoded = must_dumps(result);
    if encoded.chars().count() > TOOL_RESULT_CHARS_MAX {
        let preview: String = encoded.chars().take(TOOL_RESULT_PREVIEW_CHARS).collect();
        let mut truncated = Map::new();
        truncated.insert("status".to_owned(), Value::String("unverified".to_owned()));
        truncated.insert("truncated".to_owned(), Value::Bool(true));
        truncated.insert("preview".to_owned(), Value::String(preview));
        encoded = must_dumps(&Value::Object(truncated));
    }
    assert!(
        encoded.chars().count() <= TOOL_RESULT_CHARS_MAX,
        "truncated tool result still exceeds the cap"
    );
    encoded
}

/// Extract the validated `path` argument (schemas guarantee a string).
fn tool_path(args: &Value) -> Result<String, RuntimeError> {
    args.get("path")
        .and_then(|path| path.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RuntimeError::Invalid("Tool arguments must be an object".to_owned()))
}

/// Parse the `arguments` field of a normalized tool call.
///
/// A string is decoded as JSON (mirroring `json.loads`); anything else must
/// already be a JSON value. Failures carry the exact Python message.
fn parse_tool_arguments(arguments: Option<&Value>) -> Result<Value, String> {
    match arguments {
        None => Ok(Value::Object(Map::new())),
        Some(Value::String(text)) => {
            serde_json::from_str(text).map_err(|error| format!("Invalid tool arguments: {error}"))
        }
        Some(other) => Ok(other.clone()),
    }
}

/// Python-compatible `json.dumps(ensure_ascii=True)`; infallible for the
/// finite values the runtime handles.
fn must_dumps(value: &Value) -> String {
    phlow_mcp::dumps(value).expect("runtime values are finite JSON")
}

/// Number of role states already recorded in the report.
fn roles_len(report: &Value) -> usize {
    report
        .get("roles")
        .and_then(|roles| roles.as_array())
        .map(Vec::len)
        .unwrap_or(0)
}

/// Record a `"error"` role state and return it from the report.
fn role_error(
    report: &mut Value,
    mut state: Map<String, Value>,
    roles_len: usize,
    error: String,
) -> Value {
    state.insert("status".to_owned(), Value::String("error".to_owned()));
    state.insert("error".to_owned(), Value::String(error));
    report_mut::push(report, "roles", Value::Object(state));
    report["roles"][roles_len].clone()
}

/// Push a finished role state and return the recorded entry.
fn finish_role(report: &mut Value, state: Map<String, Value>, roles_len: usize) -> Value {
    report_mut::push(report, "roles", Value::Object(state));
    report["roles"][roles_len].clone()
}

/// Mark a role state `"unverified"` with the terminal reason.
fn role_terminal(state: &mut Map<String, Value>, error: &str) {
    state.insert("status".to_owned(), Value::String("unverified".to_owned()));
    state.insert("error".to_owned(), Value::String(error.to_owned()));
}

/// The assistant message for one turn, carrying normalized tool calls when
/// the model requested any.
fn assistant_message(content: &str, normalized: &[Value]) -> Value {
    let mut assistant = Map::new();
    assistant.insert("role".to_owned(), Value::String("assistant".to_owned()));
    assistant.insert("content".to_owned(), Value::String(content.to_owned()));
    if !normalized.is_empty() {
        assistant.insert("tool_calls".to_owned(), Value::Array(normalized.to_vec()));
    }
    Value::Object(assistant)
}

/// Finish a text-only turn: truncated summary, plus the reviewer verdict
/// when the reviewer speaks.
fn finish_text_turn(role: &str, content: &str, state: &mut Map<String, Value>) {
    let summary: String = content.chars().take(SUMMARY_CHARS_MAX).collect();
    state.insert("status".to_owned(), Value::String("ok".to_owned()));
    state.insert("summary".to_owned(), Value::String(summary));
    if role == "reviewer" {
        state.extend(reviewer_verdict(content));
    }
}

/// The per-role loop budgets, read once from the agent config.
struct RoleBudgets {
    max_iterations: usize,
    max_context_chars: usize,
    max_tool_calls: u64,
}

impl RoleBudgets {
    fn from_config(agent: &phlow_config::AgentConfig) -> Self {
        Self {
            max_iterations: agent.max_iterations() as usize,
            max_context_chars: agent.max_context_chars() as usize,
            max_tool_calls: u64::from(agent.max_tool_calls()),
        }
    }
}

impl<L: LlmTransport, E: EditorTransport> Runtime<L, E> {
    /// The validated editor context, mirroring `_editor_context`.
    ///
    /// Fails closed: the workspace must be fresh, the bridge must report
    /// `"ok"` with a string workspace that resolves to the workspace root,
    /// and the freshness API must be v1.
    fn editor_context(&mut self) -> Result<Value, RuntimeError> {
        assert!(self.editor_configured, "editor context requires a socket");
        self.workspace
            .assert_current()
            .map_err(|error| RuntimeError::EditorContext(error.to_string()))?;
        let context = self
            .editor
            .call("editor_context", &Value::Object(Map::new()));
        let workspace = context
            .get("workspace")
            .and_then(|workspace| workspace.as_str());
        let ok =
            context.get("status") == Some(&Value::String("ok".to_owned())) && workspace.is_some();
        if !ok {
            let detail = context
                .get("error")
                .and_then(|error| error.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| phlow_mcp::dumps(&context).unwrap_or_else(|_| "{}".to_owned()));
            return Err(RuntimeError::EditorContext(format!(
                "Editor context unavailable: {detail}"
            )));
        }
        let resolved =
            std::fs::canonicalize(workspace.expect("workspace is a string")).map_err(|_| {
                RuntimeError::EditorContext(
                    "Rose/Flow workspace mismatch; refusing reverse tools and edits".to_owned(),
                )
            })?;
        if resolved != self.workspace.root() {
            return Err(RuntimeError::EditorContext(
                "Rose/Flow workspace mismatch; refusing reverse tools and edits".to_owned(),
            ));
        }
        let snapshot_v1 = context.get("workspace_snapshot_version")
            == Some(&Value::Number(1.into()))
            && matches!(
                context.get("workspace_snapshot"),
                Some(Value::Object(_)) | Some(Value::Array(_))
            )
            && matches!(
                context.get("dirty_buffers"),
                Some(Value::Object(_)) | Some(Value::Array(_))
            );
        if !snapshot_v1 {
            return Err(RuntimeError::EditorContext(
                "Editor workspace freshness API v1 is unavailable; update Rose".to_owned(),
            ));
        }
        Ok(context)
    }

    /// SHA-256 fingerprint of every changed file, mirroring `_fingerprint`.
    /// Unreadable files fingerprint as `null`.
    fn fingerprint(&self) -> Map<String, Value> {
        let mut result = Map::new();
        for path in self.changed_files_sorted() {
            let digest = match self.workspace.read(&path) {
                Ok(read) => {
                    let mut hasher = Sha256::new();
                    hasher.update(read.content.as_bytes());
                    Value::String(hex_encode(&hasher.finalize()))
                }
                Err(_) => Value::Null,
            };
            result.insert(path, digest);
        }
        result
    }

    /// Run the named checks (or all) with the editor freshness gates,
    /// mirroring `_run_checks`.
    fn run_checks(&mut self, name: Option<&str>) -> Value {
        self.checked_editor_snapshot = None;
        let mut before: Option<Value> = None;
        if self.editor_configured {
            match self.editor_context() {
                Ok(context) => {
                    if is_truthy(context.get("modified")) || is_truthy(context.get("dirty_buffers"))
                    {
                        return gate_report(
                            "stale",
                            false,
                            "Unsaved editor buffers would not be checked; save manually".to_owned(),
                            "rose.editor_context",
                        );
                    }
                    before = Some(context);
                }
                Err(error) => {
                    return gate_report(
                        "unverified",
                        false,
                        error.to_string(),
                        "rose.editor_context",
                    );
                }
            }
        }
        let runner = CheckRunner::new(&self.workspace, self.config.checks().clone());
        let mut result = run_all_report_to_value(&runner.run_all(name));
        if self.editor_configured {
            let before = before.expect("editor context was fetched above");
            match self.editor_context() {
                Ok(after) => {
                    if is_truthy(after.get("modified"))
                        || is_truthy(after.get("dirty_buffers"))
                        || after.get("workspace_snapshot") != before.get("workspace_snapshot")
                    {
                        let object = result.as_object_mut().expect("check report is an object");
                        object.insert("status".to_owned(), Value::String("stale".to_owned()));
                        object.insert("verified".to_owned(), Value::Bool(false));
                        object.insert(
                            "reason".to_owned(),
                            Value::String("Editor changed while running checks".to_owned()),
                        );
                    }
                    self.checked_editor_snapshot = after.get("workspace_snapshot").cloned();
                }
                Err(error) => {
                    let object = result.as_object_mut().expect("check report is an object");
                    object.insert("status".to_owned(), Value::String("stale".to_owned()));
                    object.insert("verified".to_owned(), Value::Bool(false));
                    object.insert("reason".to_owned(), Value::String(error.to_string()));
                }
            }
        }
        result
    }

    /// Host verification: checks plus static-analysis coverage, mirroring
    /// `_verification`. Evidence is host-computed; reviewer approval never
    /// substitutes for it.
    fn verification(&mut self) -> Result<Value, RuntimeError> {
        let mut verification = self.run_checks(None);
        let checked_editor = self.checked_editor_snapshot.clone();
        let available: Vec<String> = self
            .editor
            .schemas()
            .iter()
            .filter_map(|schema| schema["function"]["name"].as_str().map(str::to_owned))
            .collect();
        let checks = verification
            .get("checks")
            .and_then(|checks| checks.as_array())
            .cloned()
            .unwrap_or_default();
        let mut coverage = Vec::new();
        for path in self.changed_files_sorted() {
            let extension = path_extension(&path);
            if NON_CODE_EXTENSIONS.contains(&extension.as_str()) {
                continue;
            }
            coverage.push(self.coverage(&path, &extension, &checks, &available));
        }
        apply_coverage_gate(&mut verification, &coverage, self.workspace.revision());
        if self.editor_configured
            && let Some(checked) = checked_editor
        {
            let current = self.editor_context()?;
            if is_truthy(current.get("dirty_buffers"))
                || !same_observed_snapshot(Some(&checked), current.get("workspace_snapshot"))
            {
                mark_stale(
                    &mut verification,
                    "Editor inputs changed during static verification",
                );
            }
        }
        Ok(verification)
    }

    /// Static-analysis evidence for one changed source file, mirroring
    /// `_coverage`.
    fn coverage(
        &mut self,
        path: &str,
        extension: &str,
        checks: &[Value],
        available: &[String],
    ) -> Value {
        assert!(
            !NON_CODE_EXTENSIONS.contains(&extension),
            "coverage is only computed for code"
        );
        let language = language_for_extension(extension);
        let evidence = check_evidence(checks, language, extension);
        let editor_evidence = self.editor_evidence(path, available);
        let lint_result = &editor_evidence[0];
        let diagnostic_result = &editor_evidence[1];
        let mut ok = !evidence.is_empty()
            || (lint_result.get("status") == Some(&Value::String("ok".to_owned()))
                && lint_result.get("verified") == Some(&Value::Bool(true)));
        if diagnostic_result
            .get("status")
            .and_then(|status| status.as_str())
            .is_some_and(|status| FAILURE_STATUSES.contains(&status))
        {
            ok = false;
        }
        if crate::prompt::has_error_diagnostics(&editor_evidence) {
            ok = false;
        }
        let mut out = Map::new();
        out.insert("path".to_owned(), Value::String(path.to_owned()));
        out.insert("language".to_owned(), Value::String(language.to_owned()));
        out.insert(
            "status".to_owned(),
            Value::String(if ok { "ok" } else { "unverified" }.to_owned()),
        );
        out.insert(
            "source".to_owned(),
            Value::String(
                if evidence.is_empty() {
                    "rose.editor"
                } else {
                    "flow.check"
                }
                .to_owned(),
            ),
        );
        out.insert(
            "checks".to_owned(),
            Value::Array(evidence.into_iter().map(Value::String).collect()),
        );
        out.insert("evidence".to_owned(), Value::Array(editor_evidence));
        out.insert(
            "scope".to_owned(),
            Value::String(
                "Configured static checks/native linters only; cached diagnostics and missing LSP are not proof"
                    .to_owned(),
            ),
        );
        Value::Object(out)
    }

    /// Gather `editor_lint`/`editor_diagnostics` evidence for one path,
    /// marking unavailable tools explicitly.
    fn editor_evidence(&mut self, path: &str, available: &[String]) -> Vec<Value> {
        let mut editor_evidence = Vec::new();
        for tool in ["editor_lint", "editor_diagnostics"] {
            let mut item = Map::new();
            item.insert("tool".to_owned(), Value::String(tool.to_owned()));
            if available.iter().any(|name| name == tool) {
                let result = self.call_tool(tool, &serde_json::json!({"path": path}), "coder");
                if let Some(fields) = result.as_object() {
                    for (key, value) in fields {
                        item.insert(key.clone(), value.clone());
                    }
                }
            } else {
                item.insert("status".to_owned(), Value::String("unavailable".to_owned()));
            }
            editor_evidence.push(Value::Object(item));
        }
        editor_evidence
    }

    /// One coder → host verification → reviewer pass, mirroring `_cycle`.
    ///
    /// Verification evidence is only valid if nothing changed while the
    /// reviewer ran: the fingerprint and the editor snapshot are rechecked
    /// after review, and any drift marks verification `"stale"` — in the
    /// returned value and in every report copy (Python mutates one dict;
    /// Rust clones, so all copies are synced).
    fn cycle(
        &mut self,
        task: &str,
        context: &Value,
        report: &mut Value,
    ) -> Result<(Value, Value, Value), RuntimeError> {
        assert!(context.get("plan").is_some(), "cycle context needs a plan");
        let coder = self.role("coder", task, context, report)?;
        let fingerprint = self.fingerprint();
        let verification = self.verification()?;
        let editor_baseline: Option<Value> = if self.editor_configured {
            Some(
                self.editor_context()?
                    .get("workspace_snapshot")
                    .cloned()
                    .unwrap_or(Value::Null),
            )
        } else {
            None
        };
        self.publish_cycle_report(report, &coder, &verification);
        let reviewer_context = Self::reviewer_context(context, &coder, report, &verification);
        let reviewer = self.role("reviewer", task, &reviewer_context, report)?;
        let verification =
            self.invalidate_after_review(verification, &fingerprint, editor_baseline.as_ref());
        Self::sync_verification(report, &verification);
        Ok((coder, reviewer, verification))
    }

    /// Publish the coder/verification state to the report before the reviewer
    /// runs, mirroring Python's `report.update(...)` in `_cycle`.
    fn publish_cycle_report(&self, report: &mut Value, coder: &Value, verification: &Value) {
        report_mut::push(report, "verification_history", verification.clone());
        report_mut::set(report, "verification", verification.clone());
        report_mut::set(
            report,
            "checks",
            verification
                .get("checks")
                .cloned()
                .unwrap_or(Value::Array(Vec::new())),
        );
        report_mut::set(
            report,
            "changed_files",
            Value::Array(
                self.changed_files_sorted()
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ),
        );
        report_mut::set(
            report,
            "summary",
            coder
                .get("summary")
                .cloned()
                .unwrap_or(Value::String(String::new())),
        );
    }

    /// The reviewer's context: plan, implementation, changed files, and the
    /// pre-review verification evidence.
    fn reviewer_context(
        context: &Value,
        coder: &Value,
        report: &Value,
        verification: &Value,
    ) -> Value {
        serde_json::json!({
            "plan": context.get("plan").cloned().unwrap_or(Value::Null),
            "implementation": coder,
            "changed_files": report.get("changed_files").cloned().unwrap_or(Value::Array(Vec::new())),
            "verification": verification,
        })
    }

    /// Re-run the staleness gates after the reviewer: any drift in the
    /// changed-file fingerprint or the editor snapshot marks the
    /// verification `"stale"`.
    fn invalidate_after_review(
        &mut self,
        mut verification: Value,
        fingerprint: &Map<String, Value>,
        editor_baseline: Option<&Value>,
    ) -> Value {
        if self.fingerprint() != *fingerprint {
            mark_stale(
                &mut verification,
                "Changed files were modified during verification/review",
            );
        }
        if self.editor_configured {
            match self.editor_context() {
                Ok(editor_context) => {
                    if is_truthy(editor_context.get("modified"))
                        || is_truthy(editor_context.get("dirty_buffers"))
                        || !same_observed_snapshot(
                            editor_baseline,
                            editor_context.get("workspace_snapshot"),
                        )
                    {
                        mark_stale(
                            &mut verification,
                            "Workspace editor snapshot changed during review",
                        );
                    }
                }
                Err(error) => mark_stale(&mut verification, &error.to_string()),
            }
        }
        verification
    }

    /// Sync the post-review verification into every report copy.
    fn sync_verification(report: &mut Value, verification: &Value) {
        report_mut::set(report, "verification", verification.clone());
        if let Some(history) = report
            .get_mut("verification_history")
            .and_then(|h| h.as_array_mut())
            && let Some(last) = history.last_mut()
        {
            *last = verification.clone();
        }
    }
}

/// Compare an observed editor snapshot pair, mirroring
/// `_same_observed_snapshot`.
///
/// Reads may add observed paths/buffers, but cannot change already-checked
/// inputs. Non-object snapshots are treated as empty (defensive: the
/// validated editor context always carries objects).
fn same_observed_snapshot(before: Option<&Value>, after: Option<&Value>) -> bool {
    let empty = Map::new();
    let before = before.and_then(|value| value.as_object()).unwrap_or(&empty);
    let after = after.and_then(|value| value.as_object()).unwrap_or(&empty);
    for (path, item) in before {
        let current = match after.get(path).and_then(|value| value.as_object()) {
            Some(current) => current,
            None => return false,
        };
        if current.get("disk") != item.get("disk") {
            return false;
        }
        let current_buffers: Vec<&Map<String, Value>> = current
            .get("buffers")
            .and_then(|buffers| buffers.as_array())
            .map(|buffers| {
                buffers
                    .iter()
                    .filter_map(|buffer| buffer.as_object())
                    .collect()
            })
            .unwrap_or_default();
        let before_buffers: Vec<&Map<String, Value>> = item
            .get("buffers")
            .and_then(|buffers| buffers.as_array())
            .map(|buffers| {
                buffers
                    .iter()
                    .filter_map(|buffer| buffer.as_object())
                    .collect()
            })
            .unwrap_or_default();
        for buffer in before_buffers {
            let bufnr = buffer.get("bufnr");
            let unchanged = current_buffers
                .iter()
                .any(|current| current.get("bufnr") == bufnr && *current == buffer);
            if !unchanged {
                return false;
            }
        }
    }
    true
}

impl<L: LlmTransport, E: EditorTransport> Runtime<L, E> {
    /// Run the bounded planner → coder → verification → reviewer loop.
    ///
    /// Mirrors `Runtime.run`: invalid tasks and a busy runtime return an
    /// `"error"` report without raising; every other failure is captured
    /// into the report as `status: "error"`. The `changed_files` ledger and
    /// `last_report` are always refreshed, and the writer lock is always
    /// released.
    pub fn run(&mut self, task: &str) -> Value {
        if task.trim().is_empty()
            || task.chars().count() > self.config.agent().max_task_chars() as usize
        {
            return error_report("Invalid or oversized task");
        }
        let writer = self.writer.clone();
        let _guard = match writer.try_lock() {
            Ok(guard) => guard,
            Err(_) => return error_report("Runtime busy: single writer"),
        };
        let mut report = new_report(task);
        match self.run_inner(task, &mut report) {
            Ok(()) => {}
            Err(error) => {
                report_mut::set(&mut report, "status", Value::String("error".to_owned()));
                report_mut::set(&mut report, "error", Value::String(error.to_string()));
            }
        }
        report_mut::set(
            &mut report,
            "changed_files",
            Value::Array(
                self.changed_files_sorted()
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ),
        );
        let mut last = Map::new();
        for key in ["status", "verified", "changed_files", "cycles"] {
            last.insert(
                key.to_owned(),
                report.get(key).cloned().unwrap_or(Value::Null),
            );
        }
        self.last_report = Some(Value::Object(last));
        report
    }

    /// The fallible body of [`Runtime::run`].
    fn run_inner(&mut self, task: &str, report: &mut Value) -> Result<(), RuntimeError> {
        if self.closed {
            return Err(RuntimeError::Closed);
        }
        self.workspace.assert_current()?;
        self.run_changed.clear();
        let planner = self.role("planner", task, &Value::Object(Map::new()), report)?;
        if planner.get("status") != Some(&Value::String("ok".to_owned())) {
            report_mut::set(report, "status", planner["status"].clone());
            report_mut::set(
                report,
                "error",
                planner
                    .get("error")
                    .cloned()
                    .unwrap_or(Value::String("Planning failed".to_owned())),
            );
            return Ok(());
        }
        let mut context = Map::new();
        context.insert("plan".to_owned(), planner["summary"].clone());
        let max_cycles = self.config.agent().max_cycles();
        for _ in 0..max_cycles {
            report_mut::add(report, "cycles", 1);
            let (coder, reviewer, verification) =
                self.cycle(task, &Value::Object(context.clone()), report)?;
            let approved = reviewer.get("approved") == Some(&Value::Bool(true))
                && reviewer.get("status") == Some(&Value::String("ok".to_owned()))
                && verification.get("verified") == Some(&Value::Bool(true));
            if coder.get("status") == Some(&Value::String("ok".to_owned())) && approved {
                report_mut::set(report, "status", Value::String("ok".to_owned()));
                report_mut::set(report, "verified", Value::Bool(true));
                break;
            }
            let failed = verification.get("status") == Some(&Value::String("failed".to_owned()))
                || reviewer.get("approved") == Some(&Value::Bool(false));
            report_mut::set(
                report,
                "status",
                Value::String(if failed { "failed" } else { "unverified" }.to_owned()),
            );
            if coder.get("status") == Some(&Value::String("error".to_owned()))
                || reviewer.get("status") == Some(&Value::String("error".to_owned()))
            {
                report_mut::set(report, "status", Value::String("error".to_owned()));
                break;
            }
            let mut next = Map::new();
            next.insert("plan".to_owned(), planner["summary"].clone());
            next.insert("previous_implementation".to_owned(), coder);
            next.insert("verification".to_owned(), verification);
            next.insert("review".to_owned(), reviewer);
            next.insert(
                "instruction".to_owned(),
                Value::String(
                    "Repair the concrete failures. Do not change trust/config.".to_owned(),
                ),
            );
            context = next;
        }
        Ok(())
    }

    /// Runtime status snapshot. Never contacts the model, mirroring
    /// `Runtime.status`.
    pub fn status(&mut self) -> Value {
        let state = match self.workspace.assert_current() {
            Ok(()) => {
                if self.closed {
                    "unavailable"
                } else {
                    "ok"
                }
            }
            Err(_) => "stale",
        };
        let editor = self.editor_status();
        let mut status = Map::new();
        status.insert("status".to_owned(), Value::String(state.to_owned()));
        status.insert(
            "workspace".to_owned(),
            Value::String(self.workspace.root().display().to_string()),
        );
        status.insert("trusted".to_owned(), Value::Bool(self.config.trusted()));
        status.insert("backend".to_owned(), Value::Object(self.backend_status()));
        status.insert("models".to_owned(), Value::Object(self.models_status()));
        status.insert("checks".to_owned(), Value::Array(self.checks_status()));
        status.insert("editor".to_owned(), editor);
        status.insert(
            "warnings".to_owned(),
            Value::Array(
                self.config
                    .warnings()
                    .iter()
                    .map(|warning| Value::String(warning.clone()))
                    .collect(),
            ),
        );
        status.insert(
            "capabilities".to_owned(),
            Value::Object(capabilities_status()),
        );
        status.insert("busy".to_owned(), Value::Bool(self.is_busy()));
        status.insert(
            "last_report".to_owned(),
            self.last_report.clone().unwrap_or(Value::Null),
        );
        Value::Object(status)
    }

    /// The editor section of the status report, attaching live context when
    /// the socket is configured.
    fn editor_status(&mut self) -> Value {
        let mut editor = self.editor.status();
        if self.editor_configured {
            match self.editor_context() {
                Ok(context) => {
                    let object = editor.as_object_mut().expect("bridge status is an object");
                    object.insert("attached".to_owned(), Value::Bool(true));
                    object.insert("workspace".to_owned(), context["workspace"].clone());
                    object.insert("context".to_owned(), context);
                }
                Err(error) => {
                    let object = editor.as_object_mut().expect("bridge status is an object");
                    object.insert("status".to_owned(), Value::String("unavailable".to_owned()));
                    object.insert("attached".to_owned(), Value::Bool(false));
                    object.insert("error".to_owned(), Value::String(error.to_string()));
                }
            }
        } else {
            editor["attached"] = Value::Bool(false);
        }
        editor
    }

    /// The backend section: Ollama endpoint with unverified connectivity
    /// (status never contacts the model).
    fn backend_status(&self) -> Map<String, Value> {
        let mut backend = Map::new();
        backend.insert("type".to_owned(), Value::String("ollama".to_owned()));
        backend.insert(
            "base_url".to_owned(),
            Value::String(self.config.ollama().base_url().to_owned()),
        );
        backend.insert(
            "connectivity".to_owned(),
            Value::String("unverified".to_owned()),
        );
        backend.insert(
            "note".to_owned(),
            Value::String("status does not contact the model".to_owned()),
        );
        backend
    }

    /// The per-role model names.
    fn models_status(&self) -> Map<String, Value> {
        let mut models = Map::new();
        for (role, model_role) in [
            ("planner", ModelRole::Planner),
            ("coder", ModelRole::Coder),
            ("reviewer", ModelRole::Reviewer),
        ] {
            models.insert(
                role.to_owned(),
                Value::String(self.config.model_for(model_role).to_owned()),
            );
        }
        models
    }

    /// The configured checks with their name/required/kind/filetypes.
    fn checks_status(&self) -> Vec<Value> {
        self.config
            .checks()
            .iter()
            .map(|(name, check)| {
                let mut item = Map::new();
                item.insert("name".to_owned(), Value::String(name.clone()));
                item.insert("required".to_owned(), Value::Bool(check.required()));
                item.insert(
                    "kind".to_owned(),
                    Value::String(check.kind().as_str().to_owned()),
                );
                item.insert(
                    "filetypes".to_owned(),
                    Value::Array(
                        check
                            .filetypes()
                            .iter()
                            .map(|filetype| Value::String(filetype.clone()))
                            .collect(),
                    ),
                );
                Value::Object(item)
            })
            .collect()
    }

    /// Switch the model for all roles, mirroring Python's `/model` command:
    /// `ollama.model` is replaced and every per-role override is reset to
    /// defaults, so later turns resolve through [`FlowConfig::model_for`].
    /// The name must be non-empty; the TUI rejects empty names first.
    pub fn select_model(&mut self, model: &str) -> Result<(), RuntimeError> {
        self.config
            .select_model(model.to_owned())
            .map_err(|error| RuntimeError::Invalid(error.to_string()))
    }

    /// Replace the reverse editor request timeout (`--editor-timeout`).
    /// Out-of-range values are rejected; the previous timeout is kept.
    pub fn set_editor_timeout(&mut self, timeout: Duration) -> Result<(), RuntimeError> {
        self.editor
            .set_timeout(timeout)
            .map_err(|error| RuntimeError::Editor(error.to_string()))
    }

    /// Installed model names from the Ollama backend (`GET /api/tags`),
    /// mirroring `OllamaClient.list_models` for the TUI's `/models` panel.
    /// Any backend failure — refused, reset, timeout, bad shape — is an
    /// error; the caller decides whether to degrade.
    pub fn list_models(&mut self) -> Result<Vec<String>, RuntimeError> {
        let backend = self.backend.as_mut().ok_or(RuntimeError::Closed)?;
        backend
            .list_models()
            .map_err(|error| RuntimeError::Backend(error.to_string()))
    }

    /// Run the named checks (or all), mirroring `Runtime.check`.
    pub fn check(&mut self, name: Option<&str>) -> Value {
        let writer = self.writer.clone();
        let _guard = match writer.try_lock() {
            Ok(guard) => guard,
            Err(_) => return error_report("Runtime busy"),
        };
        if self.closed {
            return error_report("Runtime closed");
        }
        self.run_checks(name)
    }

    /// Release the editor bridge and the model backend. Idempotent.
    pub fn close(&mut self) {
        if !self.closed {
            self.closed = true;
            self.editor.close();
            if let Some(backend) = self.backend.take() {
                backend.close();
            }
        }
    }
}
