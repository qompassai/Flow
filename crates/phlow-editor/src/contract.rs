//! Wire constants and rose.nvim result contracts.
//!
//! The Lua snippets are byte-exact with `flow/editor.py`; the bridge asserts
//! (via the fake transport in tests) that no other expression ever crosses
//! the socket.

use std::path::Path;
use std::time::Duration;

use serde_json::{Map, Value};

use crate::error::FreshnessError;

/// Lua evaluated to fetch the editor tool schemas. Byte-exact.
pub const SCHEMAS_LUA: &str = "return require('rose.tools').schemas()";
/// Lua evaluated to call one editor tool with `[name, args]`. Byte-exact.
pub const CALL_LUA: &str = "return require('rose.tools').call(...)";

/// At most this many schemas are kept; more is an error, never a truncation.
pub const SCHEMAS_MAX: usize = 64;
/// Default request timeout: 120 seconds.
pub const TIMEOUT_DEFAULT: Duration = Duration::from_secs(120);
/// Minimum request timeout: 0.1 seconds.
pub const TIMEOUT_MIN: Duration = Duration::from_millis(100);
/// Maximum request timeout: 660 seconds.
pub const TIMEOUT_MAX: Duration = Duration::from_secs(660);
/// Extra join budget the bridge allows the worker past `timeout` before it
/// declares the worker dead. Phase 4's threaded transport must honor this.
pub const WORKER_GRACE: Duration = Duration::from_secs(2);

/// The eight editor tools rose.nvim may expose.
pub const EDITOR_TOOLS: [&str; 8] = [
    "editor_context",
    "editor_diagnostics",
    "editor_symbols",
    "editor_references",
    "editor_lint",
    "editor_check",
    "editor_scip",
    "editor_debug",
];

/// The two file tools the bridge itself may offer alongside editor tools.
pub const EDITOR_BRIDGE_FILE_TOOLS: [&str; 2] = ["file_read", "file_write"];

/// `editor_debug` executes nothing; these are its allowed read-only actions.
/// Debug launch/run is manual, never a model-controlled tool: the runtime
/// rejects any other action before it reaches the bridge.
pub const DEBUG_ACTIONS: [&str; 4] = ["status", "list", "config", "discover"];

/// True when `name` is one of the eight editor tools or the two bridge file
/// tools. Unknown names are refused before anything crosses the socket.
pub fn is_allowed_tool(name: &str) -> bool {
    EDITOR_TOOLS.contains(&name) || EDITOR_BRIDGE_FILE_TOOLS.contains(&name)
}

/// True when `action` is a non-executing `editor_debug` action.
pub fn is_debug_action_allowed(action: &str) -> bool {
    DEBUG_ACTIONS.contains(&action)
}

/// Filter raw `rose.tools.schemas()` output down to well-formed editor tool
/// schemas, enforcing the 64-schema cap.
///
/// The raw result is untrusted rose.nvim output, so it is narrowed through
/// [`phlow_json::object_list`] before inspection. Mirrors
/// `EditorBridge.schemas`: a non-array is `"Rose schemas must be an array"`,
/// longer than [`SCHEMAS_MAX`] is `"Rose returned more than 64 tool
/// schemas"`, and each kept item is a `{"type": "function", "function":
/// {"name": <editor tool>, "parameters": {...}}}` dict.
pub fn filter_schemas(result: &Value) -> Result<Vec<Value>, String> {
    let items =
        phlow_json::object_list(result).map_err(|_| "Rose schemas must be an array".to_owned())?;
    if items.len() > SCHEMAS_MAX {
        return Err(format!(
            "Rose returned more than {SCHEMAS_MAX} tool schemas"
        ));
    }
    let kept = items
        .iter()
        .filter(|item| {
            item.get("type").and_then(Value::as_str) == Some("function")
                && matches!(
                    item.get("function"),
                    Some(Value::Object(function))
                        if function
                            .get("name")
                            .and_then(Value::as_str)
                            .is_some_and(is_allowed_tool)
                            && function.get("parameters").is_some_and(Value::is_object)
                )
        })
        .cloned()
        .collect();
    Ok(kept)
}

/// Check an `editor_context` result for workspace freshness.
///
/// Mirrors `Flow._editor_context`:
/// - not `{"status": "ok", "workspace": <str>}` → `Unavailable` with the
///   result's `error` (or the whole result when there is no `error` key);
/// - workspace not equal to `workspace_root` → `WorkspaceMismatch`;
/// - `version != 1`, or `snapshot`/`dirty_buffers` not object-or-array →
///   `ApiV1Missing`.
///
/// `workspace_root` must already be canonical; the editor path is
/// canonicalized before comparison so `..` and symlinks cannot smuggle a
/// mismatch past the check.
pub fn validate_editor_context(
    result: &Value,
    workspace_root: &Path,
) -> Result<(), FreshnessError> {
    let object = result.as_object();
    let ok = object.is_some_and(|map| {
        map.get("status").and_then(Value::as_str) == Some("ok")
            && map.get("workspace").and_then(Value::as_str).is_some()
    });
    if !ok {
        let detail = object
            .and_then(|map| map.get("error"))
            .map(render_detail)
            .unwrap_or_else(|| render_detail(result));
        return Err(FreshnessError::Unavailable(detail));
    }
    let map = object.expect("checked above");
    let editor_workspace = map
        .get("workspace")
        .and_then(Value::as_str)
        .expect("checked above");
    let canonical = Path::new(editor_workspace)
        .canonicalize()
        .map_err(FreshnessError::Io)?;
    if canonical != workspace_root {
        return Err(FreshnessError::WorkspaceMismatch {
            editor: editor_workspace.to_owned(),
            agent: workspace_root.to_string_lossy().into_owned(),
        });
    }
    let version_ok = map.get("version").and_then(Value::as_u64) == Some(1);
    let shape_ok = ["snapshot", "dirty_buffers"].iter().all(|key| {
        matches!(
            map.get(*key),
            Some(Value::Object(_)) | Some(Value::Array(_))
        )
    });
    if !version_ok || !shape_ok {
        return Err(FreshnessError::ApiV1Missing);
    }
    Ok(())
}

/// Render the detail for an unavailable context: the `error` string when
/// present, otherwise the whole result as compact JSON (Python's `str()`
/// on the dict).
fn render_detail(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "<unrenderable>".to_owned()),
    }
}

/// The object the bridge returns for a denied freshness check, mirroring
/// `Flow._editor_context`'s `{"status": "unavailable", "error": ...}` shape.
pub fn unavailable_object(error: &FreshnessError) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("status".to_owned(), Value::from("unavailable"));
    map.insert("error".to_owned(), Value::from(error.to_string()));
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn function_schema(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        })
    }

    #[test]
    fn lua_snippets_are_byte_exact() {
        assert_eq!(SCHEMAS_LUA, "return require('rose.tools').schemas()");
        assert_eq!(CALL_LUA, "return require('rose.tools').call(...)");
    }

    #[test]
    fn filter_schemas_rejects_non_array() {
        let err = filter_schemas(&json!({"type": "function"})).unwrap_err();
        assert_eq!(err, "Rose schemas must be an array");
    }

    #[test]
    fn filter_schemas_rejects_over_cap() {
        let many: Vec<Value> = (0..65).map(|_| function_schema("editor_lint")).collect();
        let err = filter_schemas(&Value::Array(many)).unwrap_err();
        assert_eq!(err, "Rose returned more than 64 tool schemas");
    }

    #[test]
    fn filter_schemas_keeps_only_well_formed_editor_tools() {
        let result = json!([
            function_schema("editor_lint"),
            function_schema("not_a_tool"),
            {"type": "function", "function": {"name": "editor_lint"}},
            {"type": "text", "function": {"name": "editor_lint", "parameters": {}}},
            "junk",
        ]);
        let kept = filter_schemas(&result).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0]["function"]["name"], "editor_lint");
    }

    #[test]
    fn filter_schemas_accepts_bridge_file_tools() {
        let kept = filter_schemas(&json!([function_schema("file_read")])).unwrap();
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn freshness_rejects_non_ok_context() {
        let root = std::env::temp_dir();
        let err = validate_editor_context(&json!({"status": "error", "error": "stale"}), &root)
            .unwrap_err();
        assert_eq!(err.to_string(), "Editor context unavailable: stale");
    }

    #[test]
    fn freshness_uses_whole_result_when_no_error_key() {
        let root = std::env::temp_dir();
        let result = json!({"status": "weird"});
        let err = validate_editor_context(&result, &root).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("Editor context unavailable: {}", result)
        );
    }

    #[test]
    fn freshness_rejects_workspace_mismatch() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let other = root.join("phlow-editor-other");
        std::fs::create_dir_all(&other).unwrap();
        let err = validate_editor_context(
            &json!({
                "status": "ok",
                "workspace": other.to_str().unwrap(),
                "version": 1,
                "snapshot": {},
                "dirty_buffers": [],
            }),
            &root,
        )
        .unwrap_err();
        assert!(matches!(err, FreshnessError::WorkspaceMismatch { .. }));
        assert_eq!(
            err.to_string(),
            "Rose/Flow workspace mismatch; refusing reverse tools and edits"
        );
    }

    #[test]
    fn freshness_rejects_old_api_version() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let err = validate_editor_context(
            &json!({
                "status": "ok",
                "workspace": root.to_str().unwrap(),
                "version": 0,
                "snapshot": {},
                "dirty_buffers": [],
            }),
            &root,
        )
        .unwrap_err();
        assert!(matches!(err, FreshnessError::ApiV1Missing));
        assert_eq!(
            err.to_string(),
            "Editor workspace freshness API v1 is unavailable; update Rose"
        );
    }

    #[test]
    fn freshness_rejects_bad_snapshot_shape() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let err = validate_editor_context(
            &json!({
                "status": "ok",
                "workspace": root.to_str().unwrap(),
                "version": 1,
                "snapshot": "nope",
                "dirty_buffers": [],
            }),
            &root,
        )
        .unwrap_err();
        assert!(matches!(err, FreshnessError::ApiV1Missing));
    }

    #[test]
    fn freshness_accepts_v1_context() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        validate_editor_context(
            &json!({
                "status": "ok",
                "workspace": root.to_str().unwrap(),
                "version": 1,
                "snapshot": {"a": 1},
                "dirty_buffers": [{"path": "a"}],
            }),
            &root,
        )
        .unwrap();
    }

    #[test]
    fn debug_actions_are_non_executing_only() {
        assert!(is_debug_action_allowed("status"));
        assert!(is_debug_action_allowed("list"));
        assert!(is_debug_action_allowed("config"));
        assert!(is_debug_action_allowed("discover"));
        assert!(!is_debug_action_allowed("launch"));
        assert!(!is_debug_action_allowed("run"));
        assert!(!is_debug_action_allowed("exec"));
        assert!(!is_debug_action_allowed("shell"));
    }
}
