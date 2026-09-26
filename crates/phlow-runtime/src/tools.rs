//! Tool schemas and the role/tool authorization matrix.
//!
//! Mirrors `FILE_SCHEMAS` and `Runtime.schemas` in `flow/runtime.py`.

use serde_json::{Map, Value};

/// Editor tools that never mutate: every editor tool except `editor_lint`
/// and `editor_check`.
pub const READ_ONLY_EDITOR_NAMES: [&str; 6] = [
    "editor_context",
    "editor_diagnostics",
    "editor_symbols",
    "editor_references",
    "editor_scip",
    "editor_debug",
];

/// Build one `{"type": "function", ...}` schema, mirroring `schema()`.
fn schema(name: &str, description: &str, properties: &[(&str, &str)], required: &[&str]) -> Value {
    let mut props = Map::new();
    for (prop, kind) in properties {
        let mut spec = Map::new();
        spec.insert("type".to_owned(), Value::String((*kind).to_owned()));
        props.insert((*prop).to_owned(), Value::Object(spec));
    }
    let mut parameters = Map::new();
    parameters.insert("type".to_owned(), Value::String("object".to_owned()));
    parameters.insert("properties".to_owned(), Value::Object(props));
    parameters.insert(
        "required".to_owned(),
        Value::Array(
            required
                .iter()
                .map(|name| Value::String((*name).to_owned()))
                .collect(),
        ),
    );
    parameters.insert("additionalProperties".to_owned(), Value::Bool(false));
    let mut function = Map::new();
    function.insert("name".to_owned(), Value::String(name.to_owned()));
    function.insert(
        "description".to_owned(),
        Value::String(description.to_owned()),
    );
    function.insert("parameters".to_owned(), Value::Object(parameters));
    let mut outer = Map::new();
    outer.insert("type".to_owned(), Value::String("function".to_owned()));
    outer.insert("function".to_owned(), Value::Object(function));
    Value::Object(outer)
}

/// The four built-in file/check tool schemas, mirroring `FILE_SCHEMAS`.
pub fn file_schemas() -> Vec<Value> {
    vec![
        schema(
            "file_read",
            "Read a bounded UTF-8 file relative to the workspace",
            &[("path", "string")],
            &["path"],
        ),
        schema(
            "file_list",
            "List workspace files, excluding generated/hidden Git metadata",
            &[("path", "string")],
            &[],
        ),
        schema(
            "file_write",
            "Atomically write a workspace file (coder only, trusted workspace)",
            &[("path", "string"), ("content", "string")],
            &["path", "content"],
        ),
        schema(
            "flow_check",
            "Run only operator-configured named checks",
            &[("name", "string")],
            &[],
        ),
    ]
}

/// The tool schemas visible to `role`, mirroring `Runtime.schemas`.
///
/// The planner and reviewer are read-only; only the coder sees
/// `file_write`/`flow_check`, and only when trusted. Editor tools come from
/// the bridge; `editor_debug`'s `action` enum is narrowed to `status` for
/// every role, mirroring `runtime.py`'s `schemas()` (validation runs before
/// the dispatch gate, so the gate's four actions are unreachable through
/// `call_tool`; the `launch`/`run` gate is kept as defense-in-depth).
pub fn schemas_for_role(role: &str, trusted: bool, editor_schemas: &[Value]) -> Vec<Value> {
    assert!(crate::prompt::ROLES.contains(&role), "unknown role: {role}");
    let mut result: Vec<Value> = file_schemas()
        .into_iter()
        .filter(|item| {
            role == "coder" && trusted
                || !matches!(
                    item["function"]["name"].as_str(),
                    Some("file_write" | "flow_check")
                )
        })
        .collect();
    for editor_schema in editor_schemas {
        let name = editor_schema["function"]["name"].as_str().unwrap_or("");
        let allowed = (role == "coder" && trusted) || READ_ONLY_EDITOR_NAMES.contains(&name);
        if phlow_editor::contract::EDITOR_TOOLS.contains(&name) && allowed {
            let mut narrowed = editor_schema.clone();
            if name == "editor_debug" {
                narrowed["function"]["parameters"]["properties"]["action"] = serde_json::json!({
                    "type": "string",
                    "enum": ["status"]
                });
            }
            result.push(narrowed);
        }
    }
    assert!(
        result.len() <= file_schemas().len() + phlow_editor::contract::EDITOR_TOOLS.len(),
        "schema count out of bounds"
    );
    result
}

/// Find a tool's `function` schema object by name within a role's schemas.
pub fn tool_function<'a>(schemas: &'a [Value], name: &str) -> Option<&'a Map<String, Value>> {
    schemas.iter().find_map(|item| {
        let function = item.get("function")?.as_object()?;
        if function.get("name").and_then(|name| name.as_str()) == Some(name) {
            Some(function)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_schema_shapes_match_python() {
        let schemas = file_schemas();
        assert_eq!(schemas.len(), 4);
        assert_eq!(schemas[0]["function"]["name"], "file_read");
        assert_eq!(
            schemas[0]["function"]["parameters"]["additionalProperties"],
            false
        );
        assert_eq!(
            schemas[2]["function"]["parameters"]["required"],
            serde_json::json!(["path", "content"])
        );
        assert_eq!(
            schemas[3]["function"]["description"],
            "Run only operator-configured named checks"
        );
    }

    fn fake_editor_schema(name: &str) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "editor tool",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "additionalProperties": false,
                },
            },
        })
    }

    #[test]
    fn role_matrix() {
        let editor: Vec<Value> = ["editor_lint", "editor_debug", "editor_context"]
            .iter()
            .map(|name| fake_editor_schema(name))
            .collect();
        let names = |schemas: &[Value]| -> Vec<String> {
            schemas
                .iter()
                .map(|item| item["function"]["name"].as_str().unwrap().to_owned())
                .collect()
        };
        // Untrusted coder: read-only file tools, read-only editor tools only.
        let coder = names(&schemas_for_role("coder", false, &editor));
        assert!(coder.contains(&"file_read".to_owned()));
        assert!(!coder.contains(&"file_write".to_owned()));
        assert!(!coder.contains(&"flow_check".to_owned()));
        assert!(coder.contains(&"editor_debug".to_owned()));
        assert!(!coder.contains(&"editor_lint".to_owned()));
        // Trusted coder: everything, including mutating editor tools.
        let trusted = names(&schemas_for_role("coder", true, &editor));
        assert!(trusted.contains(&"file_write".to_owned()));
        assert!(trusted.contains(&"flow_check".to_owned()));
        assert!(trusted.contains(&"editor_lint".to_owned()));
        // Planner/reviewer: read-only even when trusted.
        for role in ["planner", "reviewer"] {
            let read_only = names(&schemas_for_role(role, true, &editor));
            assert!(!read_only.contains(&"file_write".to_owned()));
            assert!(!read_only.contains(&"editor_lint".to_owned()));
            assert!(read_only.contains(&"editor_debug".to_owned()));
        }
    }

    #[test]
    fn editor_debug_action_narrowed() {
        let editor = vec![fake_editor_schema("editor_debug")];
        let schemas = schemas_for_role("coder", true, &editor);
        let debug = tool_function(&schemas, "editor_debug").unwrap();
        assert_eq!(
            debug["parameters"]["properties"]["action"],
            serde_json::json!({
                "type": "string",
                "enum": ["status"]
            })
        );
    }
}
