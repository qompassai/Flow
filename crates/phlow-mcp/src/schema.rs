//! Bounded tool-argument validation, ported from `flow/runtime.py`.
//!
//! This is the full `validate_arguments` algorithm — not a subset: an
//! explicit stack replaces recursion, every visited node is counted, and
//! nesting depth is capped, so a hostile client cannot blow the stack or
//! burn unbounded time. The message texts are byte-identical to the Python
//! `ValueError`s so `-32602` frames match.
//!
//! Phase 4 will relocate this module to `phlow-runtime` (which owns the
//! tool-dispatch contract); the logic is final and moves verbatim.

use serde_json::{Map, Value};
use std::fmt;

/// Maximum nesting depth for validated arguments.
pub const SCHEMA_DEPTH_MAX: u32 = 16;
/// Maximum schema nodes visited while validating one argument object.
pub const SCHEMA_NODES_MAX: u32 = 4096;

/// A single argument-validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// More than [`SCHEMA_NODES_MAX`] nodes visited.
    TooManyNodes,
    /// Nesting deeper than [`SCHEMA_DEPTH_MAX`].
    TooDeep,
    /// A value did not match its schema's `type` (e.g. `"Expected string arguments"`).
    ExpectedType {
        /// The schema's `type` name.
        kind: String,
    },
    /// A value was not in the schema's `enum` list.
    NotInEnum,
    /// Required keys missing or unknown keys present (strict even when the
    /// schema omits `additionalProperties: false`).
    MissingOrUnknown {
        /// Sorted missing required keys.
        missing: Vec<String>,
        /// Sorted unknown keys.
        unknown: Vec<String>,
    },
    /// The schema itself is malformed (not an object, or a non-object
    /// `properties`/`items`). Schemas here are compile-time constants, so
    /// this is unreachable in practice.
    MalformedSchema,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::TooManyNodes => {
                write!(f, "Arguments exceed {SCHEMA_NODES_MAX} schema nodes")
            }
            SchemaError::TooDeep => {
                write!(f, "Arguments exceed nesting depth {SCHEMA_DEPTH_MAX}")
            }
            SchemaError::ExpectedType { kind } => write!(f, "Expected {kind} arguments"),
            SchemaError::NotInEnum => write!(f, "Argument is not one of the allowed values"),
            SchemaError::MissingOrUnknown { missing, unknown } => {
                write!(
                    f,
                    "Missing arguments: {}; unknown arguments: {}",
                    format_python_str_list(missing),
                    format_python_str_list(unknown)
                )
            }
            SchemaError::MalformedSchema => write!(f, "Malformed argument schema"),
        }
    }
}

impl std::error::Error for SchemaError {}

/// Format a string list the way Python's `str(sorted(list))` does:
/// `['task']`, `[]`.
fn format_python_str_list(items: &[String]) -> String {
    let mut out = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(item);
        out.push('\'');
    }
    out.push(']');
    out
}

/// Validate `args` against the JSON-Schema subset Flow uses.
///
/// Faithful to `flow/runtime.py::validate_arguments`: `type` is checked
/// against the seven JSON types (with the Python caveat that a boolean is
/// never an integer/number — naturally true for `serde_json::Value`),
/// `enum` membership is enforced, and objects reject missing required keys
/// and unknown keys. Iterative: no recursion, ever.
pub fn validate_arguments(args: &Value, spec: &Value) -> Result<(), SchemaError> {
    let root = spec.as_object().ok_or(SchemaError::MalformedSchema)?;
    // (value, schema, depth); depth counts nesting below the root.
    let mut pending: Vec<(&Value, &Map<String, Value>, u32)> = vec![(args, root, 0)];
    let mut visited: u32 = 0;
    while let Some((value, schema, depth)) = pending.pop() {
        visited += 1;
        if visited > SCHEMA_NODES_MAX {
            return Err(SchemaError::TooManyNodes);
        }
        if depth > SCHEMA_DEPTH_MAX {
            return Err(SchemaError::TooDeep);
        }
        validate_node(value, schema)?;
        if let Some(object) = value.as_object() {
            let properties = schema.get("properties").and_then(|props| props.as_object());
            for (key, item) in object {
                // Unknown keys were already rejected by validate_node, so a
                // missing property schema means the schema is malformed.
                let prop_schema = properties
                    .and_then(|props| props.get(key))
                    .and_then(|prop| prop.as_object())
                    .ok_or(SchemaError::MalformedSchema)?;
                pending.push((item, prop_schema, depth + 1));
            }
        }
        if value.is_array()
            && let Some(items) = schema.get("items").and_then(|items| items.as_object())
        {
            for item in value.as_array().expect("checked is_array") {
                pending.push((item, items, depth + 1));
            }
        }
    }
    Ok(())
}

/// Check one value against its own schema node; children are queued by the caller.
fn validate_node(value: &Value, schema: &Map<String, Value>) -> Result<(), SchemaError> {
    if let Some(kind) = schema.get("type").and_then(|kind| kind.as_str()) {
        let matches = match kind {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "boolean" => value.is_boolean(),
            // serde_json::Value keeps booleans distinct from numbers, so no
            // extra bool-exclusion is needed (Python needs it explicitly).
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "null" => value.is_null(),
            // Unknown type names are ignored, like the Python original.
            _ => true,
        };
        if !matches {
            return Err(SchemaError::ExpectedType {
                kind: kind.to_owned(),
            });
        }
    }
    if let Some(allowed) = schema.get("enum").and_then(|allowed| allowed.as_array())
        && !allowed.contains(value)
    {
        return Err(SchemaError::NotInEnum);
    }
    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(|props| props.as_object());
        let mut missing: Vec<String> = schema
            .get("required")
            .and_then(|required| required.as_array())
            .map(|required| {
                required
                    .iter()
                    .filter_map(|key| key.as_str())
                    .filter(|key| !object.contains_key(*key))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let mut unknown: Vec<String> = object
            .keys()
            .filter(|key| !properties.is_some_and(|props| props.contains_key(*key)))
            .map(|key| key.to_owned())
            .collect();
        missing.sort();
        unknown.sort();
        if !missing.is_empty() || !unknown.is_empty() {
            return Err(SchemaError::MissingOrUnknown { missing, unknown });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flow_run_schema() -> Value {
        crate::protocol::tool_spec("flow_run")
            .unwrap()
            .input_schema
            .clone()
    }

    #[test]
    fn accepts_valid_arguments() {
        assert!(validate_arguments(&json!({"task": "x"}), &flow_run_schema()).is_ok());
        let status_schema = crate::protocol::tool_spec("flow_status")
            .unwrap()
            .input_schema
            .clone();
        assert!(validate_arguments(&json!({}), &status_schema).is_ok());
    }

    #[test]
    fn rejects_wrong_type_with_python_message() {
        let err = validate_arguments(&json!({"task": 1}), &flow_run_schema()).unwrap_err();
        assert_eq!(err.to_string(), "Expected string arguments");
    }

    #[test]
    fn rejects_missing_and_unknown_with_python_message() {
        let err = validate_arguments(&json!({"bogus": 1}), &flow_run_schema()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Missing arguments: ['task']; unknown arguments: ['bogus']"
        );
    }

    #[test]
    fn rejects_nonfinite_parse_constant() {
        // serde_json rejects Infinity/-Infinity/NaN at parse time, which is
        // what Python's parse_constant hook does.
        assert!(serde_json::from_str::<Value>("NaN").is_err());
        assert!(serde_json::from_str::<Value>("Infinity").is_err());
    }

    #[test]
    fn rejects_deep_nesting() {
        // Nest arrays SCHEMA_DEPTH_MAX + 1 deep; the innermost value sits at
        // depth 17, past the cap of 16.
        let mut deep_schema = json!({"type": "string"});
        for _ in 0..=SCHEMA_DEPTH_MAX {
            deep_schema = json!({"type": "array", "items": deep_schema});
        }
        let mut deep_value = json!("leaf");
        for _ in 0..=SCHEMA_DEPTH_MAX {
            deep_value = json!([deep_value]);
        }
        let err = validate_arguments(&deep_value, &deep_schema).unwrap_err();
        assert_eq!(err.to_string(), "Arguments exceed nesting depth 16");
    }

    #[test]
    fn node_cap_bounds_a_wide_array() {
        // 5000 items exceed SCHEMA_NODES_MAX = 4096 nodes. (An object would
        // trip the unknown-arguments rejection first; arrays count nodes
        // without a per-key schema.)
        let wide: Vec<Value> = (0..5000).map(Value::from).collect();
        let schema = json!({"type": "array", "items": {"type": "integer"}});
        let err = validate_arguments(&Value::Array(wide), &schema).unwrap_err();
        assert_eq!(err.to_string(), "Arguments exceed 4096 schema nodes");
    }

    #[test]
    fn enum_membership() {
        let schema = json!({"type": "string", "enum": ["a", "b"]});
        assert!(validate_arguments(&json!("a"), &schema).is_ok());
        let err = validate_arguments(&json!("c"), &schema).unwrap_err();
        assert_eq!(err.to_string(), "Argument is not one of the allowed values");
    }

    #[test]
    fn bool_is_not_an_integer() {
        let schema = json!({"type": "integer"});
        let err = validate_arguments(&json!(true), &schema).unwrap_err();
        assert_eq!(err.to_string(), "Expected integer arguments");
    }
}
