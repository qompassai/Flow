//! MCP protocol constants and the compatibility tool catalog.
//!
//! Every string, version, and error code here is part of the wire contract
//! with MCP clients (and with the Python `flow` server this port replaces).
//! Do not rename tools, renumber codes, or reword messages without bumping
//! the protocol version and updating the golden fixtures.

use crate::json_ascii::is_integer_text;
use serde_json::{Map, Number, Value, json};
use std::sync::LazyLock;

/// The protocol version this server speaks.
pub const PROTOCOL_VERSION: &str = "2025-11-25";
/// Older protocol versions a client may request; unknown versions negotiate
/// up to [`PROTOCOL_VERSION`].
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26"];
/// Largest accepted frame: 1 MiB, matching the Python server.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
/// `serverInfo.name` is `"flow"` for compatibility with existing clients,
/// even though the repository is now named `phlow`.
pub const SERVER_NAME: &str = "flow";
/// Server version reported in `serverInfo`.
pub const SERVER_VERSION: &str = "0.2.0";
/// Instructions reported on initialize.
pub const INSTRUCTIONS: &str = "Trusted workspace named checks only. Missing checks never verify.";

/// JSON-RPC parse error (malformed frame, non-finite number, over-deep nesting).
pub const PARSE_ERROR: i32 = -32700;
/// JSON-RPC invalid request (not an object, bad envelope, double initialize).
pub const INVALID_REQUEST: i32 = -32600;
/// JSON-RPC method not found.
pub const METHOD_NOT_FOUND: i32 = -32601;
/// JSON-RPC invalid params (bad initialize params, bad tool args).
pub const INVALID_PARAMS: i32 = -32602;
/// JSON-RPC internal error (defensive; the typed core has no panics to catch).
pub const INTERNAL_ERROR: i32 = -32603;
/// MCP-not-initialized: the client must initialize and send
/// `notifications/initialized` before any tool call.
pub const NOT_INITIALIZED: i32 = -32002;

/// One compatibility tool: its wire name, description, and strict input schema.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Wire name, e.g. `"flow_run"`.
    pub name: &'static str,
    /// Human description reported by `tools/list`.
    pub description: &'static str,
    /// Strict JSON Schema (`type: object`, `additionalProperties: false`).
    pub input_schema: Value,
}

static TOOL_SPECS: LazyLock<Vec<ToolSpec>> = LazyLock::new(|| {
    vec![
        ToolSpec {
            name: "flow_run",
            description: "Bounded local planner/coder/reviewer run with real verification",
            input_schema: json!({
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
                "additionalProperties": false,
            }),
        },
        ToolSpec {
            name: "flow_status",
            description: "Local health, configured models/checks, editor capabilities",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            }),
        },
        ToolSpec {
            name: "flow_check",
            description: "Run operator-configured named checks; never arbitrary argv",
            input_schema: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": false,
            }),
        },
    ]
});

/// All compatibility tools, in `tools/list` order.
pub fn tool_specs() -> &'static [ToolSpec] {
    &TOOL_SPECS
}

/// Look up a tool by wire name.
pub fn tool_spec(name: &str) -> Option<&'static ToolSpec> {
    TOOL_SPECS.iter().find(|spec| spec.name == name)
}

/// A validated JSON-RPC request id: absent (notification), string, or integer.
///
/// Booleans, floats, arrays, and objects are invalid ids; explicit `null`
/// is invalid when the key is present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestId {
    /// No `"id"` key: a notification, which never gets a reply.
    Absent,
    /// A string id, echoed back verbatim.
    Str(String),
    /// An integer id that fits `i64`, echoed back verbatim.
    Int(i64),
    /// An integer id beyond `i64`/`u64`. The `Number` keeps the source
    /// digits (via `serde_json/arbitrary_precision`) and is echoed
    /// verbatim, exactly like Python's arbitrary-precision `int`.
    BigInt(Number),
}

impl RequestId {
    /// Extract the request id from a request object.
    ///
    /// Returns `Some(Absent)` when the `"id"` key is missing (a
    /// notification), and `None` when the key is present but invalid (null,
    /// bool, float, array, object); the caller reports `Invalid Request`
    /// with a null id in that case, exactly like the Python server.
    ///
    /// Integers of any magnitude are valid ids: Python's `int` is
    /// arbitrary-precision, so `2**70` is accepted and echoed digit-for-digit.
    pub fn extract(request: &Map<String, Value>) -> Option<RequestId> {
        let raw = match request.get("id") {
            None => return Some(RequestId::Absent),
            Some(raw) => raw,
        };
        match raw {
            Value::Null => None,
            Value::Bool(_) => None,
            Value::String(text) => Some(RequestId::Str(text.clone())),
            Value::Number(number) => {
                if let Some(int) = number.as_i64() {
                    Some(RequestId::Int(int))
                } else if is_integer_text(number) {
                    // Integer literal beyond i64 (arbitrary_precision keeps
                    // the digits); floats like `1e3` are invalid ids, as in
                    // Python where they decode to `float`.
                    Some(RequestId::BigInt(number.clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Whether the `"id"` value in `request` is valid: absent, a string, or
    /// an integer of any magnitude. Mirrors the Python envelope check,
    /// including the rule that an explicit `"id": null` is invalid.
    pub fn id_is_valid(request: &Map<String, Value>) -> bool {
        match request.get("id") {
            None => true,
            Some(Value::String(_)) => true,
            // Any integer literal is a valid id, whatever its magnitude.
            Some(Value::Number(number)) => is_integer_text(number),
            Some(_) => false,
        }
    }

    /// Serialize the id for an envelope (`null` when absent).
    pub fn to_json(&self) -> Value {
        match self {
            RequestId::Absent => Value::Null,
            RequestId::Str(text) => Value::String(text.clone()),
            RequestId::Int(int) => Value::from(*int),
            RequestId::BigInt(number) => Value::Number(number.clone()),
        }
    }
}

/// Build a JSON-RPC error envelope value with the exact field order the
/// Python server emits: `jsonrpc`, `id`, `error`.
pub fn error_value(id: &RequestId, code: i32, message: &str) -> Value {
    let mut envelope = Map::new();
    envelope.insert("jsonrpc".to_owned(), Value::String("2.0".to_owned()));
    envelope.insert("id".to_owned(), id.to_json());
    let mut error = Map::new();
    error.insert("code".to_owned(), Value::from(code));
    error.insert("message".to_owned(), Value::String(message.to_owned()));
    envelope.insert("error".to_owned(), Value::Object(error));
    Value::Object(envelope)
}

/// Build a JSON-RPC result envelope value: `jsonrpc`, `id`, `result`.
pub fn result_value(id: &RequestId, result: Value) -> Value {
    let mut envelope = Map::new();
    envelope.insert("jsonrpc".to_owned(), Value::String("2.0".to_owned()));
    envelope.insert("id".to_owned(), id.to_json());
    envelope.insert("result".to_owned(), result);
    Value::Object(envelope)
}

/// Build the `tools/list` result value.
pub fn tools_list_value() -> Value {
    let tools: Vec<Value> = tool_specs()
        .iter()
        .map(|spec| {
            let mut tool = Map::new();
            tool.insert("name".to_owned(), Value::String(spec.name.to_owned()));
            tool.insert(
                "description".to_owned(),
                Value::String(spec.description.to_owned()),
            );
            tool.insert("inputSchema".to_owned(), spec.input_schema.clone());
            Value::Object(tool)
        })
        .collect();
    let mut result = Map::new();
    result.insert("tools".to_owned(), Value::Array(tools));
    Value::Object(result)
}

/// Build the `initialize` result value with the negotiated version.
pub fn initialize_value(negotiated: &str) -> Value {
    let mut server_info = Map::new();
    server_info.insert("name".to_owned(), Value::String(SERVER_NAME.to_owned()));
    server_info.insert(
        "version".to_owned(),
        Value::String(SERVER_VERSION.to_owned()),
    );
    let mut result = Map::new();
    result.insert(
        "protocolVersion".to_owned(),
        Value::String(negotiated.to_owned()),
    );
    let mut capabilities = Map::new();
    capabilities.insert("tools".to_owned(), Value::Object(Map::new()));
    result.insert("capabilities".to_owned(), Value::Object(capabilities));
    result.insert("serverInfo".to_owned(), Value::Object(server_info));
    result.insert(
        "instructions".to_owned(),
        Value::String(INSTRUCTIONS.to_owned()),
    );
    Value::Object(result)
}

/// Build the tool-call result envelope: `content` (text with the
/// ASCII-JSON result), `structuredContent`, and `isError`.
///
/// Returns [`DumpsError::NonFiniteFloat`] if the result somehow holds a
/// non-finite float; a `serde_json::Value` cannot through safe code, so the
/// caller maps this to `-32603` defensively, mirroring the Python server.
pub fn tool_result_value(
    id: &RequestId,
    result: &Value,
    is_error: bool,
) -> Result<Value, super::json_ascii::DumpsError> {
    let text = super::json_ascii::dumps(result)?;
    let mut content_item = Map::new();
    content_item.insert("type".to_owned(), Value::String("text".to_owned()));
    content_item.insert("text".to_owned(), Value::String(text));
    let mut envelope = Map::new();
    envelope.insert("jsonrpc".to_owned(), Value::String("2.0".to_owned()));
    envelope.insert("id".to_owned(), id.to_json());
    let mut result_obj = Map::new();
    result_obj.insert(
        "content".to_owned(),
        Value::Array(vec![Value::Object(content_item)]),
    );
    result_obj.insert("structuredContent".to_owned(), result.clone());
    result_obj.insert("isError".to_owned(), Value::Bool(is_error));
    envelope.insert("result".to_owned(), Value::Object(result_obj));
    Ok(Value::Object(envelope))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_ascii::dumps;

    #[test]
    fn tool_catalog_matches_python() {
        let specs = tool_specs();
        assert_eq!(
            specs.iter().map(|s| s.name).collect::<Vec<_>>(),
            ["flow_run", "flow_status", "flow_check"]
        );
        assert_eq!(
            tool_spec("flow_run").unwrap().description,
            specs[0].description
        );
        assert!(tool_spec("nope").is_none());
    }

    #[test]
    fn error_envelope_field_order() {
        let frame = dumps(&error_value(&RequestId::Int(1), -32002, "nope")).unwrap();
        assert_eq!(
            frame,
            r#"{"jsonrpc": "2.0", "id": 1, "error": {"code": -32002, "message": "nope"}}"#
        );
    }

    #[test]
    fn request_id_extraction() {
        let mut request = Map::new();
        assert_eq!(RequestId::extract(&request), Some(RequestId::Absent));
        request.insert("id".to_owned(), Value::Null);
        assert_eq!(RequestId::extract(&request), None);
        request.insert("id".to_owned(), Value::Bool(true));
        assert_eq!(RequestId::extract(&request), None);
        request.insert("id".to_owned(), Value::from(1.5f64));
        assert_eq!(RequestId::extract(&request), None);
        request.insert("id".to_owned(), Value::from(-7i64));
        assert_eq!(RequestId::extract(&request), Some(RequestId::Int(-7)));
        request.insert("id".to_owned(), Value::String("abc".to_owned()));
        assert_eq!(
            RequestId::extract(&request),
            Some(RequestId::Str("abc".to_owned()))
        );
    }

    #[test]
    fn big_int_ids_are_valid_and_echo_verbatim() {
        // 2**70: beyond u64, accepted and echoed digit-for-digit like
        // Python's arbitrary-precision int.
        let request: Value = serde_json::from_str(r#"{"id": 1180591620717411303424}"#).unwrap();
        let map = request.as_object().unwrap();
        assert!(RequestId::id_is_valid(map));
        let id = RequestId::extract(map).unwrap();
        assert!(matches!(id, RequestId::BigInt(_)));
        assert_eq!(
            dumps(&error_value(&id, -32600, "x")).unwrap(),
            r#"{"jsonrpc": "2.0", "id": 1180591620717411303424, "error": {"code": -32600, "message": "x"}}"#
        );
        // Negative big int echoes too.
        let request: Value = serde_json::from_str(r#"{"id": -1180591620717411303424}"#).unwrap();
        let map = request.as_object().unwrap();
        assert!(RequestId::id_is_valid(map));
        // A float that looks integer-valued is still an invalid id, as in
        // Python where `1e3` decodes to `float`.
        let request: Value = serde_json::from_str(r#"{"id": 1e3}"#).unwrap();
        let map = request.as_object().unwrap();
        assert!(!RequestId::id_is_valid(map));
        assert_eq!(RequestId::extract(map), None);
    }
}
