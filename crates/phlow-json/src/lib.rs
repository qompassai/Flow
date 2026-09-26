//! Checked JSON narrowing at untrusted boundaries.
//!
//! Plain words: JSON arrives from places we do not control (MCP frames, tool
//! arguments, editor responses). This crate parses it with hard caps and
//! narrows `serde_json::Value` into the shapes callers actually need,
//! returning typed errors instead of panicking.
//!
//! Port of `flow/json_types.py`. The Python `is_object` string-key check is
//! subsumed by the type system here: `serde_json::Map` keys are always
//! `String`, so there is nothing left to verify.
//!
//! # Bounds
//!
//! - [`JSON_INPUT_BYTES_MAX`]: largest input [`parse_limited`] accepts.
//! - [`JSON_DEPTH_MAX`]: deepest nesting a parsed value may contain.
//!
//! # Unsafe policy
//!
//! This crate forbids unsafe code outright.

#![forbid(unsafe_code)]

use serde_json::{Map, Value};
use std::fmt;

/// Largest JSON input [`parse_limited`] accepts, in bytes.
///
/// Matches the MCP frame cap (`MAX_FRAME_BYTES` = 1 MiB): anything bigger
/// never reaches this parser over the protocol boundary.
pub const JSON_INPUT_BYTES_MAX: usize = 1_048_576;

/// Deepest nesting (objects/arrays) a parsed value may contain.
///
/// `serde_json` enforces its own recursion limit (128) while parsing; this
/// stricter named bound is checked afterwards with an explicit stack, so no
/// recursion ever runs over attacker-controlled depth.
pub const JSON_DEPTH_MAX: usize = 64;

/// Longest field name reproduced inside an error message, in characters.
///
/// Error context is bounded: a hostile document must not be able to inflate
/// a log line without limit through a long field name.
const FIELD_CONTEXT_CHARS_MAX: usize = 64;

/// Failures from parsing or narrowing untrusted JSON.
///
/// Every variant carries bounded context only. Field names are truncated to
/// [`FIELD_CONTEXT_CHARS_MAX`] characters on display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonError {
    /// Input exceeded [`JSON_INPUT_BYTES_MAX`].
    InputTooLarge { bytes: usize, max: usize },
    /// Byte input was not valid UTF-8.
    InvalidUtf8 { bytes: usize },
    /// `serde_json` rejected the document; message is the parser's own,
    /// truncated to a bounded length.
    Parse { message: String },
    /// Nesting exceeded [`JSON_DEPTH_MAX`].
    DepthExceeded { depth: usize, max: usize },
    /// Expected a JSON object, found something else.
    NotAnObject,
    /// Expected a JSON array, found something else.
    NotAnArray,
    /// An object lacked the requested field.
    MissingField { field: String },
    /// A field held the wrong JSON type.
    UnexpectedType {
        field: String,
        expected: &'static str,
    },
    /// A number field was NaN or infinite; non-finite floats are rejected,
    /// never silently propagated (MCP contract).
    NonFiniteNumber { field: String },
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonError::InputTooLarge { bytes, max } => {
                write!(f, "JSON input too large: {bytes} bytes exceeds {max}")
            }
            JsonError::InvalidUtf8 { bytes } => {
                write!(f, "JSON input is not valid UTF-8 ({bytes} bytes)")
            }
            JsonError::Parse { message } => write!(f, "invalid JSON: {message}"),
            JsonError::DepthExceeded { depth, max } => {
                write!(f, "JSON nesting too deep: depth {depth} exceeds {max}")
            }
            JsonError::NotAnObject => write!(f, "expected a JSON object"),
            JsonError::NotAnArray => write!(f, "expected a JSON array"),
            JsonError::MissingField { field } => {
                write!(f, "missing required field {}", truncate_field(field))
            }
            JsonError::UnexpectedType { field, expected } => {
                write!(f, "field {} must be {expected}", truncate_field(field))
            }
            JsonError::NonFiniteNumber { field } => {
                write!(f, "field {} must be a finite number", truncate_field(field))
            }
        }
    }
}

impl std::error::Error for JsonError {}

/// Truncate a field name to [`FIELD_CONTEXT_CHARS_MAX`] characters on a
/// char boundary, so error context stays bounded.
fn truncate_field(field: &str) -> String {
    if field.chars().count() <= FIELD_CONTEXT_CHARS_MAX {
        return format!("{field:?}");
    }
    let truncated: String = field.chars().take(FIELD_CONTEXT_CHARS_MAX).collect();
    format!("{truncated:?}...")
}

/// Parse JSON with the crate's input and depth bounds.
///
/// Rejects oversized input *before* allocating a parser, then enforces
/// [`JSON_DEPTH_MAX`] on the parsed value with an explicit stack (no
/// recursion over attacker-controlled depth).
pub fn parse_limited(input: &str) -> Result<Value, JsonError> {
    if input.len() > JSON_INPUT_BYTES_MAX {
        return Err(JsonError::InputTooLarge {
            bytes: input.len(),
            max: JSON_INPUT_BYTES_MAX,
        });
    }
    let value: Value = serde_json::from_str(input).map_err(|err| JsonError::Parse {
        message: truncate_message(&err.to_string()),
    })?;
    check_depth(&value)?;
    check_finite(&value)?;
    Ok(value)
}

/// Parse JSON from bytes, rejecting non-UTF-8 input with a typed error.
///
/// `&str` input is UTF-8 by construction; byte input from sockets and
/// subprocesses is not, so this is the entry point for those callers.
pub fn parse_limited_bytes(input: &[u8]) -> Result<Value, JsonError> {
    let text =
        std::str::from_utf8(input).map_err(|_| JsonError::InvalidUtf8 { bytes: input.len() })?;
    parse_limited(text)
}

/// Truncate a parser message to a bounded length on a char boundary.
fn truncate_message(message: &str) -> String {
    const MESSAGE_CHARS_MAX: usize = 256;
    if message.chars().count() <= MESSAGE_CHARS_MAX {
        message.to_owned()
    } else {
        message.chars().take(MESSAGE_CHARS_MAX).collect()
    }
}

/// Enforce [`JSON_DEPTH_MAX`] with an explicit stack.
///
/// The root counts as depth 1: a value nested 64 deep is accepted, the 65th
/// level is rejected. Rejected input leaves nothing behind; there is no
/// partial state to roll back.
fn check_depth(value: &Value) -> Result<(), JsonError> {
    let mut stack: Vec<(&Value, usize)> = vec![(value, 1)];
    while let Some((node, depth)) = stack.pop() {
        if depth > JSON_DEPTH_MAX {
            return Err(JsonError::DepthExceeded {
                depth,
                max: JSON_DEPTH_MAX,
            });
        }
        match node {
            Value::Array(items) => {
                for item in items {
                    stack.push((item, depth + 1));
                }
            }
            Value::Object(map) => {
                for item in map.values() {
                    stack.push((item, depth + 1));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Reject non-finite numbers with an explicit stack walk (no recursion over
/// attacker-controlled depth).
///
/// `serde_json`'s `arbitrary_precision` feature — required by `phlow-mcp`
/// for byte-exact big-integer request IDs, and unified across the workspace
/// by Cargo's feature resolution — lets out-of-range literals like `1e999`
/// parse into a [`Number`] instead of failing at parse time. This crate's
/// contract is that non-finite floats never enter the system, so they are
/// rejected here rather than relying on the parser. Integer spellings are
/// always finite and pass through untouched.
fn check_finite(value: &Value) -> Result<(), JsonError> {
    let mut stack: Vec<&Value> = vec![value];
    while let Some(node) = stack.pop() {
        match node {
            Value::Array(items) => stack.extend(items.iter()),
            Value::Object(map) => stack.extend(map.values()),
            Value::Number(number) => {
                let text = number.to_string();
                let digits = text.strip_prefix('-').unwrap_or(&text);
                let is_integer = !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
                if !is_integer && !number.as_f64().is_some_and(f64::is_finite) {
                    return Err(JsonError::Parse {
                        message: "non-finite number".to_owned(),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// True when the value is a JSON object.
///
/// The Python original also verified string keys; `serde_json::Map` keys are
/// `String` by type, so that check has nothing left to do here.
pub fn is_object(value: &Value) -> bool {
    matches!(value, Value::Object(_))
}

/// True when the value is a JSON array.
pub fn is_array(value: &Value) -> bool {
    matches!(value, Value::Array(_))
}

/// Narrow a value to an object reference, or return [`JsonError::NotAnObject`].
pub fn object_map(value: &Value) -> Result<&Map<String, Value>, JsonError> {
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(JsonError::NotAnObject),
    }
}

/// Narrow a value to an array slice, or return [`JsonError::NotAnArray`].
pub fn object_list(value: &Value) -> Result<&[Value], JsonError> {
    match value {
        Value::Array(items) => Ok(items),
        _ => Err(JsonError::NotAnArray),
    }
}

/// Look up a field on an object, or return [`JsonError::MissingField`].
pub fn field<'a>(map: &'a Map<String, Value>, name: &str) -> Result<&'a Value, JsonError> {
    map.get(name).ok_or_else(|| JsonError::MissingField {
        field: name.to_owned(),
    })
}

/// Require a string field.
pub fn req_str<'a>(map: &'a Map<String, Value>, name: &str) -> Result<&'a str, JsonError> {
    match field(map, name)? {
        Value::String(text) => Ok(text),
        _ => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "a string",
        }),
    }
}

/// Require a boolean field. Booleans are never coerced from numbers or strings.
pub fn req_bool(map: &Map<String, Value>, name: &str) -> Result<bool, JsonError> {
    match field(map, name)? {
        Value::Bool(flag) => Ok(*flag),
        _ => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "a boolean",
        }),
    }
}

/// Require an integer field fitting in `i64`.
///
/// Floats are rejected even when whole: `1.0` is not an integer on the wire.
pub fn req_i64(map: &Map<String, Value>, name: &str) -> Result<i64, JsonError> {
    match field(map, name)? {
        Value::Number(number) => number.as_i64().ok_or_else(|| JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "an integer",
        }),
        _ => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "an integer",
        }),
    }
}

/// Require an integer field fitting in `u64`.
pub fn req_u64(map: &Map<String, Value>, name: &str) -> Result<u64, JsonError> {
    match field(map, name)? {
        Value::Number(number) => number.as_u64().ok_or_else(|| JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "an unsigned integer",
        }),
        _ => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "an unsigned integer",
        }),
    }
}

/// Require a finite float field.
///
/// Integers are accepted (they convert exactly through the small protocol
/// ranges in use); NaN and infinities are rejected per the MCP contract.
pub fn req_f64(map: &Map<String, Value>, name: &str) -> Result<f64, JsonError> {
    let number = match field(map, name)? {
        Value::Number(number) => number.as_f64().ok_or_else(|| JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "a number",
        })?,
        _ => {
            return Err(JsonError::UnexpectedType {
                field: name.to_owned(),
                expected: "a number",
            });
        }
    };
    if !number.is_finite() {
        return Err(JsonError::NonFiniteNumber {
            field: name.to_owned(),
        });
    }
    Ok(number)
}

/// Require an object field, narrowed to a map reference.
pub fn req_object<'a>(
    map: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a Map<String, Value>, JsonError> {
    match field(map, name)? {
        Value::Object(inner) => Ok(inner),
        _ => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "an object",
        }),
    }
}

/// Require an array field, narrowed to a slice.
pub fn req_array<'a>(map: &'a Map<String, Value>, name: &str) -> Result<&'a [Value], JsonError> {
    match field(map, name)? {
        Value::Array(items) => Ok(items),
        _ => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "an array",
        }),
    }
}

/// Read an optional string field: absent is `Ok(None)`, present-but-wrong is an error.
pub fn opt_str<'a>(map: &'a Map<String, Value>, name: &str) -> Result<Option<&'a str>, JsonError> {
    match map.get(name) {
        None => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(_) => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "a string",
        }),
    }
}

/// Read an optional boolean field: absent is `Ok(None)`, present-but-wrong is an error.
pub fn opt_bool(map: &Map<String, Value>, name: &str) -> Result<Option<bool>, JsonError> {
    match map.get(name) {
        None => Ok(None),
        Some(Value::Bool(flag)) => Ok(Some(*flag)),
        Some(_) => Err(JsonError::UnexpectedType {
            field: name.to_owned(),
            expected: "a boolean",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_object(text: &str) -> Map<String, Value> {
        match parse_limited(text).expect("test JSON must parse") {
            Value::Object(map) => map,
            _ => panic!("test JSON must be an object"),
        }
    }

    #[test]
    fn parses_all_scalar_shapes() {
        let value = parse_limited(r#"{"s":"x","i":1,"f":1.5,"b":true,"n":null}"#).unwrap();
        assert!(is_object(&value));
        assert!(!is_array(&value));
        let map = object_map(&value).unwrap();
        assert_eq!(req_str(map, "s").unwrap(), "x");
        assert_eq!(req_i64(map, "i").unwrap(), 1);
        assert_eq!(req_f64(map, "f").unwrap(), 1.5);
        assert!(req_bool(map, "b").unwrap());
        assert!(matches!(field(map, "n").unwrap(), Value::Null));
    }

    #[test]
    fn parses_top_level_array() {
        let value = parse_limited("[1, 2, 3]").unwrap();
        assert!(is_array(&value));
        assert!(!is_object(&value));
        assert_eq!(object_list(&value).unwrap().len(), 3);
        assert_eq!(object_map(&value), Err(JsonError::NotAnObject));
    }

    #[test]
    fn narrowing_rejects_wrong_shapes() {
        let value = parse_limited(r#"{"a": 1}"#).unwrap();
        assert_eq!(object_list(&value), Err(JsonError::NotAnArray));
        let array = parse_limited("[1]").unwrap();
        assert_eq!(object_map(&array), Err(JsonError::NotAnObject));
    }

    #[test]
    fn is_object_and_is_array_cover_every_value_kind() {
        let cases = [
            (r#"{}"#, true, false),
            (r#"[]"#, false, true),
            (r#""s""#, false, false),
            ("1", false, false),
            ("1.5", false, false),
            ("true", false, false),
            ("null", false, false),
        ];
        for (text, want_object, want_array) in cases {
            let value = parse_limited(text).unwrap();
            assert_eq!(is_object(&value), want_object, "{text}");
            assert_eq!(is_array(&value), want_array, "{text}");
        }
    }

    #[test]
    fn input_over_the_byte_cap_is_rejected_before_parsing() {
        let big = " ".repeat(JSON_INPUT_BYTES_MAX + 1);
        assert_eq!(
            parse_limited(&big),
            Err(JsonError::InputTooLarge {
                bytes: JSON_INPUT_BYTES_MAX + 1,
                max: JSON_INPUT_BYTES_MAX,
            })
        );
        // Exactly at the cap parses: a JSON string of exactly JSON_INPUT_BYTES_MAX.
        let at_cap = format!("\"{}\"", "x".repeat(JSON_INPUT_BYTES_MAX - 2));
        assert_eq!(at_cap.len(), JSON_INPUT_BYTES_MAX);
        assert!(parse_limited(&at_cap).is_ok());
    }

    #[test]
    fn non_utf8_bytes_are_rejected() {
        assert_eq!(
            parse_limited_bytes(&[0x7b, 0xff, 0x7d]),
            Err(JsonError::InvalidUtf8 { bytes: 3 })
        );
        assert!(parse_limited_bytes(br#"{"a":1}"#).is_ok());
    }

    #[test]
    fn malformed_json_reports_a_parse_error() {
        assert!(matches!(
            parse_limited("{oops"),
            Err(JsonError::Parse { .. })
        ));
        assert!(matches!(parse_limited(""), Err(JsonError::Parse { .. })));
        // Trailing garbage is not silently accepted.
        assert!(matches!(
            parse_limited("{} {}"),
            Err(JsonError::Parse { .. })
        ));
    }

    #[test]
    fn depth_at_the_limit_is_accepted_and_one_more_is_rejected() {
        // Root counts as depth 1, so 63 nested arrays reach depth 63.
        let ok_text = format!("{}{}", "[".repeat(63), "]".repeat(63));
        assert!(parse_limited(&ok_text).is_ok());
        // 64 nested arrays hold the scalar `1` at depth 65: rejected.
        let deep_text = format!("{}1{}", "[".repeat(64), "]".repeat(64));
        assert_eq!(
            parse_limited(&deep_text),
            Err(JsonError::DepthExceeded {
                depth: 65,
                max: JSON_DEPTH_MAX,
            })
        );
    }

    /// Build `{"k0":{"k1":...{"k{n-1}":1}...}}`: n `{` opens with the
    /// scalar as the innermost value, so the scalar sits at depth n+1
    /// (the root counts as depth 1).
    fn nested_object(n: usize) -> String {
        assert!(n >= 1, "need at least the root object");
        let mut text = String::from("{");
        for level in 0..n - 1 {
            text.push('"');
            text.push('k');
            text.push_str(&level.to_string());
            text.push_str("\":{");
        }
        text.push('"');
        text.push('k');
        text.push_str(&(n - 1).to_string());
        text.push_str("\":1");
        for _ in 0..n {
            text.push('}');
        }
        text
    }

    #[test]
    fn deep_objects_count_the_same_as_arrays() {
        // 63 nested objects hold the scalar at depth 64: accepted.
        assert!(parse_limited(&nested_object(63)).is_ok());
        // 64 nested objects hold the scalar at depth 65: rejected.
        assert!(matches!(
            parse_limited(&nested_object(64)),
            Err(JsonError::DepthExceeded { .. })
        ));
    }

    #[test]
    fn missing_field_is_an_error() {
        let map = parse_object(r#"{"a": 1}"#);
        assert_eq!(
            field(&map, "b"),
            Err(JsonError::MissingField {
                field: "b".to_owned()
            })
        );
    }

    #[test]
    fn wrong_types_are_rejected_per_accessor() {
        let map = parse_object(r#"{"s":"x","i":1,"b":true,"o":{},"a":[]}"#);
        assert!(matches!(
            req_str(&map, "i"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_bool(&map, "s"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_i64(&map, "s"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_u64(&map, "s"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_f64(&map, "s"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_object(&map, "a"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_array(&map, "o"),
            Err(JsonError::UnexpectedType { .. })
        ));
    }

    #[test]
    fn whole_floats_are_not_integers_and_negatives_are_not_unsigned() {
        let map = parse_object(r#"{"whole": 1.0, "neg": -1, "big": 18446744073709551615}"#);
        assert!(matches!(
            req_i64(&map, "whole"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_u64(&map, "neg"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert_eq!(req_u64(&map, "big").unwrap(), u64::MAX);
        assert!(matches!(
            req_i64(&map, "big"),
            Err(JsonError::UnexpectedType { .. })
        ));
    }

    #[test]
    fn non_finite_floats_never_enter_the_system() {
        // Out-of-range literals used to fail at parse time; with
        // arbitrary_precision they parse into a Number, so check_finite
        // rejects them here instead. Either way they never enter the system.
        assert!(matches!(
            parse_limited(r#"{"inf": 1e999}"#),
            Err(JsonError::Parse { .. })
        ));
        // req_f64 additionally refuses non-finite values as defense-in-depth
        // for programmatically built values (serde_json's public API cannot
        // construct a non-finite Number, so the parser above is the real
        // enforcement point; the check guards the MCP "never emit" contract
        // at this choke point if construction ever changes).
        let map = parse_object(r#"{"ok": 0.5, "int": 2}"#);
        assert_eq!(req_f64(&map, "ok").unwrap(), 0.5);
        assert_eq!(req_f64(&map, "int").unwrap(), 2.0);
        // The error variant still formats correctly.
        let err = JsonError::NonFiniteNumber {
            field: "x".to_owned(),
        };
        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn integers_are_accepted_as_finite_floats() {
        let map = parse_object(r#"{"n": 2}"#);
        assert_eq!(req_f64(&map, "n").unwrap(), 2.0);
    }

    #[test]
    fn nested_accessors_compose() {
        let map = parse_object(r#"{"outer": {"inner": [{"v": true}]}}"#);
        let outer = req_object(&map, "outer").unwrap();
        let inner = req_array(outer, "inner").unwrap();
        let first = object_map(&inner[0]).unwrap();
        assert!(req_bool(first, "v").unwrap());
    }

    #[test]
    fn optional_accessors_distinguish_absent_from_wrong() {
        let map = parse_object(r#"{"s": "x", "n": 1}"#);
        assert_eq!(opt_str(&map, "missing").unwrap(), None);
        assert_eq!(opt_str(&map, "s").unwrap(), Some("x"));
        assert!(matches!(
            opt_str(&map, "n"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert_eq!(opt_bool(&map, "missing").unwrap(), None);
        assert!(matches!(
            opt_bool(&map, "s"),
            Err(JsonError::UnexpectedType { .. })
        ));
    }

    #[test]
    fn booleans_are_never_coerced() {
        let map = parse_object(r#"{"zero": 0, "empty": ""}"#);
        assert!(matches!(
            req_bool(&map, "zero"),
            Err(JsonError::UnexpectedType { .. })
        ));
        assert!(matches!(
            req_bool(&map, "empty"),
            Err(JsonError::UnexpectedType { .. })
        ));
    }

    #[test]
    fn error_messages_stay_bounded_on_hostile_field_names() {
        let long_name = "f".repeat(10_000);
        let err = JsonError::MissingField {
            field: long_name.clone(),
        };
        let text = err.to_string();
        assert!(text.len() < long_name.len());
        assert!(text.contains("missing required field"));
    }

    #[test]
    fn empty_containers_narrow_cleanly() {
        let value = parse_limited("{}").unwrap();
        assert!(object_map(&value).unwrap().is_empty());
        let value = parse_limited("[]").unwrap();
        assert!(object_list(&value).unwrap().is_empty());
    }
}
