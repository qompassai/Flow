//! Python-compatible JSON serialization for MCP wire frames.
//!
//! Python's server emits every frame with
//! `json.dumps(payload, ensure_ascii=True, allow_nan=False)` (and the
//! default `", "` / `": "` separators). `serde_json::to_string` differs in
//! three ways: it emits compact separators, it passes non-ASCII through as
//! UTF-8, and its float exponent style is `1e16` where Python writes
//! `1e+16`. [`dumps`] replicates the Python form exactly so golden tests
//! can compare Rust frames byte-for-byte against Python-produced frames.
//!
//! Key order is insertion order (Python dict order). This crate enables
//! `serde_json/preserve_order` so maps built in the same insertion order
//! serialize identically.

use serde_json::Value;
use std::fmt::{self, Write};

/// Error from [`dumps`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DumpsError {
    /// Python's `allow_nan=False` rejects non-finite floats. A
    /// `serde_json::Value` cannot hold one (construction returns `None`),
    /// so this is unreachable through safe code; it exists so the
    /// serializer stays total instead of panicking.
    NonFiniteFloat,
}

impl fmt::Display for DumpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DumpsError::NonFiniteFloat => write!(f, "non-finite float is not valid JSON"),
        }
    }
}

impl std::error::Error for DumpsError {}

/// Serialize `value` like `json.dumps(value, ensure_ascii=True,
/// allow_nan=False)`.
///
/// Separators are `", "` and `": "`; every non-ASCII character is escaped
/// as `\uXXXX` (with surrogate pairs above the BMP); object keys keep
/// insertion order.
pub fn dumps(value: &Value) -> Result<String, DumpsError> {
    let mut out = String::new();
    write_value(&mut out, value)?;
    Ok(out)
}

fn write_value(out: &mut String, value: &Value) -> Result<(), DumpsError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(out, number)?,
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(out, item)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// True when the number is written as a JSON integer literal: an optional
/// leading `-` followed by ASCII digits. This is exactly the set of JSON
/// numbers Python's `json` decodes to `int` (and therefore accepts as a
/// request id); `1e3` and `100.0` decode to `float` and are invalid ids.
///
/// With `serde_json/arbitrary_precision`, `Number::to_string` keeps the
/// source text of integers beyond `u64` (smaller integers are stored
/// numerically), so this check also gates the big-int echo path.
pub(crate) fn is_integer_text(number: &serde_json::Number) -> bool {
    let text = number.to_string();
    let digits = text.strip_prefix('-').unwrap_or(&text);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn write_number(out: &mut String, number: &serde_json::Number) -> Result<(), DumpsError> {
    if let Some(int) = number.as_i64() {
        write!(out, "{int}").expect("writing to a String cannot fail");
        return Ok(());
    }
    if let Some(uint) = number.as_u64() {
        write!(out, "{uint}").expect("writing to a String cannot fail");
        return Ok(());
    }
    if is_integer_text(number) {
        // Integer beyond u64: arbitrary_precision kept the source digits;
        // echo them verbatim, like Python's arbitrary-precision int.
        out.push_str(&number.to_string());
        return Ok(());
    }
    let Some(float) = number.as_f64() else {
        // Unreachable through the server's paths (only validated ids are
        // ever serialized), but total: a non-finite float is not valid JSON.
        return Err(DumpsError::NonFiniteFloat);
    };
    if !float.is_finite() {
        return Err(DumpsError::NonFiniteFloat);
    }
    out.push_str(&format_python_float(float));
    Ok(())
}

/// Shortest round-trip float text in Python `repr` style.
///
/// `ryu` already produces the shortest round-trip digits and the same
/// fixed-vs-scientific switchover as CPython; only the exponent needs
/// reshaping (`1e16` -> `1e+16`, `1.5e-7` -> `1.5e-07`).
fn format_python_float(float: f64) -> String {
    let mut buffer = ryu::Buffer::new();
    let text = buffer.format_finite(float);
    let Some(exp_pos) = text.find('e') else {
        return text.to_owned();
    };
    let (mantissa, exponent) = text.split_at(exp_pos);
    let exp_value: i32 = exponent[1..].parse().expect("ryu emits a decimal exponent");
    format!("{mantissa}e{exp_value:+03}")
}

/// Escape a string like Python's `json.dumps(..., ensure_ascii=True)`.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ch if (ch as u32) < 0x20 || ch as u32 == 0x7f => {
                write!(out, "\\u{:04x}", ch as u32).expect("writing to a String cannot fail");
            }
            ch if (ch as u32) < 0x7f => out.push(ch),
            ch => {
                let code = ch as u32;
                if code < 0x10000 {
                    write!(out, "\\u{code:04x}").expect("writing to a String cannot fail");
                } else {
                    // Surrogate pair, exactly as CPython's ensure_ascii does.
                    let shifted = code - 0x10000;
                    let high = 0xd800 + (shifted >> 10);
                    let low = 0xdc00 + (shifted & 0x3ff);
                    write!(out, "\\u{high:04x}\\u{low:04x}")
                        .expect("writing to a String cannot fail");
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn separators_match_python_defaults() {
        assert_eq!(
            dumps(&json!({"a": 1, "b": [1, 2]})).unwrap(),
            r#"{"a": 1, "b": [1, 2]}"#
        );
    }

    #[test]
    fn ensure_ascii_escapes() {
        // Python: json.dumps("h\u00e9llo \u0001\u007f\U0001f600", ensure_ascii=True)
        let value = json!("h\u{e9}llo \u{1}\u{7f}\u{1f600}");
        assert_eq!(
            dumps(&value).unwrap(),
            r#""h\u00e9llo \u0001\u007f\ud83d\ude00""#
        );
    }

    #[test]
    fn short_escapes_match_python() {
        assert_eq!(
            dumps(&json!("a\"b\\c\nd\re\tf\x08g\x0c")).unwrap(),
            r#""a\"b\\c\nd\re\tf\bg\f""#
        );
    }

    #[test]
    fn float_exponent_style_matches_python_repr() {
        // Values checked against CPython repr(): repr(1e16) == '1e+16'.
        for (float, expected) in [
            (0.2f64, "0.2"),
            (1.0f64, "1.0"),
            (-0.0f64, "-0.0"),
            (1e16f64, "1e+16"),
            (1.5e-7f64, "1.5e-07"),
            (123456.789f64, "123456.789"),
        ] {
            assert_eq!(format_python_float(float), expected, "for {float}");
        }
    }

    #[test]
    fn insertion_order_is_preserved() {
        let mut map = serde_json::Map::new();
        map.insert("z".to_owned(), json!(1));
        map.insert("a".to_owned(), json!(2));
        assert_eq!(dumps(&Value::Object(map)).unwrap(), r#"{"z": 1, "a": 2}"#);
    }
}
