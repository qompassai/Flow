//! CPython-compatible JSON serialization.
//!
//! Ports the exact output of CPython's `json.dumps` with the default
//! `ensure_ascii=True`: separators `", "` and `": "`, `\uXXXX` escapes
//! for every character outside printable ASCII (UTF-16 surrogate pairs
//! above the BMP, mirroring `py_encode_basestring_ascii`), and the short
//! escapes `\" \\ \b \f \n \r \t` for the characters that have them.
//! [`python_json_dumps_indent2`] additionally matches `json.dumps(indent=2)`
//! (two-space indent, `[]`/`{}` for empty containers).
//!
//! Key order follows the [`serde_json::Value`] insertion order, so the
//! crate is built with `preserve_order`: Python dicts keep insertion
//! order, and the byte-identity claims below depend on it.
//!
//! Deliberate limits of the guarantee:
//! - Floats render via `serde_json`/ryu, not CPython `repr` (`1e16`
//!   renders `"1e16"`, Python writes `"1e+16"`). The port never
//!   serializes floats through these helpers (denial envelopes and
//!   search results are strings, bools, nulls, ints, arrays, objects).
//! - Non-finite floats cannot occur in a `serde_json::Value`.
//! - Astral characters use UTF-16 surrogate pairs exactly like CPython;
//!   lone surrogates cannot exist in a Rust `&str`, so there is no
//!   `"\ud800"` edge to mirror.

use std::io;

use serde::Serialize;
use serde_json::Value;
use serde_json::ser::Formatter;

/// Serialize `value` byte-identically to CPython `json.dumps(value)`.
pub fn python_json_dumps(value: &Value) -> String {
    serialize_with(PythonFormatter::compact(), value)
}

/// Serialize `value` byte-identically to CPython `json.dumps(value,
/// indent=2)`, the form Python's web-search tool returned.
pub fn python_json_dumps_indent2(value: &Value) -> String {
    serialize_with(PythonFormatter::indent2(), value)
}

fn serialize_with(formatter: PythonFormatter, value: &Value) -> String {
    let mut out = Vec::new();
    {
        let mut serializer = serde_json::ser::Serializer::with_formatter(&mut out, formatter);
        // A `Value` holds no maps with non-string keys and no non-finite
        // floats, and a `Vec` writer never fails: serialization here is
        // infallible by construction.
        let _ = value.serialize(&mut serializer);
    }
    // The formatter only ever emits ASCII bytes (string escapes are
    // `\uXXXX`, separators and brackets are ASCII literals), so UTF-8
    // conversion cannot fail; a failure would be a corrupt internal
    // relationship, not recoverable input.
    String::from_utf8(out).expect("PythonFormatter emits ASCII-only output")
}

/// A [`Formatter`] replicating CPython `json.dumps` separators and
/// `ensure_ascii=True` string escaping.
struct PythonFormatter {
    /// Whether to pretty-print with two-space indents (`indent=2`).
    pretty: bool,
    /// Current nesting depth, in indent levels.
    depth: usize,
    /// Per-level "already holds a value" flags: the top decides whether
    /// the closing bracket goes on its own indented line (`[]`/`{}` when
    /// empty, as in Python). A stack, because a nested container must not
    /// clobber its parent's flag.
    has_value: Vec<bool>,
}

impl PythonFormatter {
    fn compact() -> PythonFormatter {
        PythonFormatter {
            pretty: false,
            depth: 0,
            has_value: Vec::new(),
        }
    }

    fn indent2() -> PythonFormatter {
        PythonFormatter {
            pretty: true,
            depth: 0,
            has_value: Vec::new(),
        }
    }

    fn write_indent<W: ?Sized + io::Write>(&self, writer: &mut W) -> io::Result<()> {
        for _ in 0..self.depth {
            writer.write_all(b"  ")?;
        }
        Ok(())
    }

    /// Mark the innermost open container as non-empty. Value callbacks
    /// only arrive between balanced begin/end calls, so the stack is
    /// never empty here.
    fn mark_has_value(&mut self) {
        let top = self
            .has_value
            .last_mut()
            .expect("array value outside a container");
        *top = true;
    }

    /// Pop the closing container's flag. The serializer only emits
    /// balanced begin/end calls, so the stack is never empty here; an
    /// empty pop would be a corrupt internal relationship.
    fn pop_has_value(&mut self) -> bool {
        self.has_value
            .pop()
            .expect("end_array/end_object without matching begin")
    }

    /// Write the pretty container trailer: newline plus indent when the
    /// container held values, then the closing bracket.
    fn end_container<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        bracket: u8,
    ) -> io::Result<()> {
        self.depth -= 1;
        let has_value = self.pop_has_value();
        if self.pretty && has_value {
            writer.write_all(b"\n")?;
            self.write_indent(writer)?;
        }
        writer.write_all(&[bracket])
    }
}

/// Escape one non-ASCII or non-printable-ASCII character the way
/// `ensure_ascii=True` does: `\uXXXX`, with UTF-16 surrogate pairs above
/// the BMP. Lowercase hex digits, as CPython emits.
fn write_python_escape<W: ?Sized + io::Write>(writer: &mut W, ch: char) -> io::Result<()> {
    let code = ch as u32;
    if code > 0xFFFF {
        // Astral character: split into a UTF-16 surrogate pair.
        let value = code - 0x10000;
        let high = 0xD800 + (value >> 10);
        let low = 0xDC00 + (value & 0x3FF);
        write!(writer, "\\u{high:04x}\\u{low:04x}")
    } else {
        write!(writer, "\\u{code:04x}")
    }
    .map_err(io::Error::other)
}

impl Formatter for PythonFormatter {
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        // The serializer routes `"`, `\`, and control characters below
        // 0x20 through `write_char_escape`; what reaches this fragment is
        // printable ASCII plus anything >= 0x7F. CPython's `ESCAPE_ASCII`
        // pattern (`[\\"]|[^\ -~]`) escapes everything outside 0x20..=0x7E,
        // notably including 0x7F, which serde_json would otherwise emit raw.
        let bytes = fragment.as_bytes();
        let mut run_start = 0usize;
        for (byte_index, ch) in fragment.char_indices() {
            if ('\u{20}'..='\u{7e}').contains(&ch) {
                continue;
            }
            if byte_index > run_start {
                writer.write_all(&bytes[run_start..byte_index])?;
            }
            write_python_escape(writer, ch)?;
            run_start = byte_index + ch.len_utf8();
        }
        writer.write_all(&bytes[run_start..])
    }

    fn begin_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.depth += 1;
        self.has_value.push(false);
        writer.write_all(b"[")
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.end_container(writer, b']')
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if self.pretty {
            writer.write_all(if first { b"\n" } else { b",\n" })?;
            self.write_indent(writer)?;
        } else if !first {
            writer.write_all(b", ")?;
        }
        self.mark_has_value();
        Ok(())
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.depth += 1;
        self.has_value.push(false);
        writer.write_all(b"{")
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.end_container(writer, b'}')
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if self.pretty {
            writer.write_all(if first { b"\n" } else { b",\n" })?;
            self.write_indent(writer)?;
        } else if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        // Python's item separator is "," (no space) when indenting, ", "
        // otherwise; the key separator is always ": ".
        writer.write_all(b": ")
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.mark_has_value();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Fixture generated by driving the real Python:
    /// `json.dumps([{"title": "caf\u00e9 \U0001f600", ...}], indent=2)`.
    /// Captured 2026-09-26; see the F3 fix report for the exact command.
    const PYTHON_INDENT2: &str = "[\n  {\n    \"title\": \"caf\\u00e9 \\ud83d\\ude00\",\n    \"url\": \"https://ex.com/?q=caf\\u00e9\",\n    \"snippet\": \"na\\u00efve \\\"quoted\\\"\\nnewline\\ttab\\u0001\\u007f\"\n  }\n]";

    /// Same payload through `json.dumps(...)` (compact, default
    /// separators).
    const PYTHON_COMPACT: &str = "[{\"title\": \"caf\\u00e9 \\ud83d\\ude00\", \"url\": \"https://ex.com/?q=caf\\u00e9\", \"snippet\": \"na\\u00efve \\\"quoted\\\"\\nnewline\\ttab\\u0001\\u007f\"}]";

    fn fixture_value() -> Value {
        json!([{
            "title": "caf\u{e9} \u{1f600}",
            "url": "https://ex.com/?q=caf\u{e9}",
            "snippet": "na\u{ef}ve \"quoted\"\nnewline\ttab\u{1}\u{7f}",
        }])
    }

    #[test]
    fn compact_matches_python_json_dumps() {
        assert_eq!(python_json_dumps(&fixture_value()), PYTHON_COMPACT);
    }

    #[test]
    fn indent2_matches_python_json_dumps_indent2() {
        assert_eq!(python_json_dumps_indent2(&fixture_value()), PYTHON_INDENT2);
    }

    #[test]
    fn empty_containers_match_python() {
        assert_eq!(python_json_dumps(&json!([])), "[]");
        assert_eq!(python_json_dumps(&json!({})), "{}");
        assert_eq!(python_json_dumps_indent2(&json!([])), "[]");
        assert_eq!(python_json_dumps_indent2(&json!({})), "{}");
    }

    #[test]
    fn nested_indent_matches_python() {
        // json.dumps({"a": [1, {"b": None}]}, indent=2)
        let python = "{\n  \"a\": [\n    1,\n    {\n      \"b\": null\n    }\n  ]\n}";
        let value = json!({"a": [1, {"b": Value::Null}]});
        assert_eq!(python_json_dumps_indent2(&value), python);
    }

    #[test]
    fn nested_empty_containers_match_python() {
        // json.dumps({"a": [1, []]}, indent=2): the nested empty array
        // must not clobber the outer container's non-empty flag.
        let python = "{\n  \"a\": [\n    1,\n    []\n  ]\n}";
        let value = json!({"a": [1, []]});
        assert_eq!(python_json_dumps_indent2(&value), python);
        // And the reverse: an empty object nested in an empty object.
        assert_eq!(
            python_json_dumps_indent2(&json!({"a": {}})),
            "{\n  \"a\": {}\n}"
        );
    }

    #[test]
    fn key_order_follows_insertion_order() {
        // Python dicts keep insertion order; preserve_order must be on.
        let value = json!({"status": "error", "error": "x"});
        assert_eq!(
            python_json_dumps(&value),
            r#"{"status": "error", "error": "x"}"#
        );
    }

    #[test]
    fn control_escapes_match_python() {
        // json.dumps with control characters: short escapes where CPython
        // has them, \u00XX elsewhere, lowercase hex.
        let value = json!({"s": "\u{8}\u{c}\n\r\t\u{0}\u{1f}"});
        let expected = r#"{"s": "\b\f\n\r\t\u0000\u001f"}"#;
        assert_eq!(python_json_dumps(&value), expected);
    }
}
