//! The fail-closed LSP check stub.
//!
//! Python's `lsp_check` dispatched to a language server command implicitly,
//! which the safe runtime removed. The port keeps the result shape
//! (`status`/`clean`/`verified`/`language`/`error`) with the canonical
//! denial text, so callers that branch on `clean`/`verified` keep working.
//! There is no code path that executes a language command.

use phlow_checks::disabled::LSP_CHECK_DENIAL;
use serde_json::{Value, json};

use crate::json_compat::python_json_dumps;
use crate::registry::ToolSpec;

/// Registry name of the removed tool, unchanged from Python.
pub const LSP_CHECK_TOOL_NAME: &str = "lsp_check";

/// A language label carried in a denial is truncated to this many
/// characters; it comes from model input.
pub const LANGUAGE_CHARS_MAX: usize = 64;

/// The static tool spec: name, description, and the parameter schema.
/// Python's registered `TOOL_SPEC` had empty properties, but the tool
/// itself accepted `language` and `file_path`; the schema here documents
/// the real interface so the denial can echo the language as Python did.
/// Both are optional: the tool is disabled regardless of arguments.
pub fn tool_spec() -> ToolSpec {
    ToolSpec {
        name: LSP_CHECK_TOOL_NAME,
        description: "DISABLED: use configured named checks or editor tools",
        parameters: json!({
            "type": "object",
            "properties": {
                "language": {"type": "string", "description": "Language of the file to check"},
                "file_path": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to check",
                },
            },
        }),
    }
}

/// The denial envelope, keyed in Python dict order: `{"status":
/// "unavailable", "clean": false, "verified": false, "language":
/// <language>, "error": <LSP_CHECK_DENIAL>}`.
pub fn lsp_check_denied(language: &str) -> Value {
    let language: String = language.chars().take(LANGUAGE_CHARS_MAX).collect();
    json!({
        "status": "unavailable",
        "clean": false,
        "verified": false,
        "language": language,
        "error": LSP_CHECK_DENIAL,
    })
}

/// [`lsp_check_denied`] serialized byte-identically to the Python
/// tool's `json.dumps` output: `", "`/`": "` separators and
/// `ensure_ascii` escaping via
/// [`python_json_dumps`](crate::json_compat::python_json_dumps).
pub fn lsp_check_denied_json(language: &str) -> String {
    python_json_dumps(&lsp_check_denied(language))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Python tool's denial JSON for `run_lsp_check("rust", ...)`,
    /// captured by driving the real Python on 2026-09-26.
    const PYTHON_DENIED_RUST: &str = r#"{"status": "unavailable", "clean": false, "verified": false, "language": "rust", "error": "No implicit language command execution. Configure a named lint/typecheck in operator TOML or attach Rose's native editor tools."}"#;

    #[test]
    fn denial_matches_python_byte_for_byte() {
        assert_eq!(lsp_check_denied_json("rust"), PYTHON_DENIED_RUST);
    }

    #[test]
    fn denial_json_escapes_non_ascii_like_python() {
        // Python ensure_ascii=True: json.dumps({"language": "café", ...})
        // renders "caf\u00e9".
        let rendered = lsp_check_denied_json("café");
        assert!(rendered.contains(r#""language": "caf\u00e9""#));
        assert!(!rendered.contains("café"));
    }

    #[test]
    fn denial_json_string_parses_to_same_envelope() {
        let parsed: Value = serde_json::from_str(&lsp_check_denied_json("rust")).unwrap();
        assert_eq!(parsed, lsp_check_denied("rust"));
    }

    #[test]
    fn denial_carries_language_and_unavailable_shape() {
        let value = lsp_check_denied("go");
        assert_eq!(value["status"], "unavailable");
        assert_eq!(value["clean"], false);
        assert_eq!(value["verified"], false);
        assert_eq!(value["language"], "go");
    }

    #[test]
    fn language_label_is_bounded() {
        let long = "x".repeat(LANGUAGE_CHARS_MAX + 100);
        let value = lsp_check_denied(&long);
        assert_eq!(
            value["language"].as_str().unwrap().chars().count(),
            LANGUAGE_CHARS_MAX
        );
    }

    #[test]
    fn tool_spec_names_lsp_check() {
        assert_eq!(tool_spec().name, "lsp_check");
    }
}
