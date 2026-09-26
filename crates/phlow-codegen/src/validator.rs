//! The fail-closed code validator.
//!
//! Ports `flow/codegen/validator.py`. The Python validator dispatched to
//! the disabled `lsp_check` tool; the port dispatches through the real
//! [`ToolRegistry`](phlow_tools::ToolRegistry), so the denial envelope is
//! produced by the same code path the runtime uses. The validator never
//! executes a language command: `clean` is always false, and
//! [`CodeValidator::format_for_agent`] always renders the
//! `LSP Error: <error>` branch.

use serde_json::{Map, Value};

use crate::error::CodegenError;

/// The pretty-printed issues JSON in [`CodeValidator::format_for_agent`]
/// is cut at 2000 characters, as in Python.
pub const VALIDATOR_JSON_CHARS_MAX: usize = 2000;

/// Validates generated code against language tooling. The tooling is
/// disabled in the safe runtime, so every validation reports the
/// fail-closed denial.
pub struct CodeValidator {
    registry: phlow_tools::ToolRegistry,
}

impl CodeValidator {
    /// Build the validator over the static tool registry.
    pub fn new() -> Result<CodeValidator, CodegenError> {
        Ok(CodeValidator {
            registry: phlow_tools::ToolRegistry::with_defaults()?,
        })
    }

    /// Run the (disabled) `lsp_check` tool and return its result envelope.
    /// `file_path` is accepted for interface parity with the Python
    /// validator; the disabled tool never reads it.
    pub fn validate(&self, language: &str, file_path: Option<&str>) -> Value {
        let mut args = Map::new();
        args.insert("language".to_string(), Value::String(language.to_string()));
        if let Some(path) = file_path {
            args.insert("file_path".to_string(), Value::String(path.to_string()));
        }
        let rendered = self.registry.dispatch("lsp_check", &Value::Object(args));
        // The disabled tool always returns a JSON string; if that envelope
        // ever fails to parse, fall back to the local denial so callers
        // still see the fail-closed shape.
        rendered
            .as_str()
            .and_then(|text| serde_json::from_str(text).ok())
            .unwrap_or_else(|| phlow_tools::lsp_check::lsp_check_denied(language))
    }

    /// Whether the language tooling reports the code clean. Always false:
    /// the tooling is disabled, and an unverified claim of cleanliness
    /// would be a fabrication.
    pub fn is_clean(&self, language: &str, file_path: Option<&str>) -> bool {
        let result = self.validate(language, file_path);
        result.get("status").and_then(Value::as_str) == Some("ok")
            && result.get("clean").and_then(Value::as_bool) == Some(true)
    }

    /// Format the validation result for the agent, preserving Python's
    /// three branches: `LSP Error: <error>`, the clean checkmark (dead in
    /// the safe runtime), and `LSP Issues (<language>):` with bounded
    /// pretty JSON.
    pub fn format_for_agent(&self, language: &str, file_path: Option<&str>) -> String {
        let result = self.validate(language, file_path);
        if let Some(error) = result.get("error").and_then(Value::as_str) {
            return format!("LSP Error: {error}");
        }
        if self.is_clean(language, file_path) {
            return format!("✓ {language} code passes all checks.");
        }
        let pretty = serde_json::to_string_pretty(&result).unwrap_or_else(|_| String::from("{}"));
        let bounded: String = pretty.chars().take(VALIDATOR_JSON_CHARS_MAX).collect();
        format!("LSP Issues ({language}):\n{bounded}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical denial text the validator must surface.
    const LSP_DENIAL: &str = "No implicit language command execution. Configure a named lint/typecheck in operator TOML or attach Rose's native editor tools.";

    #[test]
    fn validate_returns_the_disabled_envelope() {
        let validator = CodeValidator::new().unwrap();
        let result = validator.validate("rust", Some("src/main.rs"));
        assert_eq!(result["status"], "unavailable");
        assert_eq!(result["clean"], false);
        assert_eq!(result["verified"], false);
        assert_eq!(result["language"], "rust");
        assert_eq!(result["error"], LSP_DENIAL);
    }

    #[test]
    fn validate_without_file_path_still_denies() {
        let validator = CodeValidator::new().unwrap();
        let result = validator.validate("python", None);
        assert_eq!(result["status"], "unavailable");
        assert_eq!(result["language"], "python");
    }

    #[test]
    fn is_clean_is_always_false() {
        let validator = CodeValidator::new().unwrap();
        assert!(!validator.is_clean("rust", Some("src/main.rs")));
        assert!(!validator.is_clean("python", None));
    }

    #[test]
    fn format_for_agent_reports_lsp_error() {
        let validator = CodeValidator::new().unwrap();
        let text = validator.format_for_agent("rust", Some("src/main.rs"));
        assert_eq!(text, format!("LSP Error: {LSP_DENIAL}"));
    }
}
