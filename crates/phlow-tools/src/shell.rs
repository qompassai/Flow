//! The fail-closed shell stub.
//!
//! Python's `make_shell_tool` returns a tool whose `run` always returns the
//! denial JSON; the Rust port keeps that shape. There is no code path that
//! spawns a shell: [`ToolSpec`] describes the tool for schema listings, and
//! [`shell_denied`] produces the exact JSON the Python tool returned.

use phlow_checks::disabled::SHELL_DENIAL;
use serde_json::{Value, json};

use crate::json_compat::python_json_dumps;
use crate::registry::ToolSpec;

/// Registry name of the removed tool, unchanged from Python.
pub const SHELL_TOOL_NAME: &str = "shell_exec";

/// The static tool spec: name, description, and an empty parameter schema,
/// mirroring Python's `TOOL_SPEC`.
pub fn tool_spec() -> ToolSpec {
    ToolSpec {
        name: SHELL_TOOL_NAME,
        description: "DISABLED: use configured named checks",
        parameters: json!({"type": "object", "properties": {}}),
    }
}

/// The denial envelope: `{"status": "error", "error": <SHELL_DENIAL>}`,
/// keyed in Python dict order.
pub fn shell_denied() -> Value {
    json!({"status": "error", "error": SHELL_DENIAL})
}

/// [`shell_denied`] serialized byte-identically to the Python tool's
/// `json.dumps` output: `", "`/`": "` separators and `ensure_ascii`
/// escaping via [`python_json_dumps`](crate::json_compat::python_json_dumps).
pub fn shell_denied_json() -> String {
    python_json_dumps(&shell_denied())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Python tool's denial JSON, captured by driving the real
    /// Python `make_shell_tool` on 2026-09-26.
    const PYTHON_DENIED: &str = r#"{"status": "error", "error": "Arbitrary commands and cwd overrides are disabled. Configure exact named check argv and use flow_check."}"#;

    #[test]
    fn denial_matches_python_byte_for_byte() {
        assert_eq!(shell_denied_json(), PYTHON_DENIED);
    }

    #[test]
    fn denial_json_string_parses_to_same_envelope() {
        let parsed: Value = serde_json::from_str(&shell_denied_json()).unwrap();
        assert_eq!(parsed, shell_denied());
    }

    #[test]
    fn tool_spec_names_shell_exec() {
        let spec = tool_spec();
        assert_eq!(spec.name, "shell_exec");
        assert_eq!(spec.parameters, json!({"type": "object", "properties": {}}));
    }
}
