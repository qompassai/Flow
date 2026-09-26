//! The static, non-executable tool registry.
//!
//! Ports `flow/tools/registry.py`. The registry describes tools (name,
//! description, JSON Schema parameters) and dispatches validated calls to
//! the built-in handlers. It never holds executable callbacks: Python's
//! `register(name, description, parameters, func)` accepted arbitrary
//! callables, but the safe runtime forbids model-reachable executable
//! authority, so the tool set is a fixed enum. [`BuiltinTool`] pairs each
//! spec with its handler, so the compiler rejects a spec without a handler
//! and vice versa.
//!
//! [`ToolRegistry::load_plugins`] always fails with the exact Python denial,
//! and [`ToolRegistry::parse_tool_call`] always returns `None`, as the
//! Python implementation did.

use serde_json::{Value, json};

use crate::error::ToolError;
use crate::web_search::WebSearch;

/// A tool's static description: name, human description, and the JSON
/// Schema its arguments are validated against before dispatch.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Registry name, e.g. `"web_search"`.
    pub name: &'static str,
    /// One-line description for schema listings.
    pub description: &'static str,
    /// JSON Schema for the tool's arguments object.
    pub parameters: Value,
}

impl ToolSpec {
    /// The OpenAI function-calling schema shape:
    /// `{"type": "function", "function": {name, description, parameters}}`.
    pub fn to_openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        })
    }
}

/// The fixed tool set: web search plus the two fail-closed stubs.
/// Executable plugins can never join this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuiltinTool {
    WebSearch,
    ShellExec,
    LspCheck,
}

impl BuiltinTool {
    /// Every tool, in registry order.
    fn all() -> [BuiltinTool; 3] {
        [
            BuiltinTool::WebSearch,
            BuiltinTool::ShellExec,
            BuiltinTool::LspCheck,
        ]
    }

    fn from_name(name: &str) -> Option<BuiltinTool> {
        BuiltinTool::all()
            .into_iter()
            .find(|tool| tool.spec().name == name)
    }

    /// The static spec, mirroring Python's `TOOL_SPEC`s.
    fn spec(self) -> ToolSpec {
        match self {
            BuiltinTool::WebSearch => ToolSpec {
                name: "web_search",
                description: "Search the web for information. Returns titles, URLs, and snippets.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return",
                        },
                    },
                    "required": ["query"],
                }),
            },
            BuiltinTool::ShellExec => crate::shell::tool_spec(),
            BuiltinTool::LspCheck => crate::lsp_check::tool_spec(),
        }
    }

    /// Run the tool's built-in handler. Arguments were already validated
    /// against [`BuiltinTool::spec`]; returns the JSON string the tool
    /// produced, as Python's tools returned JSON strings.
    fn run(self, registry: &ToolRegistry, args: &Value) -> String {
        match self {
            BuiltinTool::WebSearch => {
                let query = args.get("query").and_then(Value::as_str).unwrap_or("");
                let max_results = args.get("max_results").and_then(Value::as_i64).unwrap_or(5);
                registry.web_search.run(query, max_results)
            }
            BuiltinTool::ShellExec => crate::shell::shell_denied_json(),
            BuiltinTool::LspCheck => {
                let language = args.get("language").and_then(Value::as_str).unwrap_or("");
                crate::lsp_check::lsp_check_denied_json(language)
            }
        }
    }
}

/// The static tool registry. Owns the search backend; dispatch validates
/// arguments with the canonical MCP schema validator before running.
pub struct ToolRegistry {
    web_search: WebSearch,
}

impl ToolRegistry {
    /// Build the registry with the default DuckDuckGo search backend.
    pub fn with_defaults() -> Result<ToolRegistry, ToolError> {
        Ok(ToolRegistry {
            web_search: WebSearch::duckduckgo()?,
        })
    }

    /// Build the registry with an explicit search backend.
    pub fn with_search(web_search: WebSearch) -> ToolRegistry {
        ToolRegistry { web_search }
    }

    /// Look up a tool's spec by name.
    pub fn get(&self, name: &str) -> Option<ToolSpec> {
        BuiltinTool::from_name(name).map(|tool| tool.spec())
    }

    /// Every registered tool's spec, in registry order.
    pub fn all_specs(&self) -> Vec<ToolSpec> {
        BuiltinTool::all().iter().map(|tool| tool.spec()).collect()
    }

    /// The OpenAI function-calling schemas for every tool.
    pub fn openai_tools_schema(&self) -> Vec<Value> {
        self.all_specs()
            .iter()
            .map(ToolSpec::to_openai_schema)
            .collect()
    }

    /// `"name: description"` lines, as Python's `tool_descriptions`.
    pub fn tool_descriptions(&self) -> String {
        self.all_specs()
            .iter()
            .map(|spec| format!("{}: {}", spec.name, spec.description))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Validate `args` against the tool's schema and run the built-in
    /// handler. Returns the JSON string the tool produced, mirroring
    /// Python's `dispatch` (whose tools returned JSON strings).
    ///
    /// Unknown tools and schema violations return the
    /// `{"status": "error", ...}` envelope instead of raising.
    pub fn dispatch(&self, tool_name: &str, args: &Value) -> Value {
        let Some(tool) = BuiltinTool::from_name(tool_name) else {
            // Route through ToolError so the model-supplied name is
            // truncated at TOOL_NAME_CHARS_MAX; the message shape matches
            // Python for names within the bound.
            let error = ToolError::UnknownTool(tool_name.to_string());
            return json!({"status": "error", "error": error.to_string()});
        };
        let spec = tool.spec();
        if let Err(error) = phlow_mcp::validate_arguments(args, &spec.parameters) {
            return json!({"status": "error", "error": error.to_string()});
        }
        Value::String(tool.run(self, args))
    }

    /// Loading executable plugins is disabled in the safe runtime. Always
    /// fails with the exact Python denial.
    pub fn load_plugins(&self, _plugins_dir: &str) -> Result<(), ToolError> {
        Err(ToolError::PluginsDisabled)
    }

    /// The Python implementation never parsed free-text tool calls; always
    /// `None`.
    pub fn parse_tool_call(&self, _text: &str) -> Option<()> {
        None
    }
}

/// Loading user tools requires plugin loading, which is disabled.
/// Mirrors `flow/tools/user_tools.py`: the call fails instead of
/// registering anything.
pub fn load_user_tools(_registry: &ToolRegistry, plugins_dir: &str) -> Result<usize, ToolError> {
    let _ = plugins_dir;
    Err(ToolError::PluginsDisabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ToolRegistry {
        ToolRegistry::with_defaults().unwrap()
    }

    #[test]
    fn unknown_tool_returns_error_envelope() {
        let result = registry().dispatch("nope", &json!({}));
        assert_eq!(
            result,
            json!({"status": "error", "error": "Unknown tool: nope"})
        );
    }

    #[test]
    fn unknown_tool_name_is_truncated_at_tool_name_chars_max() {
        let long_name = "t".repeat(crate::error::TOOL_NAME_CHARS_MAX + 50);
        let result = registry().dispatch(&long_name, &json!({}));
        let shown = "t".repeat(crate::error::TOOL_NAME_CHARS_MAX);
        assert_eq!(
            result,
            json!({"status": "error", "error": format!("Unknown tool: {shown}")})
        );
    }

    #[test]
    fn missing_required_argument_returns_error_envelope() {
        let result = registry().dispatch("web_search", &json!({}));
        assert_eq!(result["status"], "error");
        let message = result["error"].as_str().unwrap();
        assert!(message.contains("query"), "unexpected message: {message}");
    }

    #[test]
    fn shell_dispatch_returns_python_denial() {
        let result = registry().dispatch("shell_exec", &json!({}));
        let rendered = result.as_str().unwrap();
        let parsed: Value = serde_json::from_str(rendered).unwrap();
        assert_eq!(parsed, crate::shell::shell_denied());
    }

    #[test]
    fn lsp_check_dispatch_returns_python_denial() {
        let result = registry().dispatch(
            "lsp_check",
            &json!({"language": "rust", "file_path": "src/main.rs"}),
        );
        let rendered = result.as_str().unwrap();
        let parsed: Value = serde_json::from_str(rendered).unwrap();
        assert_eq!(parsed["status"], "unavailable");
        assert_eq!(parsed["language"], "rust");
    }

    #[test]
    fn load_plugins_fails_with_exact_python_denial() {
        let error = registry().load_plugins("./plugins").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Executable plugins are disabled. Use configured named checks."
        );
    }

    #[test]
    fn load_user_tools_fails_closed() {
        let registry = registry();
        let error = load_user_tools(&registry, "./plugins").unwrap_err();
        assert!(matches!(error, ToolError::PluginsDisabled));
    }

    #[test]
    fn parse_tool_call_always_none() {
        assert_eq!(registry().parse_tool_call("run the tests"), None);
    }

    #[test]
    fn openai_schema_has_function_shape() {
        let schemas = registry().openai_tools_schema();
        assert_eq!(schemas.len(), 3);
        for schema in &schemas {
            assert_eq!(schema["type"], "function");
            assert!(schema["function"]["name"].is_string());
            assert!(schema["function"]["description"].is_string());
            assert!(schema["function"]["parameters"].is_object());
        }
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["web_search", "shell_exec", "lsp_check"]);
    }

    #[test]
    fn tool_descriptions_lists_all_tools() {
        let descriptions = registry().tool_descriptions();
        for name in ["web_search", "shell_exec", "lsp_check"] {
            assert!(descriptions.contains(name), "missing {name}");
        }
        assert_eq!(descriptions.lines().count(), 3);
    }
}
