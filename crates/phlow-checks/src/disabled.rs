//! Fail-closed denials for removed execution paths.
//!
//! The legacy `shell_exec` and `lsp_check` tools used to pick commands
//! implicitly. That is gone: these constructors return the exact denial the
//! Python implementation produced, as a typed value. Callers surface the
//! denial to the model; nothing here ever spawns a process.
//!
//! `phlow-tools` (Phase 5) surfaces these at the tool-registry boundary;
//! they live here so the denial text has one canonical home.

/// Denial for the removed `shell_exec` tool: arbitrary commands and cwd
/// overrides are disabled; configure exact named check argv instead.
pub const SHELL_DENIAL: &str = "Arbitrary commands and cwd overrides are disabled. \
    Configure exact named check argv and use flow_check.";

/// Denial for the removed `lsp_check` tool: no implicit language command
/// execution; configure a named lint/typecheck or attach editor tools.
pub const LSP_CHECK_DENIAL: &str = "No implicit language command execution. \
    Configure a named lint/typecheck in operator TOML or attach Rose's native editor tools.";

/// Which removed tool produced a denial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisabledTool {
    /// The removed `shell_exec` tool.
    ShellExec,
    /// The removed `lsp_check` tool.
    LspCheck,
}

impl DisabledTool {
    /// Registry name of the removed tool.
    pub fn name(self) -> &'static str {
        match self {
            DisabledTool::ShellExec => "shell_exec",
            DisabledTool::LspCheck => "lsp_check",
        }
    }
}

/// A fail-closed denial for a removed execution path.
///
/// Carries the exact denial text the Python implementation returned.
/// `status` mirrors the Python JSON envelope: `"error"` for `shell_exec`,
/// `"unavailable"` for `lsp_check`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisabledError {
    tool: DisabledTool,
    message: &'static str,
    language: Option<String>,
}

impl DisabledError {
    /// Denial for any use of the removed `shell_exec` tool.
    pub fn shell_exec() -> DisabledError {
        DisabledError {
            tool: DisabledTool::ShellExec,
            message: SHELL_DENIAL,
            language: None,
        }
    }

    /// Denial for any use of the removed `lsp_check` tool.
    pub fn lsp_check(language: impl Into<String>) -> DisabledError {
        DisabledError {
            tool: DisabledTool::LspCheck,
            message: LSP_CHECK_DENIAL,
            language: Some(language.into()),
        }
    }

    /// Which removed tool was invoked.
    pub fn tool(&self) -> DisabledTool {
        self.tool
    }

    /// The denial text, byte-identical to the Python implementation.
    pub fn message(&self) -> &'static str {
        self.message
    }

    /// Envelope status: `"error"` for shell, `"unavailable"` for lsp_check.
    pub fn status(&self) -> &'static str {
        match self.tool {
            DisabledTool::ShellExec => "error",
            DisabledTool::LspCheck => "unavailable",
        }
    }

    /// The language the caller asked about, for `lsp_check` denials.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// The denied tool never produces clean output.
    pub fn clean(&self) -> bool {
        false
    }

    /// The denied tool never verifies anything.
    pub fn verified(&self) -> bool {
        false
    }
}

impl std::fmt::Display for DisabledError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.tool.name(), self.message)
    }
}

impl std::error::Error for DisabledError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_denial_text_is_exact() {
        let denial = DisabledError::shell_exec();
        assert_eq!(denial.tool(), DisabledTool::ShellExec);
        assert_eq!(denial.tool.name(), "shell_exec");
        assert_eq!(denial.status(), "error");
        assert_eq!(
            denial.message(),
            "Arbitrary commands and cwd overrides are disabled. \
             Configure exact named check argv and use flow_check."
        );
        assert!(!denial.clean());
        assert!(!denial.verified());
        assert!(denial.language().is_none());
    }

    #[test]
    fn lsp_check_denial_text_is_exact() {
        let denial = DisabledError::lsp_check("rust");
        assert_eq!(denial.tool(), DisabledTool::LspCheck);
        assert_eq!(denial.status(), "unavailable");
        assert_eq!(
            denial.message(),
            "No implicit language command execution. \
             Configure a named lint/typecheck in operator TOML or attach \
             Rose's native editor tools."
        );
        assert_eq!(denial.language(), Some("rust"));
        assert!(!denial.clean());
        assert!(!denial.verified());
    }

    #[test]
    fn denial_display_names_tool() {
        let text = DisabledError::shell_exec().to_string();
        assert!(text.starts_with("shell_exec: "));
        assert!(text.contains("flow_check"));
    }
}
