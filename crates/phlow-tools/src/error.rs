//! Typed errors for the tool surfaces.
//!
//! Every failure a tool can report is one of these variants. Tool results
//! that cross a trust boundary (model input, HTTP, the filesystem) are
//! validated before they become values; these errors are what validation
//! and execution produce instead of panics.

use std::fmt;

/// A tool name shown in an error is at most this many characters; model
/// input is untrusted and must not flood logs.
pub const TOOL_NAME_CHARS_MAX: usize = 128;

/// An HTTP error message carried in a result is truncated to this many
/// characters; servers choose these strings, callers do not.
pub const HTTP_ERROR_CHARS_MAX: usize = 300;

/// All failure modes of the tool surfaces.
#[derive(Debug)]
pub enum ToolError {
    /// No tool is registered under the requested name.
    UnknownTool(String),
    /// Arguments failed JSON Schema validation. Carries the message.
    InvalidArguments(String),
    /// A caller asked for executable plugin loading, which is disabled.
    PluginsDisabled,
    /// An HTTP transport failure (DNS, connect, TLS, timeout).
    Http(String),
    /// The search backend answered with a non-2xx status.
    BadStatus(u16),
    /// The search backend answered with unparsable JSON.
    BadJson(String),
    /// A file operation failed inside the workspace.
    Workspace(phlow_workspace::WorkspaceError),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::UnknownTool(name) => {
                let shown: String = name.chars().take(TOOL_NAME_CHARS_MAX).collect();
                write!(f, "Unknown tool: {shown}")
            }
            ToolError::InvalidArguments(message) => write!(f, "{message}"),
            ToolError::PluginsDisabled => write!(
                f,
                "Executable plugins are disabled. Use configured named checks."
            ),
            ToolError::Http(message) => write!(f, "HTTP error: {message}"),
            ToolError::BadStatus(status) => write!(f, "search backend returned status {status}"),
            ToolError::BadJson(message) => write!(f, "search backend returned bad JSON: {message}"),
            ToolError::Workspace(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ToolError::Workspace(error) => Some(error),
            _ => None,
        }
    }
}

impl From<phlow_workspace::WorkspaceError> for ToolError {
    fn from(error: phlow_workspace::WorkspaceError) -> ToolError {
        ToolError::Workspace(error)
    }
}
