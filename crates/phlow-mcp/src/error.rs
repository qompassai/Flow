//! Crate-level error types for `phlow-mcp`.

use std::fmt;

/// A runtime failure behind a tool call.
///
/// The Python server catches any exception the runtime raises and reports
/// the tool as failed (`isError: true`) with the exception text. The
/// [`crate::server::McpRuntime`] trait models that with `Result`: `Err`
/// carries the same message text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError(pub String);

impl RuntimeError {
    /// Build from any displayable failure value.
    pub fn new(message: impl fmt::Display) -> RuntimeError {
        RuntimeError(message.to_string())
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RuntimeError {}

/// Errors from the MCP server loop itself (I/O excluded; those stay `io::Error`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpError {
    /// A frame could not be serialized (non-finite float in runtime data).
    /// The server reports `-32603` for this, mirroring the Python server.
    UnserializableResult,
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpError::UnserializableResult => write!(f, "Runtime returned non-JSON data"),
        }
    }
}

impl std::error::Error for McpError {}
