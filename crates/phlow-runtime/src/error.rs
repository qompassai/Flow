//! Typed runtime errors. Anything the model or operator sees is rendered
//! through [`RuntimeError`]'s `Display`, which carries the exact Python
//! message strings where they are wire-visible.

use std::fmt;

/// Failure of a runtime operation.
///
/// Python raises `ValueError`/`RuntimeError` with plain strings; the variants
/// here exist so call sites handle the cases the loop branches on
/// (busy, closed) while `Display` keeps the operator-visible text identical.
#[derive(Debug)]
pub enum RuntimeError {
    /// Another `run`/`check` holds the single-writer lock.
    Busy,
    /// The runtime was closed.
    Closed,
    /// Invalid task, tool name, or tool arguments. Carries the Python message.
    Invalid(String),
    /// The model backend failed.
    Backend(String),
    /// The editor bridge failed.
    Editor(String),
    /// The workspace rejected an operation.
    Workspace(String),
    /// A check run failed.
    Checks(String),
    /// The editor context was unusable. Carries the Python message.
    EditorContext(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::Busy => write!(formatter, "Runtime busy: single writer"),
            RuntimeError::Closed => write!(formatter, "Runtime closed"),
            RuntimeError::Invalid(detail) => write!(formatter, "{detail}"),
            RuntimeError::Backend(detail) => write!(formatter, "{detail}"),
            RuntimeError::Editor(detail) => write!(formatter, "{detail}"),
            RuntimeError::Workspace(detail) => write!(formatter, "{detail}"),
            RuntimeError::Checks(detail) => write!(formatter, "{detail}"),
            RuntimeError::EditorContext(detail) => write!(formatter, "{detail}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<phlow_workspace::WorkspaceError> for RuntimeError {
    fn from(error: phlow_workspace::WorkspaceError) -> Self {
        RuntimeError::Workspace(error.to_string())
    }
}

impl From<phlow_llm::LlmError> for RuntimeError {
    fn from(error: phlow_llm::LlmError) -> Self {
        RuntimeError::Backend(error.to_string())
    }
}

impl From<phlow_editor::BridgeError> for RuntimeError {
    fn from(error: phlow_editor::BridgeError) -> Self {
        RuntimeError::Editor(error.to_string())
    }
}
