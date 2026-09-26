//! Typed errors for workspace operations.
//!
//! Every failure a [`Workspace`](crate::Workspace) can report. Variants carry
//! bounded context only — never file contents, never secrets.

use std::fmt;
use std::io;

/// Failure of a confined workspace operation.
#[derive(Debug)]
pub enum WorkspaceError {
    /// A caller-supplied relative path failed validation. The message names
    /// the rule that rejected it.
    InvalidPath(String),
    /// The workspace root is not a directory.
    NotDirectory,
    /// The root was replaced or became unavailable after opening. The message
    /// tells the operator to restart; never retry against the same handle.
    Stale(&'static str),
    /// A write was attempted on a read-only (untrusted) workspace.
    ReadOnly,
    /// A write targeted operator configuration, which file tools may never
    /// modify.
    Protected,
    /// The target is not a singly-linked regular file.
    NotRegularFile,
    /// Input or output exceeded a byte cap. Carries the cap, not the data.
    TooLarge { limit_bytes: u64 },
    /// The operation needs POSIX no-follow I/O, unavailable on this platform.
    /// Secure writes fail here rather than silently weakening.
    Unavailable(&'static str),
    /// An underlying I/O failure that is none of the above.
    Io(io::Error),
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkspaceError::InvalidPath(reason) => write!(f, "invalid path: {reason}"),
            WorkspaceError::NotDirectory => write!(f, "workspace must be a directory"),
            WorkspaceError::Stale(advice) => write!(f, "stale workspace: {advice}"),
            WorkspaceError::ReadOnly => {
                write!(f, "workspace is read-only; pass --trusted to allow edits")
            }
            WorkspaceError::Protected => {
                write!(f, "operator configuration cannot be modified by file tools")
            }
            WorkspaceError::NotRegularFile => {
                write!(f, "only singly-linked regular files are allowed")
            }
            WorkspaceError::TooLarge { limit_bytes } => {
                write!(f, "data exceeds {limit_bytes} byte limit")
            }
            WorkspaceError::Unavailable(reason) => write!(f, "unavailable: {reason}"),
            WorkspaceError::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for WorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WorkspaceError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for WorkspaceError {
    fn from(err: io::Error) -> WorkspaceError {
        WorkspaceError::Io(err)
    }
}
