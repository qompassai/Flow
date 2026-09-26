//! Typed errors for the Neovim editor bridge.
//!
//! External input (socket paths, rose.nvim results, tool names) arrives here
//! as data and leaves as typed errors; nothing panics on it. Message text
//! mirrors `flow/editor.py` byte-for-byte so clients and logs keep the same
//! vocabulary.

use std::fmt;
use std::io;
use std::time::Duration;

/// What went wrong while validating the private socket path.
#[derive(Debug)]
pub enum SocketError {
    /// No socket was configured at all (empty or missing value).
    NotConfigured,
    /// On Windows the value is not a `\\.\pipe\...` named pipe.
    NotPrivatePipe,
    /// The path is relative or a symlink.
    NotAbsoluteSocket,
    /// The path is not a socket, or is not owned by the current user.
    /// Python reports both as one message; the distinction leaks nothing
    /// useful and probing either is equally hostile, so they stay merged.
    NotOwnedSocket,
    /// The socket's own mode and its parent directory are both too open.
    NotPrivate,
    /// The filesystem could not be inspected.
    Io(io::Error),
}

impl fmt::Display for SocketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SocketError::NotConfigured => write!(formatter, "No --nvim socket configured"),
            SocketError::NotPrivatePipe => {
                write!(
                    formatter,
                    "Only a private named pipe is supported on Windows"
                )
            }
            SocketError::NotAbsoluteSocket => {
                write!(
                    formatter,
                    "--nvim must be an absolute, non-symlink Unix socket"
                )
            }
            SocketError::NotOwnedSocket => {
                write!(formatter, "Neovim socket must be owned by the current user")
            }
            SocketError::NotPrivate => write!(
                formatter,
                "Neovim socket must be private (0700 parent or 0600 socket)"
            ),
            SocketError::Io(error) => write!(formatter, "Neovim socket: {error}"),
        }
    }
}

impl std::error::Error for SocketError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SocketError::Io(error) => Some(error),
            _ => None,
        }
    }
}

/// What went wrong while constructing an [`EditorBridge`](crate::EditorBridge).
#[derive(Debug)]
pub enum BridgeError {
    /// The timeout is outside the [0.1s, 660s] window.
    BadTimeout(Duration),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::BadTimeout(timeout) => write!(
                formatter,
                "editor timeout {timeout:?} is outside the 0.1s..=660s window"
            ),
        }
    }
}

impl std::error::Error for BridgeError {}

/// What the transport layer reported. The bridge converts these into the
/// `{"status": "unavailable", "error": ...}` envelope, poisoning itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// The request exceeded its deadline. The bridge reports the canonical
    /// `"Neovim bridge unavailable: Neovim request timed out"` message.
    Timeout,
    /// Any other transport failure (refused, reset, decode error).
    /// The detail is operator-visible text, never a secret.
    Failed(String),
}

impl fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::Timeout => write!(formatter, "Neovim request timed out"),
            TransportError::Failed(detail) => {
                write!(formatter, "Neovim transport failed: {detail}")
            }
        }
    }
}

impl std::error::Error for TransportError {}

/// What went wrong while checking editor-context freshness.
#[derive(Debug)]
pub enum FreshnessError {
    /// `editor_context` did not report status `"ok"` with a workspace string.
    Unavailable(String),
    /// The editor's workspace differs from the agent's workspace root.
    WorkspaceMismatch { editor: String, agent: String },
    /// The freshness payload is not the v1 shape (`version`, `snapshot`,
    /// `dirty_buffers`).
    ApiV1Missing,
    /// The workspace path could not be canonicalized for comparison.
    Io(io::Error),
}

impl fmt::Display for FreshnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FreshnessError::Unavailable(detail) => {
                write!(formatter, "Editor context unavailable: {detail}")
            }
            FreshnessError::WorkspaceMismatch { .. } => write!(
                formatter,
                "Rose/Flow workspace mismatch; refusing reverse tools and edits"
            ),
            FreshnessError::ApiV1Missing => write!(
                formatter,
                "Editor workspace freshness API v1 is unavailable; update Rose"
            ),
            FreshnessError::Io(error) => write!(formatter, "Editor workspace: {error}"),
        }
    }
}

impl std::error::Error for FreshnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FreshnessError::Io(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socket_error_messages_match_python() {
        assert_eq!(
            SocketError::NotConfigured.to_string(),
            "No --nvim socket configured"
        );
        assert_eq!(
            SocketError::NotPrivatePipe.to_string(),
            "Only a private named pipe is supported on Windows"
        );
        assert_eq!(
            SocketError::NotAbsoluteSocket.to_string(),
            "--nvim must be an absolute, non-symlink Unix socket"
        );
        assert_eq!(
            SocketError::NotOwnedSocket.to_string(),
            "Neovim socket must be owned by the current user"
        );
        assert_eq!(
            SocketError::NotPrivate.to_string(),
            "Neovim socket must be private (0700 parent or 0600 socket)"
        );
    }

    #[test]
    fn freshness_error_messages_match_python() {
        assert_eq!(
            FreshnessError::Unavailable("gone".to_owned()).to_string(),
            "Editor context unavailable: gone"
        );
        assert_eq!(
            FreshnessError::WorkspaceMismatch {
                editor: "/e".to_owned(),
                agent: "/a".to_owned(),
            }
            .to_string(),
            "Rose/Flow workspace mismatch; refusing reverse tools and edits"
        );
        assert_eq!(
            FreshnessError::ApiV1Missing.to_string(),
            "Editor workspace freshness API v1 is unavailable; update Rose"
        );
    }
}
