//! Typed errors for the self-improvement surfaces.

use std::fmt;
use std::path::PathBuf;

/// All failure modes of the feedback store, prompt evolver, and skill
/// store.
#[derive(Debug)]
pub enum SelfImproveError {
    /// The SQLite backend failed.
    Database(rusqlite::Error),
    /// A filesystem operation failed.
    Io(std::io::Error),
    /// The database path had no parent.
    BadPath(PathBuf),
    /// A text field exceeded its character budget.
    TooLong {
        /// Which field was too long.
        field: &'static str,
        /// The budget it exceeded.
        max_chars: usize,
    },
    /// A skill directory operation failed.
    Workspace(phlow_workspace::WorkspaceError),
    /// Automatic prompt evolution was requested; it is disabled.
    EvolutionDisabled,
    /// A skill mutation was requested; automatic mutation is disabled.
    SkillMutationDisabled,
    /// A Git mutation was requested; automatic Git mutation is disabled.
    GitMutationDisabled,
}

impl fmt::Display for SelfImproveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelfImproveError::Database(error) => write!(f, "feedback database error: {error}"),
            SelfImproveError::Io(error) => write!(f, "I/O error: {error}"),
            SelfImproveError::BadPath(path) => {
                write!(f, "unusable database path: {}", path.display())
            }
            SelfImproveError::TooLong { field, max_chars } => {
                write!(f, "{field} exceeds {max_chars} characters")
            }
            SelfImproveError::Workspace(error) => write!(f, "{error}"),
            SelfImproveError::EvolutionDisabled => write!(
                f,
                "Automatic prompt evolution and implicit Git commits are disabled. Review and edit operator-owned prompts manually."
            ),
            SelfImproveError::SkillMutationDisabled => {
                write!(f, "Automatic skill mutation is disabled; edit manually")
            }
            SelfImproveError::GitMutationDisabled => {
                write!(f, "Automatic Git mutation is disabled; revert manually")
            }
        }
    }
}

impl std::error::Error for SelfImproveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SelfImproveError::Database(error) => Some(error),
            SelfImproveError::Io(error) => Some(error),
            SelfImproveError::Workspace(error) => Some(error),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for SelfImproveError {
    fn from(error: rusqlite::Error) -> SelfImproveError {
        SelfImproveError::Database(error)
    }
}

impl From<phlow_workspace::WorkspaceError> for SelfImproveError {
    fn from(error: phlow_workspace::WorkspaceError) -> SelfImproveError {
        SelfImproveError::Workspace(error)
    }
}
