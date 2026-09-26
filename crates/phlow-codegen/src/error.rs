//! Typed errors for the code generation surfaces.

use std::fmt;

/// All failure modes of profile loading and code generation.
#[derive(Debug)]
pub enum CodegenError {
    /// A profile bundled with the crate failed to parse. The TOML ships
    /// with the crate, so this is a programming error, never operator
    /// input.
    EmbeddedProfile {
        /// Which bundled language profile was broken.
        language: &'static str,
        /// The parser's message.
        message: String,
    },
    /// The operator profile directory could not be listed.
    UnreadableDir(String),
    /// The operator profile directory held more entries than
    /// [`crate::profiles::PROFILE_FILES_MAX`]. The Python implementation
    /// silently ignored entries past the cap; the port refuses instead, so
    /// an operator never wonders why a profile did not load.
    TooManyProfileFiles {
        /// The cap that was exceeded.
        max: usize,
    },
    /// The tool registry backing the validator failed to build.
    Tools(phlow_tools::ToolError),
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::EmbeddedProfile { language, message } => {
                write!(f, "bundled profile for {language} is invalid: {message}")
            }
            CodegenError::UnreadableDir(message) => {
                write!(f, "cannot list profile directory: {message}")
            }
            CodegenError::TooManyProfileFiles { max } => {
                write!(f, "profile directory holds more than {max} entries")
            }
            CodegenError::Tools(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CodegenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CodegenError::Tools(error) => Some(error),
            _ => None,
        }
    }
}

impl From<phlow_tools::ToolError> for CodegenError {
    fn from(error: phlow_tools::ToolError) -> CodegenError {
        CodegenError::Tools(error)
    }
}
