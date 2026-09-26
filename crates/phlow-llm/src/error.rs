//! Typed errors for the Ollama backend.
//!
//! Message text mirrors `flow/llm/backend.py` where Python has a fixed
//! message; transport failures carry the underlying detail.

use std::fmt;

/// What went wrong talking to (or hearing back from) Ollama.
#[derive(Debug)]
pub enum LlmError {
    /// A chat payload precondition failed (empty messages, bad model name).
    /// Python raises these as `AssertionError`; here they are typed.
    BadRequest(String),
    /// The response body exceeded [`crate::transport::RESPONSE_BYTES_MAX`].
    ResponseTooLarge { limit: usize },
    /// The response was not valid JSON.
    BadJson(String),
    /// The JSON had the wrong shape (`choices[0].message` etc.).
    BadShape(String),
    /// The transport failed (refused, reset, timeout, HTTP status).
    /// The detail is operator-visible text, never a secret.
    Transport(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::BadRequest(detail) => write!(formatter, "bad chat request: {detail}"),
            LlmError::ResponseTooLarge { limit } => {
                write!(formatter, "Ollama response exceeded {limit} byte limit")
            }
            LlmError::BadJson(detail) => {
                write!(formatter, "Ollama returned invalid JSON: {detail}")
            }
            LlmError::BadShape(detail) => write!(formatter, "{detail}"),
            LlmError::Transport(detail) => write!(formatter, "Ollama transport failed: {detail}"),
        }
    }
}

impl std::error::Error for LlmError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversize_message_matches_python() {
        let error = LlmError::ResponseTooLarge {
            limit: 2 * 1024 * 1024,
        };
        assert_eq!(
            error.to_string(),
            "Ollama response exceeded 2097152 byte limit"
        );
    }
}
