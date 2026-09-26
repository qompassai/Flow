//! The Ollama backend (Phase 3: payload and response shapes).
//!
//! This crate ports `flow/llm/backend.py` and `flow/llm/prompts.py` without
//! the HTTP layer. The `httpx` client becomes the [`LlmTransport`] trait so
//! unit tests run against an in-memory fake; Phase 4 wires a real blocking
//! HTTP client behind this trait.
//!
//! Wire contract (from `flow/llm/backend.py`):
//!
//! - `POST {base}/v1/chat/completions` with `model`, `messages`,
//!   `temperature`, `stream: false`, `max_tokens`, and `tools` only when the
//!   tool list is non-empty.
//! - `max_tokens = min(8192, context_length / 2)`.
//! - `GET {base}/api/tags` lists models.
//! - Response bodies are capped at 2 MiB; larger is an error, never a
//!   truncation.
//! - No proxy environment, no redirects, explicit remote opt-in.

#![forbid(unsafe_code)]

pub mod error;
pub mod payload;
pub mod prompts;
pub mod transport;

pub use error::LlmError;
pub use payload::{
    ChatMessage, ToolCall, build_chat_payload, parse_chat_message, parse_model_list,
};
pub use prompts::{
    build_codegen_prompt, build_error_fix_prompt, default_system_prompt, load_system_prompt,
};
pub use transport::{FakeLlmTransport, LlmTransport};
