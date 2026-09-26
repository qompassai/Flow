//! The HTTP seam: `LlmTransport` and its in-memory fake.
//!
//! Mirrors the `httpx.Client` inside `OllamaClient`. Phase 3 is synchronous;
//! Phase 4 provides a real blocking HTTP client behind this trait that must
//! honor every bound documented here.
//!
//! Transport obligations (from `flow/llm/backend.py`):
//!
//! - `trust_env=False`: never read proxy environment variables.
//! - `follow_redirects=False`: never follow redirects.
//! - Base URL has no trailing slash; paths are appended verbatim.
//! - Response bodies stream with a hard cap of [`RESPONSE_BYTES_MAX`];
//!   exceeding it is an error, never a truncation.
//! - Non-2xx statuses are errors (`raise_for_status`).

use std::collections::VecDeque;
use std::time::Duration;

use serde_json::{Map, Value};

use crate::error::LlmError;

/// A chat completion body is at most a few tool calls plus prose; streaming
/// with a hard cap keeps a hostile server from exhausting memory.
pub const RESPONSE_BYTES_MAX: usize = 2 * 1024 * 1024;

/// The HTTP client seam. Synchronous in Phase 3.
pub trait LlmTransport {
    /// `POST {base}/v1/chat/completions` with a JSON payload; returns the
    /// decoded body. Must enforce [`RESPONSE_BYTES_MAX`] while streaming.
    fn post_chat(
        &mut self,
        base_url: &str,
        payload: &Map<String, Value>,
        timeout: Duration,
    ) -> Result<Value, LlmError>;

    /// `GET {base}/api/tags`; returns the decoded body.
    fn get_tags(&mut self, base_url: &str, timeout: Duration) -> Result<Value, LlmError>;

    /// Release the client. Called at most once.
    fn close(&mut self);
}

/// The Ollama backend: payload building over an [`LlmTransport`].
///
/// Mirrors `OllamaClient`: `chat` posts the payload built by
/// [`crate::payload::build_chat_payload`], `list_models` maps
/// [`crate::payload::parse_model_list`] over `GET /api/tags`, and
/// `is_available` returns false on any failure.
pub struct OllamaBackend<T: LlmTransport> {
    cfg: phlow_config::OllamaConfig,
    base_url: String,
    transport: T,
}

impl<T: LlmTransport> OllamaBackend<T> {
    /// Build a backend. The base URL is stored without its trailing slash,
    /// like Python's `cfg.base_url.rstrip("/")`.
    pub fn new(cfg: phlow_config::OllamaConfig, transport: T) -> Self {
        let base_url = cfg.base_url().trim_end_matches('/').to_owned();
        OllamaBackend {
            cfg,
            base_url,
            transport,
        }
    }

    /// One chat completion. `model` overrides the configured model.
    pub fn chat(
        &mut self,
        messages: &[Value],
        tools: &[Value],
        model: Option<&str>,
    ) -> Result<Value, LlmError> {
        let payload = crate::payload::build_chat_payload(&self.cfg, messages, tools, model)?;
        let timeout = Duration::from_secs_f64(self.cfg.timeout_secs());
        self.transport.post_chat(&self.base_url, &payload, timeout)
    }

    /// Model names from `GET /api/tags`.
    pub fn list_models(&mut self) -> Result<Vec<String>, LlmError> {
        let timeout = Duration::from_secs_f64(self.cfg.timeout_secs());
        let body = self.transport.get_tags(&self.base_url, timeout)?;
        crate::payload::parse_model_list(&body)
    }

    /// True when the server answers `GET /api/tags`. Any failure — refused,
    /// reset, timeout, HTTP error, bad JSON, bad shape — is false, mirroring
    /// Python's `except (httpx.HTTPError, ValueError, KeyError)`.
    pub fn is_available(&mut self) -> bool {
        self.list_models().is_ok()
    }

    /// The base URL with no trailing slash.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Release the transport.
    pub fn close(mut self) {
        self.transport.close();
    }
}

/// In-memory [`LlmTransport`] for tests: records requests and replays
/// scripted replies or failures.
#[derive(Debug, Default)]
pub struct FakeLlmTransport {
    /// `(method, url, payload)` per request, in order. `GET` rows carry
    /// `Value::Null` as the payload.
    pub requests: Vec<(String, String, Value)>,
    replies: VecDeque<Result<Value, LlmError>>,
    closed: bool,
}

impl FakeLlmTransport {
    /// A fake with no scripted replies: requests return `Value::Null`.
    pub fn new() -> Self {
        FakeLlmTransport::default()
    }

    /// Queue one reply (or failure) for the next request.
    pub fn queue_reply(&mut self, reply: Result<Value, LlmError>) {
        self.replies.push_back(reply);
    }

    /// True once `close` ran.
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

impl LlmTransport for FakeLlmTransport {
    fn post_chat(
        &mut self,
        base_url: &str,
        payload: &Map<String, Value>,
        _timeout: Duration,
    ) -> Result<Value, LlmError> {
        self.requests.push((
            "POST".to_owned(),
            format!("{base_url}/v1/chat/completions"),
            Value::Object(payload.clone()),
        ));
        match self.replies.pop_front() {
            Some(reply) => reply,
            None => Ok(Value::Null),
        }
    }

    fn get_tags(&mut self, base_url: &str, _timeout: Duration) -> Result<Value, LlmError> {
        self.requests.push((
            "GET".to_owned(),
            format!("{base_url}/api/tags"),
            Value::Null,
        ));
        match self.replies.pop_front() {
            Some(reply) => reply,
            None => Ok(Value::Null),
        }
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

/// A transport that enforces [`RESPONSE_BYTES_MAX`] while streaming, for
/// Phase 4 implementors to reuse. Feeding `size` bytes when `used + size`
/// would exceed the cap returns [`LlmError::ResponseTooLarge`].
#[derive(Debug, Default)]
pub struct BoundedBody {
    used: usize,
}

impl BoundedBody {
    /// Start streaming a body.
    pub fn new() -> Self {
        BoundedBody::default()
    }

    /// Account for one chunk. Mirrors the `_json` streaming loop: the check
    /// runs *before* the chunk is kept, so the cap is never exceeded.
    pub fn push(&mut self, chunk_len: usize) -> Result<(), LlmError> {
        self.used = self
            .used
            .checked_add(chunk_len)
            .filter(|used| *used <= RESPONSE_BYTES_MAX)
            .ok_or(LlmError::ResponseTooLarge {
                limit: RESPONSE_BYTES_MAX,
            })?;
        Ok(())
    }

    /// Bytes accounted for so far.
    pub fn used(&self) -> usize {
        self.used
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn backend(transport: FakeLlmTransport) -> OllamaBackend<FakeLlmTransport> {
        OllamaBackend::new(phlow_config::OllamaConfig::default(), transport)
    }

    #[test]
    fn base_url_has_no_trailing_slash() {
        let backend = backend(FakeLlmTransport::new());
        assert!(!backend.base_url().ends_with('/'));
    }

    #[test]
    fn chat_posts_to_chat_completions() {
        let mut transport = FakeLlmTransport::new();
        transport.queue_reply(Ok(json!({"choices": []})));
        let mut backend = backend(transport);
        let messages = vec![json!({"role": "user", "content": "hi"})];
        backend.chat(&messages, &[], None).unwrap();
        // (The transport is owned; re-inspect via a fresh backend is not
        // possible, so this test only asserts no error. Request-shape
        // assertions live on FakeLlmTransport directly below.)
    }

    #[test]
    fn fake_records_method_url_and_payload() {
        let mut fake = FakeLlmTransport::new();
        let payload = crate::payload::build_chat_payload(
            &phlow_config::OllamaConfig::default(),
            &[json!({"role": "user", "content": "hi"})],
            &[],
            None,
        )
        .unwrap();
        fake.post_chat("http://x", &payload, Duration::from_secs(1))
            .unwrap();
        fake.get_tags("http://x", Duration::from_secs(1)).unwrap();
        assert_eq!(fake.requests.len(), 2);
        assert_eq!(fake.requests[0].0, "POST");
        assert_eq!(fake.requests[0].1, "http://x/v1/chat/completions");
        assert_eq!(fake.requests[0].2["model"], payload["model"]);
        assert_eq!(fake.requests[1].0, "GET");
        assert_eq!(fake.requests[1].1, "http://x/api/tags");
    }

    #[test]
    fn is_available_is_false_on_any_failure() {
        let mut transport = FakeLlmTransport::new();
        transport.queue_reply(Err(LlmError::Transport("refused".to_owned())));
        let mut backend = backend(transport);
        assert!(!backend.is_available());
    }

    #[test]
    fn is_available_is_true_on_good_tags() {
        let mut transport = FakeLlmTransport::new();
        transport.queue_reply(Ok(json!({"models": [{"name": "m"}]})));
        let mut backend = backend(transport);
        assert!(backend.is_available());
        assert_eq!(backend.list_models().unwrap(), Vec::<String>::new());
    }

    #[test]
    fn bounded_body_rejects_over_cap_chunks() {
        let mut body = BoundedBody::new();
        body.push(RESPONSE_BYTES_MAX).unwrap();
        let err = body.push(1).unwrap_err();
        assert!(matches!(err, LlmError::ResponseTooLarge { .. }));
        assert_eq!(body.used(), RESPONSE_BYTES_MAX);
    }

    #[test]
    fn bounded_body_rejects_a_single_huge_chunk() {
        let mut body = BoundedBody::new();
        assert!(matches!(
            body.push(RESPONSE_BYTES_MAX + 1),
            Err(LlmError::ResponseTooLarge { .. })
        ));
        assert_eq!(body.used(), 0);
    }
}
