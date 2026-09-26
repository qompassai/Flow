//! Ollama HTTP transport over `reqwest` (blocking).
//!
//! Implements [`LlmTransport`] with the exact posture of Python's
//! `httpx.Client(timeout=..., trust_env=False, follow_redirects=False)`:
//!
//! - no proxy environment (`trust_env=False` → [`ClientBuilder::no_proxy`]),
//! - no redirects ([`redirect::Policy::none`]),
//! - TLS verification on (rustls with the Mozilla root bundle, the
//!   `certifi` equivalent — never disabled),
//! - a per-request timeout (the caller passes the role/effective timeout),
//! - a 2 MiB streaming cap enforced *while* reading, so an oversized
//!   response is cut off before it is fully buffered.
//!
//! [`LlmTransport`]: phlow_llm::LlmTransport
//! [`ClientBuilder::no_proxy`]: reqwest::blocking::ClientBuilder::no_proxy

use std::io::Read;
use std::time::Duration;

use phlow_llm::{LlmError, LlmTransport};
use serde_json::{Map, Value};

/// Python's `OLLAMA_MAX_RESPONSE_BYTES` (2 MiB): a response longer than
/// this is rejected before it is fully buffered.
const RESPONSE_BYTES_MAX: usize = 2 * 1024 * 1024;
/// Read size for the streaming response cap.
const READ_CHUNK: usize = 8192;

/// Ollama HTTP transport.
pub struct ReqwestTransport {
    client: reqwest::blocking::Client,
}

impl ReqwestTransport {
    /// Build the shared client. Timeouts are per request (the caller passes
    /// the role/effective timeout on every call), so the client carries no
    /// default timeout of its own.
    pub fn new() -> Result<Self, LlmError> {
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| LlmError::Transport(format!("HTTP client failed: {error}")))?;
        Ok(ReqwestTransport { client })
    }

    /// Read the response body with the 2 MiB streaming cap, then parse JSON.
    ///
    /// The blocking response implements [`Read`], so the cap is enforced
    /// chunk by chunk: an oversized body is cut off before it is buffered.
    fn read_capped(response: reqwest::blocking::Response) -> Result<Value, LlmError> {
        let mut response = response
            .error_for_status()
            .map_err(|error| LlmError::Transport(format!("Ollama request failed: {error}")))?;
        let mut body = Vec::new();
        let mut chunk = [0u8; READ_CHUNK];
        loop {
            let read = response
                .read(&mut chunk)
                .map_err(|error| LlmError::Transport(format!("Ollama read failed: {error}")))?;
            if read == 0 {
                break;
            }
            if body.len() + read > RESPONSE_BYTES_MAX {
                return Err(LlmError::ResponseTooLarge {
                    limit: RESPONSE_BYTES_MAX,
                });
            }
            body.extend_from_slice(&chunk[..read]);
        }
        serde_json::from_slice(&body)
            .map_err(|error| LlmError::BadJson(format!("Ollama returned invalid JSON: {error}")))
    }
}

impl LlmTransport for ReqwestTransport {
    fn post_chat(
        &mut self,
        base_url: &str,
        payload: &Map<String, Value>,
        timeout: Duration,
    ) -> Result<Value, LlmError> {
        let url = format!("{base_url}/v1/chat/completions");
        let body = serde_json::to_vec(payload)
            .map_err(|error| LlmError::BadRequest(format!("chat payload failed: {error}")))?;
        let response = self
            .client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .timeout(timeout)
            .send()
            .map_err(|error| LlmError::Transport(format!("Ollama request failed: {error}")))?;
        Self::read_capped(response)
    }

    fn get_tags(&mut self, base_url: &str, timeout: Duration) -> Result<Value, LlmError> {
        let url = format!("{base_url}/api/tags");
        let response = self
            .client
            .get(&url)
            .timeout(timeout)
            .send()
            .map_err(|error| LlmError::Transport(format!("Ollama request failed: {error}")))?;
        Self::read_capped(response)
    }

    fn close(&mut self) {
        // The blocking client owns no background threads; dropping it (with
        // the transport) releases its connection pool exactly once.
    }
}
