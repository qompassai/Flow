//! Chat payload construction and response parsing.
//!
//! Mirrors `OllamaClient.chat` (payload) and `FlowRuntime._chat` (response
//! shape). The payload key order matches Python's dict insertion order:
//! `model`, `messages`, `temperature`, `stream`, `max_tokens`, then `tools`
//! only when non-empty.

use serde_json::{Map, Value};

use crate::error::LlmError;

/// Completions are capped below the context window so the prompt always fits.
pub const COMPLETION_TOKENS_MAX: u32 = 8192;

/// One normalized model turn: the validated `(content, tool_calls)` shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    /// The assistant's text; missing or null normalizes to `""`.
    pub content: String,
    /// The requested tool calls; missing or null normalizes to `[]`.
    pub tool_calls: Vec<ToolCall>,
}

/// One normalized tool call. `function.arguments` stays a [`Value`] here;
/// the runtime parses a string-encoded arguments payload (Phase 4).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// The call id (`call["id"]`).
    pub id: String,
    /// The function name (`call["function"]["name"]`, may be missing).
    pub name: Option<String>,
    /// The raw arguments value.
    pub arguments: Value,
}

/// `max_tokens = min(8192, context_length / 2)`, exactly like Python.
pub fn max_tokens_for(context_length: u32) -> u32 {
    COMPLETION_TOKENS_MAX.min(context_length / 2)
}

/// Build the `POST /v1/chat/completions` payload.
///
/// - `model` overrides the configured model when `Some` and non-empty.
/// - `tools` is included only when the slice is non-empty (Python's
///   `if tools:`).
/// - Empty `messages` is rejected: Python asserts `len(messages) > 0`.
pub fn build_chat_payload(
    cfg: &phlow_config::OllamaConfig,
    messages: &[Value],
    tools: &[Value],
    model: Option<&str>,
) -> Result<Map<String, Value>, LlmError> {
    if messages.is_empty() {
        return Err(LlmError::BadRequest(
            "messages must not be empty".to_owned(),
        ));
    }
    let chosen = match model {
        Some(name) if !name.is_empty() => name.to_owned(),
        _ => cfg.model().to_owned(),
    };
    if chosen.is_empty() {
        return Err(LlmError::BadRequest("model must not be empty".to_owned()));
    }
    let mut payload = Map::with_capacity(6);
    payload.insert("model".to_owned(), Value::from(chosen));
    payload.insert("messages".to_owned(), Value::Array(messages.to_vec()));
    payload.insert("temperature".to_owned(), Value::from(cfg.temperature()));
    payload.insert("stream".to_owned(), Value::from(false));
    payload.insert(
        "max_tokens".to_owned(),
        Value::from(max_tokens_for(cfg.context_length())),
    );
    if !tools.is_empty() {
        payload.insert("tools".to_owned(), Value::Array(tools.to_vec()));
    }
    Ok(payload)
}

/// Parse one chat completion into the validated `(content, tool_calls)`
/// shape, mirroring `FlowRuntime._chat`:
///
/// - `response["choices"][0]["message"]` must be an object, else
///   `"Model message must be an object"`;
/// - `content` missing/null → `""`, `tool_calls` missing/null → `[]`;
/// - non-string content or non-list tool_calls →
///   `"Invalid model content/tool_calls shape"`.
pub fn parse_chat_message(response: &Value) -> Result<ChatMessage, LlmError> {
    let message = phlow_json::object_map(response)
        .ok()
        .and_then(|map| map.get("choices"))
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| phlow_json::object_map(choice).ok())
        .and_then(|choice| choice.get("message"))
        .filter(|message| message.is_object())
        .ok_or_else(|| LlmError::BadShape("Model message must be an object".to_owned()))?;
    let map = message.as_object().expect("filtered to objects");
    let content = match map.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(_) => {
            return Err(LlmError::BadShape(
                "Invalid model content/tool_calls shape".to_owned(),
            ));
        }
    };
    let tool_calls = match map.get("tool_calls") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(calls)) => calls
            .iter()
            .map(|call| {
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let function = call.get("function");
                let name = function
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let arguments = function
                    .and_then(|function| function.get("arguments"))
                    .cloned()
                    .unwrap_or(Value::Null);
                ToolCall {
                    id,
                    name,
                    arguments,
                }
            })
            .collect(),
        Some(_) => {
            return Err(LlmError::BadShape(
                "Invalid model content/tool_calls shape".to_owned(),
            ));
        }
    };
    Ok(ChatMessage {
        content,
        tool_calls,
    })
}

/// Extract model names from a `GET /api/tags` body: `[m["name"] for m in
/// body.get("models", [])]`. Like Python, an entry without a string `name`
/// is an error, not a skip: `is_available` treats it as "not available".
pub fn parse_model_list(body: &Value) -> Result<Vec<String>, LlmError> {
    let models = body
        .get("models")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    models
        .iter()
        .map(|model| {
            model
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| LlmError::BadShape("model entry is missing \"name\"".to_owned()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_config() -> phlow_config::OllamaConfig {
        phlow_config::OllamaConfig::default()
    }

    #[test]
    fn max_tokens_is_half_context_capped() {
        assert_eq!(max_tokens_for(100), 50);
        assert_eq!(max_tokens_for(16384), 8192);
        assert_eq!(max_tokens_for(100_000), 8192);
        assert_eq!(max_tokens_for(0), 0);
    }

    #[test]
    fn payload_key_order_and_values_match_python() {
        let cfg = test_config();
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let tools = vec![json!({"type": "function"})];
        let payload = build_chat_payload(&cfg, &messages, &tools, None).unwrap();
        let keys: Vec<&str> = payload.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "model",
                "messages",
                "temperature",
                "stream",
                "max_tokens",
                "tools"
            ]
        );
        assert_eq!(payload["model"], json!(cfg.model()));
        assert_eq!(payload["stream"], json!(false));
        assert_eq!(
            payload["max_tokens"],
            json!(max_tokens_for(cfg.context_length()))
        );
    }

    #[test]
    fn empty_tools_omits_the_key() {
        let cfg = test_config();
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let payload = build_chat_payload(&cfg, &messages, &[], None).unwrap();
        assert!(!payload.contains_key("tools"));
    }

    #[test]
    fn model_override_wins_when_nonempty() {
        let cfg = test_config();
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let payload = build_chat_payload(&cfg, &messages, &[], Some("other:1b")).unwrap();
        assert_eq!(payload["model"], json!("other:1b"));
        let fallback = build_chat_payload(&cfg, &messages, &[], Some("")).unwrap();
        assert_eq!(fallback["model"], json!(cfg.model()));
    }

    #[test]
    fn empty_messages_is_rejected() {
        let cfg = test_config();
        assert!(matches!(
            build_chat_payload(&cfg, &[], &[], None),
            Err(LlmError::BadRequest(_))
        ));
    }

    #[test]
    fn parse_chat_message_normalizes_missing_fields() {
        let response = json!({"choices": [{"message": {}}]});
        let parsed = parse_chat_message(&response).unwrap();
        assert_eq!(parsed.content, "");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parse_chat_message_reads_content_and_calls() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "hello",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "editor_lint",
                            "arguments": "{\"path\": \"a.rs\"}",
                        },
                    }],
                },
            }],
        });
        let parsed = parse_chat_message(&response).unwrap();
        assert_eq!(parsed.content, "hello");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "call_1");
        assert_eq!(parsed.tool_calls[0].name.as_deref(), Some("editor_lint"));
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!("{\"path\": \"a.rs\"}")
        );
    }

    #[test]
    fn parse_chat_message_rejects_bad_shapes() {
        for (response, message) in [
            (
                json!({"choices": [{"message": "nope"}]}),
                "Model message must be an object",
            ),
            (
                json!({"choices": [{"message": {"content": 42}}]}),
                "Invalid model content/tool_calls shape",
            ),
            (
                json!({"choices": [{"message": {"tool_calls": {}}}]}),
                "Invalid model content/tool_calls shape",
            ),
            (json!({}), "Model message must be an object"),
        ] {
            let err = parse_chat_message(&response).unwrap_err();
            assert_eq!(err.to_string(), message, "for {response}");
        }
    }

    #[test]
    fn parse_model_list_reads_names() {
        let body = json!({"models": [{"name": "a"}, {"name": "b"}]});
        assert_eq!(parse_model_list(&body).unwrap(), ["a", "b"]);
        assert!(parse_model_list(&json!({})).unwrap().is_empty());
    }

    #[test]
    fn parse_model_list_rejects_nameless_entries() {
        let body = json!({"models": [{"name": "a"}, {"nope": 1}]});
        assert!(matches!(
            parse_model_list(&body),
            Err(LlmError::BadShape(_))
        ));
    }
}
