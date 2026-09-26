//! Golden tests: `build_chat_payload` against Python-generated fixtures.
//!
//! The fixture payloads were built by the real `OllamaClient.chat` payload
//! logic with the default `OllamaConfig` (model `qwen2.5-coder:7b`,
//! temperature 0.2, context length 16384). ASCII escaping of the final
//! bytes is the Phase 4 transport's job; here we verify keys, key order,
//! and values.

use phlow_llm::build_chat_payload;
use serde_json::{Map, Value};

fn fixture() -> serde_json::Value {
    let text = include_str!("fixtures/llm_golden.json");
    serde_json::from_str(text).expect("fixture is valid JSON")
}

fn payload_map(with_tools: bool) -> Map<String, Value> {
    let cfg = phlow_config::OllamaConfig::default();
    let messages = vec![
        serde_json::json!({"role": "system", "content": "sys"}),
        serde_json::json!({"role": "user", "content": "h\u{e9}llo w\u{f6}rld \u{1}"}),
    ];
    let tools = if with_tools {
        vec![serde_json::json!({
            "type": "function",
            "function": {"name": "flow_run", "parameters": {}},
        })]
    } else {
        Vec::new()
    };
    build_chat_payload(&cfg, &messages, &tools, None).expect("default config is valid")
}

#[test]
fn chat_payload_matches_python() {
    let golden = fixture();
    let expected: Map<String, Value> =
        serde_json::from_str(golden["chat_payload"].as_str().unwrap()).unwrap();
    let actual = payload_map(true);
    let expected_keys: Vec<&str> = expected.keys().map(String::as_str).collect();
    let actual_keys: Vec<&str> = actual.keys().map(String::as_str).collect();
    assert_eq!(actual_keys, expected_keys, "key order must match Python");
    assert_eq!(Value::Object(actual), Value::Object(expected));
}

#[test]
fn chat_payload_without_tools_matches_python() {
    let golden = fixture();
    let expected: Map<String, Value> =
        serde_json::from_str(golden["chat_payload_no_tools"].as_str().unwrap()).unwrap();
    let actual = payload_map(false);
    assert!(!actual.contains_key("tools"));
    assert_eq!(Value::Object(actual), Value::Object(expected));
}

#[test]
fn max_tokens_fixtures_agree() {
    let golden = fixture();
    assert_eq!(
        golden["max_tokens_small_ctx"].as_str().unwrap(),
        phlow_llm::payload::max_tokens_for(100).to_string()
    );
    assert_eq!(
        golden["max_tokens_exact"].as_str().unwrap(),
        phlow_llm::payload::max_tokens_for(16384).to_string()
    );
}

#[test]
fn chat_message_fixture_parses() {
    let golden = fixture();
    let response: Value = serde_json::from_str(golden["chat_message"].as_str().unwrap()).unwrap();
    // The fixture stores the *parsed* (content, tool_calls) pair; rebuild a
    // response around it and check the parser round-trips.
    let rebuilt = serde_json::json!({
        "choices": [{
            "message": {
                "content": response["content"],
                "tool_calls": response["tool_calls"],
            },
        }],
    });
    let parsed = phlow_llm::parse_chat_message(&rebuilt).unwrap();
    assert_eq!(parsed.content, response["content"].as_str().unwrap());
    assert_eq!(
        parsed.tool_calls.len(),
        response["tool_calls"].as_array().unwrap().len()
    );
    assert_eq!(parsed.tool_calls[0].id, "c1");
    assert_eq!(parsed.tool_calls[0].name.as_deref(), Some("flow_run"));
}
