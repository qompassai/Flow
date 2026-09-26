//! Web search over the DuckDuckGo Instant Answer API or SearXNG.
//!
//! Ports `flow/tools/web_search.py`. The result shapes are preserved
//! exactly: a list of `{title, url, snippet}` objects, or `[{"error":
//! <message>}]` when the request fails. Snippet truncation widths (500 for
//! the DuckDuckGo abstract, 300 for related/SearXNG snippets, 80 for related
//! topic titles) match the Python implementation character-for-character.
//!
//! HTTP posture follows the rest of the port: rustls only, no proxy
//! environment, no redirects, and a named 10-second total timeout per
//! request. The parsing functions are pure and take fixtures in tests.

use std::io::Read;
use std::time::Duration;

use serde_json::Value;

use crate::error::ToolError;
use crate::json_compat::python_json_dumps_indent2;

/// Total timeout for one search request, mirroring `timeout=10`.
pub const SEARCH_TIMEOUT_SECS: u64 = 10;

/// A search query longer than this is rejected; servers log queries and the
/// model must not stuff unbounded text into a GET line.
pub const QUERY_CHARS_MAX: usize = 500;

/// A SearXNG base URL longer than this is rejected; it is operator config
/// and must stay a short origin, not a smuggled payload.
pub const SEARXNG_URL_CHARS_MAX: usize = 2048;

/// A search backend answer larger than this is rejected, not buffered:
/// the payloads are small JSON documents, and a hostile server must not be
/// able to exhaust memory through the search tool.
pub const SEARCH_RESPONSE_BYTES_MAX: usize = 1024 * 1024;

/// DuckDuckGo Instant Answer endpoint, unchanged from Python.
pub const DUCKDUCKGO_URL: &str = "https://api.duckduckgo.com/";

/// DuckDuckGo abstract snippets are cut at 500 characters, as in Python.
pub const ABSTRACT_CHARS_MAX: usize = 500;

/// Related-topic and SearXNG snippets are cut at 300 characters.
pub const SNIPPET_CHARS_MAX: usize = 300;

/// Related-topic titles are cut at 80 characters.
pub const RELATED_TITLE_CHARS_MAX: usize = 80;

/// A negative `max_results` is meaningless; Python's slice quirk (`[:-1]`
/// dropping the last result) is not reproduced.
pub const MAX_RESULTS_MIN: i64 = 0;

/// Which search backend answers queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchBackend {
    /// The DuckDuckGo Instant Answer API (default).
    DuckDuckGo,
    /// A self-hosted SearXNG instance at the configured base URL.
    SearXng,
}

/// One search hit. Serializes to the `{title, url, snippet}` shape the
/// Python tools returned. `title` is `None` when the backend supplied an
/// explicit JSON null (DuckDuckGo `Heading`), mirroring Python's
/// `data.get("Heading", query)`: a missing key falls back to the query
/// string, a present null stays null.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Result title (related-topic titles truncated to 80 characters);
    /// `None` renders as JSON `null`.
    pub title: Option<String>,
    /// Result URL, empty when the backend supplied none.
    pub url: String,
    /// Result snippet (truncated per backend rules).
    pub snippet: String,
}

impl SearchResult {
    /// The `{title, url, snippet}` JSON object, keyed exactly as Python's.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        })
    }
}

/// The web search tool: backend selection plus a bounded blocking client.
///
/// Build with [`WebSearch::duckduckgo`], [`WebSearch::searxng`], or
/// [`WebSearch::with_backend`]. [`WebSearch::run`] returns the JSON string
/// Python's tool returned; [`WebSearch::search`] returns typed results.
#[derive(Debug)]
pub struct WebSearch {
    backend: SearchBackend,
    searxng_url: String,
    client: reqwest::blocking::Client,
}

impl WebSearch {
    /// Build a client with the port's HTTP posture: rustls only, no proxy
    /// environment, no redirects, 10-second total timeout per request.
    fn build_client() -> Result<reqwest::blocking::Client, ToolError> {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(SEARCH_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|error| ToolError::Http(truncate_error(&error.to_string())))
    }

    /// Search via the DuckDuckGo Instant Answer API.
    pub fn duckduckgo() -> Result<WebSearch, ToolError> {
        Ok(WebSearch {
            backend: SearchBackend::DuckDuckGo,
            searxng_url: String::new(),
            client: WebSearch::build_client()?,
        })
    }

    /// Search via SearXNG at `url`. An empty `url` falls back to
    /// DuckDuckGo, mirroring Python's `make_web_search_tool` selection.
    pub fn searxng(url: &str) -> Result<WebSearch, ToolError> {
        WebSearch::with_backend(SearchBackend::SearXng, url)
    }

    /// Build for an explicit backend. A `SearXng` backend with an empty URL
    /// falls back to DuckDuckGo, as in Python.
    pub fn with_backend(backend: SearchBackend, searxng_url: &str) -> Result<WebSearch, ToolError> {
        if searxng_url.chars().count() > SEARXNG_URL_CHARS_MAX {
            return Err(ToolError::Http(format!(
                "searxng_url exceeds {SEARXNG_URL_CHARS_MAX} characters"
            )));
        }
        let backend = match backend {
            SearchBackend::SearXng if searxng_url.is_empty() => SearchBackend::DuckDuckGo,
            other => other,
        };
        Ok(WebSearch {
            backend,
            searxng_url: searxng_url.trim_end_matches('/').to_string(),
            client: WebSearch::build_client()?,
        })
    }

    /// Which backend answers queries (after the empty-URL fallback).
    pub fn backend(&self) -> SearchBackend {
        self.backend
    }

    /// Run one search and return typed results.
    ///
    /// `max_results` caps the returned hits; negative values are rejected.
    /// Transport, status, and JSON failures are [`ToolError`], which
    /// [`WebSearch::run`] renders as Python's `[{"error": ...}]` shape.
    pub fn search(&self, query: &str, max_results: i64) -> Result<Vec<SearchResult>, ToolError> {
        if max_results < MAX_RESULTS_MIN {
            return Err(ToolError::InvalidArguments(format!(
                "max_results must be >= {MAX_RESULTS_MIN}"
            )));
        }
        if query.chars().count() > QUERY_CHARS_MAX {
            return Err(ToolError::InvalidArguments(format!(
                "query exceeds {QUERY_CHARS_MAX} characters"
            )));
        }
        let limit = max_results as usize;
        match self.backend {
            SearchBackend::DuckDuckGo => {
                let payload = self.fetch(
                    DUCKDUCKGO_URL,
                    &[
                        ("q", query),
                        ("format", "json"),
                        ("no_html", "1"),
                        ("skip_disambig", "1"),
                    ],
                )?;
                Ok(parse_duckduckgo(&payload, query, limit))
            }
            SearchBackend::SearXng => {
                let url = format!("{}/search", self.searxng_url);
                let payload = self.fetch(
                    &url,
                    &[("q", query), ("format", "json"), ("categories", "general")],
                )?;
                Ok(parse_searxng(&payload, limit))
            }
        }
    }

    /// Run one search and return the JSON string Python's tool returned:
    /// `json.dumps(results, indent=2)` with `ensure_ascii` escaping, or
    /// `[{"error": <message>}]` on failure.
    pub fn run(&self, query: &str, max_results: i64) -> String {
        let value = match self.search(query, max_results) {
            Ok(results) => Value::Array(results.iter().map(SearchResult::to_json).collect()),
            Err(error) => serde_json::json!([{"error": error.to_string()}]),
        };
        python_json_dumps_indent2(&value)
    }

    /// GET `url` with `params`; non-2xx is an error (`raise_for_status`),
    /// and the body must decode as JSON within the byte cap.
    fn fetch(&self, url: &str, params: &[(&str, &str)]) -> Result<Value, ToolError> {
        let response = self
            .client
            .get(url)
            .query(params)
            .send()
            .map_err(|error| ToolError::Http(truncate_error(&error.to_string())))?;
        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::BadStatus(status.as_u16()));
        }
        let body = read_capped(response)?;
        serde_json::from_slice(&body)
            .map_err(|error| ToolError::BadJson(truncate_error(&error.to_string())))
    }
}

/// Parse a DuckDuckGo Instant Answer payload, mirroring
/// `duckduckgo_search`: the abstract first, then related topics with a
/// `Text` key, truncated to `max_results`.
///
/// `query` is the search query, already bounded by
/// [`WebSearch::search`]; a missing `Heading` falls back to it, exactly
/// like Python's `data.get("Heading", query)`. An explicit JSON null
/// stays null rather than becoming `""`.
pub fn parse_duckduckgo(payload: &Value, query: &str, max_results: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let abstract_text = payload
        .get("AbstractText")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !abstract_text.is_empty() {
        let title = match payload.get("Heading") {
            None => Some(query.to_string()),
            Some(Value::String(heading)) => Some(heading.clone()),
            // Explicit null stays null; other non-string shapes are
            // outside the DuckDuckGo contract and render as null.
            Some(_) => None,
        };
        results.push(SearchResult {
            title,
            url: payload
                .get("AbstractURL")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            snippet: truncate_chars(abstract_text, ABSTRACT_CHARS_MAX),
        });
    }
    if let Some(topics) = payload.get("RelatedTopics").and_then(Value::as_array) {
        for topic in topics {
            if results.len() >= max_results {
                break;
            }
            let Some(object) = topic.as_object() else {
                continue;
            };
            let text = object.get("Text").and_then(Value::as_str).unwrap_or("");
            if text.is_empty() {
                continue;
            }
            results.push(SearchResult {
                title: Some(truncate_chars(text, RELATED_TITLE_CHARS_MAX)),
                url: object
                    .get("FirstURL")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                snippet: truncate_chars(text, SNIPPET_CHARS_MAX),
            });
        }
    }
    results.truncate(max_results);
    results
}

/// Parse a SearXNG `/search?format=json` payload, mirroring
/// `searxng_search`: each hit becomes `{title, url, snippet}` with the
/// snippet cut at 300 characters.
pub fn parse_searxng(payload: &Value, max_results: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();
    if let Some(hits) = payload.get("results").and_then(Value::as_array) {
        for hit in hits.iter().take(max_results) {
            results.push(SearchResult {
                title: Some(
                    hit.get("title")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                ),
                url: hit
                    .get("url")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                snippet: truncate_chars(
                    hit.get("content").and_then(Value::as_str).unwrap_or(""),
                    SNIPPET_CHARS_MAX,
                ),
            });
        }
    }
    results
}

/// Truncate to `max` Unicode characters, as Python's `text[:max]` slices by
/// character, not by byte.
fn truncate_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

/// Read a response body in chunks, refusing bodies past the byte cap. A
/// hostile search backend must not exhaust memory through this tool.
fn read_capped(response: reqwest::blocking::Response) -> Result<Vec<u8>, ToolError> {
    let mut response = response;
    let mut body = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let read = response
            .read(&mut chunk)
            .map_err(|error| ToolError::Http(truncate_error(&error.to_string())))?;
        if read == 0 {
            break;
        }
        if body.len() + read > SEARCH_RESPONSE_BYTES_MAX {
            return Err(ToolError::Http(format!(
                "search response exceeds {SEARCH_RESPONSE_BYTES_MAX} bytes"
            )));
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Ok(body)
}

/// Server-chosen error strings must not flood results or logs.
fn truncate_error(message: &str) -> String {
    message
        .chars()
        .take(crate::error::HTTP_ERROR_CHARS_MAX)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DuckDuckGo fixture payload; the Python `duckduckgo_search` output
    /// for this payload is `PYTHON_DDG_RESULTS` below (Python's parse
    /// rules were verified against the real implementation).
    const DDG_INPUT: &str = r#"{"AbstractText": "Rust is a systems programming language focused on safety.", "Heading": "Rust (programming language)", "AbstractURL": "https://en.wikipedia.org/wiki/Rust_(programming_language)", "RelatedTopics": [{"Text": "Rust official site", "FirstURL": "https://www.rust-lang.org/"}, {"Text": "The Rust Book", "FirstURL": "https://doc.rust-lang.org/book/"}, {"NotText": "skipped, no Text key"}, "a string topic, skipped"]}"#;

    /// SearXNG fixture payload; the parse rules mirror the real
    /// Python `searxng_search` implementation.
    const SEARX_INPUT: &str = r#"{"results": [{"title": "Rust", "url": "https://www.rust-lang.org/", "content": "SNIPPET"}, {"title": "No content key", "url": "https://example.com/"}]}"#;

    fn ddg_payload() -> Value {
        serde_json::from_str(DDG_INPUT).unwrap()
    }

    #[test]
    fn duckduckgo_parse_matches_python() {
        let results = parse_duckduckgo(&ddg_payload(), "rust", 5);
        // Python returned 3 hits: the abstract plus the two topics that
        // carry a "Text" key.
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0],
            SearchResult {
                title: Some("Rust (programming language)".to_string()),
                url: "https://en.wikipedia.org/wiki/Rust_(programming_language)".to_string(),
                snippet: "Rust is a systems programming language focused on safety.".to_string(),
            }
        );
        assert_eq!(
            results[1],
            SearchResult {
                title: Some("Rust official site".to_string()),
                url: "https://www.rust-lang.org/".to_string(),
                snippet: "Rust official site".to_string(),
            }
        );
        assert_eq!(
            results[2],
            SearchResult {
                title: Some("The Rust Book".to_string()),
                url: "https://doc.rust-lang.org/book/".to_string(),
                snippet: "The Rust Book".to_string(),
            }
        );
    }

    #[test]
    fn duckduckgo_parse_serializes_to_python_shape() {
        let results = parse_duckduckgo(&ddg_payload(), "rust", 5);
        let value = Value::Array(results.iter().map(SearchResult::to_json).collect());
        let expected: Value = serde_json::from_str(
            r#"[{"title": "Rust (programming language)", "url": "https://en.wikipedia.org/wiki/Rust_(programming_language)", "snippet": "Rust is a systems programming language focused on safety."}, {"title": "Rust official site", "url": "https://www.rust-lang.org/", "snippet": "Rust official site"}, {"title": "The Rust Book", "url": "https://doc.rust-lang.org/book/", "snippet": "The Rust Book"}]"#,
        )
        .unwrap();
        assert_eq!(value, expected);
    }

    #[test]
    fn duckduckgo_respects_max_results() {
        let results = parse_duckduckgo(&ddg_payload(), "rust", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].title.as_deref(),
            Some("Rust (programming language)")
        );
    }

    #[test]
    fn duckduckgo_missing_heading_falls_back_to_query() {
        // Python: data.get("Heading", query).
        let payload: Value =
            serde_json::from_str(r#"{"AbstractText": "Some abstract.", "RelatedTopics": []}"#)
                .unwrap();
        let results = parse_duckduckgo(&payload, "my query", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title.as_deref(), Some("my query"));
        assert_eq!(results[0].to_json()["title"], "my query");
    }

    #[test]
    fn duckduckgo_null_heading_stays_null() {
        // Python: an explicit null is the value, not the default; it
        // serializes as JSON null rather than "".
        let payload: Value = serde_json::from_str(
            r#"{"AbstractText": "Some abstract.", "Heading": null, "RelatedTopics": []}"#,
        )
        .unwrap();
        let results = parse_duckduckgo(&payload, "my query", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, None);
        assert!(results[0].to_json()["title"].is_null());
    }

    #[test]
    fn duckduckgo_empty_abstract_yields_no_abstract_hit() {
        let payload: Value = serde_json::from_str(r#"{"RelatedTopics": []}"#).unwrap();
        assert!(parse_duckduckgo(&payload, "q", 5).is_empty());
    }

    #[test]
    fn duckduckgo_truncates_by_character_not_byte() {
        // 600 multi-byte chars; Python's text[:500] keeps 500 characters.
        let long = "\u{00e9}".repeat(600);
        let payload: Value =
            serde_json::from_str(&format!(r#"{{"AbstractText": "{long}"}}"#)).unwrap();
        let results = parse_duckduckgo(&payload, "q", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet.chars().count(), ABSTRACT_CHARS_MAX);
    }

    #[test]
    fn searxng_parse_matches_python() {
        let long = "x".repeat(400);
        let payload: Value = serde_json::from_str(&SEARX_INPUT.replace("SNIPPET", &long)).unwrap();
        let results = parse_searxng(&payload, 5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title.as_deref(), Some("Rust"));
        assert_eq!(results[0].url, "https://www.rust-lang.org/");
        // Python truncates content to 300 characters (fixtures.json).
        assert_eq!(results[0].snippet.chars().count(), SNIPPET_CHARS_MAX);
        assert_eq!(results[0].snippet, "x".repeat(300));
        assert_eq!(results[1].title.as_deref(), Some("No content key"));
        assert_eq!(results[1].snippet, "");
    }

    #[test]
    fn searxng_respects_max_results() {
        let payload: Value = serde_json::from_str(
            r#"{"results": [{"title": "a", "url": "u", "content": "c"}, {"title": "b", "url": "u", "content": "c"}]}"#,
        )
        .unwrap();
        assert_eq!(parse_searxng(&payload, 1).len(), 1);
    }

    #[test]
    fn run_json_matches_python_dumps_indent2() {
        // Byte-identity of WebSearch::run's output: json.dumps with
        // indent=2 and ensure_ascii. Expected string generated by driving
        // the real Python's json.dumps on the equivalent dicts
        // (2026-09-26).
        let results = [SearchResult {
            title: Some("café".to_string()),
            url: "https://ex.com/".to_string(),
            snippet: "naïve".to_string(),
        }];
        let value = Value::Array(results.iter().map(SearchResult::to_json).collect());
        let expected = "[\n  {\n    \"title\": \"caf\\u00e9\",\n    \"url\": \"https://ex.com/\",\n    \"snippet\": \"na\\u00efve\"\n  }\n]";
        assert_eq!(python_json_dumps_indent2(&value), expected);
    }

    #[test]
    fn query_over_bound_is_rejected() {
        let search = WebSearch::duckduckgo().unwrap();
        let long = "q".repeat(QUERY_CHARS_MAX + 1);
        let error = search.search(&long, 5).unwrap_err();
        assert!(matches!(error, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn negative_max_results_is_rejected() {
        let search = WebSearch::duckduckgo().unwrap();
        let error = search.search("rust", -1).unwrap_err();
        assert!(matches!(error, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn searxng_url_over_bound_is_rejected() {
        let long = format!("http://x/{}/", "y".repeat(SEARXNG_URL_CHARS_MAX));
        let error = WebSearch::searxng(&long).unwrap_err();
        assert!(matches!(error, ToolError::Http(_)));
    }

    #[test]
    fn empty_searxng_url_falls_back_to_duckduckgo() {
        let search = WebSearch::with_backend(SearchBackend::SearXng, "").unwrap();
        assert_eq!(search.backend(), SearchBackend::DuckDuckGo);
    }

    #[test]
    fn error_outcome_keeps_python_shape() {
        // Connection refused on localhost: fails fast, exercises the
        // [{"error": ...}] envelope Python returned on transport errors.
        let search = WebSearch::searxng("http://127.0.0.1:9").unwrap();
        let rendered = search.run("rust", 5);
        let value: Value = serde_json::from_str(&rendered).unwrap();
        let items = value.as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].get("error").and_then(Value::as_str).is_some());
    }
}
