//! The bounded app generator.
//!
//! Ports `flow/codegen/app_generator.py`. A build request becomes one
//! runtime task — `Build '<project>' using <language> <framework>...` —
//! submitted to the shared safe runtime. The generator adds no planning
//! of its own; the runtime's planner/coder/reviewer pipeline does the
//! work, and its result JSON is returned unchanged.

use phlow_editor::EditorTransport;
use phlow_llm::LlmTransport;
use phlow_runtime::Runtime;
use serde_json::Value;

/// Generate an application through the shared safe runtime.
///
/// The generator owns the runtime it submits to; construct it with the
/// runtime the CLI/TUI already configured.
pub struct AppGenerator<L: LlmTransport, E: EditorTransport> {
    runtime: Runtime<L, E>,
}

impl<L: LlmTransport, E: EditorTransport> AppGenerator<L, E> {
    /// Wrap an existing runtime. The runtime keeps its workspace, model,
    /// and trust configuration.
    pub fn new(runtime: Runtime<L, E>) -> AppGenerator<L, E> {
        AppGenerator { runtime }
    }

    /// Submit a build request and return the runtime's JSON result.
    pub fn generate(
        &mut self,
        request: &str,
        language: &str,
        framework: &str,
        project_name: &str,
    ) -> Value {
        self.runtime
            .run(&build_prompt(request, language, framework, project_name))
    }
}

/// Build the task prompt submitted to the runtime.
///
/// The prompt mirrors Python exactly: `Build {project_name!r} using
/// {language} {framework}. Use workspace-relative file tools and required
/// checks. Request: {request}`.
pub fn build_prompt(request: &str, language: &str, framework: &str, project_name: &str) -> String {
    format!(
        "Build {} using {} {}. Use workspace-relative file tools and required checks. Request: {}",
        python_repr(project_name),
        language,
        framework,
        request,
    )
}

/// Python's `repr()` for strings, used for `{project_name!r}` in the
/// prompt. Single-quote style: the string is wrapped in `'` unless it
/// contains `'` but not `"`, in which case `"` is used. Backslashes, the
/// wrapping quote, and control characters are escaped (`\\n`, `\\r`,
/// `\\t`, `\\xXX`/`\\uXXXX`/`\\UXXXXXXXX`).
///
/// Printable-ness is approximated with `char::is_control`: project names
/// are short operator-chosen labels, and the approximation only affects
/// exotic Unicode, never ASCII.
pub fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch if ch.is_control() => {
                let code = ch as u32;
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x10000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
            ch => out.push(ch),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python `repr()` outputs captured from the real interpreter
    /// (fixtures.json `reprs`).
    const REPR_CASES: &[(&str, &str)] = &[
        ("myapp", "'myapp'"),
        ("it's", "\"it's\""),
        ("say \"hi\"", "'say \"hi\"'"),
        ("back\\slash", "'back\\\\slash'"),
        ("line\nbreak", "'line\\nbreak'"),
        ("tab\there", "'tab\\there'"),
        ("uni\u{00e9}", "'uni\u{00e9}'"),
    ];

    #[test]
    fn python_repr_matches_interpreter() {
        for (input, expected) in REPR_CASES {
            assert_eq!(python_repr(input), *expected, "input: {input:?}");
        }
    }

    #[test]
    fn python_repr_escapes_control_characters() {
        assert_eq!(python_repr("a\x07b"), "'a\\x07b'");
        assert_eq!(python_repr("a\rb"), "'a\\rb'");
    }

    #[test]
    fn python_repr_empty_string() {
        assert_eq!(python_repr(""), "''");
    }

    #[test]
    fn build_prompt_matches_python_f_string() {
        // Golden prompts from the real Python f-string (golden_prompt.py).
        let cases = [
            (
                "a todo API",
                "python",
                "fastapi",
                "myapp",
                "Build 'myapp' using python fastapi. Use workspace-relative file tools and required checks. Request: a todo API",
            ),
            (
                "a web server",
                "rust",
                "axum",
                "it's",
                "Build \"it's\" using rust axum. Use workspace-relative file tools and required checks. Request: a web server",
            ),
            (
                "a CLI",
                "go",
                "bare",
                "say \"hi\"",
                "Build 'say \"hi\"' using go bare. Use workspace-relative file tools and required checks. Request: a CLI",
            ),
        ];
        for (request, language, framework, project_name, expected) in cases {
            assert_eq!(
                build_prompt(request, language, framework, project_name),
                expected,
                "project: {project_name:?}"
            );
        }
    }
}

#[cfg(test)]
mod generate_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phlow-codegen-gen-test-{prefix}-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create temp dir");
        dir
    }

    #[test]
    fn generate_surfaces_runtime_errors_as_json() {
        let dir = temp_dir("wiring");
        std::fs::write(dir.join("config.toml"), "").unwrap();
        let config = phlow_config::load_config(&phlow_config::LoadOptions {
            config_path: Some(dir.join("config.toml")),
            workspace: Some(dir.clone()),
            trusted: false,
            model: None,
        })
        .expect("test config loads");
        let mut llm = phlow_llm::FakeLlmTransport::new();
        llm.queue_reply(Err(phlow_llm::LlmError::Transport(
            "connection refused".to_string(),
        )));
        let runtime = phlow_runtime::Runtime::new(
            config,
            llm,
            phlow_editor::fake::FakeTransport::new(),
            None,
        )
        .expect("runtime builds");
        let mut generator = AppGenerator::new(runtime);
        let result = generator.generate("a todo API", "python", "fastapi", "myapp");
        assert_eq!(result["status"], "error");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
