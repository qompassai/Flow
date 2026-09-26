//! System prompt templates and prompt builders.
//!
//! Byte-exact port of `flow/llm/prompts.py`. The templates below reproduce
//! the Python source character-for-character, including the rendered JSON
//! example (Python's `{{`/`}}` escapes are already expanded here) and the
//! trailing newline. `{tool_descriptions}` and `{language_profile}` are the
//! only placeholders; `str::replace` substitutes them so the literal braces
//! in the JSON example need no escaping.

use serde_json::Value;

/// The default system prompt template, before placeholder substitution.
/// Byte-exact with `prompts.DEFAULT_SYSTEM_PROMPT`.
pub const DEFAULT_SYSTEM_PROMPT_TEMPLATE: &str = "You are Flow, an expert AI software engineering assistant running fully locally.\nYou help the user build, debug, and improve software applications.\n\n## Your Capabilities\n- Generate complete application code in any language\n- Select and use appropriate LSPs, linters, formatters, and frameworks\n- Call only supplied workspace file tools, configured named checks and optional native editor tools\n- Validate code by running LSP diagnostics and feeding errors back to yourself\n- Ask the user for clarification when you encounter uncertainty or decision points\n\n## Behavior Rules\n1. Think step-by-step before acting. Use <think>...</think> tags internally.\n2. Explain uncertainty; do not invent permission or capabilities.\n3. Always validate generated code with LSP diagnostics before declaring success.\n4. When calling a tool, output ONLY the tool call JSON \u{2014} no surrounding text.\n5. After each code generation, check for errors and iterate until clean.\n6. Be concise in output. Show diffs for edits rather than full files when possible.\n\n## Tool Calling Format\nTo call a tool, output exactly this JSON (no markdown fences):\n{\"tool\": \"tool_name\", \"args\": {\"key\": \"value\"}}\n\n## Available Tools\n{tool_descriptions}\n\n## Current Language Profile\n{language_profile}\n";

/// Render the default system prompt. Empty descriptions fall back to
/// `"None loaded yet."` / `"None selected yet."`, like Python's `or`.
pub fn default_system_prompt(tool_descriptions: &str, language_profile: &str) -> String {
    let tools = if tool_descriptions.is_empty() {
        "None loaded yet."
    } else {
        tool_descriptions
    };
    let profile = if language_profile.is_empty() {
        "None selected yet."
    } else {
        language_profile
    };
    DEFAULT_SYSTEM_PROMPT_TEMPLATE
        .replace("{tool_descriptions}", tools)
        .replace("{language_profile}", profile)
}

/// Load the system prompt from `system_prompt.md` when `skills_dir` contains
/// it, else the default. Mirrors `load_system_prompt`, minus the
/// `importlib.resources` lookup: the caller resolves the file.
///
/// NOTE for Phase 4/6: the production Python server prefers
/// `flow/skills/system_prompt.md` whenever `SYSTEM_PROMPT_FILE.is_file()`
/// and only falls back to the default template otherwise. The runtime must
/// resolve and prefer the skills file the same way (vendored copy or
/// binary-relative lookup) — silently falling back to the default would
/// change the model's system prompt.
pub fn load_system_prompt(
    system_prompt_file: Option<&str>,
    tool_descriptions: &str,
    language_profile: &str,
) -> String {
    match system_prompt_file {
        Some(template) => {
            let tools = if tool_descriptions.is_empty() {
                "None loaded yet."
            } else {
                tool_descriptions
            };
            let profile = if language_profile.is_empty() {
                "None selected yet."
            } else {
                language_profile
            };
            template
                .replace("{tool_descriptions}", tools)
                .replace("{language_profile}", profile)
        }
        None => default_system_prompt(tool_descriptions, language_profile),
    }
}

/// Build a code-generation prompt. Byte-exact with
/// `prompts.build_codegen_prompt`, including the `\n\n## Existing Files\n`
/// section when `existing_files` is non-empty (iteration order is the map's).
pub fn build_codegen_prompt(
    request: &str,
    language: &str,
    framework: &str,
    project_name: &str,
    existing_files: &[(&str, &str)],
) -> String {
    let mut files_section = String::new();
    if !existing_files.is_empty() {
        files_section.push_str("\n\n## Existing Files\n");
        for (path, content) in existing_files {
            files_section.push_str(&format!("\n### {path}\n```\n{content}\n```\n"));
        }
    }
    format!(
        "Generate a complete {language} {framework} application called \"{project_name}\".\n\
         \n\
         ## User Request\n\
         {request}\n\
         \n\
         ## Requirements\n\
         - Language: {language}\n\
         - Framework: {framework}\n\
         - Output complete, working code for all necessary files\n\
         - Include proper error handling and logging\n\
         - Follow {language} best practices and idioms\n\
         - Include a README.md with setup and run instructions\n\
         {files_section}\n\
         \n\
         ## Output Format\n\
         For each file, output:\n\
         FILE: <relative/path/to/file>\n\
         ```<language>\n\
         <content>\n\
         ```\n\
         \n\
         After writing files with the provided tools, the host runs the required named-check gate.\n"
    )
}

/// Build an LSP-error fix prompt. Byte-exact with
/// `prompts.build_error_fix_prompt`: `Line {line}: [{severity}] {message}`
/// per error, where a missing key falls back to `'?'` / `'error'` / `''`
/// and any present value renders with Python `str()` semantics.
pub fn build_error_fix_prompt(errors: &[Value], file_content: &str, file_path: &str) -> String {
    let error_lines: Vec<String> = errors
        .iter()
        .map(|error| {
            let line = error
                .get("line")
                .map(render_error_value)
                .unwrap_or_else(|| "?".to_owned());
            let severity = error
                .get("severity")
                .map(render_error_value)
                .unwrap_or_else(|| "error".to_owned());
            let message = error
                .get("message")
                .map(render_error_value)
                .unwrap_or_default();
            format!("Line {line}: [{severity}] {message}")
        })
        .collect();
    format!(
        "Fix the following LSP errors in {file_path}:\n\
         \n\
         ## Errors\n\
         {errors}\n\
         \n\
         ## Current File Content\n\
         ```\n\
         {file_content}\n\
         ```\n\
         \n\
         Output only the corrected file content in the same format.\n",
        errors = error_lines.join("\n"),
    )
}

/// Render a scalar error field like Python's `str()`: strings as-is,
/// `True`/`False` for booleans, `"None"` for null, numbers without quotes.
/// Exotic floats (`1e+20`) and compound values are best-effort — realistic
/// LSP payloads only carry strings and numbers here.
fn render_error_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => number.to_string(),
        Value::Null => "None".to_owned(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_prompt_matches_python_fallback_byte_for_byte() {
        // This pins only the DEFAULT_SYSTEM_PROMPT fallback constant. The
        // production path prefers flow/skills/system_prompt.md; see
        // system_prompt_prefers_skills_file_byte_for_byte below.
        let fixture = include_str!("../tests/fixtures/llm_golden.json");
        let golden: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let tool_desc = "- flow_run: run things\n- flow_check: check things";
        let lang_profile = "## Python\n- formatter: ruff";
        assert_eq!(
            default_system_prompt(tool_desc, lang_profile),
            golden["system_prompt_default"].as_str().unwrap()
        );
    }

    #[test]
    fn empty_descriptions_fall_back() {
        let prompt = default_system_prompt("", "");
        assert!(prompt.contains("None loaded yet."));
        assert!(prompt.contains("None selected yet."));
    }

    #[test]
    fn system_prompt_file_template_is_used_when_present() {
        let fixture = include_str!("../tests/fixtures/llm_golden.json");
        let golden: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let template = golden["system_prompt_template"].as_str().unwrap();
        let tool_desc = "- flow_run: run things\n- flow_check: check things";
        let lang_profile = "## Python\n- formatter: ruff";
        assert_eq!(
            load_system_prompt(Some(template), tool_desc, lang_profile),
            golden["system_prompt_file"].as_str().unwrap()
        );
    }

    #[test]
    fn system_prompt_prefers_skills_file_byte_for_byte() {
        // The production path: Python's load_system_prompt() reads the real
        // flow/skills/system_prompt.md (it exists, so the fallback never
        // fires in production). The fixture was generated by driving the
        // real Python function; this pins the Rust rendering byte-for-byte.
        let fixture = include_str!("../tests/fixtures/system_prompt_production.json");
        let golden: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let template = golden["template"].as_str().unwrap();
        let tool_desc = golden["tool_descriptions"].as_str().unwrap();
        let lang_profile = golden["language_profile"].as_str().unwrap();
        assert_eq!(
            load_system_prompt(Some(template), tool_desc, lang_profile),
            golden["rendered"].as_str().unwrap()
        );
    }

    #[test]
    fn codegen_prompt_matches_python_byte_for_byte() {
        let fixture = include_str!("../tests/fixtures/llm_golden.json");
        let golden: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let prompt = build_codegen_prompt(
            "add retry",
            "python",
            "stdlib",
            "demo",
            &[("a.py", "print(1)\n"), ("b.py", "print(2)\n")],
        );
        assert_eq!(prompt, golden["codegen_prompt"].as_str().unwrap());
    }

    #[test]
    fn codegen_prompt_without_files_has_no_files_section() {
        let prompt = build_codegen_prompt("x", "rust", "std", "demo", &[]);
        assert!(!prompt.contains("## Existing Files"));
        assert!(prompt.contains(
            "- Include a README.md with setup and run instructions\n\n\n## Output Format"
        ));
    }

    #[test]
    fn error_fix_prompt_matches_python_byte_for_byte() {
        let fixture = include_str!("../tests/fixtures/llm_golden.json");
        let golden: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let errors = vec![
            json!({"line": 3, "severity": "error", "message": "bad indent"}),
            json!({"severity": "warning"}),
        ];
        let prompt = build_error_fix_prompt(&errors, "print(1)\n  print(2)\n", "/tmp/x.py");
        assert_eq!(prompt, golden["error_fix_prompt"].as_str().unwrap());
    }

    #[test]
    fn error_fix_prompt_renders_null_line_like_python() {
        let prompt = build_error_fix_prompt(&[json!({"line": Value::Null})], "x", "f");
        assert!(prompt.contains("Line None: [error] "));
    }

    #[test]
    fn error_fix_prompt_uses_python_str_semantics() {
        // Verified against CPython's build_error_fix_prompt: a missing key
        // falls back ('?', 'error', ''), any present value renders with
        // str() — including None -> "None", "" -> "", True -> "True".
        for (error, expected) in [
            (json!({"line": ""}), "Line : [error] "),
            (json!({"line": true}), "Line True: [error] "),
            (json!({"line": false}), "Line False: [error] "),
            (json!({"severity": Value::Null}), "Line ?: [None] "),
            (json!({"severity": ""}), "Line ?: [] "),
            (json!({"message": Value::Null}), "Line ?: [error] None"),
            (json!({"message": true}), "Line ?: [error] True"),
            (
                json!({"line": 0, "severity": 0, "message": 0}),
                "Line 0: [0] 0",
            ),
        ] {
            let prompt = build_error_fix_prompt(&[error], "x", "f");
            assert!(
                prompt.contains(expected),
                "expected {expected:?} in:\n{prompt}"
            );
        }
    }
}
