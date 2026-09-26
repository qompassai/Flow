//! Slash-command parsing and execution, ported from `FlowApp.execute` in
//! `flow/tui/app.py`. This module has no terminal I/O: [`execute`] takes the
//! app state and one pre-trimmed input line and returns the result JSON, so
//! every command is unit-testable. The interactive loop ([`crate::app`])
//! trims lines and handles `/quit`/`/exit`/empty lines before calling
//! [`execute`], exactly as Python's `run()` loop does.
//!
//! Parsing mirrors Python's `text.partition(" ")`: the command is the text
//! up to the first ASCII space, and the argument is the untrimmed
//! remainder. A tab does not separate command from argument; extra spaces
//! are preserved in the argument.
//!
//! Command surface (exact strings preserved from the Python TUI):
//!
//! - `/status`, `/check [name]`, `/tools`, `/models`, `/model <name>`,
//!   `/build <request>`, `/help`, `/clear`
//! - `/model` accepts any non-empty name, as Python does (it assigns
//!   `cfg.ollama.model` and resets role overrides; there is no allowlist).
//! - `/plugins`, `/evolve`, `/memory`, `/feedback` are unavailable: they
//!   return `{"status": "unavailable", ...}` with the exact Python denial.
//! - Anything else, and any unknown `/command` (including `/quit`, which
//!   the loop handles), is an error.

use serde_json::{Value, json};

use crate::facade::RuntimeFacade;

/// The chat application state: the runtime facade, the installed model
/// list, and the session banner facts.
pub struct FlowApp<F> {
    facade: F,
    models: Vec<String>,
    workspace_root: String,
    trusted: bool,
}

impl<F: RuntimeFacade> FlowApp<F> {
    /// Build the app. `models` is the installed-model list shown by
    /// `/models`; Phase 6 supplies it from the Ollama backend at startup,
    /// as Python's `FlowApp` did.
    pub fn new(
        facade: F,
        models: Vec<String>,
        workspace_root: String,
        trusted: bool,
    ) -> FlowApp<F> {
        FlowApp {
            facade,
            models,
            workspace_root,
            trusted,
        }
    }

    /// Execute one pre-trimmed input line; see [`execute`].
    pub fn execute(&mut self, text: &str) -> Value {
        execute(self, text)
    }

    /// The runtime facade.
    pub fn facade(&mut self) -> &mut F {
        &mut self.facade
    }

    /// The installed models.
    pub fn models(&self) -> &[String] {
        &self.models
    }

    /// The workspace root shown in the banner.
    pub fn workspace_root(&self) -> &str {
        &self.workspace_root
    }

    /// Whether the session runs in trusted mode.
    pub fn trusted(&self) -> bool {
        self.trusted
    }
}

/// The exact denial for the unavailable legacy commands, from
/// `flow/tui/app.py`.
pub const UNAVAILABLE_MESSAGE: &str = "Legacy executable plugins, automatic prompt mutation and cross-workspace memory are disabled in the safe runtime";

/// The command list shared by `/help` and `/clear`, from `flow/tui/app.py`.
pub const COMMANDS: &[&str] = &[
    "/status",
    "/check [name]",
    "/tools",
    "/models",
    "/model <name>",
    "/build <request>",
    "/clear",
    "/quit",
];

/// The note shared by `/help` and `/clear`, from `flow/tui/app.py`.
pub const COMMANDS_NOTE: &str =
    "Every task has fresh role contexts; /clear needs no persistent cleanup";

/// Parse and execute one pre-trimmed input line. Slash commands dispatch
/// to the facade; anything else is a natural-language task passed to
/// `run` unchanged, exactly as in Python.
///
/// Like Python's `execute`, this does not trim its input and does not
/// handle `/quit`: the caller strips the line and checks for quit first.
pub fn execute<F: RuntimeFacade>(app: &mut FlowApp<F>, text: &str) -> Value {
    let (command, arg) = split_command(text);
    match command {
        "/status" => app.facade.status(),
        "/check" => app.facade.check(non_empty(arg)),
        "/tools" => json!({"status": "ok", "tools": app.facade.schemas()}),
        "/models" => json!({"status": "ok", "models": app.models}),
        "/model" => select_model(app, arg),
        "/build" => app.facade.run(&format!("Build {arg}")),
        "/help" | "/clear" => json!({
            "status": "ok",
            "commands": COMMANDS,
            "note": COMMANDS_NOTE,
        }),
        "/plugins" | "/evolve" | "/memory" | "/feedback" => {
            json!({"status": "unavailable", "error": UNAVAILABLE_MESSAGE})
        }
        _ if text.starts_with('/') => {
            json!({"status": "error", "error": "Unknown command; try /help"})
        }
        _ => app.facade.run(text),
    }
}

/// Split at the first ASCII space, mirroring Python's
/// `text.partition(" ")`: the argument keeps its exact spacing.
fn split_command(text: &str) -> (&str, &str) {
    match text.find(' ') {
        Some(index) => (&text[..index], &text[index + 1..]),
        None => (text, ""),
    }
}

/// `/model <name>`: any non-empty name is accepted, as in Python (which
/// assigns `cfg.ollama.model` and resets per-role overrides). There is no
/// allowlist.
fn select_model<F: RuntimeFacade>(app: &mut FlowApp<F>, arg: &str) -> Value {
    if arg.is_empty() {
        json!({"status": "error", "error": "Usage: /model <installed-model>"})
    } else {
        app.facade.select_model(arg);
        json!({"status": "ok", "model": arg})
    }
}

/// `None` for an empty argument, `Some` otherwise.
fn non_empty(arg: &str) -> Option<&str> {
    if arg.is_empty() { None } else { Some(arg) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facade::FakeRuntime;
    use serde_json::json;

    fn app() -> FlowApp<FakeRuntime> {
        let mut fake = FakeRuntime::new();
        fake.status_result = json!({"status": "ok", "model": "llama3.1"});
        fake.check_result = json!({"status": "ok", "checks": []});
        fake.schemas_result = vec![json!({"name": "status"})];
        FlowApp::new(
            fake,
            vec!["llama3.1".to_string(), "qwen3".to_string()],
            "/workspace".to_string(),
            false,
        )
    }

    #[test]
    fn status_returns_facade_result() {
        let mut app = app();
        assert_eq!(
            app.execute("/status"),
            json!({"status": "ok", "model": "llama3.1"})
        );
    }

    #[test]
    fn check_without_name_runs_all_checks() {
        let mut app = app();
        assert_eq!(app.execute("/check"), json!({"status": "ok", "checks": []}));
        assert_eq!(app.facade().check_names, vec![None]);
    }

    #[test]
    fn check_with_name_passes_name_through() {
        let mut app = app();
        app.execute("/check pytest");
        assert_eq!(app.facade().check_names, vec![Some("pytest".to_string())]);
    }

    #[test]
    fn check_with_tab_does_not_split_command() {
        // partition(" "): a tab is not a space, so the whole thing is the
        // command and it is unknown — as in Python.
        let mut app = app();
        assert_eq!(
            app.execute("/check\tfoo"),
            json!({"status": "error", "error": "Unknown command; try /help"})
        );
        assert!(app.facade().check_names.is_empty());
    }

    #[test]
    fn check_preserves_extra_spaces_in_arg() {
        // partition(" ") keeps the remainder untrimmed, as in Python.
        let mut app = app();
        app.execute("/check  foo");
        assert_eq!(app.facade().check_names, vec![Some(" foo".to_string())]);
    }

    #[test]
    fn tools_wraps_coder_schemas() {
        let mut app = app();
        assert_eq!(
            app.execute("/tools"),
            json!({"status": "ok", "tools": [{"name": "status"}]})
        );
    }

    #[test]
    fn models_lists_installed_models() {
        let mut app = app();
        assert_eq!(
            app.execute("/models"),
            json!({"status": "ok", "models": ["llama3.1", "qwen3"]})
        );
    }

    #[test]
    fn model_without_argument_reports_usage() {
        let mut app = app();
        assert_eq!(
            app.execute("/model"),
            json!({"status": "error", "error": "Usage: /model <installed-model>"})
        );
        assert!(app.facade().selected_models.is_empty());
    }

    #[test]
    fn model_accepts_any_non_empty_name() {
        // Python has no allowlist: it assigns cfg.ollama.model directly.
        let mut app = app();
        assert_eq!(
            app.execute("/model nope"),
            json!({"status": "ok", "model": "nope"})
        );
        assert_eq!(app.facade().selected_models, vec!["nope".to_string()]);
    }

    #[test]
    fn build_prefixes_request() {
        let mut app = app();
        app.facade()
            .queue_run_result(json!({"status": "ok", "summary": "done"}));
        assert_eq!(
            app.execute("/build a snake game"),
            json!({"status": "ok", "summary": "done"})
        );
        assert_eq!(
            app.facade().run_tasks,
            vec!["Build a snake game".to_string()]
        );
    }

    #[test]
    fn build_preserves_extra_spaces() {
        let mut app = app();
        app.execute("/build  foo");
        assert_eq!(app.facade().run_tasks, vec!["Build  foo".to_string()]);
    }

    #[test]
    fn build_without_arg_runs_bare_prefix() {
        let mut app = app();
        app.execute("/build");
        assert_eq!(app.facade().run_tasks, vec!["Build ".to_string()]);
    }

    #[test]
    fn help_and_clear_share_commands_and_note() {
        let mut app = app();
        let expected = json!({
            "status": "ok",
            "commands": COMMANDS,
            "note": COMMANDS_NOTE,
        });
        assert_eq!(app.execute("/help"), expected);
        assert_eq!(app.execute("/clear"), expected);
    }

    #[test]
    fn unavailable_commands_return_unavailable_status() {
        let mut app = app();
        for command in ["/plugins", "/evolve", "/memory", "/feedback"] {
            assert_eq!(
                app.execute(command),
                json!({"status": "unavailable", "error": UNAVAILABLE_MESSAGE}),
                "command {command}"
            );
        }
        assert_eq!(
            UNAVAILABLE_MESSAGE,
            "Legacy executable plugins, automatic prompt mutation and cross-workspace memory are disabled in the safe runtime"
        );
    }

    #[test]
    fn unknown_command_suggests_help() {
        let mut app = app();
        assert_eq!(
            app.execute("/frobnicate"),
            json!({"status": "error", "error": "Unknown command; try /help"})
        );
    }

    #[test]
    fn quit_is_unknown_to_execute() {
        // Python's execute has no quit handling; the run() loop owns it.
        let mut app = app();
        assert_eq!(
            app.execute("/quit"),
            json!({"status": "error", "error": "Unknown command; try /help"})
        );
        assert_eq!(
            app.execute("/exit"),
            json!({"status": "error", "error": "Unknown command; try /help"})
        );
    }

    #[test]
    fn natural_language_passes_through_untrimmed() {
        // execute does not trim: the loop strips before calling, as in
        // Python.
        let mut app = app();
        assert_eq!(
            app.execute("  explain lifetimes  "),
            json!({"status": "ok", "echo": "  explain lifetimes  "})
        );
        assert_eq!(
            app.facade().run_tasks,
            vec!["  explain lifetimes  ".to_string()]
        );
    }

    #[test]
    fn leading_space_makes_slash_text_natural_language() {
        // partition(" ") on "  /status" yields command "", which is not a
        // slash command, so it runs as natural language — as in Python.
        let mut app = app();
        app.execute("  /status");
        assert_eq!(app.facade().run_tasks, vec!["  /status".to_string()]);
    }
}
