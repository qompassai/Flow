//! `phlow` (and its `flow` compatibility alias): the command-line surface
//! of the safe agent runtime, ported from `flow/main.py`.
//!
//! Exit codes mirror the Python exactly: 0 success, 1 the command ran but
//! the report status is not `ok`, 2 usage/config error, 130 interrupted
//! (SIGTERM). Reports print as one ASCII-escaped JSON line, byte-compatible
//! with Python's `json.dumps(result, ensure_ascii=True)`.
//!
//! The two binaries (`src/bin/phlow.rs`, `src/bin/flow.rs`) are thin shims
//! over [`run`]; clap renders the help's Usage line from the actual
//! `argv[0]`, so the alias needs no separate parser.

#![forbid(unsafe_code)]

mod cli;
mod signal;
mod wiring;

use std::io::{self, IsTerminal as _};
use std::time::Duration;

use clap::{CommandFactory as _, Parser};
use phlow_config::{LoadOptions, load_config};
use phlow_editor::{TIMEOUT_MAX, TIMEOUT_MIN};
use phlow_mcp::McpServer;
use phlow_runtime::Runtime;
use phlow_runtime::transport::{MsgpackTransport, ReqwestTransport};
use phlow_tools::python_json_dumps;
use phlow_tui::FlowApp;
use serde_json::Value;

use cli::{Cli, Commands};
use signal::{EXIT_COMMAND_FAILED, EXIT_USAGE};
use wiring::CliRuntime;

/// Version string, byte-identical to Python's `Flow {flow.__version__}`.
/// `flow/__init__.py` declares `__version__ = "0.2.0"`; the differential
/// test below pins the two together.
pub const VERSION: &str = "0.2.0";

/// Run the CLI to completion and return the process exit code.
///
/// Owns every resource; returning runs all destructors. The only
/// `process::exit` inside is the SIGTERM watcher thread, which mirrors
/// Python abandoning in-flight work on KeyboardInterrupt.
pub fn run() -> i32 {
    // Install first, like Python installing its SIGTERM handler before
    // parsing: from here on, SIGTERM means exit 130 with the message.
    if let Err(error) = signal::install_sigterm_watcher() {
        eprintln!("Flow: {error}");
        return EXIT_USAGE;
    }
    reject_bare_double_dash();
    // clap prints usage to stderr and exits 2 on bad argv, like argparse.
    let cli = Cli::parse();
    if cli.version {
        // argparse's version action fires during parsing, winning over any
        // subcommand on the same command line.
        println!("Flow {VERSION}");
        return 0;
    }
    match dispatch(&cli) {
        Ok(code) => code,
        // Mirrors `except (ConfigError, OSError, ValueError)`: the typed
        // message, prefixed, to stderr, exit 2.
        Err(error) => {
            eprintln!("Flow: {error}");
            EXIT_USAGE
        }
    }
}

/// A lone `--` with nothing after it: argparse reports
/// `unrecognized arguments: --` (exit 2) because no positional can absorb
/// the separator. clap would otherwise treat it as "no subcommand" and
/// launch the TUI. (`phlow -- status` already exits 2 on both sides via
/// clap's own positional check, so only the bare case needs help.)
fn reject_bare_double_dash() {
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if args.len() == 2 && args[1] == "--" {
        Cli::command()
            .error(
                clap::error::ErrorKind::UnknownArgument,
                "unrecognized arguments: --",
            )
            .exit();
    }
}

/// Build the runtime and run the selected command. `Err` is always a
/// startup failure (config, transport, timeout, runtime construction);
/// command-level failures are encoded in the report's `status` field.
fn dispatch(cli: &Cli) -> Result<i32, String> {
    // Order mirrors Python's main(): load_config first, then the
    // --editor-timeout range check, so a bad config wins over a bad
    // timeout like Python's.
    let config = load_config(&LoadOptions {
        config_path: cli.config.clone(),
        workspace: cli.workspace.clone(),
        trusted: cli.trusted,
        model: cli.model.clone(),
    })
    .map_err(|error| error.to_string())?;

    if let Some(seconds) = cli.editor_timeout {
        // `contains` is false for NaN, so NaN/inf/out-of-range all land
        // here, exactly like Python's `not MIN <= v <= MAX`.
        if !(TIMEOUT_MIN.as_secs_f64()..=TIMEOUT_MAX.as_secs_f64()).contains(&seconds) {
            return Err(format!(
                "--editor-timeout must be between {} and {} seconds",
                TIMEOUT_MIN.as_secs_f64(),
                TIMEOUT_MAX.as_secs_f64(),
            ));
        }
    }

    // Captured before `config` moves into the runtime; the TUI banner needs
    // both. `to_string_lossy` is display-only (the banner), never a path
    // used for filesystem access.
    let workspace_root = config.workspace_dir().to_string_lossy().into_owned();
    let trusted = config.trusted();

    let llm_transport = ReqwestTransport::new().map_err(|error| error.to_string())?;
    // Python builds no editor transport without --nvim; the msgpack worker
    // dials lazily, so with no socket it idles untouched until exit.
    let socket_path = cli.nvim.clone().unwrap_or_default();
    let editor_transport = MsgpackTransport::new(&socket_path);
    let mut runtime = Runtime::new(config, llm_transport, editor_transport, cli.nvim.clone())
        .map_err(|error| error.to_string())?;
    if let Some(seconds) = cli.editor_timeout {
        // Guarded by the range check above: finite and within 0.1..=660.0,
        // so `from_secs_f64` cannot panic.
        debug_assert!(seconds.is_finite());
        let timeout = Duration::from_secs_f64(seconds);
        runtime
            .set_editor_timeout(timeout)
            .map_err(|error| error.to_string())?;
    }

    match &cli.command {
        Some(Commands::Serve) => serve(runtime),
        Some(Commands::Run { task }) => {
            let result = runtime.run(&task.join(" "));
            finish_report(runtime, &result)
        }
        Some(Commands::Check { .. }) => {
            let name = cli.command.as_ref().and_then(Commands::check_name);
            let result = runtime.check(name);
            finish_report(runtime, &result)
        }
        Some(Commands::Status) => {
            let result = runtime.status();
            finish_report(runtime, &result)
        }
        // No subcommand drops into the TUI, like Python's `else` branch.
        Some(Commands::Tui) | None => tui(runtime, workspace_root, trusted),
    }
}

/// Map a report's `status` field to an exit code:
/// 0 when `status == "ok"`, 1 otherwise. Mirrors Python's
/// `return 0 if result.get("status") == "ok" else 1`.
fn report_exit_code(result: &Value) -> i32 {
    match result.get("status") {
        Some(Value::String(status)) if status == "ok" => 0,
        _ => EXIT_COMMAND_FAILED,
    }
}

/// Print one ASCII-escaped JSON report line and map its `status` to an exit
/// code, mirroring `print(json.dumps(...)); return 0 if ... == "ok" else 1`.
/// The runtime is closed before returning, like the `with runtime:` block.
fn finish_report(mut runtime: wiring::ProductionRuntime, result: &Value) -> Result<i32, String> {
    println!("{}", python_json_dumps(result));
    let code = report_exit_code(result);
    runtime.close();
    Ok(code)
}

/// Newline-framed MCP over stdio. stdout carries protocol JSON exclusively;
/// `serve` flushes every frame. The server closes the runtime on every exit
/// path, mirroring Python's `finally`.
fn serve(runtime: wiring::ProductionRuntime) -> Result<i32, String> {
    let mut server = McpServer::new(CliRuntime::new(runtime));
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    match server.serve(&mut input, &mut output) {
        Ok(_) => Ok(0),
        Err(error) => Err(error.to_string()),
    }
}

/// Interactive terminal frontend. The model list is fetched eagerly because
/// the TUI takes it at construction; Python fetches it lazily inside
/// `/models`. A dead Ollama is a warning, not fatal: `/status` and friends
/// stay usable, and `/models` shows an empty list. (Deliberate, documented
/// in the porting notes.)
fn tui(
    runtime: wiring::ProductionRuntime,
    workspace_root: String,
    trusted: bool,
) -> Result<i32, String> {
    let mut facade = CliRuntime::new(runtime);
    let models = match facade.list_models() {
        Ok(models) => models,
        Err(error) => {
            eprintln!("Flow: cannot list Ollama models ({error}); /models will be empty");
            Vec::new()
        }
    };
    let mut app = FlowApp::new(facade, models, workspace_root, trusted);
    // The ratatui frontend needs a real terminal for its banner; with a
    // piped stdout the same line loop runs headless instead. Either way
    // Ctrl-D (EOF) exits 0, like Python's `except EOFError: break`.
    let outcome = if io::stdout().is_terminal() {
        phlow_tui::run(&mut app)
    } else {
        phlow_tui::run_line_mode(&mut app)
    };
    match outcome {
        Ok(()) => Ok(0),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_python_package() {
        let python_version = std::process::Command::new("python3")
            .args([
                "-c",
                "import sys; sys.path.insert(0, '/home/hatch/workspace/repos/phlow'); \
                 import flow; print(flow.__version__)",
            ])
            .output()
            .expect("python3 must run for the version cross-check");
        let python_version = String::from_utf8_lossy(&python_version.stdout);
        assert_eq!(
            format!("Flow {}", python_version.trim()),
            format!("Flow {VERSION}")
        );
        assert_eq!(VERSION, "0.2.0");
    }

    #[test]
    fn editor_timeout_bounds_render_like_python_g_format() {
        // Python: f"{0.1:g}" -> "0.1", f"{660.0:g}" -> "660".
        assert_eq!(TIMEOUT_MIN.as_secs_f64().to_string(), "0.1");
        assert_eq!(TIMEOUT_MAX.as_secs_f64().to_string(), "660");
    }

    #[test]
    fn report_exit_code_mapping() {
        assert_eq!(report_exit_code(&serde_json::json!({"status": "ok"})), 0);
        assert_eq!(
            report_exit_code(&serde_json::json!({"status": "failed"})),
            1
        );
        assert_eq!(report_exit_code(&serde_json::json!({"status": "error"})), 1);
        assert_eq!(report_exit_code(&serde_json::json!({})), 1);
        assert_eq!(report_exit_code(&serde_json::json!({"status": 0})), 1);
        assert_eq!(report_exit_code(&serde_json::json!({"status": "OK"})), 1);
    }

    #[test]
    fn interrupted_exit_code_is_130() {
        // The SIGTERM contract, asserted at the binary level too.
        assert_eq!(super::signal::EXIT_INTERRUPTED, 130);
    }
}
