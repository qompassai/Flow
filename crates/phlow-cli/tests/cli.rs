//! Subprocess contract tests for the `phlow`/`flow` binaries.
//!
//! These spawn the real binaries Cargo just built, so exit codes, stdout
//! bytes, stderr bytes, and signal behavior are the actual process
//! contract — not a re-description of the parser. Every assertion is
//! checked against the Python CLI (`flow/main.py`) where the contract
//! originates.

use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::Value;

fn phlow() -> Command {
    Command::new(env!("CARGO_BIN_EXE_phlow"))
}

fn flow_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flow"))
}

fn packaging_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("packaging")
}

fn stdout_lines(output: &std::process::Output) -> Vec<String> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
}

// --- version / usage ------------------------------------------------------

#[test]
fn version_prints_flow_0_2_0_and_exits_0() {
    let output = phlow().arg("--version").output().expect("spawn phlow");
    assert!(output.status.success());
    assert_eq!(output.stdout, b"Flow 0.2.0\n");
}

#[test]
fn version_matches_python_byte_for_byte() {
    let rust = phlow().arg("--version").output().expect("spawn phlow");
    // `flow/__init__.py` imports without the optional httpx dependency, so
    // the package version is readable even where the full CLI is not.
    let python = Command::new("python3")
        .args([
            "-c",
            "import sys; sys.path.insert(0, '/home/hatch/workspace/repos/phlow'); \
             import flow; print(f'Flow {flow.__version__}')",
        ])
        .output()
        .expect("spawn python3");
    assert_eq!(rust.stdout, python.stdout);
    assert_eq!(rust.status.code(), Some(0));
    assert_eq!(python.status.code(), Some(0));
}

#[test]
fn flow_binary_behaves_like_phlow() {
    // `--version` must be byte-identical. `--help` differs only in the
    // argv[0]-derived bin name on the Usage line (clap renders `Usage:
    // flow ...` for the compat binary), so normalize that before comparing.
    for args in [vec!["--version"], vec!["--help"], vec!["status", "--help"]] {
        let a = phlow().args(&args).output().expect("spawn phlow");
        let b = flow_bin().args(&args).output().expect("spawn flow");
        assert_eq!(a.status.code(), b.status.code(), "args: {args:?}");
        let normalize = |output: &[u8], name: &str| {
            String::from_utf8_lossy(output)
                .replace(&format!("Usage: {name}"), "Usage: BIN")
                .into_bytes()
        };
        assert_eq!(
            normalize(&a.stdout, "phlow"),
            normalize(&b.stdout, "flow"),
            "args: {args:?}"
        );
    }
}

#[test]
fn run_does_not_accept_version_like_python() {
    // Python: `flow run --version` -> argparse error, exit 2.
    for mut binary in [phlow(), flow_bin()] {
        let output = binary.arg("run").arg("--version").output().expect("spawn");
        assert_eq!(output.status.code(), Some(2));
    }
}

#[test]
fn unknown_flags_exit_2() {
    for args in [
        vec!["--bogus"],
        vec!["status", "--bogus"],
        vec!["serve", "--bogus"],
    ] {
        let output = phlow().args(&args).output().expect("spawn phlow");
        assert_eq!(output.status.code(), Some(2), "args: {args:?}");
        assert!(!output.stderr.is_empty(), "usage error must explain itself");
    }
}

#[test]
fn run_without_a_task_exits_2() {
    let output = phlow().arg("run").output().expect("spawn phlow");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn flags_work_before_and_after_the_subcommand() {
    for args in [
        vec!["--trusted", "status"],
        vec!["status", "--trusted"],
        vec!["-c", "nonexistent.toml", "status"],
    ] {
        let parsed = phlow().args(&args).output().expect("spawn phlow");
        // -c with a missing file exits 2 (config error); the others run.
        let expected = if args.contains(&"-c") { Some(2) } else { None };
        if let Some(code) = expected {
            assert_eq!(parsed.status.code(), Some(code), "args: {args:?}");
        } else {
            assert!(parsed.status.success(), "args: {args:?}");
        }
    }
}

// --- editor timeout --------------------------------------------------------

#[test]
fn editor_timeout_out_of_range_exits_2_with_python_message() {
    for value in ["0.05", "0.09", "660.01", "1000", "-1", "nan", "inf"] {
        let output = phlow()
            .args(["--editor-timeout", value, "status"])
            .output()
            .expect("spawn phlow");
        assert_eq!(output.status.code(), Some(2), "value: {value}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            stderr.as_ref(),
            "Flow: --editor-timeout must be between 0.1 and 660 seconds\n",
            "value: {value}"
        );
    }
}

#[test]
fn editor_timeout_boundaries_are_accepted() {
    for value in ["0.1", "120", "660", "660.0"] {
        let output = phlow()
            .args(["--editor-timeout", value, "status"])
            .output()
            .expect("spawn phlow");
        assert!(output.status.success(), "value: {value}");
    }
}

#[test]
fn editor_timeout_non_numeric_is_a_usage_error() {
    let output = phlow()
        .args(["--editor-timeout", "abc", "status"])
        .output()
        .expect("spawn phlow");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--editor-timeout"), "stderr: {stderr}");
}

#[test]
fn bare_double_dash_is_a_usage_error_like_python() {
    // Python: `python3 -m flow --` -> argparse `unrecognized arguments:
    // --`, exit 2. Without the special case clap would launch the TUI.
    let output = phlow().arg("--").output().expect("spawn phlow");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--"), "stderr: {stderr}");
}

#[test]
fn double_dash_before_subcommand_still_works() {
    // `phlow -- status` is a usage error on both sides (argparse rejects
    // the bare `--` position the same way); the point is the separator
    // itself does not launch the TUI.
    let output = phlow()
        .args(["--", "status"])
        .output()
        .expect("spawn phlow");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn editor_timeout_accepts_python_float_forms() {
    // " 5 ", "1_0", " 1_0 " parse like CPython's float(); the values are
    // finite and in range, so `status` runs and exits 0. ("INF",
    // "infinity", and "nan" also parse, but fail the range check — see the
    // next test.)
    for value in [" 5 ", "1_0", " 1_0 "] {
        let output = phlow()
            .args(["--editor-timeout", value, "status"])
            .output()
            .expect("spawn phlow");
        assert!(output.status.success(), "value: {value:?}");
    }
}

#[test]
fn editor_timeout_rejects_bad_underscores() {
    // "1__0", "_1", "1_", "1d5" are not valid CPython floats: usage
    // error, exit 2.
    for value in ["1__0", "_1", "1_", "1d5"] {
        let output = phlow()
            .args(["--editor-timeout", value, "status"])
            .output()
            .expect("spawn phlow");
        assert_eq!(output.status.code(), Some(2), "value: {value:?}");
    }
}

#[test]
fn editor_timeout_nan_and_inf_hit_the_range_check() {
    // "nan"/"INF" parse but are not finite: the existing range check
    // reports "--editor-timeout must be between 0.1 and 660 seconds".
    for value in ["nan", "INF", "infinity"] {
        let output = phlow()
            .args(["--editor-timeout", value, "status"])
            .output()
            .expect("spawn phlow");
        assert_eq!(output.status.code(), Some(2), "value: {value:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--editor-timeout must be between 0.1 and 660 seconds"),
            "stderr: {stderr}"
        );
    }
}

// --- reports ----------------------------------------------------------------

fn report_of(args: &[&str]) -> (std::process::Output, Value) {
    let output = phlow().args(args).output().expect("spawn phlow");
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 1, "exactly one JSON line on stdout");
    let value: Value = serde_json::from_str(&lines[0]).expect("stdout is JSON");
    (output, value)
}

#[test]
fn status_prints_one_ascii_json_line() {
    let (output, report) = report_of(&["status"]);
    assert!(report.get("status").is_some(), "report: {report}");
    assert!(output.stdout.is_ascii(), "stdout must be ASCII-escaped");
    assert!(output.stderr.is_empty(), "status must not log to stderr");
    // Exit code follows the status field, like Python.
    let expected = if report["status"] == "ok" { 0 } else { 1 };
    assert_eq!(output.status.code(), Some(expected));
}

#[test]
fn empty_workspace_flag_resolves_to_cwd_like_python() {
    // Python's argparse accepts `--workspace ""` and resolves it to the
    // caller's cwd (`base / Path("")`, flow/config.py). The resolved
    // workspace is observable in the status report's `workspace` field.
    let cwd = std::env::temp_dir().canonicalize().expect("temp dir");
    let output = phlow()
        .args(["--workspace", "", "status"])
        .current_dir(&cwd)
        .output()
        .expect("spawn phlow");
    assert_ne!(
        output.status.code(),
        Some(2),
        "empty workspace is not a usage error"
    );
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 1, "exactly one JSON line on stdout");
    let report: Value = serde_json::from_str(&lines[0]).expect("stdout is JSON");
    assert_eq!(
        report["workspace"].as_str(),
        Some(cwd.to_string_lossy().as_ref()),
        "report: {report}"
    );
}

#[test]
fn check_with_no_configured_checks_reports_cleanly() {
    let (output, report) = report_of(&["check"]);
    assert!(report.get("status").is_some());
    let expected = if report["status"] == "ok" { 0 } else { 1 };
    assert_eq!(output.status.code(), Some(expected));
    assert!(output.stdout.is_ascii());
}

#[test]
fn check_name_flag_reaches_the_runtime() {
    // No checks are configured, so any name resolves to "no checks match".
    let (output, report) = report_of(&["check", "--name", "nope"]);
    assert!(report.get("status").is_some());
    let expected = if report["status"] == "ok" { 0 } else { 1 };
    assert_eq!(output.status.code(), Some(expected));
}

#[test]
fn run_without_ollama_fails_closed_with_exit_1() {
    // No Ollama here: the backend is refused, the runtime must encode the
    // failure in the report (status != ok) and exit 1 — never panic, never
    // hang, never print a bare traceback.
    let (output, report) = report_of(&["run", "hello"]);
    assert_ne!(report["status"], "ok", "report: {report}");
    assert_eq!(output.status.code(), Some(1));
    assert!(report.get("error").is_some(), "failure must be explained");
    assert!(output.stdout.is_ascii());
}

// --- signals -----------------------------------------------------------------

#[test]
fn sigterm_during_serve_exits_130_with_python_message() {
    let child = phlow()
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn phlow serve");
    // Let the runtime finish constructing (Reqwest is lazy; the MCP loop
    // blocks on stdin immediately).
    std::thread::sleep(Duration::from_millis(500));
    assert!(child.id() > 0);
    let kill = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .output()
        .expect("kill -TERM");
    assert!(kill.status.success());
    let output = child.wait_with_output().expect("reap phlow serve");
    assert_eq!(output.status.code(), Some(130));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Flow interrupted; changes already written are not rolled back."),
        "stderr: {stderr}"
    );
}

// --- MCP serve wiring ----------------------------------------------------------

fn serve_session(frames: &[&str]) -> std::process::Output {
    let mut child = phlow()
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn phlow serve");
    {
        let stdin = child.stdin.as_mut().expect("piped stdin");
        for frame in frames {
            stdin.write_all(frame.as_bytes()).expect("write frame");
            stdin.write_all(b"\n").expect("write newline");
        }
        // EOF: the server must terminate the session and close the runtime.
    }
    child.wait_with_output().expect("reap phlow serve")
}

/// The MCP initialize handshake: `initialize` (responded) then the
/// `notifications/initialized` notification (unanswered) flips the server
/// to ready, mirroring the Python server's lifecycle gate. `initialize`
/// requires a nonempty `protocolVersion`, like the real MCP handshake.
fn handshake() -> [&'static str; 2] {
    [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    ]
}

#[test]
fn serve_answers_initialize_and_closes_on_eof() {
    let output = serve_session(&handshake()[..1]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 1, "one response frame, got: {lines:?}");
    let response: Value = serde_json::from_str(&lines[0]).expect("response is JSON");
    assert_eq!(response["id"], 1);
    assert!(response.get("result").is_some(), "response: {response}");
}

#[test]
fn serve_tools_list_exposes_flow_run() {
    let output = serve_session(&[
        handshake()[0],
        handshake()[1],
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
    ]);
    assert_eq!(output.status.code(), Some(0));
    let lines = stdout_lines(&output);
    // initialize answered; the notification is silent; tools/list answered.
    assert_eq!(lines.len(), 2, "frames: {lines:?}");
    let response: Value = serde_json::from_str(&lines[1]).expect("response is JSON");
    let tools = response["result"]["tools"].as_array().expect("tools array");
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect();
    assert!(names.contains(&"flow_run"), "tools: {names:?}");
    assert!(names.contains(&"flow_status"), "tools: {names:?}");
    assert!(names.contains(&"flow_check"), "tools: {names:?}");
}

#[test]
fn serve_rejects_calls_before_the_handshake() {
    // The lifecycle gate, mirroring Python: tools are unavailable until
    // initialize + notifications/initialized.
    let output = serve_session(&[r#"{"jsonrpc":"2.0","id":9,"method":"tools/list","params":{}}"#]);
    assert_eq!(output.status.code(), Some(0));
    let lines = stdout_lines(&output);
    let response: Value = serde_json::from_str(&lines[0]).expect("response is JSON");
    assert!(response.get("error").is_some(), "response: {response}");
}

#[test]
fn serve_tools_call_status_needs_no_model() {
    let output = serve_session(&[
        handshake()[0],
        handshake()[1],
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"flow_status","arguments":{}}}"#,
    ]);
    assert_eq!(output.status.code(), Some(0));
    let lines = stdout_lines(&output);
    let response: Value = serde_json::from_str(&lines[1]).expect("response is JSON");
    assert!(response.get("result").is_some(), "response: {response}");
}

#[test]
fn serve_rejects_malformed_frames_with_parse_error() {
    let output = serve_session(&["this is not json"]);
    assert_eq!(output.status.code(), Some(0));
    let lines = stdout_lines(&output);
    assert_eq!(lines.len(), 1);
    let response: Value = serde_json::from_str(&lines[0]).expect("response is JSON");
    assert!(response.get("error").is_some(), "response: {response}");
    assert_eq!(response["error"]["code"], -32700);
}

#[test]
fn serve_stdout_carries_protocol_json_only() {
    let requests = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
        "garbage",
    ];
    let output = serve_session(&requests);
    for (index, line) in stdout_lines(&output).iter().enumerate() {
        serde_json::from_str::<Value>(line)
            .unwrap_or_else(|_| panic!("stdout line {index} is not JSON: {line}"));
    }
}

// --- TUI default ---------------------------------------------------------------

#[test]
fn bare_invocation_enters_tui_and_exits_on_eof() {
    // No subcommand -> TUI, like Python. With stdin closed the line-mode
    // frontend exits immediately instead of blocking forever.
    let child = phlow()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn phlow");
    let output = child.wait_with_output().expect("reap phlow");
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// --- systemd / packaging ---------------------------------------------------------

fn read_unit(name: &str) -> String {
    std::fs::read_to_string(packaging_dir().join(name))
        .unwrap_or_else(|_| panic!("packaging/{name} must exist"))
}

#[test]
fn systemd_user_unit_serves_phlow_over_stdio() {
    // Accept=yes sockets spawn template instances: phlow@.service.
    let unit = read_unit("phlow@.service");
    assert!(unit.contains("ExecStart="), "must define ExecStart");
    assert!(unit.contains("phlow serve"), "must run `phlow serve`");
}

#[test]
fn systemd_units_carry_hardening_and_restart_policy() {
    for name in ["phlow@.service", "phlow-system@.service"] {
        let unit = read_unit(name);
        for directive in [
            "NoNewPrivileges=yes",
            "ProtectSystem=strict",
            "PrivateTmp=yes",
            "Restart=on-failure",
        ] {
            assert!(unit.contains(directive), "{name} must set {directive}");
        }
        assert!(
            !unit.contains("flow serve") || unit.contains("phlow serve"),
            "{name} must not invoke the legacy Python entrypoint"
        );
    }
}
