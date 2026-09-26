//! The named-check runner: exact argv, bounded output, deadline kills.
//!
//! [`CheckRunner`] executes one operator-approved check at a time. The
//! child always starts in its own process group (`process_group(0)`, a safe
//! std wrapper — no `unsafe` in this crate), with the workspace as its cwd,
//! stdin closed, and stdout/stderr captured through bounded tail buffers.
//! On timeout the whole process group is killed and reaped, so a check can
//! never outlive its report as an orphan.
//!
//! Sync only: the async Tokio executor arrives in Phase 4. Deadlines are
//! enforced by polling, which keeps every wait bounded and reaping exact.

use std::collections::{BTreeMap, VecDeque};
use std::io::Read;
use std::path::Path;
use std::process::{Child, Stdio};
use std::time::{Duration, Instant};

use phlow_config::{CHECK_ARGV_MAX, CheckConfig, FlowConfig};
use phlow_workspace::Workspace;

/// Maximum stdout/stderr retained per check, in bytes. Only the tail is
/// kept; anything more sets the `*_truncated` flag. Bounds check output the
/// way [`phlow_workspace::FILE_BYTES_MAX`] bounds file content.
pub const OUTPUT_BYTES_MAX: u64 = 16_000;
/// How often the runner polls a running check for completion, in
/// milliseconds. Small enough that timeout overshoot is negligible next to
/// the minimum 1 ms check timeout.
const CHILD_POLL_INTERVAL_MS: u64 = 1;
/// Chunk size for draining check output pipes, in bytes.
const READ_CHUNK_BYTES: usize = 8192;
/// `source` field on every check report. Kept as the compatibility
/// identifier the MCP contract uses.
const REPORT_SOURCE: &str = "flow.check";
/// Reason text when verification does not hold.
const UNVERIFIED_REASON: &str =
    "Every required check must run and pass; at least one required check must be configured";

/// Outcome status of one check run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// Exit status 0.
    Ok,
    /// Nonzero exit status.
    Failed,
    /// Exceeded its timeout; the process group was killed.
    Timeout,
    /// No such check configured, or the executable is missing.
    Unavailable,
    /// The check errored without running to a status.
    Error,
    /// The workspace is not trusted; the check did not run.
    Unverified,
    /// The workspace root was replaced; the result is untrustworthy.
    Stale,
}

impl CheckStatus {
    /// Wire string for this status.
    pub fn as_str(self) -> &'static str {
        match self {
            CheckStatus::Ok => "ok",
            CheckStatus::Failed => "failed",
            CheckStatus::Timeout => "timeout",
            CheckStatus::Unavailable => "unavailable",
            CheckStatus::Error => "error",
            CheckStatus::Unverified => "unverified",
            CheckStatus::Stale => "stale",
        }
    }
}

/// Report for one check execution.
///
/// Typed equivalent of the Python `run()` dict. The `timeout_ms` field is
/// `timeout` on the wire; units are explicit here per Tiger Style.
#[derive(Debug, Clone)]
pub struct CheckReport {
    /// Configured check name.
    pub name: String,
    /// Outcome status.
    pub status: CheckStatus,
    /// Always `"flow.check"`.
    pub source: &'static str,
    /// Exact argv that ran (or would have run).
    pub cmd: Vec<String>,
    /// Deadline in milliseconds (wire name: `timeout`).
    pub timeout_ms: u64,
    /// Whether verification requires this check to pass.
    pub required: bool,
    /// Descriptive filetypes; never silently skips required checks.
    pub filetypes: Vec<String>,
    /// What kind of verification this check performs.
    pub kind: phlow_config::CheckKind,
    /// Workspace root the check ran in.
    pub workspace: String,
    /// Workspace revision when the check ran.
    pub revision: u64,
    /// Process exit code. Negative on POSIX signal death, matching
    /// Python's `Popen.returncode`; `None` when no process ran.
    pub returncode: Option<i32>,
    /// Captured stdout tail, UTF-8 with invalid sequences replaced.
    pub stdout: String,
    /// True when stdout exceeded [`OUTPUT_BYTES_MAX`].
    pub stdout_truncated: bool,
    /// Captured stderr tail, UTF-8 with invalid sequences replaced.
    pub stderr: String,
    /// True when stderr exceeded [`OUTPUT_BYTES_MAX`].
    pub stderr_truncated: bool,
    /// Wall-clock time of the execution in milliseconds.
    pub duration_ms: u64,
    /// Human-readable failure detail; `None` when the check ran cleanly.
    pub error: Option<String>,
}

/// Aggregate verification report for [`CheckRunner::run_all`].
#[derive(Debug, Clone)]
pub struct RunAllReport {
    /// `"ok"`, `"failed"`, or `"unverified"`.
    pub status: &'static str,
    /// True only when at least one required check ran and all required
    /// checks passed.
    pub verified: bool,
    /// One report per check that ran.
    pub checks: Vec<CheckReport>,
    /// Empty when verified; otherwise the reason verification failed.
    pub reason: &'static str,
}

/// Executes the operator's named checks inside a workspace.
///
/// Holds a shared borrow of the [`Workspace`]: trust and freshness are
/// re-checked on every `run`, so a workspace that lost trust or was
/// replaced between runs cannot produce a stale "ok".
pub struct CheckRunner<'w> {
    workspace: &'w Workspace,
    checks: BTreeMap<String, CheckConfig>,
}

impl<'w> CheckRunner<'w> {
    /// Build a runner over an explicit check map.
    ///
    /// The map is expected to come from validated configuration
    /// ([`FlowConfig::checks`]); the length bound is asserted so a future
    /// caller cannot silently widen it. Prefer [`CheckRunner::from_config`].
    pub fn new(workspace: &'w Workspace, checks: BTreeMap<String, CheckConfig>) -> CheckRunner<'w> {
        // The config schema already caps the map at CHECKS_MAX; assert the
        // shape here so a future caller cannot silently widen it.
        assert!(
            checks.len() <= phlow_config::CHECKS_MAX,
            "check map exceeds the validated limit"
        );
        CheckRunner { workspace, checks }
    }

    /// Build a runner from a loaded operator configuration.
    pub fn from_config(workspace: &'w Workspace, config: &FlowConfig) -> CheckRunner<'w> {
        CheckRunner::new(workspace, config.checks().clone())
    }

    /// Run one named check to completion or timeout.
    ///
    /// Never fails with `Err`: every failure mode (unknown name, untrusted
    /// workspace, stale root, spawn failure, timeout) is a report status.
    pub fn run(&self, name: &str) -> CheckReport {
        let Some(check) = self.checks.get(name) else {
            return CheckReport::unknown(name);
        };
        let mut report = CheckReport::for_check(name, check, self.workspace);
        if !self.workspace.trusted() {
            report.set_status(CheckStatus::Unverified, "Named checks require --trusted");
            return report;
        }
        if let Err(err) = self.workspace.assert_current() {
            report.set_status(CheckStatus::Stale, &err.to_string());
            return report;
        }
        let started = Instant::now();
        let outcome = execute(self.workspace.root(), check.cmd(), check.timeout_ms());
        report.duration_ms = duration_ms_since(started);
        report.apply_outcome(outcome);
        if let Err(err) = self.workspace.assert_current() {
            // Freshness after the run overrides even a passing result: the
            // root may have been replaced mid-check.
            report.set_status(CheckStatus::Stale, &err.to_string());
        }
        report
    }

    /// Run one named check, or every configured check when `name` is `None`.
    ///
    /// `verified` is true only when at least one required check exists and
    /// every required check passed. Missing checks never verify.
    ///
    /// For a named run the named check alone determines the verdict, even
    /// when it is optional — Python parity (`flow/checks.py` sets
    /// `required = checks` in this case).
    pub fn run_all(&self, name: Option<&str>) -> RunAllReport {
        let checks: Vec<CheckReport> = match name {
            Some(one) => vec![self.run(one)],
            None => self.checks.keys().map(|key| self.run(key)).collect(),
        };
        // Python parity — for a named run, the named check alone determines
        // the verdict (flow/checks.py sets required = checks in this case),
        // even if the check is optional.
        let required: Vec<&CheckReport> = match name {
            Some(_) => checks.iter().collect(),
            None => checks.iter().filter(|report| report.required).collect(),
        };
        let verified = !required.is_empty()
            && required
                .iter()
                .all(|report| report.status == CheckStatus::Ok);
        let status = if verified {
            "ok"
        } else if required
            .iter()
            .any(|report| report.status == CheckStatus::Failed)
        {
            "failed"
        } else {
            "unverified"
        };
        RunAllReport {
            status,
            verified,
            checks,
            reason: if verified { "" } else { UNVERIFIED_REASON },
        }
    }
}

/// Error text for a check name with no configured check, byte-identical to
/// Python's `"No such configured check"` in `flow/checks.py`.
const UNKNOWN_CHECK_ERROR: &str = "No such configured check";

impl CheckReport {
    fn for_check(name: &str, check: &CheckConfig, workspace: &Workspace) -> CheckReport {
        CheckReport {
            name: name.to_string(),
            status: CheckStatus::Error,
            source: REPORT_SOURCE,
            cmd: check.cmd().to_vec(),
            timeout_ms: check.timeout_ms(),
            required: check.required(),
            filetypes: check.filetypes().to_vec(),
            kind: check.kind(),
            workspace: workspace.root().to_string_lossy().into_owned(),
            revision: workspace.revision(),
            returncode: None,
            stdout: String::new(),
            stdout_truncated: false,
            stderr: String::new(),
            stderr_truncated: false,
            duration_ms: 0,
            error: None,
        }
    }

    fn unknown(name: &str) -> CheckReport {
        CheckReport {
            name: name.to_string(),
            status: CheckStatus::Unavailable,
            source: REPORT_SOURCE,
            cmd: Vec::new(),
            timeout_ms: 0,
            required: false,
            filetypes: Vec::new(),
            kind: phlow_config::CheckKind::Check,
            workspace: String::new(),
            revision: 0,
            returncode: None,
            stdout: String::new(),
            stdout_truncated: false,
            stderr: String::new(),
            stderr_truncated: false,
            duration_ms: 0,
            error: Some(UNKNOWN_CHECK_ERROR.to_string()),
        }
    }

    /// True for the "no such configured check" placeholder. Python renders
    /// exactly this case as a compact 4-key dict (`name`, `status`,
    /// `error`, `source`), not the full `run()` shape — see
    /// `phlow_runtime::report::check_report_to_value`.
    pub fn is_unknown(&self) -> bool {
        self.error.as_deref() == Some(UNKNOWN_CHECK_ERROR)
    }

    fn set_status(&mut self, status: CheckStatus, error: &str) {
        self.status = status;
        self.error = Some(error.to_string());
    }

    fn apply_outcome(&mut self, outcome: ExecutionOutcome) {
        self.status = outcome.status;
        self.returncode = outcome.returncode;
        self.error = outcome.error;
        self.stdout = outcome.stdout.text();
        self.stdout_truncated = outcome.stdout.truncated();
        self.stderr = outcome.stderr.text();
        self.stderr_truncated = outcome.stderr.truncated();
    }
}

/// The tail of one captured output stream: at most [`OUTPUT_BYTES_MAX`]
/// trailing bytes plus the total seen, so truncation is observable.
struct Tail {
    bytes: Vec<u8>,
    total_bytes: u64,
}

impl Tail {
    fn empty() -> Tail {
        Tail {
            bytes: Vec::new(),
            total_bytes: 0,
        }
    }

    fn truncated(&self) -> bool {
        self.total_bytes > OUTPUT_BYTES_MAX
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

struct ExecutionOutcome {
    status: CheckStatus,
    returncode: Option<i32>,
    error: Option<String>,
    stdout: Tail,
    stderr: Tail,
}

impl ExecutionOutcome {
    fn failed(status: CheckStatus, error: impl Into<String>) -> ExecutionOutcome {
        ExecutionOutcome {
            status,
            returncode: None,
            error: Some(error.into()),
            stdout: Tail::empty(),
            stderr: Tail::empty(),
        }
    }
}

/// What `wait_with_deadline` observed.
enum WaitOutcome {
    /// The child exited on its own before the deadline.
    Exited(std::process::ExitStatus),
    /// The deadline passed; the process group was killed and reaped.
    TimedOut,
    /// Waiting itself failed.
    WaitFailed(String),
}

/// Run one check argv to completion or its deadline.
///
/// The child starts in its own process group so the timeout kill takes the
/// whole process tree. Stdout/stderr drain on reader threads into bounded
/// tails while the main thread polls the deadline; after the wait both
/// readers are joined, so no output and no thread outlives the report.
fn execute(root: &Path, argv: &[String], timeout_ms: u64) -> ExecutionOutcome {
    let (program, args) = match argv.split_first() {
        Some(pair) => pair,
        None => return ExecutionOutcome::failed(CheckStatus::Error, "check argv is empty"),
    };
    // The config schema already validates argv (nonempty, NUL-free,
    // CHECK_ARGV_MAX elements). Re-check the shape here so a future caller
    // cannot widen it into a panic inside Command.
    assert!(
        argv.len() <= CHECK_ARGV_MAX,
        "check argv exceeds the validated element bound"
    );
    if argv.iter().any(|arg| arg.contains('\0')) {
        return ExecutionOutcome::failed(CheckStatus::Error, "check argv contains NUL");
    }
    let mut child = match spawn_pinned(root, program, args) {
        Ok(child) => child,
        Err(outcome) => return outcome,
    };
    let (mut stdout_pipe, mut stderr_pipe) = match (child.stdout.take(), child.stderr.take()) {
        (Some(stdout), Some(stderr)) => (stdout, stderr),
        _ => {
            // We requested piped output; its absence is an internal failure.
            // Stop what we started and report it instead of hanging.
            kill_process_group(&mut child);
            let _ = child.wait();
            return ExecutionOutcome::failed(CheckStatus::Error, "failed to capture check output");
        }
    };
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let (tail_out, tail_err, wait) = std::thread::scope(|scope| {
        let reader_out = scope.spawn(|| read_tail(&mut stdout_pipe));
        let reader_err = scope.spawn(|| read_tail(&mut stderr_pipe));
        let wait = wait_with_deadline(&mut child, deadline);
        // Readers have no panic paths (I/O errors end the drain); an
        // unreachable join failure yields an empty tail, never a hang.
        let tail_out = reader_out.join().unwrap_or_else(|_| Tail::empty());
        let tail_err = reader_err.join().unwrap_or_else(|_| Tail::empty());
        (tail_out, tail_err, wait)
    });
    let (status, returncode, error) = classify_wait(wait, timeout_ms);
    ExecutionOutcome {
        status,
        returncode,
        error,
        stdout: tail_out,
        stderr: tail_err,
    }
}

/// Map what `wait_with_deadline` observed onto a report status triple.
fn classify_wait(wait: WaitOutcome, timeout_ms: u64) -> (CheckStatus, Option<i32>, Option<String>) {
    match wait {
        WaitOutcome::Exited(exit) => {
            let code = exit_code(&exit);
            if exit.success() {
                (CheckStatus::Ok, code, None)
            } else {
                (CheckStatus::Failed, code, None)
            }
        }
        WaitOutcome::TimedOut => (
            CheckStatus::Timeout,
            None,
            Some(format!("Check exceeded {timeout_ms}ms")),
        ),
        WaitOutcome::WaitFailed(message) => (CheckStatus::Error, None, Some(message)),
    }
}

/// Spawn the check process: exact argv, no shell.
///
/// The workspace pins the child's cwd; stdin is closed so a check can
/// never read from our terminal; stdout/stderr are piped for the bounded
/// drain. On Unix the child leads its own process group, so the timeout
/// kill takes the whole tree, never our group.
fn spawn_pinned(root: &Path, program: &str, args: &[String]) -> Result<Child, ExecutionOutcome> {
    let mut command = std::process::Command::new(program);
    command.args(args);
    command.current_dir(root);
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Safe std wrapper around setpgid(0, 0) in the child before exec;
        // no unsafe in this crate.
        command.process_group(0);
    }
    match command.spawn() {
        Ok(child) => Ok(child),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Err(ExecutionOutcome::failed(
            CheckStatus::Unavailable,
            err.to_string(),
        )),
        Err(err) => Err(ExecutionOutcome::failed(
            CheckStatus::Error,
            err.to_string(),
        )),
    }
}

/// Poll the child until it exits or the deadline passes.
///
/// On timeout the whole process group is signal-killed and the child is
/// reaped, so neither the child nor its grandchildren survive the report.
fn wait_with_deadline(child: &mut Child, deadline: Instant) -> WaitOutcome {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return WaitOutcome::Exited(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    kill_process_group(child);
                    return match child.wait() {
                        Ok(_) => WaitOutcome::TimedOut,
                        Err(err) => WaitOutcome::WaitFailed(err.to_string()),
                    };
                }
                std::thread::sleep(Duration::from_millis(CHILD_POLL_INTERVAL_MS));
            }
            Err(err) => return WaitOutcome::WaitFailed(err.to_string()),
        }
    }
}

/// Kill the check's process group (best-effort: an already-reaped group
/// reports ESRCH, which is the outcome we wanted anyway).
#[cfg(unix)]
fn kill_process_group(child: &mut Child) {
    use rustix::process::{Pid, Signal, kill_process_group};
    // The child leads its own process group (process_group(0) at spawn),
    // so this kills the check's tree, never ours.
    let pid = i32::try_from(child.id()).ok().and_then(Pid::from_raw);
    if let Some(pid) = pid {
        let _ = kill_process_group(pid, Signal::KILL);
    }
}

/// Non-POSIX fallback: kill just the child; no process groups exist.
#[cfg(not(unix))]
fn kill_process_group(child: &mut Child) {
    let _ = child.kill();
}

/// Exit code matching Python's `Popen.returncode`: the code on normal exit,
/// the negated signal number on POSIX signal death.
#[cfg(unix)]
fn exit_code(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .or_else(|| status.signal().map(|signal| -signal))
}

#[cfg(not(unix))]
fn exit_code(status: &std::process::ExitStatus) -> Option<i32> {
    status.code()
}

/// Drain a pipe into a bounded tail: at most [`OUTPUT_BYTES_MAX`] trailing
/// bytes are kept, but the total seen is counted so truncation is visible.
fn read_tail<R: Read>(pipe: &mut R) -> Tail {
    let mut kept: VecDeque<u8> = VecDeque::new();
    let mut total_bytes: u64 = 0;
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    loop {
        match pipe.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                total_bytes += read as u64;
                for byte in &chunk[..read] {
                    if kept.len() as u64 >= OUTPUT_BYTES_MAX {
                        kept.pop_front();
                    }
                    kept.push_back(*byte);
                }
            }
            Err(_) => break,
        }
    }
    Tail {
        bytes: kept.into_iter().collect(),
        total_bytes,
    }
}

/// Milliseconds since `started`, saturating instead of wrapping on
/// absurd clocks.
fn duration_ms_since(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlow_config::{LoadOptions, load_config};
    use phlow_workspace::Workspace;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phlow-checks-test-{prefix}-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create temp dir");
        dir
    }

    /// Load a config with the given `[checks.*]` TOML body into an existing
    /// directory, and open a workspace over it.
    fn fixture_in(
        dir: &Path,
        checks_toml: &str,
        trusted: bool,
    ) -> (Workspace, BTreeMap<String, CheckConfig>) {
        let config_path = dir.join("flow-test.toml");
        std::fs::write(&config_path, checks_toml).expect("test setup: write config");
        let config = load_config(&LoadOptions {
            config_path: Some(config_path),
            workspace: Some(dir.to_path_buf()),
            trusted,
            model: None,
        })
        .expect("test setup: load config");
        let workspace = Workspace::open(dir, trusted, &[]).expect("test setup: open workspace");
        (workspace, config.checks().clone())
    }

    /// Load a config with the given `[checks.*]` TOML body and open a
    /// workspace over the same temp dir.
    fn fixture(
        checks_toml: &str,
        trusted: bool,
    ) -> (PathBuf, Workspace, BTreeMap<String, CheckConfig>) {
        let dir = temp_dir("cfg");
        let (workspace, checks) = fixture_in(&dir, checks_toml, trusted);
        (dir, workspace, checks)
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unknown_check_is_unavailable() {
        let (dir, workspace, checks) = fixture("", true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("nope");
        assert_eq!(report.status, CheckStatus::Unavailable);
        assert_eq!(report.error.as_deref(), Some("No such configured check"));
        assert_eq!(report.source, "flow.check");
        assert_eq!(report.name, "nope");
        cleanup(&dir);
    }

    #[test]
    fn named_run_of_optional_passing_check_verifies() {
        // Python parity: for a named run the named check alone determines the
        // verdict (flow/checks.py sets required = checks in this case), even
        // when the check is optional.
        let (dir, workspace, checks) = fixture(
            "[checks.opt]\ncmd = [\"/bin/true\"]\nrequired = false\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run_all(Some("opt"));
        assert_eq!(report.status, "ok");
        assert!(report.verified);
        cleanup(&dir);
    }

    #[test]
    fn named_run_of_optional_failing_check_fails() {
        // Same Python-parity contract in the failure direction: the optional
        // check's failure determines the verdict, not "unverified".
        let (dir, workspace, checks) = fixture(
            "[checks.opt]\ncmd = [\"/bin/false\"]\nrequired = false\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run_all(Some("opt"));
        assert_eq!(report.status, "failed");
        assert!(!report.verified);
        cleanup(&dir);
    }

    #[test]
    fn untrusted_workspace_never_runs() {
        let (dir, workspace, checks) = fixture("[checks.ok]\ncmd = [\"/bin/true\"]\n", false);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("ok");
        assert_eq!(report.status, CheckStatus::Unverified);
        assert_eq!(
            report.error.as_deref(),
            Some("Named checks require --trusted")
        );
        cleanup(&dir);
    }

    #[test]
    fn passing_check_reports_ok() {
        let (dir, workspace, checks) = fixture("[checks.ok]\ncmd = [\"/bin/true\"]\n", true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("ok");
        assert_eq!(report.status, CheckStatus::Ok);
        assert_eq!(report.returncode, Some(0));
        assert!(report.error.is_none());
        assert_eq!(report.source, "flow.check");
        cleanup(&dir);
    }

    #[test]
    fn failing_check_reports_returncode() {
        let (dir, workspace, checks) = fixture(
            "[checks.bad]\ncmd = [\"/bin/sh\", \"-c\", \"exit 7\"]\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("bad");
        assert_eq!(report.status, CheckStatus::Failed);
        assert_eq!(report.returncode, Some(7));
        cleanup(&dir);
    }

    #[test]
    fn missing_executable_is_unavailable() {
        let (dir, workspace, checks) = fixture(
            "[checks.gone]\ncmd = [\"phlow-no-such-executable-fixture\"]\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("gone");
        assert_eq!(report.status, CheckStatus::Unavailable);
        assert!(report.error.is_some());
        cleanup(&dir);
    }

    #[test]
    fn argv_is_exact_with_no_shell() {
        // `;` inside an argument must stay literal: no shell interprets it.
        let (dir, workspace, checks) = fixture(
            "[checks.echo]\ncmd = [\"/bin/echo\", \"literal; echo unsafe\"]\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("echo");
        assert_eq!(report.status, CheckStatus::Ok);
        assert!(report.stdout.contains("literal; echo unsafe"));
        cleanup(&dir);
    }

    #[test]
    fn check_cwd_is_the_workspace() {
        let (dir, workspace, checks) =
            fixture("[checks.pwd]\ncmd = [\"/bin/sh\", \"-c\", \"pwd\"]\n", true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("pwd");
        assert_eq!(report.status, CheckStatus::Ok);
        assert!(report.stdout.trim_end() == dir.to_string_lossy());
        cleanup(&dir);
    }

    #[test]
    fn output_is_a_bounded_tail() {
        let (dir, workspace, checks) = fixture(
            "[checks.noisy]\ncmd = [\"/bin/sh\", \"-c\", \"yes | head -c 100000\"]\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("noisy");
        assert_eq!(report.status, CheckStatus::Ok);
        assert!(report.stdout_truncated);
        assert!(report.stdout.chars().count() <= OUTPUT_BYTES_MAX as usize);
        assert!(!report.stdout.is_empty());
        cleanup(&dir);
    }

    #[test]
    fn stderr_is_captured_separately() {
        let (dir, workspace, checks) = fixture(
            "[checks.err]\ncmd = [\"/bin/sh\", \"-c\", \"echo oops >&2; exit 3\"]\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("err");
        assert_eq!(report.status, CheckStatus::Failed);
        assert_eq!(report.returncode, Some(3));
        assert!(report.stderr.contains("oops"));
        assert!(!report.stdout.contains("oops"));
        cleanup(&dir);
    }

    #[test]
    fn timeout_kills_the_check() {
        let (dir, workspace, checks) = fixture(
            "[checks.slow]\ncmd = [\"/bin/sleep\", \"10\"]\ntimeout = 50\n",
            true,
        );
        let runner = CheckRunner::new(&workspace, checks);
        let started = Instant::now();
        let report = runner.run("slow");
        let elapsed = started.elapsed();
        assert_eq!(report.status, CheckStatus::Timeout);
        assert_eq!(report.error.as_deref(), Some("Check exceeded 50ms"));
        assert!(report.returncode.is_none());
        assert!(
            elapsed < Duration::from_secs(3),
            "timeout did not bound the run"
        );
        cleanup(&dir);
    }

    #[test]
    fn timeout_kills_the_process_group() {
        // The check spawns a grandchild that would write a file after 1s;
        // the timeout kill must take the whole group, so the file never
        // appears. Ports test_timeout_kills_check_process_group.
        let dir = temp_dir("group");
        let leak = dir.join("leak.txt");
        let script = format!("( sleep 1; touch \"{}\" ) & exec sleep 20", leak.display());
        let toml = format!(
            "[checks.tree]\ncmd = [\"/bin/sh\", \"-c\", {:?}, \"sh\"]\ntimeout = 100\n",
            script
        );
        let (workspace, checks) = fixture_in(&dir, &toml, true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("tree");
        assert_eq!(report.status, CheckStatus::Timeout);
        std::thread::sleep(Duration::from_millis(1500));
        assert!(!leak.exists(), "grandchild survived the process-group kill");
        cleanup(&dir);
    }

    #[test]
    fn report_carries_check_fields() {
        let toml = "[checks.lint]\ncmd = [\"/bin/true\"]\nkind = \"lint\"\nrequired = false\n\
                    filetypes = [\"rust\"]\ntimeout = 1234\n";
        let (dir, workspace, checks) = fixture(toml, true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("lint");
        assert_eq!(report.status, CheckStatus::Ok);
        assert_eq!(report.cmd, vec!["/bin/true".to_string()]);
        assert_eq!(report.timeout_ms, 1234);
        assert!(!report.required);
        assert_eq!(report.filetypes, vec!["rust".to_string()]);
        assert_eq!(report.kind, phlow_config::CheckKind::Lint);
        assert_eq!(report.workspace, dir.to_string_lossy());
        assert_eq!(report.revision, workspace.revision());
        cleanup(&dir);
    }

    #[test]
    fn default_timeout_matches_python() {
        let (dir, workspace, checks) = fixture("[checks.ok]\ncmd = [\"/bin/true\"]\n", true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run("ok");
        assert_eq!(report.timeout_ms, 60_000);
        cleanup(&dir);
    }

    #[test]
    fn checks_run_in_sorted_name_order() {
        let toml = "[checks.b]\ncmd = [\"/bin/true\"]\n[checks.a]\ncmd = [\"/bin/true\"]\n";
        let (dir, workspace, checks) = fixture(toml, true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run_all(None);
        let names: Vec<&str> = report
            .checks
            .iter()
            .map(|check| check.name.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b"]);
        cleanup(&dir);
    }

    #[test]
    fn run_all_verified_needs_required_passing() {
        let (dir, workspace, checks) = fixture("[checks.ok]\ncmd = [\"/bin/true\"]\n", true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run_all(None);
        assert!(report.verified);
        assert_eq!(report.status, "ok");
        assert_eq!(report.reason, "");
        cleanup(&dir);
    }

    #[test]
    fn run_all_empty_and_optional_never_verify() {
        let (dir, workspace, checks) = fixture("", true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run_all(None);
        assert!(!report.verified);
        assert_eq!(report.status, "unverified");
        assert!(!report.reason.is_empty());

        let (dir2, workspace2, checks2) = fixture(
            "[checks.maybe]\ncmd = [\"/bin/true\"]\nrequired = false\n",
            true,
        );
        let runner2 = CheckRunner::new(&workspace2, checks2);
        let report2 = runner2.run_all(None);
        assert!(!report2.verified);
        assert_eq!(report2.status, "unverified");
        cleanup(&dir);
        cleanup(&dir2);
    }

    #[test]
    fn run_all_failed_required_is_failed() {
        let toml = "[checks.bad]\ncmd = [\"/bin/sh\", \"-c\", \"exit 1\"]\n";
        let (dir, workspace, checks) = fixture(toml, true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run_all(None);
        assert!(!report.verified);
        assert_eq!(report.status, "failed");
        cleanup(&dir);
    }

    #[test]
    fn run_all_with_name_runs_one() {
        let toml = "[checks.a]\ncmd = [\"/bin/true\"]\n[checks.b]\ncmd = [\"/bin/true\"]\n";
        let (dir, workspace, checks) = fixture(toml, true);
        let runner = CheckRunner::new(&workspace, checks);
        let report = runner.run_all(Some("b"));
        assert_eq!(report.checks.len(), 1);
        assert_eq!(report.checks[0].name, "b");
        // A single required passing check verifies.
        assert!(report.verified);
        let missing = runner.run_all(Some("nope"));
        assert_eq!(missing.checks[0].status, CheckStatus::Unavailable);
        assert!(!missing.verified);
        cleanup(&dir);
    }

    #[test]
    fn stale_workspace_poison_the_report() {
        let dir = temp_dir("stalecheck");
        let (workspace, checks) = fixture_in(&dir, "[checks.ok]\ncmd = [\"/bin/true\"]\n", true);
        let runner = CheckRunner::new(&workspace, checks);
        // Replace the root out from under the runner.
        std::fs::remove_dir_all(&dir).expect("setup");
        std::fs::create_dir(&dir).expect("setup");
        let report = runner.run("ok");
        assert_eq!(report.status, CheckStatus::Stale);
        assert!(report.error.is_some());
        cleanup(&dir);
    }

    #[test]
    fn from_config_builds_runner() {
        let (dir, workspace, checks) = fixture("[checks.ok]\ncmd = [\"/bin/true\"]\n", true);
        let config_path = dir.join("flow-test.toml");
        let config = load_config(&LoadOptions {
            config_path: Some(config_path),
            workspace: Some(dir.clone()),
            trusted: true,
            model: None,
        })
        .expect("setup");
        let runner = CheckRunner::from_config(&workspace, &config);
        assert_eq!(runner.run("ok").status, CheckStatus::Ok);
        assert_eq!(checks.len(), 1);
        cleanup(&dir);
    }

    #[test]
    fn status_strings_match_wire() {
        assert_eq!(CheckStatus::Ok.as_str(), "ok");
        assert_eq!(CheckStatus::Failed.as_str(), "failed");
        assert_eq!(CheckStatus::Timeout.as_str(), "timeout");
        assert_eq!(CheckStatus::Unavailable.as_str(), "unavailable");
        assert_eq!(CheckStatus::Error.as_str(), "error");
        assert_eq!(CheckStatus::Unverified.as_str(), "unverified");
        assert_eq!(CheckStatus::Stale.as_str(), "stale");
    }
}
