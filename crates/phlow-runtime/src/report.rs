//! Report shapes: the exact JSON the CLI, TUI, and MCP surfaces emit.
//!
//! Mirrors `_new_report` in `flow/runtime.py` and the `run`/`run_all` dicts
//! in `flow/checks.py`. Key insertion order matches Python's dict order so
//! serialized output is byte-compatible.

use phlow_checks::{CheckReport, RunAllReport};
use phlow_workspace::{ListResult, ReadResult, WriteResult};
use serde_json::{Map, Value};

/// Build a fresh run report, mirroring `_new_report`.
pub fn new_report(task: &str) -> Value {
    let mut report = Map::new();
    report.insert("status".to_owned(), Value::String("unverified".to_owned()));
    report.insert("verified".to_owned(), Value::Bool(false));
    report.insert("task".to_owned(), Value::String(task.to_owned()));
    report.insert("roles".to_owned(), Value::Array(Vec::new()));
    report.insert("events".to_owned(), Value::Array(Vec::new()));
    report.insert("changed_files".to_owned(), Value::Array(Vec::new()));
    report.insert("checks".to_owned(), Value::Array(Vec::new()));
    report.insert("verification".to_owned(), Value::Object(Map::new()));
    report.insert("model_calls".to_owned(), Value::Number(0.into()));
    report.insert("tool_calls".to_owned(), Value::Number(0.into()));
    report.insert("cycles".to_owned(), Value::Number(0.into()));
    report.insert("summary".to_owned(), Value::String(String::new()));
    report.insert("verification_history".to_owned(), Value::Array(Vec::new()));
    Value::Object(report)
}

/// Convert a [`CheckReport`] to its Python `run()` dict shape.
///
/// Key order matches Python: `name, source, cmd, timeout, required,
/// filetypes, kind, workspace, revision`, then the outcome keys.
///
/// A report for an unknown check name is the exception: Python's `run()`
/// returns a compact 4-key dict for it, exactly
/// `{"name": ..., "status": "unavailable", "error": "No such configured
/// check", "source": "flow.check"}` — not the full report shape.
pub fn check_report_to_value(report: &CheckReport) -> Value {
    if report.is_unknown() {
        let mut out = Map::new();
        out.insert("name".to_owned(), Value::String(report.name.clone()));
        out.insert(
            "status".to_owned(),
            Value::String(report.status.as_str().to_owned()),
        );
        out.insert(
            "error".to_owned(),
            Value::String(report.error.clone().unwrap_or_default()),
        );
        out.insert("source".to_owned(), Value::String(report.source.to_owned()));
        return Value::Object(out);
    }
    let mut out = Map::new();
    out.insert("name".to_owned(), Value::String(report.name.clone()));
    out.insert("source".to_owned(), Value::String(report.source.to_owned()));
    out.insert(
        "cmd".to_owned(),
        Value::Array(
            report
                .cmd
                .iter()
                .map(|arg| Value::String(arg.clone()))
                .collect(),
        ),
    );
    out.insert(
        "timeout".to_owned(),
        Value::Number(report.timeout_ms.into()),
    );
    out.insert("required".to_owned(), Value::Bool(report.required));
    out.insert(
        "filetypes".to_owned(),
        Value::Array(
            report
                .filetypes
                .iter()
                .map(|filetype| Value::String(filetype.clone()))
                .collect(),
        ),
    );
    out.insert(
        "kind".to_owned(),
        Value::String(report.kind.as_str().to_owned()),
    );
    out.insert(
        "workspace".to_owned(),
        Value::String(report.workspace.clone()),
    );
    out.insert("revision".to_owned(), Value::Number(report.revision.into()));
    out.insert(
        "status".to_owned(),
        Value::String(report.status.as_str().to_owned()),
    );
    if let Some(returncode) = report.returncode {
        out.insert("returncode".to_owned(), Value::Number(returncode.into()));
    }
    out.insert("stdout".to_owned(), Value::String(report.stdout.clone()));
    out.insert(
        "stdout_truncated".to_owned(),
        Value::Bool(report.stdout_truncated),
    );
    out.insert("stderr".to_owned(), Value::String(report.stderr.clone()));
    out.insert(
        "stderr_truncated".to_owned(),
        Value::Bool(report.stderr_truncated),
    );
    out.insert(
        "duration_ms".to_owned(),
        Value::Number(report.duration_ms.into()),
    );
    if let Some(error) = &report.error {
        out.insert("error".to_owned(), Value::String(error.clone()));
    }
    Value::Object(out)
}

/// Convert a [`RunAllReport`] to its Python `run_all()` dict shape.
pub fn run_all_report_to_value(report: &RunAllReport) -> Value {
    let mut out = Map::new();
    out.insert("status".to_owned(), Value::String(report.status.to_owned()));
    out.insert("verified".to_owned(), Value::Bool(report.verified));
    out.insert(
        "checks".to_owned(),
        Value::Array(report.checks.iter().map(check_report_to_value).collect()),
    );
    out.insert("reason".to_owned(), Value::String(report.reason.to_owned()));
    Value::Object(out)
}

/// Convert a [`ReadResult`] to its Python `read()` dict shape.
pub fn read_result_to_value(result: &ReadResult) -> Value {
    let mut out = Map::new();
    out.insert("status".to_owned(), Value::String("ok".to_owned()));
    out.insert("path".to_owned(), Value::String(result.path.clone()));
    out.insert("content".to_owned(), Value::String(result.content.clone()));
    out.insert("bytes".to_owned(), Value::Number(result.bytes.into()));
    Value::Object(out)
}

/// Convert a [`WriteResult`] to its Python `write()` dict shape.
pub fn write_result_to_value(result: &WriteResult) -> Value {
    let mut out = Map::new();
    out.insert("status".to_owned(), Value::String("ok".to_owned()));
    out.insert("path".to_owned(), Value::String(result.path.clone()));
    out.insert("bytes".to_owned(), Value::Number(result.bytes.into()));
    out.insert("revision".to_owned(), Value::Number(result.revision.into()));
    Value::Object(out)
}

/// Convert a [`ListResult`] to its Python `list()` dict shape.
pub fn list_result_to_value(path: &str, result: &ListResult) -> Value {
    let mut out = Map::new();
    out.insert("status".to_owned(), Value::String("ok".to_owned()));
    out.insert("path".to_owned(), Value::String(path.to_owned()));
    out.insert(
        "files".to_owned(),
        Value::Array(
            result
                .files
                .iter()
                .map(|file| Value::String(file.clone()))
                .collect(),
        ),
    );
    out.insert("truncated".to_owned(), Value::Bool(result.truncated));
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlow_checks::CheckRunner;

    #[test]
    fn new_report_keys_and_order() {
        let report = new_report("task");
        let object = report.as_object().unwrap();
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "status",
                "verified",
                "task",
                "roles",
                "events",
                "changed_files",
                "checks",
                "verification",
                "model_calls",
                "tool_calls",
                "cycles",
                "summary",
                "verification_history",
            ]
        );
        assert_eq!(report["status"], "unverified");
        assert_eq!(report["task"], "task");
    }

    #[test]
    fn unknown_check_renders_compact_four_key_dict() {
        // Python's `run()` for an unknown name returns exactly
        // {"name": ..., "status": "unavailable",
        //  "error": "No such configured check", "source": "flow.check"} —
        // the full `run()` shape is only for configured checks.
        let dir =
            std::env::temp_dir().join(format!("phlow-report-test-{}-unknown", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test temp dir");
        let workspace =
            phlow_workspace::Workspace::open(&dir, false, &[]).expect("test setup: open workspace");
        let runner = CheckRunner::new(&workspace, std::collections::BTreeMap::new());
        let report = runner.run("nope");
        assert!(report.is_unknown());
        let value = check_report_to_value(&report);
        let object = value.as_object().expect("report is an object");
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, ["name", "status", "error", "source"]);
        assert_eq!(value["name"], "nope");
        assert_eq!(value["status"], "unavailable");
        assert_eq!(value["error"], "No such configured check");
        assert_eq!(value["source"], "flow.check");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn configured_check_keeps_full_report_shape() {
        // A configured check that fails still serializes with the full
        // key set — only the unknown-name placeholder is compact.
        let dir =
            std::env::temp_dir().join(format!("phlow-report-test-{}-full", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test temp dir");
        let config_path = dir.join("test.toml");
        std::fs::write(
            &config_path,
            "[checks.broken]\ncmd=[\"/nonexistent-binary-xyz\"]\n",
        )
        .expect("write test config");
        let config = phlow_config::load_config(&phlow_config::LoadOptions {
            config_path: Some(config_path),
            workspace: Some(dir.clone()),
            ..phlow_config::LoadOptions::default()
        })
        .expect("test setup: load config");
        let workspace =
            phlow_workspace::Workspace::open(&dir, false, &[]).expect("test setup: open workspace");
        let runner = CheckRunner::new(&workspace, config.checks().clone());
        let report = runner.run("broken");
        assert!(!report.is_unknown());
        let value = check_report_to_value(&report);
        let object = value.as_object().expect("report is an object");
        assert!(object.contains_key("cmd"), "full shape keeps cmd");
        assert!(object.contains_key("timeout"), "full shape keeps timeout");
        assert!(object.contains_key("stdout"), "full shape keeps stdout");
        assert_eq!(value["source"], "flow.check");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
