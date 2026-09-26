//! Named, operator-approved checks for the phlow agent runtime.
//!
//! A [`CheckRunner`](runner::CheckRunner) executes the exact argv from the
//! operator's TOML — no shell, no model-chosen command, cwd, or
//! environment. Each check runs to completion or its configured timeout;
//! on timeout the whole process group is killed and reaped, so no orphan
//! keeps running after the report.
//!
//! The legacy `shell_exec` and `lsp_check` tools are fail-closed here:
//! [`disabled`] preserves their exact denial behavior as typed values,
//! never as silently skipped execution.

#![forbid(unsafe_code)]

pub mod disabled;
pub mod runner;

pub use disabled::{DisabledError, DisabledTool, LSP_CHECK_DENIAL, SHELL_DENIAL};
pub use phlow_config::CheckKind;
pub use runner::{CheckReport, CheckRunner, CheckStatus, OUTPUT_BYTES_MAX, RunAllReport};
