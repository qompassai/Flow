//! The safe agent runtime: bounded planner → coder → host verification →
//! reviewer over one [`Workspace`], one [`OllamaBackend`], and one
//! [`EditorBridge`].
//!
//! Mirrors `flow/runtime.py`. The runtime is the single authority shared by
//! the CLI, TUI, and MCP surfaces; everything model-visible crosses it.
//!
//! Safety contract (same as Python):
//!
//! - Single writer: at most one `run`/`check` executes at a time; a second
//!   caller gets an `"error"` report (`"Runtime busy: single writer"` /
//!   `"Runtime busy"`), never a second mutable run.
//! - Bounded work: context characters, model iterations, tool calls, cycles,
//!   argument depth (16) and argument nodes (4,096) are all capped with named
//!   constants. No unbounded recursion anywhere.
//! - Role gates: the planner and reviewer are read-only; only the coder gets
//!   `file_write`/`flow_check`, and only when `--trusted`.
//! - Host verification: reviewer approval never substitutes for host-run
//!   checks. Changed files are fingerprinted before and after review; any
//!   drift marks verification `"stale"`.
//! - `editor_debug` `launch`/`run` is manual, not an agent tool.
//!
//! [`Workspace`]: phlow_workspace::Workspace
//! [`OllamaBackend`]: phlow_llm::OllamaBackend
//! [`EditorBridge`]: phlow_editor::EditorBridge

#![forbid(unsafe_code)]

pub mod error;
pub mod prompt;
pub mod report;
pub mod runtime;
pub mod tools;
pub mod transport;

pub use error::RuntimeError;
pub use prompt::{
    VENDORED_SYSTEM_PROMPT, has_error_diagnostics, normalize_tool_calls, reviewer_verdict,
    system_prompt_for_role,
};
pub use report::{
    check_report_to_value, list_result_to_value, new_report, read_result_to_value,
    run_all_report_to_value, write_result_to_value,
};
pub use runtime::Runtime;
pub use tools::{READ_ONLY_EDITOR_NAMES, file_schemas, schemas_for_role};
pub use transport::{MsgpackTransport, ReqwestTransport};
