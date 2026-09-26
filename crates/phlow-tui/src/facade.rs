//! The narrow runtime surface the TUI drives.
//!
//! `RuntimeFacade` exposes exactly what the chat commands need: `status`,
//! `check`, `schemas`, `run`, and model selection. Command logic programs
//! against this trait, so tests use [`FakeRuntime`] without a live backend.
//!
//! The trait is deliberately not implemented for the shared safe runtime
//! (`phlow_runtime::Runtime`) in this phase: the runtime does not expose
//! model listing or model switching, and the trait must not pretend
//! otherwise. Phase 6 wires the real transport and config ownership.

use std::collections::VecDeque;

use serde_json::Value;

/// The runtime operations the TUI commands need.
pub trait RuntimeFacade {
    /// The runtime status report.
    fn status(&mut self) -> Value;
    /// Run one named check, or all checks when `name` is `None`.
    fn check(&mut self, name: Option<&str>) -> Value;
    /// The model-facing tool schemas (the `coder` role surface).
    fn schemas(&mut self) -> Vec<Value>;
    /// Run one natural-language task; returns the result JSON.
    fn run(&mut self, task: &str) -> Value;
    /// Switch the active model. Python's `/model` assigns the name and
    /// resets per-role model overrides to defaults; implementations
    /// targeting the real runtime must do the same. The fake records the
    /// request.
    fn select_model(&mut self, model: &str);
}

/// An in-memory [`RuntimeFacade`] for tests: scripted results, recorded
/// calls.
#[derive(Debug, Default)]
pub struct FakeRuntime {
    /// Result returned by [`FakeRuntime::status`].
    pub status_result: Value,
    /// Result returned by [`FakeRuntime::check`].
    pub check_result: Value,
    /// Schemas returned by [`FakeRuntime::schemas`].
    pub schemas_result: Vec<Value>,
    /// Results returned by successive [`FakeRuntime::run`] calls; when
    /// empty, `run` returns `{"status": "ok", "echo": <task>}`.
    pub run_results: VecDeque<Value>,
    /// Tasks passed to [`FakeRuntime::run`], in order.
    pub run_tasks: Vec<String>,
    /// Check names passed to [`FakeRuntime::check`], in order.
    pub check_names: Vec<Option<String>>,
    /// Models passed to [`FakeRuntime::select_model`], in order.
    pub selected_models: Vec<String>,
}

impl FakeRuntime {
    /// A fake returning null results until scripted.
    pub fn new() -> FakeRuntime {
        FakeRuntime::default()
    }

    /// Queue one result for the next [`FakeRuntime::run`] call.
    pub fn queue_run_result(&mut self, result: Value) {
        self.run_results.push_back(result);
    }
}

impl RuntimeFacade for FakeRuntime {
    fn status(&mut self) -> Value {
        self.status_result.clone()
    }

    fn check(&mut self, name: Option<&str>) -> Value {
        self.check_names.push(name.map(str::to_string));
        self.check_result.clone()
    }

    fn schemas(&mut self) -> Vec<Value> {
        self.schemas_result.clone()
    }

    fn run(&mut self, task: &str) -> Value {
        self.run_tasks.push(task.to_string());
        self.run_results
            .pop_front()
            .unwrap_or_else(|| serde_json::json!({"status": "ok", "echo": task}))
    }

    fn select_model(&mut self, model: &str) {
        self.selected_models.push(model.to_string());
    }
}
