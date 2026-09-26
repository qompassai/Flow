//! Thin orchestrator over the safe runtime. Mirrors
//! `flow/agent/orchestrator.py`.
//!
//! Python validates `isinstance(runtime, Runtime)` at construction; Rust
//! enforces that statically through the type parameter. `run` delegates to
//! [`phlow_runtime::Runtime::run`]: the runtime's budgets, containment, and
//! host verification always apply.

use phlow_llm::transport::LlmTransport;
use phlow_runtime::Runtime;

/// Drives one [`Runtime`]: `run` forwards the user message, `close`
/// releases the runtime idempotently.
pub struct Orchestrator<L: LlmTransport, E: phlow_editor::bridge::EditorTransport> {
    runtime: Runtime<L, E>,
}

impl<L: LlmTransport, E: phlow_editor::bridge::EditorTransport> Orchestrator<L, E> {
    /// Wrap an initialized safe runtime.
    pub fn new(runtime: Runtime<L, E>) -> Self {
        Self { runtime }
    }

    /// Run the bounded planner → coder → verification → reviewer loop for
    /// one user message, returning the runtime's report object.
    pub fn run(&mut self, user_message: &str) -> serde_json::Value {
        self.runtime.run(user_message)
    }

    /// Release the runtime. Idempotent, like
    /// [`phlow_runtime::Runtime::close`].
    pub fn close(&mut self) {
        self.runtime.close();
    }
}
