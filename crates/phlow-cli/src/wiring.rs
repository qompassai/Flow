//! Production wiring: the shared safe runtime behind both the MCP server
//! and the TUI facade.
//!
//! Phase 5 deliberately left `RuntimeFacade` unimplemented for the real
//! runtime (it exposed no model listing or switching). This module closes
//! that seam: [`CliRuntime`] owns a
//! `Runtime<ReqwestTransport, MsgpackTransport>` and implements
//! [`McpRuntime`](phlow_mcp::McpRuntime) for `serve` and
//! [`RuntimeFacade`](phlow_tui::RuntimeFacade) for the TUI.
//!
//! Facade decisions, each mirroring the Python `FlowApp`:
//! - `schemas()` drives the `coder` role surface — Python's
//!   `Runtime.schemas()` defaults to `role="coder"`.
//! - `select_model(name)` assigns `ollama.model` and resets per-role
//!   overrides, exactly like Python's `/model` handler.

use phlow_mcp::{McpRuntime, RuntimeError as McpRuntimeError};
use phlow_runtime::transport::{MsgpackTransport, ReqwestTransport};
use phlow_runtime::{Runtime, RuntimeError};
use phlow_tui::RuntimeFacade;
use serde_json::Value;

/// The concrete production runtime: real Ollama HTTP transport plus the
/// real Neovim msgpack transport.
pub type ProductionRuntime = Runtime<ReqwestTransport, MsgpackTransport>;

/// Newtype so the two foreign traits (`McpRuntime`, `RuntimeFacade`) can
/// both be implemented for the production runtime.
pub struct CliRuntime(pub ProductionRuntime);

impl CliRuntime {
    /// Wrap a built runtime.
    pub fn new(runtime: ProductionRuntime) -> CliRuntime {
        CliRuntime(runtime)
    }

    /// Installed model names from the Ollama backend (`GET /api/tags`).
    pub fn list_models(&mut self) -> Result<Vec<String>, RuntimeError> {
        self.0.list_models()
    }
}

impl McpRuntime for CliRuntime {
    // `run`/`status`/`check` never fail as Rust operations — the runtime
    // encodes failure in the report Value, exactly like Python — but the
    // trait demands a fallible error type, so map the runtime error into
    // the MCP error string.
    fn run(&mut self, task: &str) -> Result<Value, McpRuntimeError> {
        Ok(self.0.run(task))
    }

    fn status(&mut self) -> Result<Value, McpRuntimeError> {
        Ok(self.0.status())
    }

    fn check(&mut self, name: Option<&str>) -> Result<Value, McpRuntimeError> {
        Ok(self.0.check(name))
    }

    fn close(&mut self) {
        self.0.close();
    }
}

impl RuntimeFacade for CliRuntime {
    fn status(&mut self) -> Value {
        self.0.status()
    }

    fn check(&mut self, name: Option<&str>) -> Value {
        self.0.check(name)
    }

    fn schemas(&mut self) -> Vec<Value> {
        // Python's `Runtime.schemas()` defaults to the coder role.
        self.0.schemas("coder")
    }

    fn run(&mut self, task: &str) -> Value {
        self.0.run(task)
    }

    fn select_model(&mut self, model: &str) {
        // The TUI rejects empty names before calling, so failure here is
        // unexpected; report it loudly rather than dropping the error.
        if let Err(error) = self.0.select_model(model) {
            eprintln!("Flow: cannot switch model: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    // Wiring is exercised by the integration tests (tests/cli.rs) against
    // the real runtime. Unit-testing here would only re-test trait
    // signatures.
}
