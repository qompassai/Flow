//! Validated operator configuration for the phlow agent runtime.
//!
//! Plain words: the operator's TOML file says where the workspace is, which
//! Ollama server to talk to, how hard the agent may work, and which named
//! checks verify its output. This crate reads that file, rejects anything
//! the schema does not define (including `trusted` and the legacy
//! `shell`/`plugins`/`self_improve` keys), enforces every numeric bound, and
//! hands back a [`FlowConfig`] whose fields can only be read, never mutated
//! back into an invalid state.
//!
//! Port of `flow/config.py`. Project configuration is never auto-discovered:
//! only an explicit `--config` path or `$XDG_CONFIG_HOME/flow/config.toml`
//! is loaded, and workspace-local `config.toml`/`.flow.toml` files produce
//! warnings instead of configuration.
//!
//! # Bounds
//!
//! - [`load::CONFIG_FILE_BYTES_MAX`]: largest config file accepted.
//! - [`model::CHECKS_MAX`], [`model::CHECK_ARGV_MAX`],
//!   [`model::CHECK_NAME_CHARS_MAX`], [`model::CHECK_TIMEOUT_MS_MAX`]: check
//!   table limits.
//!
//! # Unsafe policy
//!
//! This crate forbids unsafe code outright.

#![forbid(unsafe_code)]

mod error;
mod load;
mod model;

pub use error::ConfigError;
pub use load::{CONFIG_FILE_BYTES_MAX, LoadOptions, load_config};
pub use model::{
    AGENT_MAX_CONTEXT_CHARS_MAX, AGENT_MAX_CONTEXT_CHARS_MIN, AGENT_MAX_CYCLES_MAX,
    AGENT_MAX_CYCLES_MIN, AGENT_MAX_ITERATIONS_MAX, AGENT_MAX_ITERATIONS_MIN,
    AGENT_MAX_TASK_CHARS_MAX, AGENT_MAX_TASK_CHARS_MIN, AGENT_MAX_TOOL_CALLS_MAX,
    AGENT_MAX_TOOL_CALLS_MIN, AgentConfig, CHECK_ARGV_MAX, CHECK_KINDS, CHECK_NAME_CHARS_MAX,
    CHECK_TIMEOUT_MS_DEFAULT, CHECK_TIMEOUT_MS_MAX, CHECK_TIMEOUT_MS_MIN, CHECKS_MAX, CheckConfig,
    CheckKind, FlowConfig, ModelRole, ModelsConfig, OLLAMA_CONTEXT_LENGTH_MAX,
    OLLAMA_CONTEXT_LENGTH_MIN, OLLAMA_TEMPERATURE_MAX, OLLAMA_TEMPERATURE_MIN,
    OLLAMA_TIMEOUT_SECS_MAX, OLLAMA_TIMEOUT_SECS_MIN, OllamaConfig,
};
