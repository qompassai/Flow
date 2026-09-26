//! Validated configuration model.
//!
//! Every struct here is a *validated* value: fields are private, so the only
//! way to obtain one is through the constructors in [`crate::load`], which
//! enforce every bound below. Readers use the getters; nothing can mutate a
//! loaded config back into an invalid state.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::error::ConfigError;

// --- Named bounds (units in the name) --------------------------------------
// These mirror flow/config.py. The const assertions below are the Rust form
// of that module's module-level `assert`s: the limits must stay sane.

/// Maximum number of named checks.
pub const CHECKS_MAX: usize = 32;
/// Maximum argv elements per check.
pub const CHECK_ARGV_MAX: usize = 128;
/// Maximum check-name length, in characters.
pub const CHECK_NAME_CHARS_MAX: usize = 64;
/// Minimum check timeout, in milliseconds.
pub const CHECK_TIMEOUT_MS_MIN: u64 = 1;
/// Maximum check timeout, in milliseconds.
pub const CHECK_TIMEOUT_MS_MAX: u64 = 600_000;
/// Default check timeout, in milliseconds (mirrors Python's `timeout: int = 60000`).
pub const CHECK_TIMEOUT_MS_DEFAULT: u64 = 60_000;

const _: () = assert!(
    0 < CHECKS_MAX && CHECKS_MAX <= 256,
    "CHECKS_MAX out of range"
);
const _: () = assert!(0 < CHECK_ARGV_MAX, "CHECK_ARGV_MAX must be positive");
const _: () = assert!(
    0 < CHECK_NAME_CHARS_MAX,
    "CHECK_NAME_CHARS_MAX must be positive"
);
const _: () = assert!(
    0 < CHECK_TIMEOUT_MS_MAX,
    "CHECK_TIMEOUT_MS_MAX must be positive"
);
const _: () = assert!(
    CHECK_TIMEOUT_MS_MIN <= CHECK_TIMEOUT_MS_MAX,
    "check timeout range is empty"
);
const _: () = assert!(
    CHECK_TIMEOUT_MS_MIN <= CHECK_TIMEOUT_MS_DEFAULT
        && CHECK_TIMEOUT_MS_DEFAULT <= CHECK_TIMEOUT_MS_MAX,
    "CHECK_TIMEOUT_MS_DEFAULT outside [MIN, MAX]"
);

/// Supported check kinds, in canonical order.
pub const CHECK_KINDS: &[&str] = &["check", "test", "lint", "typecheck", "diagnostics", "build"];

/// Ollama numeric bounds, mirroring `flow/config.py::_number` calls.
pub const OLLAMA_TEMPERATURE_MIN: f64 = 0.0;
pub const OLLAMA_TEMPERATURE_MAX: f64 = 2.0;
pub const OLLAMA_TIMEOUT_SECS_MIN: f64 = 0.1;
pub const OLLAMA_TIMEOUT_SECS_MAX: f64 = 600.0;
pub const OLLAMA_CONTEXT_LENGTH_MIN: u32 = 1024;
pub const OLLAMA_CONTEXT_LENGTH_MAX: u32 = 131_072;

/// Agent numeric bounds, mirroring `flow/config.py`.
pub const AGENT_MAX_ITERATIONS_MIN: u32 = 1;
pub const AGENT_MAX_ITERATIONS_MAX: u32 = 32;
pub const AGENT_MAX_CYCLES_MIN: u32 = 1;
pub const AGENT_MAX_CYCLES_MAX: u32 = 5;
pub const AGENT_MAX_TOOL_CALLS_MIN: u32 = 1;
pub const AGENT_MAX_TOOL_CALLS_MAX: u32 = 256;
pub const AGENT_MAX_CONTEXT_CHARS_MIN: u32 = 4_096;
pub const AGENT_MAX_CONTEXT_CHARS_MAX: u32 = 1_000_000;
pub const AGENT_MAX_TASK_CHARS_MIN: u32 = 1;
pub const AGENT_MAX_TASK_CHARS_MAX: u32 = 100_000;

/// A named check's kind. Closed enum: unknown kinds are rejected at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckKind {
    Check,
    Test,
    Lint,
    Typecheck,
    Diagnostics,
    Build,
}

impl CheckKind {
    /// Parse a kind name; `None` means the schema rejects it.
    pub fn parse(name: &str) -> Option<CheckKind> {
        match name {
            "check" => Some(CheckKind::Check),
            "test" => Some(CheckKind::Test),
            "lint" => Some(CheckKind::Lint),
            "typecheck" => Some(CheckKind::Typecheck),
            "diagnostics" => Some(CheckKind::Diagnostics),
            "build" => Some(CheckKind::Build),
            _ => None,
        }
    }

    /// Canonical wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            CheckKind::Check => "check",
            CheckKind::Test => "test",
            CheckKind::Lint => "lint",
            CheckKind::Typecheck => "typecheck",
            CheckKind::Diagnostics => "diagnostics",
            CheckKind::Build => "build",
        }
    }
}

/// One named check: exact argv, no shell, no model-chosen command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckConfig {
    cmd: Vec<String>,
    timeout_ms: u64,
    required: bool,
    filetypes: Vec<String>,
    kind: CheckKind,
}

impl CheckConfig {
    /// Build a check from already-validated parts. Callers inside this crate
    /// validate first; the constructor asserts the validated shape so a
    /// future caller cannot silently widen it.
    pub(crate) fn new(
        cmd: Vec<String>,
        timeout_ms: u64,
        required: bool,
        filetypes: Vec<String>,
        kind: CheckKind,
    ) -> CheckConfig {
        assert!(
            !cmd.is_empty() && cmd.len() <= CHECK_ARGV_MAX,
            "check argv must be 1..=CHECK_ARGV_MAX elements"
        );
        assert!(
            cmd.iter().all(|arg| !arg.is_empty() && !arg.contains('\0')),
            "check argv elements must be nonempty and NUL-free"
        );
        assert!(
            (CHECK_TIMEOUT_MS_MIN..=CHECK_TIMEOUT_MS_MAX).contains(&timeout_ms),
            "check timeout out of range"
        );
        CheckConfig {
            cmd,
            timeout_ms,
            required,
            filetypes,
            kind,
        }
    }

    /// Exact argv, executed without a shell.
    pub fn cmd(&self) -> &[String] {
        &self.cmd
    }
    /// Timeout in milliseconds.
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }
    /// Whether verification fails when this check is missing or failing.
    pub fn required(&self) -> bool {
        self.required
    }
    /// Descriptive filetypes; never silently skips required checks.
    pub fn filetypes(&self) -> &[String] {
        &self.filetypes
    }
    /// What kind of verification this check performs.
    pub fn kind(&self) -> CheckKind {
        self.kind
    }
}

/// Ollama backend settings.
#[derive(Debug, Clone, PartialEq)]
pub struct OllamaConfig {
    base_url: String,
    model: String,
    temperature: f64,
    context_length: u32,
    timeout_secs: f64,
    allow_remote: bool,
}

impl Default for OllamaConfig {
    fn default() -> OllamaConfig {
        OllamaConfig {
            base_url: "http://127.0.0.1:11434".to_owned(),
            model: "qwen2.5-coder:7b".to_owned(),
            temperature: 0.2,
            context_length: 16384,
            timeout_secs: 120.0,
            allow_remote: false,
        }
    }
}

impl OllamaConfig {
    pub(crate) fn validated(
        base_url: String,
        model: String,
        temperature: f64,
        context_length: u32,
        timeout_secs: f64,
        allow_remote: bool,
    ) -> OllamaConfig {
        assert!(!model.trim().is_empty(), "ollama.model must be nonempty");
        assert!(
            (OLLAMA_TEMPERATURE_MIN..=OLLAMA_TEMPERATURE_MAX).contains(&temperature),
            "ollama.temperature out of range"
        );
        assert!(
            (OLLAMA_CONTEXT_LENGTH_MIN..=OLLAMA_CONTEXT_LENGTH_MAX).contains(&context_length),
            "ollama.context_length out of range"
        );
        assert!(
            (OLLAMA_TIMEOUT_SECS_MIN..=OLLAMA_TIMEOUT_SECS_MAX).contains(&timeout_secs),
            "ollama.timeout out of range"
        );
        OllamaConfig {
            base_url,
            model,
            temperature,
            context_length,
            timeout_secs,
            allow_remote,
        }
    }

    /// Return a copy with a different model.
    ///
    /// Used for the `--model` CLI override. The model must be nonempty; the
    /// caller validates CLI input first and this re-runs the constructor
    /// assertions, so the validated invariant cannot be bypassed.
    pub(crate) fn with_model(&self, model: String) -> OllamaConfig {
        OllamaConfig::validated(
            self.base_url.clone(),
            model,
            self.temperature,
            self.context_length,
            self.timeout_secs,
            self.allow_remote,
        )
    }

    /// HTTP(S) origin of the Ollama daemon; loopback unless `allow_remote`.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    /// Default model; roles with empty names inherit this.
    pub fn model(&self) -> &str {
        &self.model
    }
    /// Replace the default model, mirroring Python's `/model` assignment
    /// (`runtime.cfg.ollama.model = arg`). The name must be non-empty; the
    /// TUI rejects empty names before calling, so a failure here is an
    /// operator-input error, not an internal invariant violation.
    pub fn set_model(&mut self, model: String) -> Result<(), ConfigError> {
        if model.trim().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "ollama.model".to_owned(),
                reason: "must be a nonempty string".to_owned(),
            });
        }
        self.model = model;
        Ok(())
    }
    /// Sampling temperature, 0–2.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    /// Model context window, 1024–131072 tokens.
    pub fn context_length(&self) -> u32 {
        self.context_length
    }
    /// HTTP timeout in seconds, 0.1–600.
    pub fn timeout_secs(&self) -> f64 {
        self.timeout_secs
    }
    /// Whether a non-loopback `base_url` is explicitly permitted.
    pub fn allow_remote(&self) -> bool {
        self.allow_remote
    }
}

/// Per-role model overrides. Empty strings inherit `ollama.model`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelsConfig {
    planner: String,
    coder: String,
    reviewer: String,
}

impl ModelsConfig {
    pub(crate) fn validated(planner: String, coder: String, reviewer: String) -> ModelsConfig {
        ModelsConfig {
            planner,
            coder,
            reviewer,
        }
    }

    /// Model for the planner role; empty means "inherit `ollama.model`".
    pub fn planner(&self) -> &str {
        &self.planner
    }
    /// Model for the coder role; empty means "inherit `ollama.model`".
    pub fn coder(&self) -> &str {
        &self.coder
    }
    /// Model for the reviewer role; empty means "inherit `ollama.model`".
    pub fn reviewer(&self) -> &str {
        &self.reviewer
    }
}

/// Agent loop budgets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    max_iterations: u32,
    max_cycles: u32,
    max_tool_calls: u32,
    max_context_chars: u32,
    max_task_chars: u32,
}

impl Default for AgentConfig {
    fn default() -> AgentConfig {
        AgentConfig {
            max_iterations: 6,
            max_cycles: 2,
            max_tool_calls: 64,
            max_context_chars: 100_000,
            max_task_chars: 16_000,
        }
    }
}

impl AgentConfig {
    pub(crate) fn validated(
        max_iterations: u32,
        max_cycles: u32,
        max_tool_calls: u32,
        max_context_chars: u32,
        max_task_chars: u32,
    ) -> AgentConfig {
        assert!(
            (AGENT_MAX_ITERATIONS_MIN..=AGENT_MAX_ITERATIONS_MAX).contains(&max_iterations),
            "agent.max_iterations out of range"
        );
        assert!(
            (AGENT_MAX_CYCLES_MIN..=AGENT_MAX_CYCLES_MAX).contains(&max_cycles),
            "agent.max_cycles out of range"
        );
        assert!(
            (AGENT_MAX_TOOL_CALLS_MIN..=AGENT_MAX_TOOL_CALLS_MAX).contains(&max_tool_calls),
            "agent.max_tool_calls out of range"
        );
        assert!(
            (AGENT_MAX_CONTEXT_CHARS_MIN..=AGENT_MAX_CONTEXT_CHARS_MAX)
                .contains(&max_context_chars),
            "agent.max_context_chars out of range"
        );
        assert!(
            (AGENT_MAX_TASK_CHARS_MIN..=AGENT_MAX_TASK_CHARS_MAX).contains(&max_task_chars),
            "agent.max_task_chars out of range"
        );
        AgentConfig {
            max_iterations,
            max_cycles,
            max_tool_calls,
            max_context_chars,
            max_task_chars,
        }
    }

    /// Model turns per role, per cycle, 1–32.
    pub fn max_iterations(&self) -> u32 {
        self.max_iterations
    }
    /// Planner→coder→reviewer cycles, 1–5.
    pub fn max_cycles(&self) -> u32 {
        self.max_cycles
    }
    /// Global tool-call budget across all roles and cycles, 1–256.
    pub fn max_tool_calls(&self) -> u32 {
        self.max_tool_calls
    }
    /// Context window cap in characters, 4096–1_000_000.
    pub fn max_context_chars(&self) -> u32 {
        self.max_context_chars
    }
    /// Task description cap in characters, 1–100_000.
    pub fn max_task_chars(&self) -> u32 {
        self.max_task_chars
    }
}

/// Agent roles that can carry a model override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    Planner,
    Coder,
    Reviewer,
}

/// The validated operator configuration.
///
/// Constructed only by [`crate::load::load_config`]. `trusted` is CLI/API
/// only: it is never read from TOML (a `trusted` key in TOML is rejected as
/// an unknown option).
#[derive(Debug, Clone, PartialEq)]
pub struct FlowConfig {
    ollama: OllamaConfig,
    models: ModelsConfig,
    agent: AgentConfig,
    checks: BTreeMap<String, CheckConfig>,
    workspace_dir: PathBuf,
    trusted: bool,
    config_path: Option<PathBuf>,
    warnings: Vec<String>,
}

impl FlowConfig {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validated(
        ollama: OllamaConfig,
        models: ModelsConfig,
        agent: AgentConfig,
        checks: BTreeMap<String, CheckConfig>,
        workspace_dir: PathBuf,
        trusted: bool,
        config_path: Option<PathBuf>,
        warnings: Vec<String>,
    ) -> FlowConfig {
        assert!(
            workspace_dir.is_absolute(),
            "workspace_dir must be absolute"
        );
        assert!(checks.len() <= CHECKS_MAX, "too many checks");
        FlowConfig {
            ollama,
            models,
            agent,
            checks,
            workspace_dir,
            trusted,
            config_path,
            warnings,
        }
    }

    /// Ollama backend settings.
    pub fn ollama(&self) -> &OllamaConfig {
        &self.ollama
    }
    /// Per-role model overrides.
    pub fn models(&self) -> &ModelsConfig {
        &self.models
    }
    /// Agent loop budgets.
    pub fn agent(&self) -> &AgentConfig {
        &self.agent
    }
    /// Named checks, sorted by name for deterministic iteration.
    pub fn checks(&self) -> &BTreeMap<String, CheckConfig> {
        &self.checks
    }
    /// Resolved absolute workspace directory; guaranteed to exist.
    pub fn workspace_dir(&self) -> &PathBuf {
        &self.workspace_dir
    }
    /// Execution trust. Only ever set from the CLI/API, never from TOML.
    pub fn trusted(&self) -> bool {
        self.trusted
    }
    /// Config file that was loaded, if any.
    pub fn config_path(&self) -> Option<&PathBuf> {
        self.config_path.as_ref()
    }
    /// Non-fatal notices, e.g. ignored workspace-local config files.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Model for a role: the role's override, or `ollama.model` when empty.
    ///
    /// Mirrors `FlowConfig.model_for`.
    pub fn model_for(&self, role: ModelRole) -> &str {
        let named = match role {
            ModelRole::Planner => self.models.planner.as_str(),
            ModelRole::Coder => self.models.coder.as_str(),
            ModelRole::Reviewer => self.models.reviewer.as_str(),
        };
        if named.is_empty() {
            self.ollama.model.as_str()
        } else {
            named
        }
    }

    /// Switch the active model, mirroring Python's `/model` command exactly:
    /// assign `ollama.model`, then reset every per-role override to defaults
    /// so all roles inherit the new default. Later runs resolve through
    /// [`FlowConfig::model_for`], so the change takes effect immediately.
    pub fn select_model(&mut self, model: String) -> Result<(), ConfigError> {
        self.ollama.set_model(model)?;
        self.models = ModelsConfig::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_kind_round_trips_all_spellings() {
        for name in CHECK_KINDS {
            let kind = CheckKind::parse(name).expect("known kind");
            assert_eq!(kind.as_str(), *name);
        }
        assert_eq!(CheckKind::parse("pretend"), None);
        assert_eq!(CheckKind::parse("CHECK"), None);
        assert_eq!(CheckKind::parse(""), None);
    }

    #[test]
    fn model_for_inherits_empty_roles() {
        let cfg = FlowConfig::validated(
            OllamaConfig::default(),
            ModelsConfig::validated("small".to_owned(), String::new(), String::new()),
            AgentConfig::default(),
            BTreeMap::new(),
            PathBuf::from("/tmp"),
            false,
            None,
            Vec::new(),
        );
        assert_eq!(cfg.model_for(ModelRole::Planner), "small");
        assert_eq!(cfg.model_for(ModelRole::Coder), "qwen2.5-coder:7b");
        assert_eq!(cfg.model_for(ModelRole::Reviewer), "qwen2.5-coder:7b");
    }

    #[test]
    fn defaults_match_the_python_dataclasses() {
        let ollama = OllamaConfig::default();
        assert_eq!(ollama.base_url(), "http://127.0.0.1:11434");
        assert_eq!(ollama.model(), "qwen2.5-coder:7b");
        assert_eq!(ollama.temperature(), 0.2);
        assert_eq!(ollama.context_length(), 16384);
        assert_eq!(ollama.timeout_secs(), 120.0);
        assert!(!ollama.allow_remote());

        let agent = AgentConfig::default();
        assert_eq!(agent.max_iterations(), 6);
        assert_eq!(agent.max_cycles(), 2);
        assert_eq!(agent.max_tool_calls(), 64);
        assert_eq!(agent.max_context_chars(), 100_000);
        assert_eq!(agent.max_task_chars(), 16_000);

        let models = ModelsConfig::default();
        assert_eq!(models.planner(), "");
        assert_eq!(models.coder(), "");
        assert_eq!(models.reviewer(), "");
    }
}
