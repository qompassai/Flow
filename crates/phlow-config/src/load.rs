//! Loading and validating operator configuration.
//!
//! Trust model, mirroring `flow/config.py`:
//!
//! - An explicit `--config` path approves its contents but grants no
//!   execution trust (`trusted` stays false unless the CLI says otherwise).
//! - Only `$XDG_CONFIG_HOME/flow/config.toml` is auto-loaded. No
//!   workspace-local `config.toml`/`.flow.toml` is ever consumed
//!   automatically; when one exists and was not explicitly selected, it
//!   produces a warning, not configuration.
//! - `trusted` is CLI/API-only. A `trusted` key in TOML is rejected as an
//!   unknown option, exactly like the legacy `shell`/`plugins`/
//!   `self_improve` keys.
//!
//! Validation order mirrors the Python: unknown-key rejection, per-section
//! parsing with bounds, workspace anchoring, `--model` override, then the
//! workspace-must-exist check and project-config warnings.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::error::{ConfigError, invalid_value, io_error, unknown_option};
use crate::model::{
    AGENT_MAX_CONTEXT_CHARS_MAX, AGENT_MAX_CONTEXT_CHARS_MIN, AGENT_MAX_CYCLES_MAX,
    AGENT_MAX_CYCLES_MIN, AGENT_MAX_ITERATIONS_MAX, AGENT_MAX_ITERATIONS_MIN,
    AGENT_MAX_TASK_CHARS_MAX, AGENT_MAX_TASK_CHARS_MIN, AGENT_MAX_TOOL_CALLS_MAX,
    AGENT_MAX_TOOL_CALLS_MIN, AgentConfig, CHECK_ARGV_MAX, CHECK_NAME_CHARS_MAX,
    CHECK_TIMEOUT_MS_DEFAULT, CHECK_TIMEOUT_MS_MAX, CHECK_TIMEOUT_MS_MIN, CHECKS_MAX, CheckConfig,
    CheckKind, FlowConfig, ModelsConfig, OLLAMA_CONTEXT_LENGTH_MAX, OLLAMA_CONTEXT_LENGTH_MIN,
    OLLAMA_TEMPERATURE_MAX, OLLAMA_TEMPERATURE_MIN, OLLAMA_TIMEOUT_SECS_MAX,
    OLLAMA_TIMEOUT_SECS_MIN, OllamaConfig,
};

/// Largest config file accepted, in bytes.
///
/// Operator config is small; a 1 MiB cap bounds the read before parsing.
/// (Python reads the whole file; the cap is a deliberate hardening.)
pub const CONFIG_FILE_BYTES_MAX: u64 = 1_048_576;

/// Root keys the schema defines. Anything else — including `trusted` and the
/// legacy `shell`/`plugins`/`self_improve` keys — is rejected.
const ROOT_KEYS: &[&str] = &["workspace_dir", "ollama", "models", "agent", "checks"];

/// `at` label for root-level unknown options, naming the legacy keys the
/// Python error message calls out.
const ROOT_AT: &str = "config root (legacy shell/plugins/self_improve options are not supported)";

/// Options for [`load_config`]. Every field mirrors a `load_config` keyword
/// argument in `flow/config.py`.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Explicit `--config` path. Approves the file's contents; grants no trust.
    pub config_path: Option<PathBuf>,
    /// `--workspace` override. Wins over the TOML `workspace_dir`.
    pub workspace: Option<PathBuf>,
    /// Execution trust. CLI/API only; never read from TOML.
    pub trusted: bool,
    /// `--model` override. Replaces `ollama.model` and clears every role
    /// override, so all roles inherit the override.
    pub model: Option<String>,
}

/// Load and validate the operator configuration.
///
/// Accepted inputs: an explicit config path, or nothing (auto-loads the XDG
/// default when it exists). Rejected inputs: malformed TOML, unknown keys,
/// out-of-range values, non-loopback Ollama URLs without explicit opt-in,
/// missing workspaces. Rejected validation leaves nothing behind; there is
/// no partial config to observe.
pub fn load_config(options: &LoadOptions) -> Result<FlowConfig, ConfigError> {
    // 1. Resolve which file to read.
    let config_path = match &options.config_path {
        Some(path) => normalize_path(path)?,
        None => default_config_path()?,
    };
    let explicit = options.config_path.is_some();

    // 2. Read and parse. The default path is only read when it exists;
    //    an explicit path must exist.
    let file_exists = std::fs::metadata(&config_path).is_ok();
    let raw: toml::map::Map<String, toml::Value> = if explicit || file_exists {
        read_toml_table(&config_path)?
    } else {
        toml::map::Map::new()
    };
    let config_path_opt = if !raw.is_empty() || file_exists {
        Some(config_path.clone())
    } else {
        None
    };

    // 3. Unknown root keys fail closed (this also rejects `trusted` and the
    //    legacy shell/plugins/self_improve keys).
    reject_unknown_keys(&raw, ROOT_KEYS, ROOT_AT)?;

    // 4. Sections.
    let ollama = parse_ollama(table_section(&raw, "ollama")?)?;
    let models = parse_models(table_section(&raw, "models")?)?;
    let agent = parse_agent(table_section(&raw, "agent")?)?;
    let checks = parse_checks(table_section(&raw, "checks")?)?;

    // 5. Workspace anchoring: config-relative when the TOML names it,
    //    caller-cwd-relative otherwise (including the CLI override).
    let raw_workspace_dir = match raw.get("workspace_dir") {
        None => None,
        Some(value) => {
            // The empty string is accepted, matching Python: `base /
            // Path("")` resolves to `base` (the config file's parent dir)
            // before `validate_config`'s nonempty check ever sees it, so
            // the observable contract is success with the config-relative
            // directory, not a usage error.
            Some(parse_string(value, "workspace_dir")?)
        }
    };
    let home = home_dir();
    let selected = match &options.workspace {
        Some(workspace) => expand_tilde(workspace, home.as_deref())?,
        None => expand_tilde(
            Path::new(raw_workspace_dir.as_deref().unwrap_or(".")),
            home.as_deref(),
        )?,
    };
    let base = if options.workspace.is_none() && raw_workspace_dir.is_some() {
        config_path
            .parent()
            .map_or_else(|| PathBuf::from("/"), std::path::Path::to_path_buf)
    } else {
        std::env::current_dir().map_err(|err| ConfigError::CurrentDir {
            message: err.to_string(),
        })?
    };
    // `PathBuf::push` with an absolute `selected` replaces `base`, exactly
    // like Python's `base / Path(selected)` on an absolute right side.
    let joined = base.join(&selected);
    let workspace_dir =
        std::fs::canonicalize(&joined).map_err(|_| ConfigError::WorkspaceMissing {
            path: joined.clone(),
        })?;
    if !workspace_dir.is_dir() {
        return Err(ConfigError::WorkspaceMissing {
            path: workspace_dir,
        });
    }

    // 6. `--model` override: replaces ollama.model and clears every role.
    if let Some(model) = &options.model
        && model.trim().is_empty()
    {
        return Err(invalid_value("ollama.model", "must be a nonempty string"));
    }
    let ollama = match &options.model {
        Some(model) => ollama.with_model(model.clone()),
        None => ollama,
    };
    let models = if options.model.is_some() {
        ModelsConfig::default()
    } else {
        models
    };

    // 7. Warn about workspace-local configs that were NOT selected.
    let mut warnings = Vec::new();
    for local_name in ["config.toml", ".flow.toml"] {
        let local = workspace_dir.join(local_name);
        if !local.exists() {
            continue;
        }
        let same_as_selected = match (&config_path_opt, std::fs::canonicalize(&local)) {
            (Some(selected_path), Ok(local_canonical)) => std::fs::canonicalize(selected_path)
                .map(|selected_canonical| selected_canonical == local_canonical)
                .unwrap_or(false),
            _ => false,
        };
        if !same_as_selected {
            warnings.push(format!(
                "Ignored project configuration: {}; use --config explicitly",
                local.display()
            ));
        }
    }

    Ok(FlowConfig::validated(
        ollama,
        models,
        agent,
        checks,
        workspace_dir,
        options.trusted,
        config_path_opt,
        warnings,
    ))
}

/// Marker toml-rs places between a source-echo line number and the echoed
/// text (e.g. `1 | <text>`).
const TOML_SOURCE_ECHO_MARKER: char = '|';

/// True for a toml-rs source-echo line: leading whitespace, a line number,
/// optional whitespace, then the [`TOML_SOURCE_ECHO_MARKER`] separator
/// (e.g. `  1 | <text>`).
fn is_toml_source_line(line: &str) -> bool {
    let mut chars = line.trim_start().chars().peekable();
    let mut saw_digit = false;
    while matches!(chars.peek(), Some(c) if c.is_ascii_digit()) {
        chars.next();
        saw_digit = true;
    }
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
    saw_digit && matches!(chars.next(), Some(TOML_SOURCE_ECHO_MARKER))
}

/// True for a toml-rs annotation line: leading whitespace followed only by
/// caret/marker runs (e.g. `  |`, `  |     ^`). Carries no message text.
fn is_toml_annotation_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c.is_whitespace() || matches!(c, '^' | '|' | '='))
}

/// Strip source-echo lines from a toml-rs parse error message.
///
/// toml-rs renders failures with the offending source line (`1 | <text>`)
/// plus caret annotation lines, so storing the message verbatim echoes
/// operator config text into diagnostics. The first line (message plus
/// line/column) is kept; echo and annotation lines are dropped. The error
/// still names the failure location, and the crate's 128-char truncation
/// still bounds the result at display time.
fn strip_source_echo(message: &str) -> String {
    let mut kept = String::new();
    for (index, line) in message.lines().enumerate() {
        if index > 0 && (is_toml_source_line(line) || is_toml_annotation_line(line)) {
            continue;
        }
        if index > 0 {
            kept.push('\n');
        }
        kept.push_str(line);
    }
    kept
}

/// Read a file, cap its size, require UTF-8, and parse it as a TOML table.
fn read_toml_table(path: &Path) -> Result<toml::map::Map<String, toml::Value>, ConfigError> {
    let file = std::fs::File::open(path).map_err(|err| io_error(path.to_path_buf(), &err))?;
    // The cap bounds the read itself: read at most one byte past
    // CONFIG_FILE_BYTES_MAX so an oversized file is detected without ever
    // materializing it.
    let mut limited = file.take(CONFIG_FILE_BYTES_MAX + 1);
    let mut bytes = Vec::new();
    limited
        .read_to_end(&mut bytes)
        .map_err(|err| io_error(path.to_path_buf(), &err))?;
    if bytes.len() as u64 > CONFIG_FILE_BYTES_MAX {
        return Err(ConfigError::Parse {
            path: path.to_path_buf(),
            message: format!("configuration file exceeds {CONFIG_FILE_BYTES_MAX} bytes"),
        });
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| ConfigError::Parse {
        path: path.to_path_buf(),
        message: "configuration file is not valid UTF-8".to_owned(),
    })?;
    let value: toml::Value = toml::from_str(text).map_err(|err| ConfigError::Parse {
        path: path.to_path_buf(),
        message: strip_source_echo(&err.to_string()),
    })?;
    match value {
        toml::Value::Table(table) => Ok(table),
        _ => Err(ConfigError::Parse {
            path: path.to_path_buf(),
            message: "configuration root must be a TOML table".to_owned(),
        }),
    }
}

/// Fetch an optional table section; a present-but-not-a-table value is an error.
fn table_section<'a>(
    raw: &'a toml::map::Map<String, toml::Value>,
    name: &str,
) -> Result<Option<&'a toml::map::Map<String, toml::Value>>, ConfigError> {
    match raw.get(name) {
        None => Ok(None),
        Some(toml::Value::Table(table)) => Ok(Some(table)),
        Some(_) => Err(invalid_value(name, "must be a TOML table")),
    }
}

/// Reject keys outside `known`, reporting them sorted like the Python.
fn reject_unknown_keys(
    table: &toml::map::Map<String, toml::Value>,
    known: &[&str],
    at: &str,
) -> Result<(), ConfigError> {
    let mut unknown: Vec<&str> = table
        .keys()
        .map(String::as_str)
        .filter(|key| !known.contains(key))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    Err(unknown_option(at, unknown.join(", ")))
}

// --- Scalar field parsers -------------------------------------------------
// Each mirrors flow/config.py::_number's strictness: booleans are rejected
// for numeric fields (TOML distinguishes them, as does Python's
// `type(value) not in (int,)` check), NaN/inf are rejected, ranges are
// inclusive.

fn parse_string(value: &toml::Value, field: &str) -> Result<String, ConfigError> {
    match value {
        toml::Value::String(text) => Ok(text.clone()),
        _ => Err(invalid_value(field, "must be a string")),
    }
}

fn parse_nonempty_string(value: &toml::Value, field: &str) -> Result<String, ConfigError> {
    let text = parse_string(value, field)?;
    if text.trim().is_empty() {
        return Err(invalid_value(field, "must be a nonempty string"));
    }
    Ok(text)
}

fn parse_bool(value: &toml::Value, field: &str) -> Result<bool, ConfigError> {
    match value {
        toml::Value::Boolean(flag) => Ok(*flag),
        _ => Err(invalid_value(field, "must be a boolean")),
    }
}

fn parse_bounded_f64(
    value: &toml::Value,
    field: &str,
    min: f64,
    max: f64,
) -> Result<f64, ConfigError> {
    debug_assert!(min <= max, "bound range must not be empty");
    // Python's `_number` uses one message for every failure mode (wrong
    // type, NaN/inf, out of range), so the type rejection below shares the
    // range message instead of inventing a second one.
    let range = format!("must be a number between {min} and {max}");
    let number = match value {
        toml::Value::Integer(int) => *int as f64,
        toml::Value::Float(float) => *float,
        _ => return Err(invalid_value(field, range)),
    };
    if !number.is_finite() || number < min || number > max {
        return Err(invalid_value(field, range));
    }
    Ok(number)
}

fn parse_bounded_u32(
    value: &toml::Value,
    field: &str,
    min: u32,
    max: u32,
) -> Result<u32, ConfigError> {
    debug_assert!(min <= max, "bound range must not be empty");
    // Like `parse_bounded_f64`: Python's `_number(..., integer=True)` uses
    // one message for every failure mode, including a non-integer type.
    let range = format!("must be an integer between {min} and {max}");
    match value {
        toml::Value::Integer(int) => {
            if *int < i64::from(min) || *int > i64::from(max) {
                Err(invalid_value(field, range))
            } else {
                Ok(*int as u32)
            }
        }
        _ => Err(invalid_value(field, range)),
    }
}

fn parse_timeout_ms(value: &toml::Value, field: &str) -> Result<u64, ConfigError> {
    // Python's `_number(..., integer=True)` uses one message for every
    // failure mode, including a non-integer type.
    let range =
        format!("must be an integer between {CHECK_TIMEOUT_MS_MIN} and {CHECK_TIMEOUT_MS_MAX}");
    match value {
        toml::Value::Integer(int) => {
            if *int < CHECK_TIMEOUT_MS_MIN as i64 || *int > CHECK_TIMEOUT_MS_MAX as i64 {
                Err(invalid_value(field, range))
            } else {
                Ok(*int as u64)
            }
        }
        _ => Err(invalid_value(field, range)),
    }
}

fn parse_string_array(value: &toml::Value, field: &str) -> Result<Vec<String>, ConfigError> {
    let items = match value {
        toml::Value::Array(items) => items,
        _ => return Err(invalid_value(field, "must be a string array")),
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            toml::Value::String(text) => out.push(text.clone()),
            _ => return Err(invalid_value(field, "must be a string array")),
        }
    }
    Ok(out)
}

/// Parse check argv: a nonempty string array, at most [`CHECK_ARGV_MAX`]
/// elements, no empty args, no NUL bytes.
fn parse_argv(value: &toml::Value, field: &str) -> Result<Vec<String>, ConfigError> {
    let not_argv = || invalid_value(field, "must be a nonempty string argv array");
    let items = match value {
        toml::Value::Array(items) => items,
        _ => return Err(not_argv()),
    };
    if items.is_empty() || items.len() > CHECK_ARGV_MAX {
        return Err(not_argv());
    }
    let mut argv = Vec::with_capacity(items.len());
    for item in items {
        match item {
            toml::Value::String(arg) if !arg.is_empty() && !arg.contains('\0') => {
                argv.push(arg.clone());
            }
            _ => return Err(not_argv()),
        }
    }
    Ok(argv)
}

// --- Section parsers --------------------------------------------------------

fn parse_ollama(
    table: Option<&toml::map::Map<String, toml::Value>>,
) -> Result<OllamaConfig, ConfigError> {
    const KNOWN: &[&str] = &[
        "base_url",
        "model",
        "temperature",
        "context_length",
        "timeout",
        "allow_remote",
    ];
    let defaults = OllamaConfig::default();
    let Some(table) = table else {
        return Ok(defaults);
    };
    reject_unknown_keys(table, KNOWN, "ollama")?;

    let base_url = match table.get("base_url") {
        Some(toml::Value::String(text)) => text.clone(),
        // Python: "ollama.base_url must be a URL string". An empty string
        // is not rejected here: it flows into `validate_base_url`, which
        // reports it as `Invalid ollama.base_url: ...` like Python's
        // urlsplit validation does.
        Some(_) => return Err(invalid_value("ollama.base_url", "must be a URL string")),
        None => defaults.base_url().to_owned(),
    };
    let model = match table.get("model") {
        Some(value) => parse_nonempty_string(value, "ollama.model")?,
        None => defaults.model().to_owned(),
    };
    let temperature = match table.get("temperature") {
        Some(value) => parse_bounded_f64(
            value,
            "ollama.temperature",
            OLLAMA_TEMPERATURE_MIN,
            OLLAMA_TEMPERATURE_MAX,
        )?,
        None => defaults.temperature(),
    };
    let context_length = match table.get("context_length") {
        Some(value) => parse_bounded_u32(
            value,
            "ollama.context_length",
            OLLAMA_CONTEXT_LENGTH_MIN,
            OLLAMA_CONTEXT_LENGTH_MAX,
        )?,
        None => defaults.context_length(),
    };
    let timeout_secs = match table.get("timeout") {
        Some(value) => parse_bounded_f64(
            value,
            "ollama.timeout",
            OLLAMA_TIMEOUT_SECS_MIN,
            OLLAMA_TIMEOUT_SECS_MAX,
        )?,
        None => defaults.timeout_secs(),
    };
    let allow_remote = match table.get("allow_remote") {
        Some(toml::Value::Boolean(flag)) => *flag,
        // Python validates both trust flags together:
        // "trust flags must be booleans".
        Some(_) => return Err(invalid_value("trust flags", "must be booleans")),
        None => defaults.allow_remote(),
    };

    validate_base_url(&base_url, allow_remote)?;
    Ok(OllamaConfig::validated(
        base_url,
        model,
        temperature,
        context_length,
        timeout_secs,
        allow_remote,
    ))
}

fn parse_models(
    table: Option<&toml::map::Map<String, toml::Value>>,
) -> Result<ModelsConfig, ConfigError> {
    const KNOWN: &[&str] = &["planner", "coder", "reviewer"];
    let Some(table) = table else {
        return Ok(ModelsConfig::default());
    };
    reject_unknown_keys(table, KNOWN, "models")?;
    let planner = match table.get("planner") {
        Some(value) => parse_string(value, "models.planner")?,
        None => String::new(),
    };
    let coder = match table.get("coder") {
        Some(value) => parse_string(value, "models.coder")?,
        None => String::new(),
    };
    let reviewer = match table.get("reviewer") {
        Some(value) => parse_string(value, "models.reviewer")?,
        None => String::new(),
    };
    Ok(ModelsConfig::validated(planner, coder, reviewer))
}

fn parse_agent(
    table: Option<&toml::map::Map<String, toml::Value>>,
) -> Result<AgentConfig, ConfigError> {
    const KNOWN: &[&str] = &[
        "max_iterations",
        "max_cycles",
        "max_tool_calls",
        "max_context_chars",
        "max_task_chars",
    ];
    let defaults = AgentConfig::default();
    let Some(table) = table else {
        return Ok(defaults);
    };
    reject_unknown_keys(table, KNOWN, "agent")?;
    let get = |name: &str, min: u32, max: u32, fallback: u32| -> Result<u32, ConfigError> {
        match table.get(name) {
            Some(value) => parse_bounded_u32(value, &format!("agent.{name}"), min, max),
            None => Ok(fallback),
        }
    };
    Ok(AgentConfig::validated(
        get(
            "max_iterations",
            AGENT_MAX_ITERATIONS_MIN,
            AGENT_MAX_ITERATIONS_MAX,
            defaults.max_iterations(),
        )?,
        get(
            "max_cycles",
            AGENT_MAX_CYCLES_MIN,
            AGENT_MAX_CYCLES_MAX,
            defaults.max_cycles(),
        )?,
        get(
            "max_tool_calls",
            AGENT_MAX_TOOL_CALLS_MIN,
            AGENT_MAX_TOOL_CALLS_MAX,
            defaults.max_tool_calls(),
        )?,
        get(
            "max_context_chars",
            AGENT_MAX_CONTEXT_CHARS_MIN,
            AGENT_MAX_CONTEXT_CHARS_MAX,
            defaults.max_context_chars(),
        )?,
        get(
            "max_task_chars",
            AGENT_MAX_TASK_CHARS_MIN,
            AGENT_MAX_TASK_CHARS_MAX,
            defaults.max_task_chars(),
        )?,
    ))
}

fn parse_checks(
    table: Option<&toml::map::Map<String, toml::Value>>,
) -> Result<BTreeMap<String, CheckConfig>, ConfigError> {
    let mut checks = BTreeMap::new();
    let Some(table) = table else {
        return Ok(checks);
    };
    if table.len() > CHECKS_MAX {
        // Python: f"At most {CHECKS_MAX} named checks are supported". The
        // sentence is split across field/reason so the "{field} {reason}"
        // display renders it byte-identically (a "checks ..." field prefix
        // here would read as the malformed "checks at most ...").
        return Err(invalid_value(
            "At most",
            format!("{CHECKS_MAX} named checks are supported"),
        ));
    }
    for (name, value) in table {
        let check_table = match value {
            toml::Value::Table(inner) => inner,
            _ => {
                return Err(invalid_value(
                    format!("checks.{name}"),
                    "must be a check table",
                ));
            }
        };
        checks.insert(name.clone(), parse_check(name, check_table)?);
    }
    Ok(checks)
}

fn parse_check(
    name: &str,
    table: &toml::map::Map<String, toml::Value>,
) -> Result<CheckConfig, ConfigError> {
    const KNOWN: &[&str] = &["cmd", "timeout", "required", "filetypes", "kind"];
    if !valid_check_name(name) {
        // Python: f"Invalid check name: {name!r}". The colon lives at the
        // end of the field so the "{field} {reason}" display renders the
        // message byte-identically, with Python-repr single quotes.
        return Err(invalid_value(
            "Invalid check name:",
            crate::error::repr_str(name),
        ));
    }
    let at = format!("checks.{name}");
    reject_unknown_keys(table, KNOWN, &at)?;

    let cmd = match table.get("cmd") {
        Some(value) => parse_argv(value, &format!("checks.{name}.cmd"))?,
        None => {
            return Err(invalid_value(
                format!("checks.{name}.cmd"),
                "must be a nonempty string argv array",
            ));
        }
    };
    let timeout_ms = match table.get("timeout") {
        Some(value) => parse_timeout_ms(value, &format!("checks.{name}.timeout"))?,
        None => CHECK_TIMEOUT_MS_DEFAULT,
    };
    let required = match table.get("required") {
        Some(value) => parse_bool(value, &format!("checks.{name}.required"))?,
        None => true,
    };
    let filetypes = match table.get("filetypes") {
        Some(value) => parse_string_array(value, &format!("checks.{name}.filetypes"))?,
        None => Vec::new(),
    };
    let kind = match table.get("kind") {
        Some(value) => {
            let kind_name = parse_string(value, &format!("checks.{name}.kind"))?;
            CheckKind::parse(&kind_name).ok_or_else(|| {
                invalid_value(
                    format!("checks.{name}.kind"),
                    "is not a supported check kind",
                )
            })?
        }
        None => CheckKind::Check,
    };

    Ok(CheckConfig::new(cmd, timeout_ms, required, filetypes, kind))
}

/// Check names match `[A-Za-z0-9][A-Za-z0-9_.-]*` (full match) and are at most
/// [`CHECK_NAME_CHARS_MAX`] characters. Hand-rolled: no regex crate needed
/// for a character class this small.
fn valid_check_name(name: &str) -> bool {
    if name.chars().count() > CHECK_NAME_CHARS_MAX {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

// --- URL validation ---------------------------------------------------------
// Mirrors flow/config.py's urlsplit-based checks: http/https origin only, no
// credentials/query/fragment/non-root path, and non-loopback hosts require
// explicit allow_remote.

fn validate_base_url(raw: &str, allow_remote: bool) -> Result<(), ConfigError> {
    let (scheme, rest) = raw
        .split_once("://")
        .ok_or_else(|| invalid_base_url("must be an HTTP(S) origin URL"))?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return Err(invalid_base_url("scheme must be http or https"));
    }
    // Query or fragment anywhere is rejected (Python checks url.query/url.fragment).
    if rest.contains(['?', '#']) {
        return Err(invalid_base_url(
            "must not carry credentials, query, or fragment",
        ));
    }
    let (authority, path) = match rest.find('/') {
        Some(index) => (&rest[..index], &rest[index..]),
        None => (rest, ""),
    };
    if !path.is_empty() && path != "/" {
        return Err(invalid_base_url("must be an origin: no path beyond \"/\""));
    }
    // Any '@' means userinfo (Python checks url.username/url.password).
    if authority.contains('@') {
        return Err(invalid_base_url(
            "must not carry credentials, query, or fragment",
        ));
    }
    let (host, _port) = split_host_port(authority)?;
    if host.is_empty() {
        return Err(invalid_base_url("must have a hostname"));
    }
    let lower = host.to_ascii_lowercase();
    let loopback = match lower.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => lower == "localhost",
    };
    if !loopback && !allow_remote {
        return Err(invalid_base_url(
            "non-loopback Ollama requires explicit ollama.allow_remote=true",
        ));
    }
    Ok(())
}

/// Build the `Invalid ollama.base_url: <reason>` error, matching Python's
/// `ConfigError(f"Invalid ollama.base_url: {exc}")` shape. The reasons stay
/// granular (each Rust check names its own failure); only the prefix is
/// Python's. The colon lives at the end of the field so the
/// `"{field} {reason}"` display renders it byte-identically.
fn invalid_base_url(reason: impl Into<String>) -> ConfigError {
    invalid_value("Invalid ollama.base_url:", reason)
}

/// Split an authority into host and optional port.
///
/// Handles `host`, `host:port`, `[::1]`, and `[::1]:port`. The port must be
/// numeric and fit in `u16` (Python's `url.port` raises `ValueError` outside
/// 0–65535, and on a missing or non-numeric port after a colon). A colon with
/// a missing or non-numeric port is rejected here, fail-closed, rather than
/// being smuggled through as part of the hostname.
fn split_host_port(authority: &str) -> Result<(&str, Option<u16>), ConfigError> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, after) = rest
            .split_once(']')
            .ok_or_else(|| invalid_base_url("malformed IPv6 authority"))?;
        if after.is_empty() {
            return Ok((host, None));
        }
        let digits = after
            .strip_prefix(':')
            .ok_or_else(|| invalid_base_url("malformed IPv6 authority"))?;
        return Ok((host, Some(parse_port(digits)?)));
    }
    match authority.rsplit_once(':') {
        Some((host, digits))
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) =>
        {
            Ok((host, Some(parse_port(digits)?)))
        }
        // A colon is present but the port part is missing or non-numeric
        // (e.g. `host:` or `host:abc`): Python's `url.port` raises here, so
        // reject instead of accepting the whole authority as a bare host.
        Some((_host, _digits)) => Err(invalid_base_url("port out of range 0-65535")),
        None => Ok((authority, None)),
    }
}

fn parse_port(digits: &str) -> Result<u16, ConfigError> {
    digits
        .parse::<u16>()
        .map_err(|_| invalid_base_url("port out of range 0-65535"))
}

// --- Path handling ----------------------------------------------------------

/// `$HOME`, when set.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Expand a leading `~` (only `~` or `~/...`, via `$HOME`).
///
/// `~user` forms are left untouched and documented as unsupported: resolving
/// other users' homes would reach outside the operator's own directories.
fn expand_tilde(path: &Path, home: Option<&Path>) -> Result<PathBuf, ConfigError> {
    let mut components = path.components();
    match components.next() {
        Some(Component::Normal(first)) if first == "~" => {
            let home = home.ok_or(ConfigError::NoHomeDirectory)?;
            Ok(home.join(components.as_path()))
        }
        _ => Ok(path.to_path_buf()),
    }
}

/// Make a path absolute against the process cwd and resolve `.`/`..`
/// lexically, without touching the filesystem.
///
/// Python uses `Path.resolve()` (non-strict): for paths that exist it may
/// also resolve symlinks, but this function never fails on missing paths.
/// The workspace directory itself is canonicalized strictly later, so the
/// security-relevant resolution still happens where it matters.
fn lexical_absolute(path: &Path) -> Result<PathBuf, ConfigError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let cwd = std::env::current_dir().map_err(|err| ConfigError::CurrentDir {
            message: err.to_string(),
        })?;
        cwd.join(path)
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            _ => out.push(component.as_os_str()),
        }
    }
    Ok(out)
}

/// Explicit config paths: `~`-expand, then absolutize.
fn normalize_path(path: &Path) -> Result<PathBuf, ConfigError> {
    let expanded = expand_tilde(path, home_dir().as_deref())?;
    lexical_absolute(&expanded)
}

/// Pure XDG default-path resolution, split out so tests never mutate the
/// process environment (`std::env::set_var` is `unsafe` in edition 2024).
///
/// Deviation from Python, documented: an empty `XDG_CONFIG_HOME` falls back
/// to the default per the XDG spec; Python would use the empty string.
fn xdg_default_config_path(
    xdg_config_home: Option<&str>,
    home: Option<&Path>,
) -> Result<PathBuf, ConfigError> {
    let base: PathBuf = match xdg_config_home {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => home.ok_or(ConfigError::NoHomeDirectory)?.join(".config"),
    };
    lexical_absolute(&base.join("flow").join("config.toml"))
}

/// Read `$XDG_CONFIG_HOME` / `$HOME` from the environment and resolve the
/// default config path.
fn default_config_path() -> Result<PathBuf, ConfigError> {
    let xdg = std::env::var_os("XDG_CONFIG_HOME");
    let xdg_str = xdg.as_deref().and_then(|s| s.to_str());
    xdg_default_config_path(xdg_str, home_dir().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Fresh unique temp dir per test (tests run in parallel threads).
    fn test_dir(name: &str) -> PathBuf {
        let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "phlow-config-test-{}-{}-{id}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test temp dir");
        dir
    }

    fn write_config(dir: &Path, name: &str, text: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, text).expect("write test config");
        path
    }

    fn load_text(dir: &Path, text: &str) -> Result<FlowConfig, ConfigError> {
        let path = write_config(dir, "test.toml", text);
        load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
    }

    fn load_ok(dir: &Path, text: &str) -> FlowConfig {
        load_text(dir, text).expect("test config must load")
    }

    #[test]
    fn empty_config_loads_all_defaults() {
        let dir = test_dir("defaults");
        let cfg = load_ok(&dir, "");
        assert_eq!(cfg.ollama().base_url(), "http://127.0.0.1:11434");
        assert_eq!(cfg.ollama().model(), "qwen2.5-coder:7b");
        assert_eq!(cfg.agent().max_iterations(), 6);
        assert!(cfg.checks().is_empty());
        assert!(!cfg.trusted());
        assert!(cfg.warnings().is_empty());
        assert!(cfg.config_path().is_some());
        assert_eq!(cfg.workspace_dir(), &dir.canonicalize().unwrap());
    }

    #[test]
    fn loads_the_repos_real_config_toml() {
        // The repo's own config.toml must parse: it is the live contract.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest.parent().and_then(|p| p.parent()).unwrap();
        let config_path = repo_root.join("config.toml");
        assert!(config_path.exists(), "repo config.toml must exist");
        let dir = test_dir("real-config");
        let cfg = load_config(&LoadOptions {
            config_path: Some(config_path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .expect("repo config.toml must load");
        assert_eq!(cfg.ollama().model(), "qwen2.5-coder:7b");
        assert_eq!(cfg.agent().max_iterations(), 6);
        assert_eq!(cfg.agent().max_tool_calls(), 64);
        let names: Vec<&str> = cfg.checks().keys().map(String::as_str).collect();
        assert!(names.contains(&"tests"));
        assert!(names.contains(&"lint"));
        let tests = &cfg.checks()["tests"];
        assert_eq!(tests.kind(), CheckKind::Test);
        assert!(tests.required());
        assert_eq!(tests.timeout_ms(), 120_000);
    }

    #[test]
    fn unknown_root_keys_fail_closed() {
        let dir = test_dir("unknown-root");
        // Mirrors tests/test_security_config.py's invalid-config parametrize list.
        // Root-level unknown keys name the legacy keys in the message.
        let bad_root = [
            "trusted=true\n",
            "[tools]\nshell_allowed_commands=[\"python\"]\n",
            "shell=true\n",
            "plugins=true\n",
            "self_improve=true\n",
            "[tui]\nworkspace_dir=\".\"\n",
        ];
        for text in bad_root {
            let err = load_text(&dir, text).expect_err("unknown key must fail");
            let message = err.to_string();
            assert!(message.contains("unknown option"), "{text}: {message}");
            assert!(message.contains("not supported"), "{text}: {message}");
        }
        // Section-level unknown keys fail too, without the legacy note
        // (matching the Python's "Unknown {section} options" message).
        let err = load_text(&dir, "[ollama]\nworkspace_dir=\".\"\n")
            .expect_err("unknown section key must fail");
        assert!(err.to_string().contains("unknown option"));
    }

    #[test]
    fn unknown_section_keys_fail_closed() {
        let dir = test_dir("unknown-section");
        let err = load_text(&dir, "[checks.evil]\ncmd=[\"python\"]\ncwd=\"/etc\"\n")
            .expect_err("unknown check key must fail");
        assert!(err.to_string().contains("cwd"));
    }

    #[test]
    fn check_argv_must_be_a_nonempty_string_array() {
        let dir = test_dir("check-argv");
        let bad = [
            "[checks.evil]\ncmd=\"python -c print(1)\"\n", // string, not array
            "[checks.evil]\ncmd=[]\n",                     // empty
            "[checks.evil]\ncmd=[\"python\", 42]\n",       // non-string element
            "[checks.evil]\ncmd=[\"\"]\n",                 // empty arg
        ];
        for text in bad {
            load_text(&dir, text).expect_err("bad argv must fail");
        }
        let too_many = format!(
            "[checks.evil]\ncmd=[{}]\n",
            (0..CHECK_ARGV_MAX + 1)
                .map(|i| format!("\"arg{i}\""))
                .collect::<Vec<_>>()
                .join(",")
        );
        load_text(&dir, &too_many).expect_err("129 argv elements must fail");
        let max_ok = format!(
            "[checks.evil]\ncmd=[{}]\n",
            (0..CHECK_ARGV_MAX)
                .map(|i| format!("\"arg{i}\""))
                .collect::<Vec<_>>()
                .join(",")
        );
        let cfg = load_ok(&dir, &max_ok);
        assert_eq!(cfg.checks()["evil"].cmd().len(), CHECK_ARGV_MAX);
    }

    #[test]
    fn check_scalar_validation() {
        let dir = test_dir("check-scalars");
        let bad = [
            "[checks.evil]\ncmd=[\"python\"]\ntimeout=-1\n",
            "[checks.evil]\ncmd=[\"python\"]\ntimeout=0\n",
            "[checks.evil]\ncmd=[\"python\"]\ntimeout=600001\n",
            "[checks.evil]\ncmd=[\"python\"]\ntimeout=1.5\n",
            "[checks.evil]\ncmd=[\"python\"]\nrequired=\"yes\"\n",
            "[checks.evil]\ncmd=[\"python\"]\nkind=\"pretend\"\n",
            "[checks.evil]\ncmd=[\"python\"]\nkind=\"CHECK\"\n",
            "[checks.evil]\ncmd=[\"python\"]\nfiletypes=\"python\"\n",
            "[checks.evil]\ncmd=[\"python\"]\nfiletypes=[42]\n",
        ];
        for text in bad {
            load_text(&dir, text).expect_err("bad check scalar must fail");
        }
        // Boundary timeouts are accepted.
        let cfg = load_ok(
            &dir,
            "[checks.a]\ncmd=[\"x\"]\ntimeout=1\n[checks.b]\ncmd=[\"x\"]\ntimeout=600000\n",
        );
        assert_eq!(cfg.checks()["a"].timeout_ms(), 1);
        assert_eq!(cfg.checks()["b"].timeout_ms(), 600_000);
    }

    #[test]
    fn check_name_rules() {
        let dir = test_dir("check-names");
        let bad = ["-bad", ".bad", "bad name", "", "ünicode"];
        for name in bad {
            let text = format!("[checks.\"{name}\"]\ncmd=[\"x\"]\n");
            load_text(&dir, &text).expect_err("bad check name must fail");
        }
        let long = "a".repeat(CHECK_NAME_CHARS_MAX + 1);
        load_text(&dir, &format!("[checks.\"{long}\"]\ncmd=[\"x\"]\n"))
            .expect_err("65-char name must fail");
        // Dotted names must be quoted in TOML; unquoted dots nest tables.
        let cfg = load_ok(
            &dir,
            "[checks.\"ok-name_1.x\"]\ncmd=[\"x\"]\nkind=\"lint\"\nrequired=false\nfiletypes=[\"rust\"]\n",
        );
        let check = &cfg.checks()["ok-name_1.x"];
        assert_eq!(check.kind(), CheckKind::Lint);
        assert!(!check.required());
        assert_eq!(check.filetypes(), &["rust".to_owned()]);
        assert_eq!(check.timeout_ms(), 60_000);
    }

    #[test]
    fn at_most_32_checks() {
        let dir = test_dir("checks-max");
        let mut text = String::new();
        for i in 0..=CHECKS_MAX {
            text.push_str(&format!("[checks.c{i}]\ncmd=[\"x\"]\n"));
        }
        load_text(&dir, &text).expect_err("33 checks must fail");
        let mut text = String::new();
        for i in 0..CHECKS_MAX {
            text.push_str(&format!("[checks.c{i}]\ncmd=[\"x\"]\n"));
        }
        assert_eq!(load_ok(&dir, &text).checks().len(), CHECKS_MAX);
    }

    #[test]
    fn agent_bounds_reject_bools_and_out_of_range() {
        let dir = test_dir("agent-bounds");
        // Mirrors the Python parametrize cases.
        load_text(&dir, "[agent]\nmax_iterations=true\n").expect_err("bool is not an integer");
        load_text(&dir, "[agent]\nmax_cycles=1000\n").expect_err("max_cycles=1000 must fail");
        load_text(&dir, "[agent]\nmax_iterations=0\n").expect_err("0 must fail");
        load_text(&dir, "[agent]\nmax_iterations=33\n").expect_err("33 must fail");
        load_text(&dir, "[agent]\nmax_tool_calls=257\n").expect_err("257 must fail");
        load_text(&dir, "[agent]\nmax_context_chars=4095\n").expect_err("4095 must fail");
        load_text(&dir, "[agent]\nmax_context_chars=1000001\n").expect_err("1000001 must fail");
        load_text(&dir, "[agent]\nmax_task_chars=0\n").expect_err("0 must fail");
        load_text(&dir, "[agent]\nmax_task_chars=100001\n").expect_err("100001 must fail");
        load_text(&dir, "[agent]\nmax_iterations=1.0\n").expect_err("float is not an integer");
        let cfg = load_ok(
            &dir,
            "[agent]\nmax_iterations=32\nmax_cycles=5\nmax_tool_calls=256\nmax_context_chars=1000000\nmax_task_chars=100000\n",
        );
        assert_eq!(cfg.agent().max_iterations(), 32);
        assert_eq!(cfg.agent().max_cycles(), 5);
        assert_eq!(cfg.agent().max_tool_calls(), 256);
        assert_eq!(cfg.agent().max_context_chars(), 1_000_000);
        assert_eq!(cfg.agent().max_task_chars(), 100_000);
    }

    #[test]
    fn ollama_url_rules() {
        let dir = test_dir("ollama-url");
        // Mirrors tests/test_security_config.py's invalid-config cases.
        let bad = [
            "http://evil.example",
            "http://127.0.0.1@evil.example",
            "http://127.0.0.1:11434/redirect",
            "http://127.0.0.1:11434/?q=1",
            "http://127.0.0.1:11434/#frag",
            "ftp://127.0.0.1:11434",
            "not-a-url",
            "http://",
            "http://127.0.0.1:99999",
            "http://[::1",
        ];
        for url in bad {
            let text = format!("[ollama]\nbase_url=\"{url}\"\n");
            load_text(&dir, &text).expect_err("bad base_url must fail: {url}");
        }
        // Loopback forms are accepted without opt-in.
        for url in [
            "http://127.0.0.1:11434",
            "http://127.0.0.1:11434/",
            "http://localhost:11434",
            "http://[::1]:11434",
            "https://127.0.0.1/",
            "HTTP://127.0.0.1:11434",
        ] {
            let text = format!("[ollama]\nbase_url=\"{url}\"\n");
            let cfg = load_ok(&dir, &text);
            assert!(!cfg.ollama().allow_remote(), "{url}");
        }
        // Non-loopback requires explicit opt-in, exactly like the Python test.
        let text = "[ollama]\nbase_url=\"https://private-ollama.example\"\nallow_remote=true\n";
        let cfg = load_ok(&dir, text);
        assert!(cfg.ollama().allow_remote());
        assert_eq!(cfg.ollama().base_url(), "https://private-ollama.example");
    }

    #[test]
    fn ollama_numeric_bounds() {
        let dir = test_dir("ollama-numeric");
        load_text(&dir, "[ollama]\ntimeout=nan\n").expect_err("NaN timeout must fail");
        load_text(&dir, "[ollama]\ntimeout=inf\n").expect_err("inf timeout must fail");
        load_text(&dir, "[ollama]\ntemperature=true\n").expect_err("bool temperature must fail");
        load_text(&dir, "[ollama]\ntemperature=2.1\n").expect_err("temperature > 2 must fail");
        load_text(&dir, "[ollama]\ntemperature=-0.1\n").expect_err("temperature < 0 must fail");
        load_text(&dir, "[ollama]\ntimeout=0.09\n").expect_err("timeout < 0.1 must fail");
        load_text(&dir, "[ollama]\ntimeout=601\n").expect_err("timeout > 600 must fail");
        load_text(&dir, "[ollama]\ncontext_length=1023\n")
            .expect_err("context_length < 1024 must fail");
        load_text(&dir, "[ollama]\ncontext_length=131073\n").expect_err("too large must fail");
        load_text(&dir, "[ollama]\ncontext_length=1.5\n")
            .expect_err("float context_length must fail");
        load_text(&dir, "[ollama]\nallow_remote=\"yes\"\n").expect_err("string bool must fail");
        let cfg = load_ok(
            &dir,
            "[ollama]\ntemperature=2\ncontext_length=1024\ntimeout=600\n",
        );
        assert_eq!(cfg.ollama().temperature(), 2.0);
        assert_eq!(cfg.ollama().context_length(), 1024);
        assert_eq!(cfg.ollama().timeout_secs(), 600.0);
    }

    #[test]
    fn explicit_config_does_not_grant_trust() {
        let dir = test_dir("trust");
        let path = write_config(&dir, "operator.toml", "workspace_dir=\".\"\n");
        let plain = load_config(&LoadOptions {
            config_path: Some(path.clone()),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .unwrap();
        assert!(!plain.trusted());
        let trusted = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            trusted: true,
            ..LoadOptions::default()
        })
        .unwrap();
        assert!(trusted.trusted());
    }

    #[test]
    fn model_override_replaces_every_role() {
        let dir = test_dir("model-override");
        let text = "[models]\nplanner=\"small\"\ncoder=\"large\"\n";
        let path = write_config(&dir, "operator.toml", text);
        let cfg = load_config(&LoadOptions {
            config_path: Some(path.clone()),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .unwrap();
        assert_eq!(cfg.model_for(crate::model::ModelRole::Planner), "small");
        assert_eq!(cfg.model_for(crate::model::ModelRole::Coder), "large");
        assert_eq!(
            cfg.model_for(crate::model::ModelRole::Reviewer),
            "qwen2.5-coder:7b"
        );
        let overridden = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            model: Some("override".to_owned()),
            ..LoadOptions::default()
        })
        .unwrap();
        // --model deliberately clears every role override.
        assert_eq!(
            overridden.model_for(crate::model::ModelRole::Planner),
            "override"
        );
        assert_eq!(
            overridden.model_for(crate::model::ModelRole::Reviewer),
            "override"
        );
        load_config(&LoadOptions {
            config_path: Some(write_config(&dir, "empty-model.toml", "")),
            workspace: Some(dir.to_path_buf()),
            model: Some("  ".to_owned()),
            ..LoadOptions::default()
        })
        .expect_err("empty --model must fail");
    }

    #[test]
    fn workspace_is_config_relative_not_process_cwd() {
        // Mirrors the Python test of the same name.
        let dir = test_dir("anchoring");
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();
        let path = write_config(&dir, "operator.toml", "workspace_dir=\"project\"\n");
        let cfg = load_config(&LoadOptions {
            config_path: Some(path),
            ..LoadOptions::default()
        })
        .unwrap();
        assert_eq!(cfg.workspace_dir(), &project.canonicalize().unwrap());
    }

    #[test]
    fn workspace_override_wins_over_toml() {
        let dir = test_dir("override-wins");
        let other = test_dir("override-other");
        let path = write_config(&dir, "operator.toml", "workspace_dir=\".\"\n");
        let cfg = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(other.clone()),
            ..LoadOptions::default()
        })
        .unwrap();
        assert_eq!(cfg.workspace_dir(), &other.canonicalize().unwrap());
    }

    #[test]
    fn missing_explicit_config_is_an_error() {
        let dir = test_dir("missing-config");
        let err = load_config(&LoadOptions {
            config_path: Some(dir.join("nope.toml")),
            ..LoadOptions::default()
        })
        .expect_err("missing explicit config must fail");
        assert!(matches!(err, ConfigError::Io { .. }));
    }

    #[test]
    fn missing_config_renders_cpython_errno_shape() {
        // Python: "Cannot load configuration /nonexistent.toml: [Errno 2]
        // No such file or directory: '/nonexistent.toml'".
        let dir = test_dir("missing-config-errno");
        let missing = dir.join("nope.toml");
        let err = load_config(&LoadOptions {
            config_path: Some(missing.clone()),
            ..LoadOptions::default()
        })
        .expect_err("missing explicit config must fail");
        let text = err.to_string();
        let missing_str = missing.to_string_lossy();
        assert!(
            text.starts_with(&format!(
                "Cannot load configuration {missing_str}: [Errno 2]"
            )),
            "unexpected: {text}"
        );
        assert!(
            text.ends_with(&format!(": '{missing_str}'")),
            "unexpected: {text}"
        );
        assert!(text.contains("No such file or directory"));
        assert!(
            !text.contains("(os error 2)"),
            "errno printed twice: {text}"
        );
    }

    #[test]
    fn directory_as_config_renders_cpython_errno_shape() {
        // Python: "Cannot load configuration /tmp: [Errno 21] Is a
        // directory: '/tmp'". Opening a directory for reading fails with
        // EISDIR on Linux.
        let dir = test_dir("directory-config-errno");
        let err = load_config(&LoadOptions {
            config_path: Some(dir.clone()),
            ..LoadOptions::default()
        })
        .expect_err("directory as config must fail");
        let text = err.to_string();
        let dir_str = dir.to_string_lossy();
        assert!(
            text.starts_with(&format!("Cannot load configuration {dir_str}: [Errno 21]")),
            "unexpected: {text}"
        );
        assert!(
            text.ends_with(&format!(": '{dir_str}'")),
            "unexpected: {text}"
        );
        assert!(text.contains("Is a directory"));
    }

    #[test]
    fn current_dir_failure_is_not_a_config_load_error() {
        // A `current_dir()` failure must not be misreported as
        // "Cannot load configuration ...".
        let err = ConfigError::CurrentDir {
            message: "Too many open files (os error 24)".to_owned(),
        };
        let text = err.to_string();
        assert!(
            !text.contains("Cannot load configuration"),
            "unexpected: {text}"
        );
        assert!(text.contains("current directory"));
    }

    #[test]
    fn missing_workspace_is_an_error() {
        let dir = test_dir("missing-workspace");
        // No CLI override here: the TOML workspace_dir is config-relative,
        // so "nope" resolves under the config dir and does not exist.
        let path = write_config(&dir, "test.toml", "workspace_dir=\"nope\"\n");
        let err = load_config(&LoadOptions {
            config_path: Some(path),
            ..LoadOptions::default()
        })
        .expect_err("missing workspace must fail");
        assert!(matches!(err, ConfigError::WorkspaceMissing { .. }));
    }

    #[test]
    fn missing_cli_workspace_override_is_an_error() {
        // The CLI override is caller-cwd-relative; a missing dir still
        // fails closed even though the empty string is now accepted.
        let dir = test_dir("missing-cli-workspace");
        let path = write_config(&dir, "test.toml", "");
        let err = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(PathBuf::from("definitely-not-here")),
            ..LoadOptions::default()
        })
        .expect_err("missing CLI workspace must fail");
        assert!(matches!(err, ConfigError::WorkspaceMissing { .. }));
    }

    #[test]
    fn workspace_dir_type_rules() {
        let dir = test_dir("workspace-type");
        load_text(&dir, "workspace_dir=123\n").expect_err("non-string workspace_dir must fail");
        // Python parity (verified 2026-09-26): `workspace_dir = ""`
        // resolves to the config file's parent dir — it is not an error.
        // No CLI override here, so the TOML value is config-relative.
        let path = write_config(&dir, "test.toml", "workspace_dir=\"\"\n");
        let cfg = load_config(&LoadOptions {
            config_path: Some(path),
            ..LoadOptions::default()
        })
        .expect("empty workspace_dir must resolve to the config parent");
        assert_eq!(cfg.workspace_dir(), &dir.canonicalize().unwrap());
    }

    #[test]
    fn malformed_toml_and_non_utf8_fail_closed() {
        let dir = test_dir("malformed");
        let err = load_text(&dir, "[ollama\n").expect_err("malformed TOML must fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
        let path = dir.join("bad-utf8.toml");
        std::fs::write(&path, [0x5b, 0xff, 0x5d]).unwrap();
        let err = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .expect_err("non-UTF8 must fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn oversized_config_file_is_rejected() {
        let dir = test_dir("oversized");
        let path = dir.join("big.toml");
        let big = "x".repeat(CONFIG_FILE_BYTES_MAX as usize + 1);
        std::fs::write(&path, big).unwrap();
        let err = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .expect_err("oversized config must fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn project_configs_are_ignored_with_a_warning() {
        let dir = test_dir("project-warning");
        // A workspace-local config.toml that was NOT explicitly selected.
        std::fs::write(dir.join("config.toml"), "[ollama]\nmodel=\"evil\"\n").unwrap();
        let other = write_config(&dir, "operator.toml", "");
        let cfg = load_config(&LoadOptions {
            config_path: Some(other),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .unwrap();
        assert_eq!(cfg.warnings().len(), 1);
        assert!(cfg.warnings()[0].contains("Ignored project configuration"));
        assert!(cfg.warnings()[0].contains("use --config explicitly"));
        // The local file did NOT take effect.
        assert_eq!(cfg.ollama().model(), "qwen2.5-coder:7b");
    }

    #[test]
    fn selected_config_produces_no_warning_for_itself() {
        let dir = test_dir("no-self-warning");
        let path = write_config(&dir, "config.toml", "");
        let cfg = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .unwrap();
        assert!(cfg.warnings().is_empty());
    }

    #[test]
    fn xdg_default_path_resolution() {
        // Pure function: no process-environment mutation in tests.
        let home = Path::new("/home/tester");
        let path = xdg_default_config_path(Some("/tmp/xdg"), Some(home)).unwrap();
        assert!(path.ends_with("flow/config.toml"));
        assert!(path.starts_with("/tmp/xdg"));
        let path = xdg_default_config_path(None, Some(home)).unwrap();
        assert!(path.starts_with("/home/tester/.config"));
        // Empty XDG_CONFIG_HOME falls back per the XDG spec (documented
        // deviation from Python, which would use the empty string).
        let path = xdg_default_config_path(Some(""), Some(home)).unwrap();
        assert!(path.starts_with("/home/tester/.config"));
        let err = xdg_default_config_path(None, None).expect_err("no home must fail");
        assert_eq!(err, ConfigError::NoHomeDirectory);
    }

    #[test]
    fn tilde_expansion_rules() {
        let home = Path::new("/home/tester");
        assert_eq!(
            expand_tilde(Path::new("~/proj"), Some(home)).unwrap(),
            PathBuf::from("/home/tester/proj")
        );
        assert_eq!(
            expand_tilde(Path::new("~"), Some(home)).unwrap(),
            PathBuf::from("/home/tester")
        );
        assert_eq!(
            expand_tilde(Path::new("/abs/path"), Some(home)).unwrap(),
            PathBuf::from("/abs/path")
        );
        // ~user is left untouched (documented limitation).
        assert_eq!(
            expand_tilde(Path::new("~other/x"), Some(home)).unwrap(),
            PathBuf::from("~other/x")
        );
        let err = expand_tilde(Path::new("~/x"), None).expect_err("no home must fail");
        assert_eq!(err, ConfigError::NoHomeDirectory);
    }

    #[test]
    fn lexical_absolute_resolves_dot_segments() {
        let cwd = std::env::current_dir().unwrap();
        let resolved = lexical_absolute(Path::new("a/./b/../c")).unwrap();
        assert_eq!(resolved, cwd.join("a").join("c"));
        assert!(
            lexical_absolute(Path::new("/x/../y"))
                .unwrap()
                .is_absolute()
        );
    }

    #[test]
    fn strip_source_echo_drops_echo_and_annotation_lines() {
        let raw = "TOML parse error at line 1, column 25\n  |\n1 | sk-fake-SECRET\n  |     ^";
        let clean = strip_source_echo(raw);
        assert!(clean.contains("TOML parse error at line 1, column 25"));
        assert!(!clean.contains("sk-fake-SECRET"));
        assert!(!clean.contains('|'));
        // A single-line message passes through untouched.
        assert_eq!(strip_source_echo("plain message"), "plain message");
    }

    #[test]
    fn parse_error_names_location_without_echoing_source() {
        let dir = test_dir("parse-error-secret");
        let path = write_config(&dir, "secret.toml", "sk-fake-SECRET-abc123XYZ");
        let err = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .expect_err("malformed config must fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
        let text = err.to_string();
        assert!(
            !text.contains("sk-fake-SECRET-abc123XYZ"),
            "parse error must not echo config text: {text}"
        );
        assert!(text.contains("line 1"), "parse error must name the line");
    }

    #[test]
    fn oversized_config_fires_size_gate_not_parser() {
        let dir = test_dir("oversized-gate");
        let path = dir.join("big.toml");
        std::fs::write(&path, "x".repeat(CONFIG_FILE_BYTES_MAX as usize + 1)).unwrap();
        let err = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .expect_err("oversized config must fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
        assert!(
            err.to_string().contains("exceeds"),
            "the size gate must fire before parsing"
        );
    }

    #[test]
    fn config_at_exact_size_cap_passes_the_size_gate() {
        let dir = test_dir("exact-cap");
        let path = dir.join("exact.toml");
        // Exactly CONFIG_FILE_BYTES_MAX bytes of valid TOML: one long comment.
        let body = "x".repeat(CONFIG_FILE_BYTES_MAX as usize - 2);
        std::fs::write(&path, format!("#{body}\n")).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            CONFIG_FILE_BYTES_MAX
        );
        let cfg = load_config(&LoadOptions {
            config_path: Some(path),
            workspace: Some(dir.to_path_buf()),
            ..LoadOptions::default()
        })
        .expect("config at exactly the cap must pass the size gate");
        assert!(cfg.checks().is_empty());
    }
}
