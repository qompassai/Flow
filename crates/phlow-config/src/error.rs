//! Typed configuration errors.
//!
//! Every failure to load or validate operator configuration becomes a
//! [`ConfigError`]. External input (TOML text, option names, values) is
//! reported with bounded context; nothing here panics on operator input.

use std::fmt;
use std::path::PathBuf;

/// Longest option name or value fragment reproduced in an error message, in
/// characters. Operator config is trusted text, but error context is still
/// bounded so a hostile file cannot inflate diagnostics without limit.
const CONTEXT_CHARS_MAX: usize = 128;

/// Truncate to [`CONTEXT_CHARS_MAX`] characters on a char boundary.
fn truncate(text: &str) -> String {
    if text.chars().count() <= CONTEXT_CHARS_MAX {
        text.to_owned()
    } else {
        text.chars().take(CONTEXT_CHARS_MAX).collect()
    }
}

/// All the ways loading or validating configuration can fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The config file could not be read. Displayed in CPython's
    /// `[Errno N] message: 'path'` shape, prefixed with
    /// `Cannot load configuration <path>: ` exactly like Python's
    /// `load_config`. `code` is the OS errno when the failure came from
    /// the OS (`io::Error::raw_os_error`); `message` is the OS message
    /// text. At display time the trailing ` (os error N)` suffix is
    /// stripped so the errno is not printed twice, and the path is
    /// rendered Python-`repr`-style (single-quoted, `\\` and `\'`
    /// escaped). When `code` is `None` (a non-OS I/O failure) the
    /// `[Errno N]` prefix is omitted.
    Io {
        path: PathBuf,
        code: Option<i32>,
        message: String,
    },
    /// The process working directory could not be determined. Separate
    /// from [`ConfigError::Io`] so a `current_dir()` failure is never
    /// misreported as a config-file load failure.
    CurrentDir { message: String },
    /// The file was read but is not valid TOML (or not UTF-8).
    Parse { path: PathBuf, message: String },
    /// A TOML key the schema does not define. `at` names the table
    /// ("config root", "ollama", "checks.<name>", ...).
    UnknownOption { at: String, option: String },
    /// A value failed validation. `field` is a dotted path like
    /// "ollama.temperature"; `reason` states the bound. Displayed as
    /// `"{field} {reason}"`, byte-matching Python's plain `ConfigError`
    /// strings (e.g. `"ollama.model must be a nonempty string"`).
    ///
    /// Two Python messages carry their own `X: detail` shape instead of a
    /// field prefix; for those the colon lives at the end of `field`
    /// (e.g. `field = "Invalid check name:"`, `reason = "'foo'"` renders
    /// `"Invalid check name: 'foo'"`).
    InvalidValue { field: String, reason: String },
    /// The selected workspace does not exist or is not a directory.
    WorkspaceMissing { path: PathBuf },
    /// `$HOME` is unset, so `~` expansion and the XDG default have no base.
    NoHomeDirectory,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io {
                path,
                code,
                message,
            } => {
                // CPython's OSError str: "[Errno 2] No such file or
                // directory: '/nonexistent.toml'". Rust's io::Error
                // Display is "No such file or directory (os error 2)";
                // the parenthesized suffix is stripped so the errno
                // appears exactly once.
                let detail = match code {
                    Some(code) => {
                        let suffix = format!(" (os error {code})");
                        let bare = message.strip_suffix(suffix.as_str()).unwrap_or(message);
                        format!("[Errno {code}] {}", truncate(bare))
                    }
                    None => truncate(message),
                };
                write!(
                    f,
                    "Cannot load configuration {}: {}: '{}'",
                    path.display(),
                    detail,
                    escape_for_repr(&path.to_string_lossy())
                )
            }
            ConfigError::CurrentDir { message } => {
                write!(
                    f,
                    "cannot determine current directory: {}",
                    truncate(message)
                )
            }
            ConfigError::Parse { path, message } => {
                write!(
                    f,
                    "cannot parse configuration {}: {}",
                    path.display(),
                    truncate(message)
                )
            }
            ConfigError::UnknownOption { at, option } => {
                write!(f, "unknown option {} in {}", truncate(option), at)
            }
            ConfigError::InvalidValue { field, reason } => {
                write!(f, "{} {}", truncate(field), truncate(reason))
            }
            ConfigError::WorkspaceMissing { path } => {
                write!(
                    f,
                    "Workspace must already be a directory: {}",
                    path.display()
                )
            }
            ConfigError::NoHomeDirectory => {
                write!(f, "cannot expand home directory: $HOME is not set")
            }
        }
    }
}

/// Escape a path the way Python's `repr()` escapes a string: backslashes
/// and single quotes gain a backslash, then the caller wraps the result in
/// single quotes.
fn escape_for_repr(text: &str) -> String {
    text.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Render like Python's `repr()` of a string: single-quoted with backslash
/// and quote escapes (e.g. `"Invalid check name: 'foo'"`).
pub(crate) fn repr_str(text: &str) -> String {
    format!("'{}'", escape_for_repr(text))
}

impl std::error::Error for ConfigError {}

/// Convenience: build an [`ConfigError::InvalidValue`] for a dotted field path.
pub(crate) fn invalid_value(field: impl Into<String>, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidValue {
        field: field.into(),
        reason: reason.into(),
    }
}

/// Convenience: build an [`ConfigError::Io`] from a failed file read,
/// capturing the OS errno so the display can render CPython's
/// `[Errno N] message: 'path'` shape.
pub(crate) fn io_error(path: PathBuf, err: &std::io::Error) -> ConfigError {
    ConfigError::Io {
        path,
        code: err.raw_os_error(),
        message: err.to_string(),
    }
}

/// Convenience: build an [`ConfigError::UnknownOption`].
pub(crate) fn unknown_option(at: impl Into<String>, option: impl Into<String>) -> ConfigError {
    ConfigError::UnknownOption {
        at: at.into(),
        option: option.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_stays_bounded_on_long_context() {
        let long = "x".repeat(10_000);
        let err = invalid_value(long.clone(), long.clone());
        let text = err.to_string();
        assert!(text.len() < long.len() * 2);
        // "{field} {reason}" shape: no "invalid" prefix, no colon.
        assert!(!text.starts_with("invalid "));
        assert!(!text.contains(": "));
    }

    #[test]
    fn invalid_value_renders_field_then_reason() {
        let err = invalid_value("ollama.model", "must be a nonempty string");
        assert_eq!(err.to_string(), "ollama.model must be a nonempty string");
    }

    #[test]
    fn io_renders_cpython_errno_shape() {
        let err = ConfigError::Io {
            path: PathBuf::from("/nonexistent.toml"),
            code: Some(2),
            message: "No such file or directory (os error 2)".to_owned(),
        };
        assert_eq!(
            err.to_string(),
            "Cannot load configuration /nonexistent.toml: \
             [Errno 2] No such file or directory: '/nonexistent.toml'"
        );
    }

    #[test]
    fn io_without_errno_omits_bracket_prefix() {
        let err = ConfigError::Io {
            path: PathBuf::from("/tmp/x.toml"),
            code: None,
            message: "stream did not contain valid UTF-8".to_owned(),
        };
        assert_eq!(
            err.to_string(),
            "Cannot load configuration /tmp/x.toml: \
             stream did not contain valid UTF-8: '/tmp/x.toml'"
        );
    }

    #[test]
    fn workspace_missing_is_capitalized_like_python() {
        let err = ConfigError::WorkspaceMissing {
            path: PathBuf::from("/nope"),
        };
        assert_eq!(
            err.to_string(),
            "Workspace must already be a directory: /nope"
        );
    }

    #[test]
    fn unknown_option_names_table_and_key() {
        let err = unknown_option("config root", "shell");
        let text = err.to_string();
        assert!(text.contains("shell"));
        assert!(text.contains("config root"));
    }
}
