//! Tool surfaces for the safe runtime.
//!
//! A static, non-executable registry: `web_search` (DuckDuckGo Instant
//! Answer or SearXNG over bounded HTTP), the fail-closed `shell_exec` and
//! `lsp_check` stubs, and read-only file operations. Executable plugin
//! loading is disabled; see [`ToolRegistry::load_plugins`].
//!
//! # Limits
//!
//! - Search requests: 10-second total timeout, no proxy, no redirects.
//! - Queries: [`web_search::QUERY_CHARS_MAX`] characters.
//! - Snippets: 500/300/80 characters per the Python truncation rules.
//! - Tool names in errors: [`error::TOOL_NAME_CHARS_MAX`] characters.

#![forbid(unsafe_code)]

pub mod error;
pub mod file_ops;
pub mod json_compat;
pub mod lsp_check;
pub mod registry;
pub mod shell;
pub mod web_search;

pub use error::ToolError;
pub use file_ops::FileOps;
pub use json_compat::{python_json_dumps, python_json_dumps_indent2};
pub use registry::{ToolRegistry, ToolSpec, load_user_tools};
pub use web_search::{SearchBackend, SearchResult, WebSearch};
