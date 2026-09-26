//! The `phlow` terminal chat interface, ported from `flow/tui.py`.
//!
//! Command parsing and execution ([`commands`]) are separated from terminal
//! I/O ([`app`]) so every command is unit-testable without a terminal.
//! Panels ([`panels`]) are pure ratatui widget constructors, tested through
//! [`TestBackend`](ratatui::backend::TestBackend).
//!
//! # Limits
//!
//! - Panel truncation bounds live in [`panels`]:
//!   [`panels::TOOL_RESULT_CHARS_MAX_DEFAULT`],
//!   [`panels::MEMORY_WHEN_CHARS_MAX`],
//!   [`panels::MEMORY_QUERY_CHARS_MAX`],
//!   [`panels::MEMORY_RESPONSE_CHARS_MAX`],
//!   [`panels::TOOL_DESCRIPTION_CHARS_MAX`].
//! - No background tasks: one line in, one panel out. Input lines are
//!   capped at [`app::INPUT_LINE_BYTES_MAX`] bytes.

#![forbid(unsafe_code)]

pub mod app;
pub mod commands;
pub mod facade;
pub mod panels;

pub use app::{LineAction, classify_line, read_line, run, run_line_mode};
pub use commands::{COMMANDS, COMMANDS_NOTE, FlowApp, UNAVAILABLE_MESSAGE, execute};
pub use facade::{FakeRuntime, RuntimeFacade};
