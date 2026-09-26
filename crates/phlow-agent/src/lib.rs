//! The agent layer: bounded conversation context, the orchestrator, and the
//! SQLite FTS5 memory store. Mirrors `flow/agent/`.
//!
//! The orchestrator drives [`phlow_runtime::Runtime`]; it never bypasses the
//! runtime's budgets, containment, or host verification.

#![forbid(unsafe_code)]

pub mod context;
pub mod memory;
pub mod orchestrator;

pub use context::{ChatMessage, ConversationContext, DEFAULT_MAX_MESSAGES};
pub use memory::{
    HIT_QUERY_CHARS, HIT_RESPONSE_CHARS, MemoryEntry, MemoryError, MemoryStore, QUERY_CHARS_MAX,
    RECENT_MAX, RESPONSE_CHARS_MAX, SEARCH_TERM_CHARS_MAX, TAG_CHARS_MAX, TAGS_MAX, TOP_K_MAX,
};
pub use orchestrator::Orchestrator;
