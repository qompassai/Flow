//! Self-improvement surfaces: the SQLite feedback store, the disabled
//! prompt evolver, and the read-only skill store.
//!
//! Nothing here mutates prompts, skills, or Git history automatically.
//! Feedback is recorded for operator review; evolution and mutation always
//! fail with the exact Python denials.
//!
//! # Limits
//!
//! - Feedback text fields: [`feedback::SESSION_ID_CHARS_MAX`],
//!   [`feedback::COMMENT_CHARS_MAX`], [`feedback::PROMPT_USED_CHARS_MAX`],
//!   [`feedback::OUTCOME_CHARS_MAX`] characters.
//! - Low-rated listing: [`feedback::LOW_RATED_LIMIT_MAX`] rows.
//! - Skill names: [`skill_store::SKILL_NAME_CHARS_MAX`] characters;
//!   listings: the workspace's bounded recursive file list (at most
//!   `phlow_workspace::LIST_FILES_MAX` files, as in Python).

#![forbid(unsafe_code)]

pub mod error;
pub mod feedback;
pub mod prompt_evolver;
pub mod skill_store;

pub use error::SelfImproveError;
pub use feedback::{FeedbackEntry, FeedbackStore};
pub use prompt_evolver::PromptEvolver;
pub use skill_store::SkillStore;
