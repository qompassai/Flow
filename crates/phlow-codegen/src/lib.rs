//! Code generation surfaces: embedded language profiles, the fail-closed
//! code validator, and the bounded app generator.
//!
//! # Limits
//!
//! - Operator profile directories: [`profiles::PROFILE_FILES_MAX`]
//!   entries; more is an error.
//! - One profile file: [`profiles::PROFILE_TOML_BYTES_MAX`] bytes.
//! - Validator issue JSON: [`validator::VALIDATOR_JSON_CHARS_MAX`]
//!   characters.

#![forbid(unsafe_code)]

pub mod app_generator;
pub mod error;
pub mod profiles;
pub mod validator;

pub use app_generator::{AppGenerator, build_prompt, python_repr};
pub use error::CodegenError;
pub use profiles::{LanguageProfile, builtin_profiles, load_profiles, profile_summary};
pub use validator::CodeValidator;
