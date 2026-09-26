//! Inference-serving primitives for the phlow agent runtime.
//!
//! Plain words: these are the serving-layer ideas worth stealing from the
//! DeepSeek-V4.1-Flash paper (arXiv:2609.19969), reimplemented as small,
//! dependency-free Rust modules. Nothing here trains a model or runs one:
//! each module captures a *deployment* trick that works with any engine.
//!
//! - [`tiered_cache`]: two-tier KV lifetime. Global entries persist to disk
//!   with a long TTL; local entries live in memory with a short TTL and are
//!   rebuilt by replaying only the most recent entries (SWA Bounded Replay).
//! - [`two_stage`]: coarse-to-fine candidate ranking. A cheap scorer picks a
//!   bounded candidate pool; the expensive scorer only ever sees pool items
//!   (Hierarchical Sparse Indexer, minus the training).
//! - [`speculative`]: draft-then-verify orchestration with confidence-scheduled
//!   acceptance and a throughput-table-driven verify-length scheduler
//!   (DSpark's serving idea, without the trained drafter).
//! - [`kv_policy`]: quantization precision policy as data: 4-bit is fine for
//!   global KV, local/SWA KV stays at higher precision, quantize after RoPE.
//!
//! # Safety and bounds
//!
//! Every module names its limits in `SCREAMING_SNAKE_CASE` with units,
//! returns typed errors on invalid input, and asserts internal invariants.
//! Concurrency is synchronous (`std::sync` primitives only): there is no
//! async runtime yet, so `tokio` integration is explicitly future work.
//!
//! # Unsafe policy
//!
//! This crate forbids unsafe code outright.

#![forbid(unsafe_code)]

pub mod kv_policy;
pub mod speculative;
pub mod tiered_cache;
pub mod two_stage;
