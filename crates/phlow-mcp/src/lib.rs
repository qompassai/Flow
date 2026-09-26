//! Newline-delimited JSON-RPC server for the phlow agent runtime.
//!
//! This crate is a byte-faithful port of `flow/mcp.py`. It speaks the same
//! wire protocol: newline-delimited JSON-RPC 2.0 on stdin/stdout, protocol
//! version `2025-11-25` (negotiating down from older supported versions),
//! the same three compatibility tools (`flow_run`, `flow_status`,
//! `flow_check`), the same six error codes, and the same 1 MiB frame cap.
//!
//! Wire compatibility notes:
//!
//! * Frames are serialized with [`json_ascii::dumps`], which replicates
//!   Python's `json.dumps(payload, ensure_ascii=True, allow_nan=False)`:
//!   `", "`/`": "` separators, ASCII-only escaping, insertion-ordered
//!   keys. Golden tests replay Python-produced frames byte-for-byte.
//! * Stdout carries *only* protocol JSON. The [`McpRuntime`] trait documents
//!   that implementations must not write to stdout; the server itself never
//!   logs there.
//! * The runtime behind the server is synchronous here ([`McpRuntime`]); the
//!   async runtime arrives in Phase 4 and reuses this framing core unchanged.

#![forbid(unsafe_code)]

pub mod error;
pub mod json_ascii;
pub mod protocol;
pub mod schema;
pub mod server;

pub use error::{McpError, RuntimeError};
pub use json_ascii::{DumpsError, dumps};
pub use protocol::{
    INSTRUCTIONS, INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST, MAX_FRAME_BYTES,
    METHOD_NOT_FOUND, NOT_INITIALIZED, PARSE_ERROR, PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS, ToolSpec, tool_spec, tool_specs,
};
pub use schema::{SCHEMA_DEPTH_MAX, SCHEMA_NODES_MAX, SchemaError, validate_arguments};
pub use server::{FakeRuntime, FrameRead, McpRuntime, McpServer, ServeEnd};
