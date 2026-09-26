//! The private Neovim socket bridge (Phase 3: synchronous transport seam).
//!
//! This crate ports `flow/editor.py` without the `pynvim` wire layer. The
//! bridge never opens TCP: on Unix it speaks to a private Unix socket, on
//! Windows to a private named pipe. The `pynvim` MsgPack-RPC session becomes
//! the [`EditorTransport`] trait, so unit tests run against an in-memory
//! fake; Phase 4 wires a real `rmp`/`rmpv` transport behind this trait.
//!
//! Wire contract (byte-exact with `flow/editor.py`):
//!
//! - Schemas: `"return require('rose.tools').schemas()"` with no arguments.
//! - Calls: `"return require('rose.tools').call(...)"` with `[name, args]`.
//! - At most 64 schemas are kept; the 65th is an error, not a truncation.
//! - A timeout poisons the bridge until it is dropped and rebuilt.
//! - Exactly one owning worker issues requests (Phase 4 detail; the Phase 3
//!   fake is single-threaded by construction).

#![forbid(unsafe_code)]

pub mod bridge;
pub mod contract;
pub mod error;
pub mod fake;
pub mod socket;

pub use bridge::{EditorBridge, EditorTransport};
pub use contract::{
    CALL_LUA, DEBUG_ACTIONS, EDITOR_BRIDGE_FILE_TOOLS, EDITOR_TOOLS, SCHEMAS_LUA, SCHEMAS_MAX,
    TIMEOUT_DEFAULT, TIMEOUT_MAX, TIMEOUT_MIN, WORKER_GRACE,
};
pub use error::{BridgeError, FreshnessError, SocketError, TransportError};
pub use fake::FakeTransport;
pub use socket::validate_socket;
