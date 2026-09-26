//! Real transports behind the Phase 3 seams.
//!
//! - [`MsgpackTransport`]: hand-rolled msgpack-RPC over the Neovim private
//!   socket, implementing [`EditorTransport`]. Only `nvim_exec_lua` with the
//!   two audited Lua expressions ever crosses the socket.
//! - [`ReqwestTransport`]: Ollama HTTP over `reqwest` (blocking, rustls,
//!   no proxies, no redirects), implementing [`LlmTransport`].
//!
//! [`EditorTransport`]: phlow_editor::EditorTransport
//! [`LlmTransport`]: phlow_llm::LlmTransport

mod http;
mod msgpack;

pub use http::ReqwestTransport;
pub use msgpack::MsgpackTransport;
