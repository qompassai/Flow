//! Confined file operations for the phlow agent runtime.
//!
//! A [`Workspace`] pins one root directory and mediates every file access the
//! agent's file tools perform. Model-supplied paths are untrusted input:
//! they are validated component-by-component (no absolute paths, no `..`,
//! no `.git`, no symlinks, no multiply-linked files) and then opened with
//! descriptor-relative, no-follow syscalls so a path cannot be swapped
//! between validation and use.
//!
//! Safety > performance > developer experience. All bounds are named
//! constants with units; every failure is a typed [`WorkspaceError`], never
//! a panic; the crate is `#![forbid(unsafe_code)]` — the `rustix` dependency
//! is what makes no-follow I/O expressible without `unsafe`.

#![forbid(unsafe_code)]

mod error;
mod workspace;

pub use error::WorkspaceError;
pub use workspace::{
    FILE_BYTES_MAX, LIST_FILES_MAX, LIST_VISITED_MAX, ListResult, PATH_DEPTH_MAX, PATH_LENGTH_MAX,
    ReadResult, SKIP_DIRS, Workspace, WriteResult,
};
