//! The read-only file operation tools.
//!
//! Ports the `make_file_ops_tools` factories from
//! `flow/tools/file_ops.py`, minus the write path's operator trust: reads
//! and listings go through the descriptor-relative, root-pinned
//! [`Workspace`](phlow_workspace::Workspace). Every call returns a JSON
//! string, as Python's `wrap` did; failures become
//! `{"status": "error", "error": <message>}`.
//!
//! Deletion stays disabled: [`FileOps::file_delete`] returns the denial.

use std::path::Path;

use serde_json::json;

use crate::error::ToolError;

/// The file tools never delete; Python returned this exact denial.
pub const FILE_DELETE_DENIAL: &str = "Deletion tools are disabled";

/// Read-only file operations over one workspace root.
///
/// `trusted` selects the workspace's write mode for [`FileOps::file_write`];
/// reads and listings work either way, exactly like the Python tools.
pub struct FileOps {
    workspace: phlow_workspace::Workspace,
}

impl FileOps {
    /// Open the file tools over `workspace_dir`. Fails when the directory
    /// is missing or not a directory.
    pub fn open(workspace_dir: &Path, trusted: bool) -> Result<FileOps, ToolError> {
        let workspace = phlow_workspace::Workspace::open(workspace_dir, trusted, &[])?;
        Ok(FileOps { workspace })
    }

    /// Read a workspace-relative file; returns the result JSON string.
    pub fn file_read(&self, path: &str) -> String {
        match self.workspace.read(path) {
            Ok(result) => json!({
                "status": "ok",
                "path": result.path,
                "content": result.content,
                "bytes": result.bytes,
            })
            .to_string(),
            Err(error) => denied(&error.to_string()),
        }
    }

    /// Write a workspace-relative file atomically; returns the result JSON
    /// string. Fails on a read-only (untrusted) workspace.
    pub fn file_write(&self, path: &str, content: &str) -> String {
        match self.workspace.write(path, content) {
            Ok(result) => json!({
                "status": "ok",
                "path": result.path,
                "bytes": result.bytes,
            })
            .to_string(),
            Err(error) => denied(&error.to_string()),
        }
    }

    /// List files under a workspace-relative directory; returns the result
    /// JSON string.
    pub fn file_list(&self, directory: &str) -> String {
        match self.workspace.list(directory) {
            Ok(result) => json!({
                "status": "ok",
                "files": result.files,
                "truncated": result.truncated,
            })
            .to_string(),
            Err(error) => denied(&error.to_string()),
        }
    }

    /// Deletion is disabled in the safe runtime; always the denial JSON.
    pub fn file_delete(&self, _path: &str) -> String {
        denied(FILE_DELETE_DENIAL)
    }
}

/// Python's `wrap` error shape.
fn denied(message: &str) -> String {
    json!({"status": "error", "error": message}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phlow-tools-test-{prefix}-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create temp dir");
        dir
    }

    #[test]
    fn read_write_list_round_trip() {
        let dir = temp_dir("rw");
        let tools = FileOps::open(&dir, true).unwrap();
        let written: Value = serde_json::from_str(&tools.file_write("a.txt", "hello")).unwrap();
        assert_eq!(written["status"], "ok");
        let read: Value = serde_json::from_str(&tools.file_read("a.txt")).unwrap();
        assert_eq!(read["status"], "ok");
        assert_eq!(read["content"], "hello");
        let listed: Value = serde_json::from_str(&tools.file_list(".")).unwrap();
        assert_eq!(listed["status"], "ok");
        assert!(
            listed["files"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == "a.txt")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_missing_file_returns_error_envelope() {
        let dir = temp_dir("missing");
        let tools = FileOps::open(&dir, false).unwrap();
        let result: Value = serde_json::from_str(&tools.file_read("nope.txt")).unwrap();
        assert_eq!(result["status"], "error");
        assert!(result["error"].is_string());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_on_untrusted_workspace_is_denied() {
        let dir = temp_dir("readonly");
        let tools = FileOps::open(&dir, false).unwrap();
        let result: Value = serde_json::from_str(&tools.file_write("a.txt", "x")).unwrap();
        assert_eq!(result["status"], "error");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_is_always_denied() {
        let dir = temp_dir("delete");
        let tools = FileOps::open(&dir, true).unwrap();
        let result: Value = serde_json::from_str(&tools.file_delete("a.txt")).unwrap();
        assert_eq!(
            result,
            serde_json::json!({"status": "error", "error": FILE_DELETE_DENIAL})
        );
        assert_eq!(FILE_DELETE_DENIAL, "Deletion tools are disabled");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_rejects_missing_root() {
        let missing = std::env::temp_dir().join("phlow-tools-test-no-such-dir-xyz");
        let _ = std::fs::remove_dir_all(&missing);
        assert!(FileOps::open(&missing, false).is_err());
    }
}
