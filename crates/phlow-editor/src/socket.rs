//! Private socket validation.
//!
//! The bridge never touches TCP. On Unix the `--nvim` value must be an
//! absolute, non-symlink Unix socket owned by the current user, with either a
//! 0600 socket or a 0700 parent directory. On Windows it must be a
//! `\\.\pipe\...` named pipe. Validation is lazy: Python checks on the first
//! request, not in the constructor, and the bridge mirrors that.

#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, MetadataExt};
#[cfg(unix)]
use std::path::Path;

use crate::error::SocketError;

/// Validate a configured `--nvim` socket value.
///
/// Mirrors `EditorBridge._validate_socket`. The empty string counts as "not
/// configured", matching Python's `if not self.socket`.
pub fn validate_socket(socket: &str) -> Result<(), SocketError> {
    if socket.is_empty() {
        return Err(SocketError::NotConfigured);
    }
    validate_socket_platform(socket)
}

/// Windows: the value must be a private `\\.\pipe\...` named pipe; nothing
/// else is accepted. The transport layer opens the pipe itself.
#[cfg(windows)]
fn validate_socket_platform(socket: &str) -> Result<(), SocketError> {
    if !socket.starts_with("\\\\.\\pipe\\") {
        return Err(SocketError::NotPrivatePipe);
    }
    Ok(())
}

/// Unix: the value must be an absolute, non-symlink Unix socket owned by the
/// current user, with either a 0600 socket or a 0700 parent directory.
#[cfg(unix)]
fn validate_socket_platform(socket: &str) -> Result<(), SocketError> {
    let path = Path::new(socket);
    // Absolute first, like Python (`path.is_absolute()` precedes `stat`):
    // a nonexistent relative path reports the canonical message, not an
    // I/O error.
    if !path.is_absolute() {
        return Err(SocketError::NotAbsoluteSocket);
    }
    // `symlink_metadata` (not `metadata`) so a symlink is seen as a symlink.
    let link_info = std::fs::symlink_metadata(path).map_err(SocketError::Io)?;
    if link_info.file_type().is_symlink() {
        return Err(SocketError::NotAbsoluteSocket);
    }
    let info = std::fs::metadata(path).map_err(SocketError::Io)?;
    let is_socket = info.file_type().is_socket();
    let owned_by_me = info.uid() == rustix::process::getuid().as_raw();
    if !is_socket || !owned_by_me {
        return Err(SocketError::NotOwnedSocket);
    }
    // A 0700 parent protects sockets whose own mode is affected by
    // Neovim/umask: private is (socket has no group/other bits) OR (parent
    // is owned by us and has no group/other bits).
    let socket_mode_open = info.mode() & 0o077 != 0;
    if socket_mode_open {
        let parent = path.parent().ok_or(SocketError::NotAbsoluteSocket)?;
        let parent_info = std::fs::metadata(parent).map_err(SocketError::Io)?;
        let parent_ok = parent_info.uid() == rustix::process::getuid().as_raw()
            && parent_info.mode() & 0o077 == 0;
        if !parent_ok {
            return Err(SocketError::NotPrivate);
        }
    }
    Ok(())
}

// Socket-permission semantics are Unix-specific; the Windows variant is a
// pure string check validated by inspection.
#[cfg(unix)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("phlow-editor-test-{}-{}", tag, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn empty_socket_is_not_configured() {
        assert!(matches!(
            validate_socket(""),
            Err(SocketError::NotConfigured)
        ));
    }

    #[test]
    fn relative_path_is_rejected() {
        // The absolute-path check runs before any filesystem access, so a
        // nonexistent relative path is NotAbsoluteSocket (Python's error
        // ordering), never a bare Io error.
        let err = validate_socket("relative/socket").unwrap_err();
        assert!(matches!(err, SocketError::NotAbsoluteSocket));
    }

    #[test]
    fn symlink_is_rejected() {
        let dir = unique_dir("symlink");
        let real = dir.join("real.sock");
        let _listener = UnixListener::bind(&real).unwrap();
        let link = dir.join("link.sock");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let err = validate_socket(link.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, SocketError::NotAbsoluteSocket), "{err}");
    }

    #[test]
    fn regular_file_is_not_a_socket() {
        let dir = unique_dir("regular");
        let file = dir.join("plain.txt");
        std::fs::write(&file, b"x").unwrap();
        let err = validate_socket(file.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, SocketError::NotOwnedSocket), "{err}");
    }

    #[test]
    fn private_socket_passes() {
        use std::os::unix::fs::PermissionsExt;
        let dir = unique_dir("private");
        let path = dir.join("nvim.sock");
        let _listener = UnixListener::bind(&path).unwrap();
        // /tmp is world-open, so the socket itself must be 0600.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        validate_socket(path.to_str().unwrap()).unwrap();
    }

    #[test]
    fn world_open_socket_with_open_parent_fails() {
        use std::os::unix::fs::PermissionsExt;
        let dir = unique_dir("open");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = dir.join("nvim.sock");
        let _listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o777)).unwrap();
        let err = validate_socket(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, SocketError::NotPrivate), "{err}");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[test]
    fn open_socket_with_private_parent_passes() {
        use std::os::unix::fs::PermissionsExt;
        let dir = unique_dir("privparent");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = dir.join("nvim.sock");
        let _listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o777)).unwrap();
        validate_socket(path.to_str().unwrap()).unwrap();
    }

    #[test]
    fn missing_socket_is_io_not_panic() {
        let missing = std::env::temp_dir().join("phlow-editor-test-nope/missing.sock");
        assert!(matches!(
            validate_socket(missing.to_str().unwrap()),
            Err(SocketError::Io(_))
        ));
    }
}
