//! The confined workspace: one pinned root, validated paths, no-follow I/O.
//!
//! [`Workspace::open`] canonicalizes the root strictly and captures its
//! device/inode identity. Every later operation re-checks that identity
//! ([`Workspace::assert_current`]), so replacing the root directory
//! mid-session is detected instead of silently followed.
//!
//! Path handling is two layers, mirroring the threat model:
//!
//! 1. [`Workspace::path`] validates the *string*: nonempty, NUL-free,
//!    within [`PATH_LENGTH_MAX`] characters and [`PATH_DEPTH_MAX`]
//!    components, relative, no Windows drive or separator, no `..`, no
//!    `.git`, and no symlink in any component (checked with
//!    `symlink_metadata`, which never follows).
//! 2. [`Workspace::read`] and [`Workspace::write`] then open the file
//!    descriptor-relative from the root with `O_NOFOLLOW`, so a component
//!    swapped for a symlink *between* validation and open still fails
//!    instead of escaping.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rustix::fd::OwnedFd;
use rustix::fs::{AtFlags, FileType, Mode, OFlags};

use crate::WorkspaceError;

/// Largest file the file tools will read or write, in bytes. Model-facing
/// file tools are for source code; anything larger is not read into context
/// or rewritten wholesale.
pub const FILE_BYTES_MAX: u64 = 256 * 1024;
/// Maximum number of path components in a model-supplied relative path.
/// Bounds the component-by-component traversal loop and the number of
/// descriptors a single open can touch.
pub const PATH_DEPTH_MAX: usize = 32;
/// Maximum length of a model-supplied relative path, in characters.
/// Oversize is a validation error, not an assertion: paths are input.
pub const PATH_LENGTH_MAX: usize = 1024;
/// Maximum files returned by [`Workspace::list`]. Listing stops early
/// rather than walking an arbitrarily large tree on every tool call.
pub const LIST_FILES_MAX: usize = 500;
/// Maximum directory entries visited by one [`Workspace::list`] call.
pub const LIST_VISITED_MAX: usize = 5000;
/// Maximum entries read from a single directory by [`Workspace::list`],
/// in entries. Reading is lazy and capped so one gigantic directory cannot
/// force an unbounded allocation before the visit cap is consulted.
pub const LIST_DIR_ENTRIES_MAX: usize = 5000;
/// Directories never descended into by [`Workspace::list`].
pub const SKIP_DIRS: &[&str] = &[
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
];

const _: () = assert!(FILE_BYTES_MAX > 0, "file cap must be positive");
const _: () = assert!(
    PATH_DEPTH_MAX > 0 && PATH_DEPTH_MAX < PATH_LENGTH_MAX,
    "depth bound must fit inside the length bound"
);
const _: () = assert!(
    LIST_FILES_MAX > 0 && LIST_FILES_MAX <= LIST_VISITED_MAX,
    "file cap must fit inside the visit cap"
);
const _: () = assert!(
    LIST_DIR_ENTRIES_MAX > 0 && LIST_DIR_ENTRIES_MAX <= LIST_VISITED_MAX,
    "per-directory read cap must be positive and fit inside the visit cap"
);

/// Permissions for directories the workspace creates while walking to a
/// write target. Matches the Python implementation (`0o755`).
const MKDIR_MODE: u32 = 0o755;
/// Permissions for files the workspace creates. An overwritten file keeps
/// its existing mode instead.
const CREATE_MODE: u32 = 0o644;
/// Prefix for atomic-write temporary files, created with `O_EXCL` beside
/// the target so the rename is atomic on the same filesystem.
const TEMP_NAME_PREFIX: &str = ".phlow-write-";

/// Counter that makes atomic-write temporary names unique within a process.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Result of [`Workspace::read`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadResult {
    /// The requested path, as supplied.
    pub path: String,
    /// File content, UTF-8 with invalid sequences replaced.
    pub content: String,
    /// Byte length of the file content.
    pub bytes: u64,
}

/// Result of [`Workspace::write`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteResult {
    /// The written path, as supplied.
    pub path: String,
    /// Byte length written.
    pub bytes: u64,
    /// Workspace revision after this write.
    pub revision: u64,
}

/// Result of [`Workspace::list`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListResult {
    /// Relative paths of listed regular files, in deterministic order.
    pub files: Vec<String>,
    /// True when a cap stopped the walk early.
    pub truncated: bool,
}

/// Mutable workspace state behind one lock.
///
/// The lock guards `changed_files` and `revision` only. It is never held
/// during I/O: operations resolve and transfer bytes first, then take the
/// lock for a short ledger update. Single lock, so no lock ordering to
/// document.
struct WorkspaceState {
    changed_files: BTreeSet<String>,
    revision: u64,
}

/// A confined root directory.
///
/// Construct with [`Workspace::open`]. `read` and `write` need POSIX
/// no-follow I/O; on other platforms they return
/// [`WorkspaceError::Unavailable`] rather than silently weakening.
/// `path` and `list` work everywhere.
pub struct Workspace {
    root: PathBuf,
    trusted: bool,
    identity: RootIdentity,
    protected: BTreeSet<PathBuf>,
    posix: bool,
    state: Mutex<WorkspaceState>,
}

/// How the root's identity is pinned, so replacement is detectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootIdentity {
    /// POSIX: device + inode captured at open.
    DevIno { dev: u64, ino: u64 },
    /// Non-POSIX fallback: no stable identity available.
    #[cfg_attr(unix, allow(dead_code))]
    Unpinned,
}

/// What one directory entry contributes to a listing walk.
enum ListStep {
    /// A singly-linked regular file: its workspace-relative display path.
    Record(String),
    /// A directory to descend into: its workspace-relative path.
    Descend(PathBuf),
    /// Ignored: symlink, [`SKIP_DIRS`] member, non-regular, multiply-linked,
    /// or deeper than [`PATH_DEPTH_MAX`].
    Skip,
}

/// Classify one directory entry for [`Workspace::list`].
///
/// Symlinks never descend and are never recorded; only singly-linked
/// regular files are recorded.
fn classify_entry(
    entry: &std::fs::DirEntry,
    current_rel: &Path,
    depth: usize,
) -> Result<ListStep, WorkspaceError> {
    let name = entry.file_name();
    let meta = std::fs::symlink_metadata(entry.path()).map_err(WorkspaceError::Io)?;
    if meta.file_type().is_symlink() {
        return Ok(ListStep::Skip);
    }
    if meta.file_type().is_dir() {
        if SKIP_DIRS.contains(&name.to_string_lossy().as_ref()) || depth >= PATH_DEPTH_MAX {
            return Ok(ListStep::Skip);
        }
        return Ok(ListStep::Descend(current_rel.join(&name)));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if !meta.file_type().is_file() || meta.nlink() > 1 {
            return Ok(ListStep::Skip);
        }
    }
    #[cfg(not(unix))]
    {
        if !meta.file_type().is_file() {
            return Ok(ListStep::Skip);
        }
    }
    let display = current_rel.join(&name).to_string_lossy().into_owned();
    Ok(ListStep::Record(display))
}

/// Read one directory's entries in sorted name order, so listings are
/// deterministic across runs and filesystems.
///
/// Reads lazily and stops at [`LIST_DIR_ENTRIES_MAX`] entries; the
/// returned flag is true when the cap tripped, so the caller can mark the
/// listing truncated instead of silently dropping entries.
fn sorted_entries(absolute: &Path) -> Result<(Vec<std::fs::DirEntry>, bool), WorkspaceError> {
    let mut read = std::fs::read_dir(absolute).map_err(WorkspaceError::Io)?;
    let mut entries: Vec<std::fs::DirEntry> = Vec::new();
    for entry in read.by_ref().take(LIST_DIR_ENTRIES_MAX) {
        entries.push(entry.map_err(WorkspaceError::Io)?);
    }
    let dir_truncated = match read.next() {
        Some(Ok(_)) => true,
        Some(Err(err)) => return Err(WorkspaceError::Io(err)),
        None => false,
    };
    entries.sort_by_key(|entry| entry.file_name());
    Ok((entries, dir_truncated))
}

/// Anchor a protected path to the pinned root without letting a missing
/// path silently drop its protection.
///
/// Tries `canonicalize()` first (strict: resolves symlinks). When the path
/// does not exist yet, falls back to lexical normalization of
/// `root.join(item)` for relative items (or `item` itself when absolute):
/// `.` components are dropped and `..` pops one component. Over-matching
/// is the fail-closed direction here: a protected path that never exists
/// protects nothing real, while an unanchored fallback could never match
/// the absolute candidates and would fail open.
fn anchor_protected(root: &Path, item: &Path) -> PathBuf {
    if let Ok(canonical) = item.canonicalize() {
        return canonical;
    }
    let joined = if item.is_absolute() {
        item.to_path_buf()
    } else {
        root.join(item)
    };
    let mut anchored = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                anchored.pop();
            }
            other => anchored.push(other.as_os_str()),
        }
    }
    anchored
}

/// Open the parent directory of `relative`, descriptor-relative from the
/// root with `O_NOFOLLOW` at every level.
///
/// Missing intermediate directories are created (mode [`MKDIR_MODE`])
/// only when `write` is true. Returns the parent directory fd and the
/// final component name.
impl Workspace {
    /// Open a workspace at `root`.
    ///
    /// Accepted: an existing directory (symlinks in `root` itself are
    /// resolved strictly — the target is pinned, not the link). Rejected:
    /// missing paths and non-directories.
    ///
    /// `trusted` gates writes; `protected` lists extra paths the file tools
    /// may never modify (absolute, or relative to the root; the root's
    /// `config.toml` and `.flow.toml` are always protected).
    pub fn open(
        root: &Path,
        trusted: bool,
        protected: &[PathBuf],
    ) -> Result<Workspace, WorkspaceError> {
        let canonical = root.canonicalize().map_err(WorkspaceError::Io)?;
        if !canonical.is_dir() {
            return Err(WorkspaceError::NotDirectory);
        }
        let identity = stat_identity(&canonical).map_err(WorkspaceError::Io)?;
        let mut protected_set: BTreeSet<PathBuf> = protected
            .iter()
            .map(|item| anchor_protected(&canonical, item))
            .collect();
        protected_set.insert(canonical.join("config.toml"));
        protected_set.insert(canonical.join(".flow.toml"));
        Ok(Workspace {
            root: canonical,
            trusted,
            identity,
            protected: protected_set,
            posix: cfg!(unix),
            state: Mutex::new(WorkspaceState {
                changed_files: BTreeSet::new(),
                revision: 0,
            }),
        })
    }

    /// The pinned root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether writes are allowed (operator passed `--trusted`).
    pub fn trusted(&self) -> bool {
        self.trusted
    }

    /// Revision counter, bumped once per successful write.
    pub fn revision(&self) -> u64 {
        self.state.lock().map(|state| state.revision).unwrap_or(0)
    }

    /// Snapshot of files changed by [`Workspace::write`], in sorted order.
    pub fn changed_files(&self) -> Vec<String> {
        self.state
            .lock()
            .map(|state| state.changed_files.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Re-check that the root is still the directory captured at open.
    ///
    /// Fails when the root became a symlink, was replaced (device/inode
    /// changed), or is unavailable. Callers must treat failure as "restart",
    /// never retry against the same handle.
    pub fn assert_current(&self) -> Result<(), WorkspaceError> {
        let meta = std::fs::symlink_metadata(&self.root)
            .map_err(|_| WorkspaceError::Stale("root is unavailable; restart Flow"))?;
        if meta.file_type().is_symlink() {
            return Err(WorkspaceError::Stale("root was replaced; restart Flow"));
        }
        match self.identity {
            RootIdentity::DevIno { dev, ino } => {
                let now = stat_identity(&self.root)
                    .map_err(|_| WorkspaceError::Stale("root is unavailable; restart Flow"))?;
                let fresh = match now {
                    RootIdentity::DevIno {
                        dev: now_dev,
                        ino: now_ino,
                    } => now_dev == dev && now_ino == ino,
                    RootIdentity::Unpinned => true,
                };
                if !fresh {
                    return Err(WorkspaceError::Stale("root was replaced; restart Flow"));
                }
            }
            RootIdentity::Unpinned => {}
        }
        Ok(())
    }

    /// Validate a model-supplied relative path and resolve it under the root.
    ///
    /// Accepted: nonempty relative paths within [`PATH_LENGTH_MAX`]
    /// characters and [`PATH_DEPTH_MAX`] components, containing no `..`,
    /// no `.git`, no symlinks, and no multiply-linked final file.
    /// Rejected inputs describe the violated rule; `write = true` additionally
    /// requires a trusted workspace and rejects protected paths.
    pub fn path(&self, relative: &str, write: bool) -> Result<PathBuf, WorkspaceError> {
        self.assert_current()?;
        let parts = split_relative(relative)?;
        // Walk component by component with symlink_metadata (never follows),
        // so a symlink anywhere in the path is rejected before use.
        let mut candidate = self.root.clone();
        for part in &parts {
            candidate.push(part);
            match std::fs::symlink_metadata(&candidate) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    return Err(WorkspaceError::InvalidPath(
                        "symlinks are not allowed in file paths".to_string(),
                    ));
                }
                Ok(_) | Err(_) => {}
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if candidate.is_file()
                && let Ok(meta) = std::fs::metadata(&candidate)
                && meta.nlink() > 1
            {
                return Err(WorkspaceError::InvalidPath(
                    "multiply linked files are not allowed in file paths".to_string(),
                ));
            }
        }
        if write {
            if !self.trusted {
                return Err(WorkspaceError::ReadOnly);
            }
            if self.protected.contains(&candidate) {
                return Err(WorkspaceError::Protected);
            }
        }
        debug_assert!(
            candidate.starts_with(&self.root),
            "validated path escaped the root"
        );
        Ok(candidate)
    }

    /// Read a file through the workspace.
    ///
    /// Opens the parent descriptor-relative with `O_NOFOLLOW` and the file
    /// itself with `O_NOFOLLOW`, then requires a singly-linked regular file.
    /// Reads at most [`FILE_BYTES_MAX`] + 1 bytes so oversize is detected
    /// without buffering more.
    pub fn read(&self, relative: &str) -> Result<ReadResult, WorkspaceError> {
        if !self.posix {
            return Err(WorkspaceError::Unavailable(
                "secure descriptor-relative file I/O requires POSIX",
            ));
        }
        let (parent, name) = self.open_parent(relative, false)?;
        let fd = rustix::fs::openat(
            &parent,
            name.as_str(),
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(syscall_err)?;
        let info = rustix::fs::fstat(&fd).map_err(syscall_err)?;
        if FileType::from_raw_mode(info.st_mode) != FileType::RegularFile {
            return Err(WorkspaceError::NotRegularFile);
        }
        #[cfg(unix)]
        if info.st_nlink > 1 {
            return Err(WorkspaceError::NotRegularFile);
        }
        let file = std::fs::File::from(fd);
        let mut data = Vec::new();
        file.take(FILE_BYTES_MAX + 1)
            .read_to_end(&mut data)
            .map_err(WorkspaceError::Io)?;
        if data.len() as u64 > FILE_BYTES_MAX {
            return Err(WorkspaceError::TooLarge {
                limit_bytes: FILE_BYTES_MAX,
            });
        }
        let bytes = data.len() as u64;
        Ok(ReadResult {
            path: relative.to_string(),
            content: String::from_utf8_lossy(&data).into_owned(),
            bytes,
        })
    }

    /// Write a file through the workspace.
    ///
    /// Requires a trusted workspace. Writes to an exclusive temporary file
    /// beside the target (`O_CREAT | O_EXCL | O_NOFOLLOW`), fsyncs, re-checks
    /// root freshness, then atomically renames over the target. The temp
    /// file is removed on every path, success or failure.
    pub fn write(&self, relative: &str, content: &str) -> Result<WriteResult, WorkspaceError> {
        let data = content.as_bytes();
        if data.len() as u64 > FILE_BYTES_MAX {
            return Err(WorkspaceError::TooLarge {
                limit_bytes: FILE_BYTES_MAX,
            });
        }
        if !self.posix {
            return Err(WorkspaceError::Unavailable(
                "secure descriptor-relative file I/O requires POSIX",
            ));
        }
        let (parent, name) = self.open_parent(relative, true)?;
        let temp_name = temp_file_name();
        let outcome = self.write_temp(&parent, &name, &temp_name, data);
        // Remove the temp file on every path; after a successful rename it
        // is already gone and the unlink fails harmlessly.
        let _ = rustix::fs::unlinkat(&parent, temp_name.as_str(), AtFlags::empty());
        outcome?;
        let bytes = data.len() as u64;
        let revision = self.record_write(relative);
        Ok(WriteResult {
            path: relative.to_string(),
            bytes,
            revision,
        })
    }

    /// List regular files under a workspace-relative directory.
    ///
    /// Walks with an explicit stack (no recursion), never descends into
    /// symlinked directories or [`SKIP_DIRS`], and only reports singly-linked
    /// regular files. Stops early with `truncated = true` at
    /// [`LIST_FILES_MAX`] files, [`LIST_VISITED_MAX`] visited entries, or
    /// [`LIST_DIR_ENTRIES_MAX`] entries read from a single directory.
    pub fn list(&self, relative: &str) -> Result<ListResult, WorkspaceError> {
        let directory = self.path(relative, false)?;
        if !directory.is_dir() {
            return Err(WorkspaceError::InvalidPath(
                "directory not found".to_string(),
            ));
        }
        // Scope the walk to the requested directory, not the root: strip
        // the pinned root to recover its workspace-relative form.
        let start_rel = directory
            .strip_prefix(&self.root)
            .map_err(|_| WorkspaceError::InvalidPath("directory not found".to_string()))?;
        let start_depth = start_rel.components().count();
        let mut files: Vec<String> = Vec::new();
        let mut visited: usize = 0;
        let mut truncated = false;
        // Stack of (relative dir, depth). LIFO with reversed push order keeps
        // the walk deterministic.
        let mut stack: Vec<(PathBuf, usize)> = vec![(start_rel.to_path_buf(), start_depth)];
        while let Some((current_rel, depth)) = stack.pop() {
            let absolute = self.root.join(&current_rel);
            let mut child_dirs: Vec<PathBuf> = Vec::new();
            let (entries, dir_truncated) = sorted_entries(&absolute)?;
            truncated = truncated || dir_truncated;
            for entry in entries {
                visited += 1;
                match classify_entry(&entry, &current_rel, depth)? {
                    ListStep::Skip => {}
                    ListStep::Descend(dir) => child_dirs.push(dir),
                    ListStep::Record(display) => {
                        // Re-validate through path(): keeps listing
                        // consistent with what read() would accept.
                        if self.path(&display, false).is_ok() {
                            files.push(display);
                        }
                    }
                }
                if files.len() >= LIST_FILES_MAX || visited >= LIST_VISITED_MAX {
                    truncated = true;
                    break;
                }
            }
            if truncated {
                break;
            }
            // Reversed push keeps the pop order sorted: deterministic.
            for dir in child_dirs.into_iter().rev() {
                stack.push((dir, depth + 1));
            }
        }
        Ok(ListResult { files, truncated })
    }

    fn open_parent(
        &self,
        relative: &str,
        write: bool,
    ) -> Result<(OwnedFd, String), WorkspaceError> {
        let target = self.path(relative, write)?;
        let mut parts: Vec<String> = target
            .strip_prefix(&self.root)
            .map_err(|_| WorkspaceError::InvalidPath("path is outside workspace".to_string()))?
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect();
        if parts.is_empty() {
            return Err(WorkspaceError::InvalidPath(
                "expected a file path, not workspace root".to_string(),
            ));
        }
        assert!(
            parts.len() <= PATH_DEPTH_MAX,
            "path() already enforces the depth bound"
        );
        let name = parts.pop().unwrap_or_default();
        let directory_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW;
        let mut fd =
            rustix::fs::open(&self.root, directory_flags, Mode::empty()).map_err(syscall_err)?;
        let root_info = rustix::fs::fstat(&fd).map_err(syscall_err)?;
        if !identity_matches(&root_info, &self.identity) {
            return Err(WorkspaceError::Stale("root was replaced; restart Flow"));
        }
        for part in &parts {
            let child = match rustix::fs::openat(&fd, part.as_str(), directory_flags, Mode::empty())
            {
                Ok(child) => child,
                Err(rustix::io::Errno::NOENT) if write => {
                    rustix::fs::mkdirat(&fd, part.as_str(), Mode::from_bits_truncate(MKDIR_MODE))
                        .map_err(syscall_err)?;
                    rustix::fs::openat(&fd, part.as_str(), directory_flags, Mode::empty())
                        .map_err(syscall_err)?
                }
                Err(err) => return Err(WorkspaceError::Io(err.into())),
            };
            fd = child;
        }
        Ok((fd, name))
    }

    /// Write `data` to an exclusive temp file beside `name`, fsync, re-check
    /// freshness, and atomically rename over the target.
    fn write_temp(
        &self,
        parent: &OwnedFd,
        name: &str,
        temp_name: &str,
        data: &[u8],
    ) -> Result<(), WorkspaceError> {
        let mode = match rustix::fs::statat(parent, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(info) => {
                if FileType::from_raw_mode(info.st_mode) != FileType::RegularFile {
                    return Err(WorkspaceError::NotRegularFile);
                }
                #[cfg(unix)]
                if info.st_nlink > 1 {
                    return Err(WorkspaceError::NotRegularFile);
                }
                info.st_mode & 0o777
            }
            Err(rustix::io::Errno::NOENT) => CREATE_MODE,
            Err(err) => return Err(WorkspaceError::Io(err.into())),
        };
        let mut temp = std::fs::File::from(
            rustix::fs::openat(
                parent,
                temp_name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW,
                Mode::from_bits_truncate(mode),
            )
            .map_err(syscall_err)?,
        );
        temp.write_all(data).map_err(WorkspaceError::Io)?;
        temp.sync_all().map_err(WorkspaceError::Io)?;
        // Re-check freshness after the bytes are durable but before the
        // rename becomes visible.
        self.assert_current()?;
        rustix::fs::renameat(parent, temp_name, parent, name).map_err(syscall_err)?;
        Ok(())
    }

    /// Record a successful write in the ledger. Returns the new revision.
    fn record_write(&self, relative: &str) -> u64 {
        let normalized = Path::new(relative)
            .components()
            .filter(|component| !matches!(component, Component::CurDir))
            .collect::<PathBuf>()
            .to_string_lossy()
            .into_owned();
        let mut state = self.state.lock().unwrap_or_else(|poisoned| {
            // A poisoned ledger means a previous write panicked mid-update;
            // the data is still consistent (insert + increment are simple),
            // so continue rather than failing the write.
            poisoned.into_inner()
        });
        state.changed_files.insert(normalized);
        // Saturating: 2^64 writes is physically unreachable, and the ledger
        // must never panic.
        state.revision = state.revision.saturating_add(1);
        state.revision
    }
}

/// Split and validate a model-supplied relative path into components.
///
/// Rejects: empty strings, NUL bytes, over-long paths, absolute paths,
/// Windows drives and separators, `..`, `.git`, and excess depth.
fn split_relative(relative: &str) -> Result<Vec<String>, WorkspaceError> {
    if relative.is_empty() || relative.contains('\0') {
        return Err(WorkspaceError::InvalidPath(
            "path must be a nonempty relative string".to_string(),
        ));
    }
    if relative.chars().count() > PATH_LENGTH_MAX {
        return Err(WorkspaceError::InvalidPath(format!(
            "path exceeds {PATH_LENGTH_MAX} characters"
        )));
    }
    let path = Path::new(relative);
    if path.is_absolute() || has_windows_drive(relative) || relative.contains('\\') {
        return Err(WorkspaceError::InvalidPath(
            "absolute and Windows-style paths are not allowed".to_string(),
        ));
    }
    let mut parts: Vec<String> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => {
                if part == OsStr::new(".git") {
                    return Err(WorkspaceError::InvalidPath(
                        "parent traversal and .git access are not allowed".to_string(),
                    ));
                }
                parts.push(part.to_string_lossy().into_owned());
            }
            Component::ParentDir => {
                return Err(WorkspaceError::InvalidPath(
                    "parent traversal and .git access are not allowed".to_string(),
                ));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(WorkspaceError::InvalidPath(
                    "absolute and Windows-style paths are not allowed".to_string(),
                ));
            }
        }
    }
    if parts.len() > PATH_DEPTH_MAX {
        return Err(WorkspaceError::InvalidPath(format!(
            "path exceeds {PATH_DEPTH_MAX} components"
        )));
    }
    Ok(parts)
}

/// Detect a Windows drive (`C:`) or UNC prefix (`\\server`) without
/// depending on the host platform's path parser.
fn has_windows_drive(relative: &str) -> bool {
    let bytes = relative.as_bytes();
    (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
        || relative.starts_with("\\\\")
}

/// Convert a rustix syscall error into a workspace error.
fn syscall_err(err: rustix::io::Errno) -> WorkspaceError {
    WorkspaceError::Io(err.into())
}

/// Capture the root's pinned identity: device + inode on POSIX.
fn stat_identity(path: &Path) -> std::io::Result<RootIdentity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path)?;
        Ok(RootIdentity::DevIno {
            dev: meta.dev(),
            ino: meta.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(RootIdentity::Unpinned)
    }
}

/// Compare a freshly statted root against the pinned identity.
fn identity_matches(info: &rustix::fs::Stat, identity: &RootIdentity) -> bool {
    match identity {
        RootIdentity::DevIno { dev, ino } => info.st_dev == *dev && info.st_ino == *ino,
        RootIdentity::Unpinned => true,
    }
}

/// Build a unique temporary file name for atomic writes.
fn temp_file_name() -> String {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    format!("{TEMP_NAME_PREFIX}{}-{counter}-{nanos}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Create a fresh empty directory under the system temp dir.
    fn temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phlow-ws-test-{prefix}-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create temp dir");
        dir
    }

    fn open_trusted(root: &Path) -> Workspace {
        Workspace::open(root, true, &[]).expect("test setup: open workspace")
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn open_rejects_missing_root() {
        let missing = std::env::temp_dir().join("phlow-ws-test-no-such-dir-xyz");
        let _ = std::fs::remove_dir_all(&missing);
        assert!(Workspace::open(&missing, false, &[]).is_err());
    }

    #[test]
    fn open_rejects_file_root() {
        let dir = temp_dir("fileroot");
        let file = dir.join("f");
        std::fs::write(&file, b"x").expect("setup");
        assert!(matches!(
            Workspace::open(&file, false, &[]),
            Err(WorkspaceError::NotDirectory)
        ));
        cleanup(&dir);
    }

    #[test]
    fn path_rejects_bad_shapes() {
        let dir = temp_dir("shapes");
        let ws = open_trusted(&dir);
        for bad in [
            "",
            "/absolute",
            "a/../b",
            "..",
            "../escape",
            "a\\.git",
            ".git/config",
            "sub/.git/x",
            "C:\\windows",
            "C:/windows",
            "\\\\server\\share",
            "nul\0byte",
        ] {
            assert!(
                ws.path(bad, false).is_err(),
                "expected rejection for {bad:?}"
            );
        }
        let too_long = "a".repeat(PATH_LENGTH_MAX + 1);
        assert!(ws.path(&too_long, false).is_err());
        let deep: String = (0..PATH_DEPTH_MAX + 1)
            .map(|i| format!("d{i}"))
            .collect::<Vec<_>>()
            .join("/");
        assert!(ws.path(&deep, false).is_err());
        // Boundary: exactly at the limits is accepted.
        let at_limit = "a".repeat(PATH_LENGTH_MAX);
        assert!(ws.path(&at_limit, false).is_ok());
        cleanup(&dir);
    }

    #[test]
    fn path_rejects_symlink_components() {
        let dir = temp_dir("symlink");
        let real = dir.join("real");
        std::fs::create_dir(&real).expect("setup");
        std::os::unix::fs::symlink(&real, dir.join("link")).expect("setup");
        std::fs::write(real.join("f.txt"), b"data").expect("setup");
        let ws = open_trusted(&dir);
        assert!(ws.path("link/f.txt", false).is_err());
        assert!(ws.path("link", false).is_err());
        assert!(ws.path("real/f.txt", false).is_ok());
        cleanup(&dir);
    }

    #[test]
    fn path_rejects_multiply_linked_files() {
        let dir = temp_dir("hardlink");
        let file = dir.join("orig.txt");
        std::fs::write(&file, b"data").expect("setup");
        std::fs::hard_link(&file, dir.join("alias.txt")).expect("setup");
        let ws = open_trusted(&dir);
        assert!(ws.path("alias.txt", false).is_err());
        assert!(ws.path("orig.txt", false).is_err());
        cleanup(&dir);
    }

    #[test]
    fn read_round_trip() {
        let dir = temp_dir("roundtrip");
        std::fs::write(dir.join("hello.txt"), b"hello").expect("setup");
        let ws = open_trusted(&dir);
        let result = ws.read("hello.txt").expect("read");
        assert_eq!(result.content, "hello");
        assert_eq!(result.bytes, 5);
        assert_eq!(result.path, "hello.txt");
        cleanup(&dir);
    }

    #[test]
    fn read_rejects_symlink_final_component() {
        let dir = temp_dir("readlink");
        std::fs::write(dir.join("real.txt"), b"secret").expect("setup");
        std::os::unix::fs::symlink("real.txt", dir.join("link.txt")).expect("setup");
        let ws = open_trusted(&dir);
        // path() rejects it first; even if it didn't, O_NOFOLLOW would.
        assert!(ws.read("link.txt").is_err());
        cleanup(&dir);
    }

    #[test]
    fn read_rejects_oversize() {
        let dir = temp_dir("oversize");
        let big = vec![b'x'; FILE_BYTES_MAX as usize + 1];
        std::fs::write(dir.join("big.bin"), &big).expect("setup");
        let ws = open_trusted(&dir);
        assert!(matches!(
            ws.read("big.bin"),
            Err(WorkspaceError::TooLarge { .. })
        ));
        // Exactly at the cap reads fine.
        let exact = vec![b'y'; FILE_BYTES_MAX as usize];
        std::fs::write(dir.join("exact.bin"), &exact).expect("setup");
        assert_eq!(ws.read("exact.bin").expect("read").bytes, FILE_BYTES_MAX);
        cleanup(&dir);
    }

    #[test]
    fn read_replaces_invalid_utf8() {
        let dir = temp_dir("utf8");
        std::fs::write(dir.join("bin.txt"), [0xff, 0xfe, b'a']).expect("setup");
        let ws = open_trusted(&dir);
        let result = ws.read("bin.txt").expect("read");
        assert_eq!(result.content, "��a");
        cleanup(&dir);
    }

    #[test]
    fn write_round_trip_and_revision() {
        let dir = temp_dir("write");
        let ws = open_trusted(&dir);
        let first = ws.write("sub/dir/note.txt", "one").expect("write");
        assert_eq!(first.bytes, 3);
        assert_eq!(first.revision, 1);
        assert_eq!(ws.read("sub/dir/note.txt").expect("read").content, "one");
        let second = ws.write("sub/dir/note.txt", "two!").expect("write");
        assert_eq!(second.revision, 2);
        assert_eq!(ws.changed_files(), vec!["sub/dir/note.txt".to_string()]);
        // No temp files left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("scan")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(TEMP_NAME_PREFIX)
            })
            .collect();
        assert!(leftovers.is_empty());
        cleanup(&dir);
    }

    #[test]
    fn write_is_atomic_visible() {
        let dir = temp_dir("atomic");
        let ws = open_trusted(&dir);
        ws.write("f.txt", "before").expect("write");
        // Overwrite keeps the file present with exactly one of the contents
        // at every instant; readers never see a torn file.
        ws.write("f.txt", "after").expect("write");
        assert_eq!(ws.read("f.txt").expect("read").content, "after");
        cleanup(&dir);
    }

    #[test]
    fn write_rejects_when_untrusted() {
        let dir = temp_dir("readonly");
        let ws = Workspace::open(&dir, false, &[]).expect("open");
        assert!(matches!(
            ws.write("f.txt", "x"),
            Err(WorkspaceError::ReadOnly)
        ));
        // Reads still work.
        std::fs::write(dir.join("r.txt"), b"r").expect("setup");
        assert!(ws.read("r.txt").is_ok());
        cleanup(&dir);
    }

    #[test]
    fn write_rejects_protected_config() {
        let dir = temp_dir("protected");
        std::fs::write(dir.join("config.toml"), b"[x]").expect("setup");
        let ws = open_trusted(&dir);
        assert!(matches!(
            ws.write("config.toml", "evil = true"),
            Err(WorkspaceError::Protected)
        ));
        assert!(matches!(
            ws.write(".flow.toml", "evil = true"),
            Err(WorkspaceError::Protected)
        ));
        // Custom protected paths are honored too.
        let keep = dir.join("keep.txt");
        std::fs::write(&keep, b"k").expect("setup");
        let ws2 = Workspace::open(&dir, true, &[keep]).expect("open");
        assert!(matches!(
            ws2.write("keep.txt", "x"),
            Err(WorkspaceError::Protected)
        ));
        cleanup(&dir);
    }

    #[test]
    fn write_rejects_nonexistent_protected_path() {
        // Regression: protecting a path that does not exist yet must still
        // block writes to it. The anchor falls back to lexical
        // normalization (fail-closed); it must never silently no-op.
        let dir = temp_dir("protectedfuture");
        let future = dir.join("future.txt");
        let ws = Workspace::open(&dir, true, std::slice::from_ref(&future)).expect("open");
        assert!(matches!(
            ws.write("future.txt", "evil = true"),
            Err(WorkspaceError::Protected)
        ));
        // Rejected validation leaves published state unchanged.
        assert!(!future.exists());
        assert_eq!(ws.revision(), 0);
        cleanup(&dir);
    }

    #[test]
    fn write_rejects_oversize_content() {
        let dir = temp_dir("woversize");
        let ws = open_trusted(&dir);
        let big = "x".repeat(FILE_BYTES_MAX as usize + 1);
        assert!(matches!(
            ws.write("big.txt", &big),
            Err(WorkspaceError::TooLarge { .. })
        ));
        cleanup(&dir);
    }

    #[test]
    fn write_refuses_to_follow_symlink() {
        let dir = temp_dir("wlink");
        let outside = temp_dir("wlink-outside");
        std::os::unix::fs::symlink(&outside, dir.join("escape")).expect("setup");
        let ws = open_trusted(&dir);
        assert!(ws.write("escape/owned.txt", "pwned").is_err());
        assert!(!outside.join("owned.txt").exists());
        cleanup(&dir);
        cleanup(&outside);
    }

    #[test]
    fn symlink_swap_never_escapes() {
        // The race this crate exists to close: a component that is a real
        // directory during validation but a symlink to outside during open
        // (or vice versa) must never serve outside content. The swap uses
        // rename so the symlink is really installed, unlike remove+create
        // which cannot replace a non-empty directory.
        let dir = temp_dir("race");
        let evil_base = temp_dir("race-evil");
        let good = dir.join("race");
        std::fs::create_dir(&good).expect("setup");
        // Written once: no truncate/rewrite window for the reader to see.
        std::fs::write(good.join("file.txt"), b"good").expect("setup");
        let evil = evil_base.join("evil");
        std::fs::create_dir(&evil).expect("setup");
        std::fs::write(evil.join("file.txt"), b"EVIL").expect("setup");
        let ws = open_trusted(&dir);

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopper = stop.clone();
        let race_path = dir.join("race");
        let stash_path = dir.join("race-stash");
        let evil_target = evil.clone();
        let swapper = std::thread::spawn(move || {
            let mut as_link = false;
            while !stopper.load(Ordering::SeqCst) {
                if as_link {
                    // Real dir -> symlink to outside. There is a window
                    // where `race` is missing; reads then fail, which is
                    // safe and already tolerated below.
                    if std::fs::rename(&race_path, &stash_path).is_ok() {
                        let _ = std::os::unix::fs::symlink(&evil_target, &race_path);
                    }
                } else {
                    // Symlink -> real dir: unlink the link, rename back.
                    let _ = std::fs::remove_file(&race_path);
                    let _ = std::fs::rename(&stash_path, &race_path);
                }
                as_link = !as_link;
            }
            // Leave the tree clean for the final deterministic assertion.
            let _ = std::fs::remove_file(&race_path);
            let _ = std::fs::rename(&stash_path, &race_path);
        });
        // Liveness (at least one good read landing in the race window) is a
        // scheduler assumption, not a security property; the property under
        // test is that no read ever escapes.
        for _ in 0..400 {
            if let Ok(result) = ws.read("race/file.txt") {
                // The good file. Outside content must never appear.
                assert_eq!(result.content, "good", "escaped the workspace!");
            }
        }
        stop.store(true, Ordering::SeqCst);
        swapper.join().expect("swapper thread");
        // The swapper restored the real directory; the final state is
        // deterministic.
        let final_content = ws.read("race/file.txt").expect("final read").content;
        assert_eq!(final_content, "good");
        cleanup(&dir);
        cleanup(&evil_base);
    }

    #[test]
    fn list_scopes_to_subdirectory() {
        // Regression: list("src") must list src/, not the whole workspace.
        let dir = temp_dir("listscope");
        std::fs::create_dir_all(dir.join("src")).expect("setup");
        std::fs::write(dir.join("src").join("a.rs"), b"a").expect("setup");
        std::fs::write(dir.join("top.txt"), b"t").expect("setup");
        let ws = open_trusted(&dir);
        let result = ws.list("src").expect("list");
        assert!(!result.truncated);
        assert_eq!(result.files, vec!["src/a.rs".to_string()]);
        // The root listing still sees everything.
        let root = ws.list(".").expect("list");
        assert!(root.files.contains(&"src/a.rs".to_string()));
        assert!(root.files.contains(&"top.txt".to_string()));
        // Missing directories are rejected, not walked as the root.
        assert!(ws.list("nope").is_err());
        cleanup(&dir);
    }

    #[test]
    fn sorted_entries_caps_single_directory_reads() {
        // Regression: one directory must not force an unbounded entry
        // allocation before the visit cap is consulted. The read stops at
        // LIST_DIR_ENTRIES_MAX and reports the truncation.
        let dir = temp_dir("bigdir");
        // Fail fast with an environment-attributed message if the temp
        // filesystem is exhausted: the fixture below needs 5001 writable
        // files, and a mid-loop ENOSPC would otherwise masquerade as a
        // cap bug (observed once when /tmp was full).
        let probe = dir.join(".probe");
        std::fs::write(&probe, b"p").expect("test environment: temp dir not writable");
        std::fs::remove_file(&probe).expect("test environment: temp dir not writable");
        for index in 0..=LIST_DIR_ENTRIES_MAX {
            std::fs::write(dir.join(format!("f{index:05}.txt")), b"x").expect("setup");
        }
        let (entries, dir_truncated) = sorted_entries(&dir).expect("read entries");
        assert!(dir_truncated);
        assert_eq!(entries.len(), LIST_DIR_ENTRIES_MAX);
        // The capped read is still returned in sorted name order.
        assert!(
            entries
                .windows(2)
                .all(|pair| { pair[0].file_name() <= pair[1].file_name() })
        );
        // The flag propagates through list(): the listing is truncated.
        let ws = open_trusted(&dir);
        assert!(ws.list(".").expect("list").truncated);
        cleanup(&dir);
    }

    #[test]
    fn list_files_skips_and_caps() {
        let dir = temp_dir("list");
        std::fs::create_dir_all(dir.join("src")).expect("setup");
        std::fs::write(dir.join("src").join("a.rs"), b"a").expect("setup");
        std::fs::write(dir.join("top.txt"), b"t").expect("setup");
        std::fs::create_dir(dir.join(".git")).expect("setup");
        std::fs::write(dir.join(".git").join("hidden"), b"h").expect("setup");
        std::os::unix::fs::symlink("src/a.rs", dir.join("link.rs")).expect("setup");
        let ws = open_trusted(&dir);
        let result = ws.list(".").expect("list");
        assert!(!result.truncated);
        assert!(result.files.contains(&"src/a.rs".to_string()));
        assert!(result.files.contains(&"top.txt".to_string()));
        assert!(!result.files.iter().any(|f| f.contains(".git")));
        assert!(!result.files.iter().any(|f| f.contains("link.rs")));
        // Deterministic order.
        let again = ws.list(".").expect("list");
        assert_eq!(result.files, again.files);
        cleanup(&dir);
    }

    #[test]
    fn list_rejects_missing_dir() {
        let dir = temp_dir("listmissing");
        let ws = open_trusted(&dir);
        assert!(ws.list("nope").is_err());
        std::fs::write(dir.join("f.txt"), b"x").expect("setup");
        assert!(ws.list("f.txt").is_err());
        cleanup(&dir);
    }

    #[test]
    fn assert_current_detects_replacement() {
        let dir = temp_dir("stale");
        let ws = open_trusted(&dir);
        ws.assert_current().expect("fresh");
        // Replace the root directory with a new one at the same path.
        std::fs::remove_dir_all(&dir).expect("setup");
        std::fs::create_dir(&dir).expect("setup");
        assert!(ws.assert_current().is_err());
        cleanup(&dir);
    }
}
