"""Confined file operations, with descriptor-relative no-follow traversal on POSIX.

This protects Flow's file tools, not project tests from executing trusted code.
All symlinks and multiply linked regular files are deliberately rejected.
"""

from __future__ import annotations

import os
import stat
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath

# Model-facing file tools are for source code; anything larger is not something the coder
# should read into its context or rewrite wholesale.
FILE_BYTES_MAX = 256 * 1024
# Relative paths are walked component by component with no-follow opens, so their depth
# and length bound both the traversal loop and the number of descriptors touched.
PATH_DEPTH_MAX = 32
PATH_LENGTH_MAX = 1024
# Listing stops early rather than walking an arbitrarily large tree on every tool call.
LIST_FILES_MAX = 500
LIST_VISITED_MAX = 5000
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}

assert 0 < FILE_BYTES_MAX
assert 0 < PATH_DEPTH_MAX < PATH_LENGTH_MAX
assert 0 < LIST_FILES_MAX <= LIST_VISITED_MAX


class WorkspaceError(ValueError):
    pass


class Workspace:
    def __init__(self, root: str | Path, *, trusted: bool = False, protected=()):
        assert isinstance(root, (str, Path))
        assert isinstance(trusted, bool)
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError("Workspace must be a directory")
        self.trusted = trusted
        self._identity = self._stat_identity()
        self._lock = threading.RLock()
        self.changed_files: set[str] = set()
        self.revision = 0
        self._protected = {Path(item).resolve() for item in protected if item}
        self._protected.update({self.root / "config.toml", self.root / ".flow.toml"})
        # POSIX is the supported write platform; do not silently weaken race protection.
        self._posix = os.name == "posix" and hasattr(os, "O_NOFOLLOW")

    def _stat_identity(self):
        info = self.root.stat()
        return info.st_dev, info.st_ino

    def assert_current(self):
        try:
            if self.root.is_symlink() or self._stat_identity() != self._identity:
                raise WorkspaceError("stale workspace: root was replaced; restart Flow")
        except OSError as exc:
            raise WorkspaceError("stale workspace: root is unavailable; restart Flow") from exc

    def path(self, relative: str, *, write: bool = False) -> Path:
        assert isinstance(write, bool)
        self.assert_current()
        if not isinstance(relative, str) or not relative or "\0" in relative:
            raise WorkspaceError("path must be a nonempty relative string")
        # Model-supplied paths are operating input, so oversize is an error, not an assert.
        if len(relative) > PATH_LENGTH_MAX:
            raise WorkspaceError(f"path exceeds {PATH_LENGTH_MAX} characters")
        rel = Path(relative)
        if rel.is_absolute() or PureWindowsPath(relative).drive or "\\" in relative:
            raise WorkspaceError("absolute and Windows-style paths are not allowed")
        if ".." in rel.parts or ".git" in rel.parts:
            raise WorkspaceError("parent traversal and .git access are not allowed")
        if len(rel.parts) > PATH_DEPTH_MAX:
            raise WorkspaceError(f"path exceeds {PATH_DEPTH_MAX} components")
        candidate = self.root / rel
        try:
            candidate.resolve().relative_to(self.root)
        except (ValueError, RuntimeError, OSError) as exc:
            raise WorkspaceError("path is outside workspace") from exc
        current = self.root
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                raise WorkspaceError("symlinks are not allowed in file paths")
        if candidate.is_file() and candidate.stat().st_nlink > 1:
            raise WorkspaceError("multiply linked files are not allowed in file paths")
        if write:
            if not self.trusted:
                raise WorkspaceError("workspace is read-only; pass --trusted to allow edits")
            if candidate in self._protected:
                raise WorkspaceError("operator configuration cannot be modified by file tools")
        assert candidate.is_relative_to(self.root)
        return candidate

    @contextmanager
    def _parent(self, relative: str, *, write: bool = False):
        path = self.path(relative, write=write)
        if path == self.root:
            raise WorkspaceError("expected a file path, not workspace root")
        if not self._posix:
            raise WorkspaceError("secure descriptor-relative file I/O requires POSIX")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(self.root, flags)
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != self._identity:
                raise WorkspaceError("stale workspace")
            parts = path.relative_to(self.root).parts
            assert 0 < len(parts) <= PATH_DEPTH_MAX
            for part in parts[:-1]:
                try:
                    child = os.open(part, flags, dir_fd=fd)
                except FileNotFoundError:
                    if not write:
                        raise
                    os.mkdir(part, mode=0o755, dir_fd=fd)
                    child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd, parts[-1]
        finally:
            os.close(fd)

    def read(self, path: str) -> dict:
        with self._lock, self._parent(path) as (parent, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
                    raise WorkspaceError("only singly-linked regular files can be read")
                # Read one byte past the limit so oversize is detected without buffering more.
                data = handle.read(FILE_BYTES_MAX + 1)
            if len(data) > FILE_BYTES_MAX:
                raise WorkspaceError(f"file exceeds {FILE_BYTES_MAX} byte limit")
            assert len(data) <= FILE_BYTES_MAX
            return {
                "status": "ok",
                "path": path,
                "content": data.decode("utf-8", errors="replace"),
                "bytes": len(data),
            }

    def write(self, path: str, content: str) -> dict:
        if not isinstance(content, str):
            raise WorkspaceError("content must be a string")
        data = content.encode("utf-8")
        if len(data) > FILE_BYTES_MAX:
            raise WorkspaceError(f"content exceeds {FILE_BYTES_MAX} byte limit")
        with self._lock, self._parent(path, write=True) as (parent, name):
            mode = 0o644
            try:
                old = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISREG(old.st_mode) or old.st_nlink > 1:
                    raise WorkspaceError("only singly-linked regular files can be overwritten")
                mode = stat.S_IMODE(old.st_mode) & 0o777
            except FileNotFoundError:
                pass
            temp = f".flow-write-{uuid.uuid4().hex}"
            fd = os.open(
                temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=parent
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                self.assert_current()
                os.replace(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(temp, dir_fd=parent)
                except FileNotFoundError:
                    pass
            self.changed_files.add(str(Path(path)))
            self.revision += 1
            assert self.revision > 0
            return {"status": "ok", "path": path, "bytes": len(data), "revision": self.revision}

    def list(self, path: str = ".") -> dict:
        directory = self.path(path)
        if not directory.is_dir():
            raise WorkspaceError("directory not found")
        files: list[str] = []
        visited = 0
        # fwalk holds directory descriptors and never follows symlink directories.
        for base, dirs, names, fd in os.fwalk(directory, follow_symlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
            for name in sorted(names):
                visited += 1
                relative = str((Path(base) / name).relative_to(self.root))
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    self.path(relative)
                    if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        files.append(relative)
                except (OSError, WorkspaceError):
                    continue
                if len(files) >= LIST_FILES_MAX:
                    return {"status": "ok", "files": files, "truncated": True}
                if visited >= LIST_VISITED_MAX:
                    return {"status": "ok", "files": files, "truncated": True}
        assert len(files) < LIST_FILES_MAX
        assert visited < LIST_VISITED_MAX
        return {"status": "ok", "files": files, "truncated": False}
