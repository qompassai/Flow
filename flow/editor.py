"""Optional Flow -> Rose reverse bridge. One owning thread, fixed Lua, no TCP."""

from __future__ import annotations

import concurrent.futures
import copy
import os
import stat
import sys
import threading
from pathlib import Path

EDITOR_NAMES = frozenset(
    {
        "editor_context",
        "editor_diagnostics",
        "editor_symbols",
        "editor_references",
        "editor_lint",
        "editor_check",
        "editor_scip",
        "editor_debug",
    }
)
READ_ONLY_EDITOR_NAMES = EDITOR_NAMES - {"editor_lint", "editor_check"}
BRIDGE_FILE_NAMES = frozenset({"file_read", "file_write"})
SCHEMAS_LUA = "return require('rose.tools').schemas()"
CALL_LUA = "return require('rose.tools').call(...)"
# Bounds mirror the CLI: a request may take at most --editor-timeout, and the worker gets a
# short grace period beyond that to unwind before the bridge is declared unavailable.
TIMEOUT_S_DEFAULT = 120.0
TIMEOUT_S_MIN = 0.1
TIMEOUT_S_MAX = 660.0
WORKER_GRACE_S = 2.0
# Rose advertises a small, fixed tool set; more entries indicate a broken or foreign server.
SCHEMAS_MAX = 64

assert TIMEOUT_S_MIN < TIMEOUT_S_DEFAULT <= TIMEOUT_S_MAX
assert WORKER_GRACE_S > 0


class EditorBridge:
    """pynvim is attached, used and closed exclusively on this worker.

    A bounded async msgpack request is serviced by pynvim's owning event loop.
    Timeouts make the bridge unavailable for further calls until restarted.
    """

    def __init__(self, socket: str | None, *, timeout_s: float = TIMEOUT_S_DEFAULT):
        assert socket is None or isinstance(socket, str)
        assert isinstance(timeout_s, (int, float))
        assert TIMEOUT_S_MIN <= timeout_s <= TIMEOUT_S_MAX
        self.socket = socket
        self.timeout_s = float(timeout_s)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="flow-editor"
        )
        self._nvim = None
        self._session = None
        self._schemas = None
        self._failure = None
        self._closed = False
        self._mutex = threading.Lock()
        if socket:
            try:
                # pynvim 0.5/0.6 remembers threading.current_thread() at import time
                # as its signal-handling "main_thread". Import on the actual main
                # thread BEFORE the worker attaches. No global monkeypatch required.
                if (
                    "pynvim" not in sys.modules
                    and threading.current_thread() is not threading.main_thread()
                ):
                    raise RuntimeError("First pynvim import must occur on the main thread")
                import pynvim  # noqa: F401
            except (ImportError, RuntimeError) as exc:
                self._failure = f"Editor unavailable: install flow[editor]; {exc}"

    def _validate_socket(self):
        if not self.socket:
            raise ValueError("No --nvim socket configured")
        if os.name != "posix":
            if not self.socket.startswith("\\\\.\\pipe\\"):
                raise ValueError("Only a private named pipe is supported on Windows")
            return
        path = Path(self.socket)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("--nvim must be an absolute, non-symlink Unix socket")
        info = path.stat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Neovim socket must be owned by the current user")
        # A 0700 parent protects sockets whose own mode is affected by Neovim/umask.
        parent = path.parent.stat()
        if info.st_mode & 0o077 and (parent.st_uid != os.getuid() or parent.st_mode & 0o077):
            raise ValueError("Neovim socket must be private (0700 parent or 0600 socket)")

    def _request(self, expression: str, args: list):
        assert expression in {SCHEMAS_LUA, CALL_LUA}
        assert isinstance(args, list)
        if self._session is None:
            self._validate_socket()
            import pynvim

            # Public low-level attach APIs let us bound the initial API handshake too.
            self._session = pynvim.socket_session(self.socket)
        session = self._session
        timed_out = threading.Event()

        def interrupt():
            timed_out.set()
            session.threadsafe_call(session.stop)

        timer = threading.Timer(self.timeout_s, interrupt)
        timer.daemon = True
        timer.start()
        try:
            if self._nvim is None:
                import pynvim

                self._nvim = pynvim.Nvim.from_session(session).with_decode(True)
            result = self._nvim.exec_lua(expression, *args)
            if timed_out.is_set():
                raise TimeoutError("Neovim request timed out")
            return result
        except Exception:
            if timed_out.is_set():
                raise TimeoutError("Neovim request timed out") from None
            raise
        finally:
            timer.cancel()
            timer.join()

    def _submit(self, expression: str, args: list):
        with self._mutex:
            if self._closed or self._failure:
                raise RuntimeError(self._failure or "Editor bridge closed")
            future = self._executor.submit(self._request, expression, args)
            try:
                return future.result(timeout=self.timeout_s + WORKER_GRACE_S)
            except Exception as exc:
                self._failure = f"Neovim bridge unavailable: {exc}"
                raise RuntimeError(self._failure) from exc

    def schemas(self) -> list[dict]:
        if not self.socket:
            return []
        if self._schemas is None:
            try:
                result = self._submit(SCHEMAS_LUA, [])
                if not isinstance(result, list):
                    raise ValueError("Rose schemas must be an array")
                if len(result) > SCHEMAS_MAX:
                    raise ValueError(f"Rose returned more than {SCHEMAS_MAX} tool schemas")
                self._schemas = [
                    schema
                    for schema in result
                    if isinstance(schema, dict)
                    and schema.get("type") == "function"
                    and isinstance(schema.get("function"), dict)
                    and schema["function"].get("name") in EDITOR_NAMES
                    and isinstance(schema["function"].get("parameters"), dict)
                ]
            except Exception as exc:
                self._failure = str(exc)
                self._schemas = []
        assert len(self._schemas) <= SCHEMAS_MAX
        return copy.deepcopy(self._schemas)

    def call(self, name: str, args: dict) -> dict:
        assert isinstance(name, str)
        if name not in EDITOR_NAMES | BRIDGE_FILE_NAMES:
            return {"status": "error", "error": "Editor tool is not allowlisted"}
        if not isinstance(args, dict):
            return {"status": "error", "error": "Editor arguments must be an object"}
        if not self.socket:
            return {"status": "unavailable", "error": "No --nvim socket configured"}
        try:
            result = self._submit(CALL_LUA, [name, args])
            if not isinstance(result, dict):
                return {"status": "error", "error": "Rose returned a non-object tool result"}
            return result
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}

    def status(self) -> dict:
        schemas = self.schemas()
        return {
            "status": "ok" if schemas else "unavailable",
            "tools": [schema["function"]["name"] for schema in schemas],
            "error": self._failure or (None if schemas else "No editor bridge configured"),
        }

    def close(self):
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            if self._session is not None:
                # Public threadsafe scheduling interrupts a pending request immediately;
                # all actual session methods still execute on their owning worker.
                self._session.threadsafe_call(self._session.stop)
                self._executor.submit(self._session.close)
        self._executor.shutdown(wait=True, cancel_futures=False)
