"""Named, operator-approved checks. No model-controlled argv, shell, cwd or environment."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict

from flow.config import CHECKS_MAX, CheckConfig
from flow.workspace import Workspace, WorkspaceError

# Only the tail of each stream is reported, because failures print their cause last and a
# noisy check must not be able to inflate a report (or the MCP frame carrying it) unboundedly.
OUTPUT_BYTES_MAX = 16000
MILLISECONDS_PER_SECOND = 1000

assert OUTPUT_BYTES_MAX > 0


class CheckRunner:
    def __init__(self, workspace: Workspace, checks: dict[str, CheckConfig]):
        assert isinstance(workspace, Workspace)
        assert isinstance(checks, dict)
        assert len(checks) <= CHECKS_MAX
        for name, check in checks.items():
            assert isinstance(name, str)
            assert isinstance(check, CheckConfig)
        self.workspace = workspace
        self.checks = dict(checks)

    def run(self, name: str) -> dict:
        assert isinstance(name, str)
        if name not in self.checks:
            return {
                "name": name,
                "status": "unavailable",
                "error": "No such configured check",
                "source": "flow.check",
            }
        check = self.checks[name]
        result = {
            "name": name,
            "source": "flow.check",
            **asdict(check),
            "workspace": str(self.workspace.root),
            "revision": self.workspace.revision,
        }
        if not self.workspace.trusted:
            return {**result, "status": "unverified", "error": "Named checks require --trusted"}
        try:
            self.workspace.assert_current()
        except WorkspaceError as exc:
            return {**result, "status": "stale", "error": str(exc)}
        started = time.monotonic()
        # Temporary files avoid unbounded memory consumption by noisy child processes.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            result.update(self._execute(check, stdout, stderr))
            result.update(_tail("stdout", stdout))
            result.update(_tail("stderr", stderr))
        result["duration_ms"] = round((time.monotonic() - started) * MILLISECONDS_PER_SECOND)
        assert result["duration_ms"] >= 0
        try:
            self.workspace.assert_current()
        except WorkspaceError as exc:
            result.update(status="stale", error=str(exc))
        assert "status" in result
        return result

    def _execute(self, check: CheckConfig, stdout, stderr) -> dict:
        """Run one check to completion or timeout, always reaping its whole process group."""
        assert check.timeout > 0
        outcome: dict = {}
        process = None
        try:
            # Process groups ensure a timed-out check's grandchildren are terminated on POSIX.
            process = subprocess.Popen(
                check.cmd,
                cwd=self.workspace.root,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=os.name == "posix",
            )
            process.wait(timeout=check.timeout / MILLISECONDS_PER_SECOND)
            outcome["status"] = "ok" if process.returncode == 0 else "failed"
            outcome["returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            outcome.update(status="timeout", error=f"Check exceeded {check.timeout}ms")
        except FileNotFoundError as exc:
            outcome.update(status="unavailable", error=str(exc))
        except OSError as exc:
            outcome.update(status="error", error=str(exc))
        finally:
            if process is not None:
                # Also reap descendants that outlive an otherwise successful parent.
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    elif process.poll() is None:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
        assert outcome["status"] in {"ok", "failed", "timeout", "unavailable", "error"}
        return outcome

    def run_all(self, name: str | None = None) -> dict:
        assert name is None or isinstance(name, str)
        if name is not None:
            checks = [self.run(name)]
            required = checks
        else:
            checks = [self.run(configured) for configured in self.checks]
            required = [check for check in checks if check.get("required")]
        assert len(checks) <= CHECKS_MAX
        verified = bool(required) and all(check["status"] == "ok" for check in required)
        status = (
            "ok"
            if verified
            else (
                "failed" if any(check["status"] == "failed" for check in required) else "unverified"
            )
        )
        return {
            "status": status,
            "verified": verified,
            "checks": checks,
            "reason": ""
            if verified
            else "Every required check must run and pass; "
            "at least one required check must be configured",
        }


def _tail(stream_name: str, handle) -> dict:
    """Decode at most OUTPUT_BYTES_MAX trailing bytes of a captured stream."""
    assert stream_name in {"stdout", "stderr"}
    size = handle.tell()
    assert size >= 0
    handle.seek(max(0, size - OUTPUT_BYTES_MAX))
    text = handle.read(OUTPUT_BYTES_MAX).decode("utf-8", errors="replace")
    assert len(text) <= OUTPUT_BYTES_MAX
    return {stream_name: text, f"{stream_name}_truncated": size > OUTPUT_BYTES_MAX}
