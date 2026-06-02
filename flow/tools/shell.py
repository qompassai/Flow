"""Shell execution tool with allowlist and timeout."""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path


def make_shell_tool(allowed_commands: list[str], timeout: int = 30, workspace: str = "."):
    """Factory returning a sandboxed shell execution function."""
    allowed_set = set(allowed_commands)

    def run(command: str, cwd: str | None = None) -> str:
        parts = shlex.split(command)
        if not parts:
            return json.dumps({"error": "Empty command"})

        base_cmd = parts[0]
        if base_cmd not in allowed_set:
            return json.dumps({
                "error": f"Command '{base_cmd}' not in allowlist. Allowed: {sorted(allowed_set)}"
            })

        work_dir = Path(cwd) if cwd else Path(workspace)

        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )
            return json.dumps({
                "stdout": result.stdout[-4000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"Command timed out after {timeout}s"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return run


TOOL_SPEC = {
    "name": "shell_exec",
    "description": "Execute a shell command in the workspace directory. Only allowed commands can run.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "cwd": {"type": "string", "description": "Working directory (relative to workspace)"},
        },
        "required": ["command"],
    },
}
