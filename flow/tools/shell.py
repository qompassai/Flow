"""Removed unsafe compatibility API. Executable basenames are not a security boundary."""

import json


def make_shell_tool(allowed_commands=(), timeout=30, workspace="."):
    """Fail closed even when legacy callers supply python/bash in an allowlist."""

    def denied(command, cwd=None):
        return json.dumps(
            {
                "status": "error",
                "error": "Arbitrary commands and cwd overrides are disabled. "
                "Configure exact named check argv and use flow_check.",
            }
        )

    return denied


TOOL_SPEC = {
    "name": "shell_exec",
    "description": "DISABLED: use configured named checks",
    "parameters": {"type": "object", "properties": {}},
}
