"""Legacy inference-based command selection is disabled; checks must be explicitly named."""

import json


def run_lsp_check(language: str, file_path=None, workspace=".") -> str:
    return json.dumps(
        {
            "status": "unavailable",
            "clean": False,
            "verified": False,
            "language": language,
            "error": "No implicit language command execution. Configure a named lint/typecheck "
            "in operator TOML or attach Rose's native editor tools.",
        }
    )


TOOL_SPEC = {
    "name": "lsp_check",
    "description": "DISABLED: use configured named checks or editor tools",
    "parameters": {"type": "object", "properties": {}},
}
