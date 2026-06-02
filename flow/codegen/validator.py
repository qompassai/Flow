"""Code validator — wraps LSP check and feeds results to the agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from flow.tools.registry import ToolRegistry

console = Console()


class CodeValidator:
    """Runs LSP checks and formats results for the agent."""

    def __init__(self, registry: "ToolRegistry", workspace: str = "."):
        self.registry = registry
        self.workspace = Path(workspace)

    def validate(self, language: str, file_path: str | None = None) -> dict:
        """Run LSP check and return structured result."""
        result_str = self.registry.dispatch(
            "lsp_check",
            {"language": language, "file_path": file_path},
        )
        try:
            return json.loads(result_str) if isinstance(result_str, str) else result_str
        except Exception as e:
            return {"error": str(e)}

    def is_clean(self, language: str, file_path: str | None = None) -> bool:
        """Return True if code passes LSP check with no errors."""
        result = self.validate(language, file_path)
        return result.get("clean", False)

    def format_for_agent(self, language: str, file_path: str | None = None) -> str:
        """Run LSP check and return a human-readable string for the agent prompt."""
        result = self.validate(language, file_path)
        if result.get("error"):
            return f"LSP Error: {result['error']}"
        if result.get("clean"):
            return f"✓ {language} code passes all checks."
        diagnostics = result.get("diagnostics", result.get("raw_output", "Unknown errors"))
        return f"LSP Issues ({language}):\n{json.dumps(diagnostics, indent=2)[:2000]}"
