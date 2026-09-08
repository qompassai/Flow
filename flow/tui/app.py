"""Rich + prompt_toolkit interface to the exact CLI/MCP runtime (no legacy tool path)."""

from __future__ import annotations

import json

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel

from flow.config import load_config
from flow.runtime import Runtime


class FlowApp:
    def __init__(self, config_path=None, *, runtime=None, workspace=None, trusted=False):
        self.runtime = runtime or Runtime(
            load_config(config_path, workspace=workspace, trusted=trusted)
        )
        self.cfg = self.runtime.cfg
        self.workspace = self.runtime.workspace.root
        self.console = Console()

    def execute(self, text: str):
        """Small testable command handler; natural-language and /build share the agent."""
        command, _, arg = text.partition(" ")
        if command == "/status":
            return self.runtime.status()
        if command == "/check":
            return self.runtime.check(arg or None)
        if command == "/tools":
            return {"status": "ok", "tools": self.runtime.schemas()}
        if command == "/models":
            return {"status": "ok", "models": self.runtime.backend.list_models()}
        if command == "/model":
            if not arg:
                return {"status": "error", "error": "Usage: /model <installed-model>"}
            from flow.config import ModelsConfig

            self.runtime.cfg.ollama.model = arg
            self.runtime.cfg.models = ModelsConfig()
            return {"status": "ok", "model": arg}
        if command in {"/plugins", "/evolve", "/memory", "/feedback"}:
            return {
                "status": "unavailable",
                "error": "Legacy executable plugins, automatic prompt "
                "mutation and cross-workspace memory are disabled in the safe runtime",
            }
        if command in {"/help", "/clear"}:
            return {
                "status": "ok",
                "commands": [
                    "/status",
                    "/check [name]",
                    "/tools",
                    "/models",
                    "/model <name>",
                    "/build <request>",
                    "/clear",
                    "/quit",
                ],
                "note": "Every task has fresh role contexts; /clear needs no persistent cleanup",
            }
        if command == "/build":
            return self.runtime.run("Build " + arg)
        if text.startswith("/"):
            return {"status": "error", "error": "Unknown command; try /help"}
        return self.runtime.run(text)

    def run(self):
        self.console.print(
            Panel(
                "Flow · local planner / coder / reviewer\n"
                "Type /help for commands. /quit or Ctrl-D to exit.",
                title="Flow",
            )
        )
        self.console.print(f"Workspace: {self.workspace}", markup=False)
        self.console.print(
            "Trusted writes/checks enabled" if self.cfg.trusted else "Read-only mode"
        )
        session = PromptSession()
        try:
            while True:
                try:
                    text = session.prompt("flow > ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
                if text in {"/quit", "/exit"}:
                    break
                if not text:
                    continue
                try:
                    result = self.execute(text)
                    self.console.print(
                        Panel(json.dumps(result, indent=2), title="Flow result"), markup=False
                    )
                except KeyboardInterrupt:
                    self.console.print("Interrupted; saved edits remain.", markup=False)
                except Exception as exc:
                    self.console.print(f"Error: {exc}", markup=False)
        finally:
            self.runtime.close()
