"""Tool registry and dispatcher."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

console = Console()


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable,
    ) -> None:
        self._tools[name] = Tool(name, description, parameters, func)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool_descriptions(self) -> str:
        lines = []
        for tool in self._tools.values():
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in tool.parameters.get("properties", {}).items()
            )
            lines.append(f"- **{tool.name}**({params}): {tool.description}")
        return "\n".join(lines)

    def openai_tools_schema(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def dispatch(self, tool_name: str, args: dict) -> Any:
        tool = self.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return tool.call(**args)
        except Exception as e:
            return {"error": f"Tool {tool_name} failed: {e}"}

    def load_plugins(self, plugins_dir: str | Path) -> None:
        """Load user-defined tool plugins from a directory."""
        plugins_path = Path(plugins_dir)
        if not plugins_path.exists():
            return

        for py_file in plugins_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                # Look for TOOL_SPEC dict and run() function
                if hasattr(mod, "TOOL_SPEC") and hasattr(mod, "run"):
                    spec_data = mod.TOOL_SPEC
                    self.register(
                        name=spec_data["name"],
                        description=spec_data["description"],
                        parameters=spec_data.get("parameters", {"type": "object", "properties": {}}),
                        func=mod.run,
                    )
                    console.print(f"[green]Loaded plugin:[/green] {spec_data['name']}")
                else:
                    console.print(
                        f"[yellow]Plugin {py_file.name} missing TOOL_SPEC or run(),[/yellow] skipping."
                    )
            except Exception as e:
                console.print(f"[red]Failed to load plugin {py_file.name}:[/red] {e}")

    def parse_tool_call(self, text: str) -> tuple[str, dict] | None:
        """Parse a tool call from LLM output text."""
        text = text.strip()
        # Try direct JSON parse
        try:
            data = json.loads(text)
            if "tool" in data and "args" in data:
                return data["tool"], data["args"]
        except json.JSONDecodeError:
            pass

        # Try finding JSON in text
        pattern = r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}[^{}]*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if "tool" in data and "args" in data:
                    return data["tool"], data["args"]
            except json.JSONDecodeError:
                continue

        return None
