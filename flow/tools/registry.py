"""In-process registry for explicitly supplied Python callbacks; no executable plugin discovery."""

from __future__ import annotations

from typing import Callable


class Tool:
    def __init__(self, name: str, description: str, parameters: dict, func: Callable):
        self.name, self.description, self.parameters, self.func = (
            name,
            description,
            parameters,
            func,
        )

    def to_openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, **kwargs):
        return self.func(**kwargs)


class ToolRegistry:
    """Legacy Python API only. Runtime does not accept custom executable callbacks."""

    def __init__(self):
        self._tools = {}

    def register(self, name, description, parameters, func):
        self._tools[name] = Tool(name, description, parameters, func)

    def get(self, name):
        return self._tools.get(name)

    def all_tools(self):
        return list(self._tools.values())

    def openai_tools_schema(self):
        return [t.to_openai_schema() for t in self.all_tools()]

    def tool_descriptions(self):
        return "\n".join(f"{t.name}: {t.description}" for t in self.all_tools())

    def dispatch(self, tool_name, args):
        from flow.runtime import validate_arguments

        try:
            tool = self.get(tool_name)
            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")
            validate_arguments(args, tool.parameters)
            return tool.call(**args)
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def load_plugins(self, plugins_dir):
        raise RuntimeError("Executable plugins are disabled. Use configured named checks.")

    def parse_tool_call(self, text):
        # Text never acquires executable authority. Use native tool_calls with ids.
        return None
