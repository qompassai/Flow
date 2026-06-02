"""
Example user-defined tool plugin for Flow.

To create your own tool:
1. Copy this file to plugins/your_tool_name.py
2. Fill in TOOL_SPEC with your tool's name, description, and parameter schema
3. Implement the run() function
4. Flow will automatically load it on startup or when you run /plugins

The run() function receives keyword arguments matching the parameters in TOOL_SPEC
and must return a string (JSON is recommended for structured data).
"""

TOOL_SPEC = {
    "name": "example_greet",
    "description": "Greet someone by name. Replace this with your actual tool.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name to greet",
            },
        },
        "required": ["name"],
    },
}


def run(name: str) -> str:
    """Your tool logic goes here. Must return a string."""
    return f"Hello, {name}! This is a custom Flow tool."
