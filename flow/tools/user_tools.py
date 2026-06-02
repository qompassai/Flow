"""User tools loader — convenience wrapper around registry.load_plugins."""
from __future__ import annotations

from pathlib import Path
from flow.tools.registry import ToolRegistry


def load_user_tools(registry: ToolRegistry, plugins_dir: str | Path = "plugins") -> int:
    """Load all user-defined tools from the plugins directory.
    
    Returns the number of tools successfully loaded.
    """
    before = len(registry.all_tools())
    registry.load_plugins(plugins_dir)
    after = len(registry.all_tools())
    return after - before
