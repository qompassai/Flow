"""Configuration loader for Flow."""
import os
import tomllib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "flow" / "config.toml"
LOCAL_CONFIG_PATH = Path("config.toml")


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:14b"
    temperature: float = 0.2
    context_length: int = 16384
    timeout: int = 120


@dataclass
class AgentConfig:
    max_iterations: int = 20
    pause_on_uncertainty: bool = True
    uncertainty_threshold: float = 0.7
    human_in_loop_keywords: list[str] = field(default_factory=lambda: [
        "unsure", "unclear", "multiple options", "which approach",
        "depends on", "preference", "decision needed"
    ])


@dataclass
class ToolsConfig:
    web_search_backend: str = "duckduckgo"  # "duckduckgo" | "searxng"
    searxng_url: str = "http://localhost:8080"
    shell_timeout: int = 30
    shell_allowed_commands: list[str] = field(default_factory=lambda: [
        "cargo", "python", "python3", "pip", "rustc", "npm", "node",
        "go", "lua", "bash", "sh", "git", "make", "cmake", "meson",
        "ninja", "gcc", "clang", "ruff", "black", "mypy", "pylint",
        "eslint", "prettier", "gofmt", "rustfmt", "stylua"
    ])
    plugins_dir: str = "plugins"


@dataclass
class SelfImproveConfig:
    enabled: bool = True
    feedback_after_session: bool = True
    skill_git_repo: str = "skills"
    auto_commit_skills: bool = True


@dataclass
class TuiConfig:
    theme: str = "dark"
    show_thinking: bool = True
    max_output_lines: int = 200


@dataclass
class FlowConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    self_improve: SelfImproveConfig = field(default_factory=SelfImproveConfig)
    tui: TuiConfig = field(default_factory=TuiConfig)
    workspace_dir: str = "."
    memory_db: str = str(Path.home() / ".local" / "share" / "flow" / "memory.db")


def load_config(path: Path | None = None) -> FlowConfig:
    """Load config from TOML file, falling back to defaults."""
    cfg_path = path or (LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH)

    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            raw = tomllib.load(f)

    cfg = FlowConfig()

    if "ollama" in raw:
        for k, v in raw["ollama"].items():
            if hasattr(cfg.ollama, k):
                setattr(cfg.ollama, k, v)

    if "agent" in raw:
        for k, v in raw["agent"].items():
            if hasattr(cfg.agent, k):
                setattr(cfg.agent, k, v)

    if "tools" in raw:
        for k, v in raw["tools"].items():
            if hasattr(cfg.tools, k):
                setattr(cfg.tools, k, v)

    if "self_improve" in raw:
        for k, v in raw["self_improve"].items():
            if hasattr(cfg.self_improve, k):
                setattr(cfg.self_improve, k, v)

    if "tui" in raw:
        for k, v in raw["tui"].items():
            if hasattr(cfg.tui, k):
                setattr(cfg.tui, k, v)

    if "workspace_dir" in raw:
        cfg.workspace_dir = raw["workspace_dir"]
    if "memory_db" in raw:
        cfg.memory_db = raw["memory_db"]

    return cfg
