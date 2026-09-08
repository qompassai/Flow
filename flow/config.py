"""Validated operator configuration. Project configuration is never auto-discovered."""

from __future__ import annotations

import ipaddress
import math
import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit

# Named checks are the only executable surface; keeping their count and argv short bounds the
# work `flow check` can be asked to do and the size of every status/report payload.
CHECKS_MAX = 32
CHECK_ARGV_MAX = 128
CHECK_NAME_LENGTH_MAX = 64
CHECK_TIMEOUT_MS_MAX = 600000
CHECK_KINDS = frozenset({"check", "test", "lint", "typecheck", "diagnostics", "build"})

assert 0 < CHECKS_MAX <= 256
assert 0 < CHECK_ARGV_MAX
assert 0 < CHECK_NAME_LENGTH_MAX
assert 0 < CHECK_TIMEOUT_MS_MAX


class ConfigError(ValueError):
    pass


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.2
    context_length: int = 16384
    timeout: float = 120
    allow_remote: bool = False


@dataclass
class ModelsConfig:
    planner: str = ""
    coder: str = ""
    reviewer: str = ""


@dataclass
class AgentConfig:
    max_iterations: int = 6  # model turns per role, per cycle
    max_cycles: int = 2
    max_tool_calls: int = 64  # global budget, across all roles and cycles
    max_context_chars: int = 100000
    max_task_chars: int = 16000
    # Sequential role scheduling is intentional for on-device memory and single-writer safety.


@dataclass(frozen=True)
class CheckConfig:
    cmd: tuple[str, ...]
    timeout: int = 60000  # milliseconds, same units as Rose
    required: bool = True
    filetypes: tuple[str, ...] = ()  # descriptive, never silently skips required checks
    kind: str = "check"  # lint/typecheck/diagnostics provide static language verification


@dataclass
class FlowConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    checks: dict[str, CheckConfig] = field(default_factory=dict)
    workspace_dir: str = "."
    trusted: bool = False  # ONLY supplied by the CLI/API, never accepted from TOML
    config_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def model_for(self, role: str) -> str:
        return getattr(self.models, role) or self.ollama.model


def _number(value, name: str, minimum: float, maximum: float, integer=False):
    assert minimum <= maximum
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigError(
            f"{name} must be {'an integer' if integer else 'a number'} "
            f"between {minimum} and {maximum}"
        )


def validate_config(cfg: FlowConfig) -> None:
    if not isinstance(cfg.workspace_dir, str) or not cfg.workspace_dir:
        raise ConfigError("workspace_dir must be a nonempty string at the TOML root")
    for key in ("model",):
        if not isinstance(getattr(cfg.ollama, key), str) or not getattr(cfg.ollama, key).strip():
            raise ConfigError(f"ollama.{key} must be a nonempty string")
    for role in ("planner", "coder", "reviewer"):
        if not isinstance(getattr(cfg.models, role), str):
            raise ConfigError(f"models.{role} must be a string")
    if type(cfg.ollama.allow_remote) is not bool or type(cfg.trusted) is not bool:
        raise ConfigError("trust flags must be booleans")
    if not isinstance(cfg.ollama.base_url, str):
        raise ConfigError("ollama.base_url must be a URL string")
    try:
        url = urlsplit(cfg.ollama.base_url)
        _ = url.port
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("expected HTTP(S) origin without credentials, query or path")
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            local = url.hostname == "localhost"
        if not local and not cfg.ollama.allow_remote:
            raise ValueError("non-loopback Ollama requires explicit ollama.allow_remote=true")
    except ValueError as exc:
        raise ConfigError(f"Invalid ollama.base_url: {exc}") from exc
    _number(cfg.ollama.temperature, "ollama.temperature", 0, 2)
    _number(cfg.ollama.timeout, "ollama.timeout", 0.1, 600)
    _number(cfg.ollama.context_length, "ollama.context_length", 1024, 131072, True)
    for name, low, high in (
        ("max_iterations", 1, 32),
        ("max_cycles", 1, 5),
        ("max_tool_calls", 1, 256),
        ("max_context_chars", 4096, 1000000),
        ("max_task_chars", 1, 100000),
    ):
        _number(getattr(cfg.agent, name), f"agent.{name}", low, high, True)
    if len(cfg.checks) > CHECKS_MAX:
        raise ConfigError(f"At most {CHECKS_MAX} named checks are supported")
    for name, check in cfg.checks.items():
        _validate_check(name, check)


def _validate_check(name: str, check: CheckConfig) -> None:
    if not isinstance(name, str) or len(name) > CHECK_NAME_LENGTH_MAX:
        raise ConfigError(f"Invalid check name: {name!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ConfigError(f"Invalid check name: {name!r}")
    if not isinstance(check, CheckConfig):
        raise ConfigError(f"checks.{name} must be a check table")
    if not check.cmd or len(check.cmd) > CHECK_ARGV_MAX:
        raise ConfigError(f"checks.{name}.cmd must be a nonempty string argv array")
    if not all(isinstance(arg, str) and arg and "\0" not in arg for arg in check.cmd):
        raise ConfigError(f"checks.{name}.cmd must be a nonempty string argv array")
    _number(check.timeout, f"checks.{name}.timeout", 1, CHECK_TIMEOUT_MS_MAX, True)
    if type(check.required) is not bool:
        raise ConfigError(f"checks.{name}.required must be a boolean")
    if not all(isinstance(filetype, str) for filetype in check.filetypes):
        raise ConfigError(f"checks.{name}.filetypes must be a string array")
    if check.kind not in CHECK_KINDS:
        raise ConfigError(f"checks.{name}.kind is not a supported check kind")


def _section(cls, raw: dict, name: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"{name} must be a TOML table")
    unknown = set(raw) - {field_info.name for field_info in fields(cls)}
    if unknown:
        raise ConfigError(f"Unknown {name} options: {', '.join(sorted(unknown))}")
    return cls(**raw)


def load_config(
    path: str | Path | None = None,
    *,
    workspace: str | Path | None = None,
    trusted: bool = False,
    model: str | None = None,
) -> FlowConfig:
    """An explicit --config approves its contents, but does not grant execution trust.

    Only the user's XDG config is auto-loaded. No workspace-local plugin, prompt or
    TOML is automatically consumed, including when --trusted is set.
    """
    default = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "flow/config.toml"
    cfg_path = Path(path).expanduser().resolve() if path is not None else default.resolve()
    raw = {}
    if path is not None or cfg_path.exists():
        try:
            with cfg_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"Cannot load configuration {cfg_path}: {exc}") from exc
    unknown = set(raw) - {"workspace_dir", "ollama", "models", "agent", "checks"}
    if unknown:
        raise ConfigError(
            f"Unknown config options: {', '.join(sorted(unknown))}. "
            "Legacy shell/plugins/self_improve options are not supported."
        )
    cfg = FlowConfig(
        ollama=_section(OllamaConfig, raw.get("ollama", {}), "ollama"),
        models=_section(ModelsConfig, raw.get("models", {}), "models"),
        agent=_section(AgentConfig, raw.get("agent", {}), "agent"),
        trusted=trusted,
        config_path=str(cfg_path) if raw or cfg_path.exists() else None,
    )
    checks = raw.get("checks", {})
    if not isinstance(checks, dict):
        raise ConfigError("checks must be a table of named commands")
    if len(checks) > CHECKS_MAX:
        raise ConfigError(f"At most {CHECKS_MAX} named checks are supported")
    for name, values in checks.items():
        check = _section(CheckConfig, values, f"checks.{name}")
        if not isinstance(check.cmd, list) or not isinstance(check.filetypes, (tuple, list)):
            raise ConfigError(f"checks.{name}.cmd and filetypes must be arrays")
        cfg.checks[name] = CheckConfig(
            tuple(check.cmd), check.timeout, check.required, tuple(check.filetypes), check.kind
        )
    selected = workspace if workspace is not None else raw.get("workspace_dir", ".")
    if not isinstance(selected, (str, Path)):
        raise ConfigError("workspace_dir must be a string")
    # Config-relative paths are anchored to the config, CLI-relative paths to the caller.
    base = cfg_path.parent if workspace is None and "workspace_dir" in raw else Path.cwd()
    cfg.workspace_dir = str((base / Path(selected).expanduser()).resolve())
    if model is not None:
        cfg.ollama.model = model
        cfg.models = ModelsConfig()  # --model deliberately overrides every role
    validate_config(cfg)
    root = Path(cfg.workspace_dir)
    if not root.is_dir():
        raise ConfigError(f"Workspace must already be a directory: {root}")
    for local in (root / "config.toml", root / ".flow.toml"):
        if local.exists() and str(local.resolve()) != cfg.config_path:
            cfg.warnings.append(f"Ignored project configuration: {local}; use --config explicitly")
    return cfg
