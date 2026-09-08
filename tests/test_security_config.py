from __future__ import annotations

import json
import os

import pytest

from flow.config import ConfigError, load_config
from flow.tools.file_ops import make_file_ops_tools
from flow.tools.registry import ToolRegistry
from flow.tools.shell import make_shell_tool
from flow.workspace import FILE_BYTES_MAX, Workspace, WorkspaceError


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return root


def test_project_config_and_plugins_never_auto_loaded(isolated):
    (isolated / "config.toml").write_text('[ollama]\nbase_url="http://evil.example"\n')
    plugins = isolated / "plugins"
    plugins.mkdir()
    (plugins / "evil.py").write_text("raise Exception('executed project plugin')")
    cfg = load_config(trusted=True)
    assert cfg.ollama.base_url == "http://127.0.0.1:11434"
    assert cfg.warnings and cfg.config_path is None
    with pytest.raises(RuntimeError, match="disabled"):
        ToolRegistry().load_plugins(plugins)


def test_explicit_config_does_not_grant_trust_and_workspace_override_wins(isolated):
    other = isolated.parent / "other"
    other.mkdir()
    path = isolated / "operator.toml"
    path.write_text('workspace_dir="."\n[models]\nplanner="small"\ncoder="large"\n')
    cfg = load_config(path, workspace=other)
    assert cfg.workspace_dir == str(other)
    assert cfg.trusted is False
    assert cfg.model_for("planner") == "small"
    assert cfg.model_for("coder") == "large"
    assert cfg.model_for("reviewer") == cfg.ollama.model
    assert load_config(path, model="override").model_for("planner") == "override"


def test_workspace_is_config_relative_not_process_cwd(isolated):
    sub = isolated / "project"
    sub.mkdir()
    config = isolated / "operator.toml"
    config.write_text('workspace_dir="project"\n')
    assert load_config(config).workspace_dir == str(sub)


@pytest.mark.parametrize(
    "text",
    [
        '[tui]\nworkspace_dir="."\n',
        '[ollama]\nworkspace_dir="."\n',
        "trusted=true\n",
        '[tools]\nshell_allowed_commands=["python"]\n',
        '[checks.evil]\ncmd="python -c print(1)"\n',
        '[checks.evil]\ncmd=["python"]\ncwd="/etc"\n',
        '[checks.evil]\ncmd=["python"]\ntimeout=-1\n',
        '[checks.evil]\ncmd=["python"]\nrequired="yes"\n',
        '[checks.evil]\ncmd=["python"]\nkind="pretend"\n',
        "[checks.evil]\ncmd=[]\n",
        "[agent]\nmax_iterations=true\n",
        "[agent]\nmax_cycles=1000\n",
        "[ollama]\ntimeout=nan\n",
        '[ollama]\nbase_url="http://evil.example"\n',
        '[ollama]\nbase_url="http://127.0.0.1@evil.example"\n',
        '[ollama]\nbase_url="http://127.0.0.1:11434/redirect"\n',
        "workspace_dir=123\n",
    ],
)
def test_invalid_and_legacy_configuration_fails_closed(isolated, text):
    path = isolated / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)


def test_remote_is_only_explicit_operator_opt_in(isolated):
    path = isolated / "config.toml"
    path.write_text('[ollama]\nbase_url="https://private-ollama.example"\nallow_remote=true\n')
    assert load_config(path).ollama.allow_remote


def test_missing_config_or_workspace_is_an_error(isolated):
    with pytest.raises(ConfigError):
        load_config(isolated / "missing.toml")
    with pytest.raises(ConfigError):
        load_config(workspace=isolated / "missing")


@pytest.mark.parametrize(
    "path",
    [
        "../project-other/secret",
        "/etc/passwd",
        "a/../../secret",
        r"C:\Windows\win.ini",
        r"..\escape",
        ".git/config",
        "a\0b",
    ],
)
def test_containment_is_not_string_prefix(tmp_path, path):
    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "project-other").mkdir()
    ws = Workspace(root, trusted=True)
    with pytest.raises((ValueError, OSError)):
        ws.write(path, "no")


def test_symlinks_existing_and_missing_targets_blocked(tmp_path):
    root, other = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    other.mkdir()
    (other / "secret").write_text("outside")
    (root / "out").symlink_to(other, target_is_directory=True)
    (root / "in").write_text("inside")
    (root / "alias").symlink_to(root / "in")
    (root / "missing").symlink_to(other / "not-created")
    ws = Workspace(root, trusted=True)
    for path in ["out/secret", "out/new/deep", "alias", "missing"]:
        with pytest.raises((ValueError, OSError)):
            ws.write(path, "no")
        with pytest.raises((ValueError, OSError)):
            ws.read(path)
    assert not (other / "not-created").exists()
    assert "out/secret" not in ws.list()["files"]


def test_hardlinks_and_nonregular_files_are_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("outside")
    os.link(secret, root / "hard")
    os.mkfifo(root / "fifo")
    ws = Workspace(root, trusted=True)
    for path in ["hard", "fifo"]:
        with pytest.raises(WorkspaceError):
            ws.read(path)
        with pytest.raises(WorkspaceError):
            ws.write(path, "no")
    assert secret.read_text() == "outside"


def test_atomic_write_limits_config_and_root_staleness(tmp_path):
    ws = Workspace(tmp_path, trusted=True, protected=[tmp_path / "operator.toml"])
    assert ws.write("nested/file.py", "hello")["status"] == "ok"
    assert ws.read("nested/file.py")["content"] == "hello"
    assert ws.changed_files == {"nested/file.py"}
    assert not list((tmp_path / "nested").glob(".flow-write-*"))
    for path in ["config.toml", ".flow.toml", "operator.toml"]:
        with pytest.raises(WorkspaceError, match="configuration"):
            ws.write(path, "no")
    with pytest.raises(WorkspaceError, match="limit"):
        ws.write("nested/file.py", "x" * (FILE_BYTES_MAX + 1))
    assert ws.read("nested/file.py")["content"] == "hello"
    moved = tmp_path.with_name(tmp_path.name + "-moved")
    tmp_path.rename(moved)
    tmp_path.mkdir()
    with pytest.raises(WorkspaceError, match="stale"):
        ws.write("escaped", "no")


def test_readonly_and_legacy_executable_factory_denials(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="read-only"):
        ws.write("a", "no")
    read, write, listing, delete = make_file_ops_tools(str(tmp_path), trusted=True)
    assert json.loads(write("a", "ok"))["status"] == "ok"
    assert json.loads(read("a"))["content"] == "ok"
    assert json.loads(delete("a"))["status"] == "error"
    assert "a" in json.loads(listing())["files"]
    shell = make_shell_tool(["python", "bash", "sh"], workspace=str(tmp_path))
    for command in ['python -c \'open("pwned","w").write("x")\'', "bash -c 'touch pwned'"]:
        assert json.loads(shell(command, cwd="/tmp"))["status"] == "error"
    assert not (tmp_path / "pwned").exists()


def test_root_fd_blocks_final_component_symlink_swap(tmp_path, monkeypatch):
    """Swap the destination after path validation: atomic replace must not follow it."""
    ws = Workspace(tmp_path, trusted=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_text("secret")
    original = os.replace

    def replace(src, dst, **kwargs):
        (tmp_path / "file").symlink_to(outside)
        return original(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    ws.write("file", "safe")
    assert outside.read_text() == "secret"
    assert (tmp_path / "file").read_text() == "safe"
