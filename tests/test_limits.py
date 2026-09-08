"""Explicit limits and programmer-error assertions introduced by the Tiger Style pass.

Each test drives one bound across the valid/invalid boundary: the largest accepted input
succeeds and the smallest oversize input is rejected with the documented error type.
"""

from __future__ import annotations

import http.server
import io
import json
import sys
import threading

import pytest

from flow import checks as checks_module
from flow import mcp as mcp_module
from flow import runtime as runtime_module
from flow import workspace as workspace_module
from flow.checks import CheckRunner
from flow.config import CHECKS_MAX, CheckConfig, ConfigError, load_config
from flow.editor import TIMEOUT_S_MAX, TIMEOUT_S_MIN, EditorBridge
from flow.llm.backend import RESPONSE_BYTES_MAX, OllamaClient
from flow.mcp import MAX_FRAME_BYTES, MCPServer
from flow.runtime import SCHEMA_DEPTH_MAX, SCHEMA_NODES_MAX, validate_arguments
from flow.workspace import PATH_DEPTH_MAX, PATH_LENGTH_MAX, Workspace, WorkspaceError


def nested_schema(depth: int) -> dict:
    schema: dict = {"type": "string"}
    for _ in range(depth):
        schema = {"type": "object", "properties": {"child": schema}, "required": ["child"]}
    return schema


def nested_value(depth: int) -> object:
    value: object = "leaf"
    for _ in range(depth):
        value = {"child": value}
    return value


def test_validate_arguments_depth_limit_is_iterative_and_exact():
    validate_arguments(nested_value(SCHEMA_DEPTH_MAX), nested_schema(SCHEMA_DEPTH_MAX))
    with pytest.raises(ValueError, match="nesting depth"):
        validate_arguments(nested_value(SCHEMA_DEPTH_MAX + 1), nested_schema(SCHEMA_DEPTH_MAX + 1))
    # Far deeper than Python's recursion limit would allow with the old recursive walk.
    deep = SCHEMA_NODES_MAX * 2
    with pytest.raises(ValueError, match="nesting depth"):
        validate_arguments(nested_value(deep), nested_schema(deep))


def test_validate_arguments_node_limit():
    schema = {"type": "array", "items": {"type": "integer"}}
    validate_arguments(list(range(SCHEMA_NODES_MAX - 1)), schema)
    with pytest.raises(ValueError, match="schema nodes"):
        validate_arguments(list(range(SCHEMA_NODES_MAX)), schema)
    with pytest.raises(ValueError, match="Expected integer"):
        validate_arguments([True], schema)


def test_workspace_path_length_and_depth_limits(tmp_path):
    ws = Workspace(tmp_path)
    ws.path("a/" * (PATH_DEPTH_MAX - 1) + "leaf")
    with pytest.raises(WorkspaceError, match="components"):
        ws.path("a/" * PATH_DEPTH_MAX + "leaf")
    ws.path("x" * PATH_LENGTH_MAX)
    with pytest.raises(WorkspaceError, match="characters"):
        ws.path("x" * (PATH_LENGTH_MAX + 1))


def test_workspace_list_stops_at_file_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_module, "LIST_FILES_MAX", 3)
    monkeypatch.setattr(workspace_module, "LIST_VISITED_MAX", 3)
    for index in range(4):
        (tmp_path / f"file{index}.txt").write_text("x")
    listed = Workspace(tmp_path).list()
    assert listed["truncated"] is True
    assert len(listed["files"]) == 3
    (tmp_path / "file3.txt").unlink()
    monkeypatch.setattr(workspace_module, "LIST_FILES_MAX", 500)
    monkeypatch.setattr(workspace_module, "LIST_VISITED_MAX", 5000)
    listed = Workspace(tmp_path).list()
    assert listed["truncated"] is False
    assert sorted(listed["files"]) == ["file0.txt", "file1.txt", "file2.txt"]


def test_check_count_limit_in_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    (tmp_path / "ws").mkdir()

    def config_with(count: int) -> str:
        lines = ['workspace_dir="ws"']
        for index in range(count):
            lines.append(f'[checks.c{index}]\ncmd=["true"]')
        path = tmp_path / f"config{count}.toml"
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    assert len(load_config(config_with(CHECKS_MAX)).checks) == CHECKS_MAX
    with pytest.raises(ConfigError, match=f"At most {CHECKS_MAX}"):
        load_config(config_with(CHECKS_MAX + 1))
    with pytest.raises(ConfigError, match="Invalid check name"):
        load_config(str(_write(tmp_path, f'[checks.{"n" * 65}]\ncmd=["true"]\n')))


def _write(tmp_path, text: str):
    path = tmp_path / "bad.toml"
    path.write_text('workspace_dir="ws"\n' + text)
    return path


def test_check_runner_asserts_programmer_errors_and_bounds_output(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        CheckRunner("not a workspace", {})  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        CheckRunner(Workspace(tmp_path), {"bad": ("not", "a", "CheckConfig")})  # type: ignore
    monkeypatch.setattr(checks_module, "OUTPUT_BYTES_MAX", 10)
    checks = {"noisy": CheckConfig((sys.executable, "-c", "print('a' * 50)"))}
    result = CheckRunner(Workspace(tmp_path, trusted=True), checks).run("noisy")
    assert result["status"] == "ok"
    assert len(result["stdout"]) <= 10
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is False


def test_editor_timeout_bounds_are_asserted(tmp_path):
    EditorBridge(None, timeout_s=TIMEOUT_S_MIN).close()
    EditorBridge(None, timeout_s=TIMEOUT_S_MAX).close()
    with pytest.raises(AssertionError):
        EditorBridge(None, timeout_s=TIMEOUT_S_MIN / 2)
    with pytest.raises(AssertionError):
        EditorBridge(None, timeout_s=TIMEOUT_S_MAX + 1)
    with pytest.raises(AssertionError):
        EditorBridge(None, timeout_s="120")  # type: ignore[arg-type]


class Recorder:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_mcp_frame_limit_closes_connection_and_counts_frames():
    runtime = Recorder()
    server = MCPServer(runtime)
    ping = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
    oversize = b"{" + b" " * MAX_FRAME_BYTES + b"}\n"
    source = io.BytesIO(ping + b"\n\n" + oversize + ping + b"\n")
    target = io.StringIO()
    server.serve(source, target)
    replies = [json.loads(line) for line in target.getvalue().splitlines()]
    assert replies[0]["result"] == {}
    assert replies[1]["error"]["code"] == -32700
    assert len(replies) == 2, "oversize frame terminates the transport"
    assert server.frames_read == 3
    assert runtime.closed is True
    assert mcp_module.MAX_FRAME_BYTES == 1024 * 1024


def test_runtime_constants_relationships():
    assert runtime_module.TOOL_RESULT_PREVIEW_CHARS < runtime_module.TOOL_RESULT_CHARS_MAX
    assert runtime_module.SCHEMA_DEPTH_MAX < runtime_module.SCHEMA_NODES_MAX
    assert set(runtime_module.STATIC_CHECK_KINDS) <= {"lint", "typecheck", "diagnostics"}


class HugeOllama(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        body = b'{"models": [' + b"0," * (RESPONSE_BYTES_MAX // 2) + b"0]}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_backend_response_byte_limit():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HugeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from flow.config import OllamaConfig

        client = OllamaClient(OllamaConfig(base_url=f"http://127.0.0.1:{server.server_port}"))
        with pytest.raises(ValueError, match="byte limit"):
            client.list_models()
        assert client.is_available() is False
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
