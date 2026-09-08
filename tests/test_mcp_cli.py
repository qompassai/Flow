from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flow.mcp import MAX_FRAME_BYTES, PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]


def request(method, params=None, ident=1):
    result = {"jsonrpc": "2.0", "method": method}
    if ident != "notification":
        result["id"] = ident
    if params is not None:
        result["params"] = params
    return result


def initialized():
    return [
        request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        ),
        request("notifications/initialized", ident="notification"),
    ]


def execute(tmp_path, requests, *, config=None, trusted=False):
    command = [sys.executable, "-m", "flow", "serve", "--workspace", str(tmp_path)]
    if config:
        command.extend(["--config", str(config)])
    if trusted:
        command.append("--trusted")
    data = (
        requests
        if isinstance(requests, bytes)
        else ("\n".join(json.dumps(r) for r in requests) + "\n").encode()
    )
    result = subprocess.run(
        command,
        input=data,
        capture_output=True,
        timeout=15,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT), "XDG_CONFIG_HOME": str(tmp_path / "xdg")},
    )
    return result, [json.loads(line) for line in result.stdout.splitlines()]


def test_real_subprocess_initialize_status_list_ping_and_clean_eof(tmp_path):
    requests = initialized() + [
        request("tools/list", {}, "list"),
        request("tools/call", {"name": "flow_status", "arguments": {}}, "status"),
        request("ping", {}, "ping"),
    ]
    result, frames = execute(tmp_path, requests)
    assert result.returncode == 0 and result.stderr == b""
    assert [f["id"] for f in frames] == [1, "list", "status", "ping"]
    assert frames[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert {t["name"] for t in frames[1]["result"]["tools"]} == {
        "flow_run",
        "flow_check",
        "flow_status",
    }
    tool = frames[2]["result"]
    assert json.loads(tool["content"][0]["text"]) == tool["structuredContent"]
    assert tool["structuredContent"]["workspace"] == str(tmp_path)
    assert tool["structuredContent"]["backend"]["connectivity"] == "unverified"
    assert tool["isError"] is False


def test_fragmented_request_is_framed_by_newline_not_read_boundary(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    process = subprocess.Popen(
        [sys.executable, "-m", "flow", "serve", "--workspace", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=tmp_path,
    )
    data = (json.dumps(initialized()[0]) + "\n").encode()
    process.stdin.write(data[:15])
    process.stdin.flush()
    process.stdin.write(data[15:])
    process.stdin.flush()
    out, err = process.communicate(timeout=10)
    frames = [json.loads(line) for line in out.splitlines()]
    assert len(frames) == 1 and frames[0]["id"] == 1
    assert not err and process.returncode == 0


def test_parse_errors_recover_and_protocol_stdout_is_clean(tmp_path):
    data = b'not-json\n[]\n{"jsonrpc":"2.0","method":"ping","id":true}\n'
    data += (json.dumps(request("ping", {}, "alive")) + "\n").encode()
    result, frames = execute(tmp_path, data)
    assert [r.get("error", {}).get("code") for r in frames[:3]] == [-32700, -32600, -32600]
    assert frames[-1] == {"jsonrpc": "2.0", "id": "alive", "result": {}}
    assert not result.stderr


def test_unsupported_protocol_and_lifecycle_errors(tmp_path):
    _, frames = execute(
        tmp_path,
        [
            request("tools/list", {}, 1),
            request("initialize", {"protocolVersion": "1900-01-01"}, 2),
            *initialized(),
            request("initialize", {"protocolVersion": PROTOCOL_VERSION}, 3),
            request("no-such-method", {}, 4),
        ],
    )
    assert [frames[i]["error"]["code"] for i in [0, 1, 3, 4]] == [
        -32002,
        -32602,
        -32600,
        -32601,
    ]


@pytest.mark.parametrize(
    "params",
    [
        {"name": "shell_exec", "arguments": {"command": "python -c pass"}},
        {"name": "flow_check", "arguments": {"name": "x", "cwd": "/"}},
        {"name": "flow_check", "arguments": {"argv": ["python", "-c", "pass"]}},
        {"name": "flow_status", "arguments": []},
        {"name": "flow_run", "arguments": {"task": 123}},
        {"name": "flow_run", "arguments": {}},
        {"name": "flow_status", "arguments": {}, "cwd": "/"},
    ],
)
def test_mcp_rejects_unknown_tools_and_argument_injection(tmp_path, params):
    _, frames = execute(tmp_path, initialized() + [request("tools/call", params, 2)])
    assert frames[-1]["error"]["code"] == -32602


def test_notification_never_executes_a_tool(tmp_path):
    config = tmp_path / "operator.toml"
    config.write_text(
        "[checks.write]\ncmd="
        + json.dumps(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('pwned').write_text('bad')",
            ]
        )
        + "\n"
    )
    _, frames = execute(
        tmp_path,
        initialized()
        + [
            request(
                "tools/call", {"name": "flow_check", "arguments": {"name": "write"}}, "notification"
            ),
            request("notifications/whatever", {}, "notification"),
            request("ping", {}, 3),
        ],
        config=config,
        trusted=True,
    )
    assert len(frames) == 2
    assert not (tmp_path / "pwned").exists()


def test_real_configured_check_and_missing_check_envelope(tmp_path):
    config = tmp_path / "operator.toml"
    config.write_text(
        "[checks.smoke]\ncmd="
        + json.dumps(
            [
                sys.executable,
                "-c",
                "print('check output is inside JSON only')",
            ]
        )
        + "\ntimeout=1000\n"
    )
    result, frames = execute(
        tmp_path,
        initialized()
        + [
            request("tools/call", {"name": "flow_check", "arguments": {"name": "smoke"}}, 2),
            request("tools/call", {"name": "flow_check", "arguments": {"name": "missing"}}, 3),
        ],
        config=config,
        trusted=True,
    )
    assert result.stderr == b""
    assert frames[1]["result"]["isError"] is False
    assert frames[1]["result"]["structuredContent"]["verified"] is True
    assert frames[2]["result"]["isError"] is True
    assert frames[2]["result"]["structuredContent"]["verified"] is False
    assert len(frames) == 3


def test_oversized_frame_reports_error_and_closes_boundedly(tmp_path):
    result, frames = execute(tmp_path, b"x" * (MAX_FRAME_BYTES + 2) + b"\n")
    assert result.returncode == 0
    assert len(frames) == 1 and frames[0]["error"]["code"] == -32700


def test_cli_workspace_override_no_stale_registration_and_exit_codes(tmp_path):
    other = tmp_path / "actual"
    other.mkdir()
    config = tmp_path / "operator.toml"
    config.write_text(
        'workspace_dir="."\n[checks.cwd]\ncmd='
        + json.dumps(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; assert Path.cwd() == Path({str(other)!r})",
            ]
        )
        + "\n"
    )
    command = [
        sys.executable,
        "-m",
        "flow",
        "--config",
        str(config),
        "--trusted",
        "check",
        "--workspace",
        str(other),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified"]
    missing = subprocess.run(
        [sys.executable, "-m", "flow", "check", "--workspace", str(other)],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "xdg")},
    )
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["status"] == "unverified"


def test_sigterm_unwinds_named_check_process_group(tmp_path):
    config = tmp_path / "operator.toml"
    child = "import time; from pathlib import Path; time.sleep(1); Path('orphan').write_text('bad')"
    code = (
        f"import subprocess,time; from pathlib import Path; "
        f"subprocess.Popen([{sys.executable!r},'-c',{child!r}]); "
        "Path('started').write_text('yes'); time.sleep(10)"
    )
    config.write_text(
        "[checks.long]\ncmd=" + json.dumps([sys.executable, "-c", code]) + "\ntimeout=15000\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "flow",
            "serve",
            "--workspace",
            str(tmp_path),
            "--trusted",
            "--config",
            str(config),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        env={**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "xdg")},
    )
    try:
        data = initialized() + [
            request("tools/call", {"name": "flow_check", "arguments": {"name": "long"}}, 2),
        ]
        process.stdin.write(("\n".join(json.dumps(v) for v in data) + "\n").encode())
        process.stdin.flush()
        deadline = time.monotonic() + 5
        while not (tmp_path / "started").exists():
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(0.01)
        process.terminate()
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 130
        assert b"interrupted" in stderr
        assert len(stdout.splitlines()) == 1  # initialize only, no invented check response
        time.sleep(1.1)
        assert not (tmp_path / "orphan").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
