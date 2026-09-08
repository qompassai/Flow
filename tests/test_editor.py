from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from flow.editor import EditorBridge

NVIM = os.environ.get("NVIM") or shutil.which("nvim")


@pytest.fixture
def nvim(tmp_path):
    pynvim = pytest.importorskip("pynvim")
    if not NVIM:
        pytest.skip("Neovim executable not present; set NVIM to run real editor tests")
    sock = tmp_path / "editor.sock"
    log = (tmp_path / "nvim.log").open("wb")
    process = subprocess.Popen(
        [NVIM, "--headless", "-u", "NONE", "--listen", str(sock)], stdout=log, stderr=log
    )
    try:
        deadline = time.monotonic() + 5
        while not sock.exists():
            assert process.poll() is None and time.monotonic() < deadline, "Neovim failed to start"
            time.sleep(0.01)
        client = pynvim.attach("socket", path=str(sock))
        client.exec_lua("""
            package.preload['rose.tools'] = function()
              return {
                schemas = function()
                  return {
                    {type='function',['function']={name='editor_context',description='fixture',
                      parameters={type='object',properties={path={type='string'}}}}},
                    {type='function',['function']={name='editor_debug',description='fixture',
                      parameters={type='object',properties={action={type='string'}}}}},
                    {type='function',['function']={name='evil_lua',parameters={type='object'}}}
                  }
                end,
                call = function(name,args)
                  if args.path == 'slow' then vim.wait(1000, function() return false end) end
                  return {status='ok',name=name,args=args,workspace=vim.fn.getcwd()}
                end,
              }
            end
        """)
        yield sock, client
        try:
            client.command("qa!")
        except Exception:
            pass
        client.close()
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def test_real_nvim_allowlist_parameters_and_owning_thread(nvim):
    sock, client = nvim
    bridge = EditorBridge(str(sock), timeout_s=2)
    try:
        assert bridge.status()["status"] == "ok"
        assert bridge.status()["tools"] == ["editor_context", "editor_debug"]
        injected = "'); vim.g.pwned = true; --"
        result = bridge.call("editor_context", {"path": injected})
        assert result["args"]["path"] == injected
        assert not client.exec_lua("return vim.g.pwned")
        assert bridge.call("evil_lua", {})["status"] == "error"
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: bridge.call("editor_context", {}), range(6)))
        assert all(r["status"] == "ok" for r in results)
    finally:
        bridge.close()


def test_real_nvim_timeout_does_not_leave_live_worker(nvim):
    sock, client = nvim
    bridge = EditorBridge(str(sock), timeout_s=0.1)
    started = time.monotonic()
    try:
        result = bridge.call("editor_context", {"path": "slow"})
        assert result["status"] == "unavailable"
        assert "timed out" in result["error"]
    finally:
        bridge.close()
    assert time.monotonic() - started < 2
    assert bridge.call("editor_context", {})["status"] == "unavailable"


def test_silent_socket_handshake_is_bounded(tmp_path):
    pytest.importorskip("pynvim")
    sockpath = str(tmp_path / "silent.sock")
    server = socket.socket(socket.AF_UNIX)
    server.bind(sockpath)
    server.listen(1)
    done = threading.Event()

    def accept():
        client, _ = server.accept()
        done.wait(3)
        client.close()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    start = time.monotonic()
    bridge = EditorBridge(sockpath, timeout_s=0.1)
    try:
        assert bridge.status()["status"] == "unavailable"
        assert "timed out" in bridge.status()["error"]
    finally:
        bridge.close()
        done.set()
        thread.join(2)
        server.close()
    assert time.monotonic() - start < 2


@pytest.mark.parametrize("path", ["http://127.0.0.1:6666", "127.0.0.1:6666", "relative.sock"])
def test_tcp_or_relative_sockets_are_never_allowed(path):
    bridge = EditorBridge(path, timeout_s=0.1)
    try:
        assert bridge.status()["status"] == "unavailable"
    finally:
        bridge.close()


def test_public_socket_in_unprotected_directory_is_rejected(tmp_path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    os.chmod(public, 0o755)
    sock = socket.socket(socket.AF_UNIX)
    path = public / "nvim.sock"
    sock.bind(str(path))
    os.chmod(path, 0o666)
    bridge = EditorBridge(str(path), timeout_s=0.1)
    try:
        assert "private" in bridge.status()["error"]
    finally:
        bridge.close()
        sock.close()
