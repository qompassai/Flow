from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import httpx
import pytest

from flow.config import OllamaConfig
from flow.llm.backend import OllamaClient
from flow.llm.prompts import load_system_prompt

ROOT = Path(__file__).resolve().parents[1]


class FakeOllama(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.payloads.append(payload)
        if self.server.redirect:
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/cloud")
            self.end_headers()
            return
        response = {"choices": [{"message": {"role": "assistant", "content": "fixture"}}]}
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_response(503)
        self.end_headers()


@pytest.fixture
def http_fixture():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    server.payloads, server.redirect = [], False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_actual_local_http_model_payload_ignores_proxy_environment(http_fixture, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    cfg = OllamaConfig(base_url=f"http://127.0.0.1:{http_fixture.server_port}", timeout=1)
    client = OllamaClient(cfg)
    try:
        result = client.chat([{"role": "user", "content": "fixture"}], model="role-specific")
        assert result["choices"][0]["message"]["content"] == "fixture"
        assert http_fixture.payloads[-1]["model"] == "role-specific"
        assert http_fixture.payloads[-1]["stream"] is False
        assert client.is_available() is False  # an HTTP 503 is not availability
        http_fixture.redirect = True
        with pytest.raises(httpx.HTTPStatusError):
            client.chat([{"role": "user", "content": "no remote redirect"}])
    finally:
        client.close()


def test_bundled_prompts_do_not_read_cwd_project_skill_injection(tmp_path, monkeypatch):
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills/system_prompt.md").write_text("UNTRUSTED EXECUTABLE POLICY")
    monkeypatch.chdir(tmp_path)
    prompt = load_system_prompt()
    assert "UNTRUSTED EXECUTABLE POLICY" not in prompt
    assert "No arbitrary commands" in prompt


def test_wheel_contains_and_loads_bundled_resources_outside_repository(tmp_path):
    wheel_dir = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheel_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(wheel_dir.glob("flow-*.whl"))
    install = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "flow/skills/system_prompt.md" in names
        assert "flow/skills/language_profiles/python.toml" in names
        archive.extractall(install)
    working = tmp_path / "unrelated-project"
    working.mkdir()
    code = (
        "import flow; from flow.llm.prompts import load_system_prompt; "
        "from flow.codegen.language_profiles import load_profiles; "
        f"assert flow.__file__.startswith({str(install)!r}); "
        "assert 'No arbitrary commands' in load_system_prompt(); "
        "assert load_profiles()['python'].linter == 'ruff'; "
        "print(flow.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=working,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(install)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.2.0"
