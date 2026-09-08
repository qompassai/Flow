from __future__ import annotations

import sys
import time

from flow.checks import CheckRunner
from flow.config import CheckConfig
from flow.workspace import Workspace


def runner(root, checks, trusted=True):
    return CheckRunner(Workspace(root, trusted=trusted), checks)


def test_missing_optional_and_empty_checks_never_verify(tmp_path):
    checks = runner(tmp_path, {})
    assert checks.run_all()["verified"] is False
    assert checks.run_all("missing")["checks"][0]["status"] == "unavailable"
    checks = runner(
        tmp_path, {"optional": CheckConfig((sys.executable, "-c", "pass"), required=False)}
    )
    assert checks.run_all()["verified"] is False


def test_configured_check_executes_only_exact_argv_in_pinned_workspace(tmp_path):
    script = "from pathlib import Path; print(Path.cwd()); print('literal; echo unsafe')"
    checks = runner(tmp_path, {"smoke": CheckConfig((sys.executable, "-c", script))})
    result = checks.run_all()
    assert result["verified"] is True
    assert str(tmp_path) in result["checks"][0]["stdout"]
    assert "literal; echo unsafe" in result["checks"][0]["stdout"]
    assert result["checks"][0]["source"] == "flow.check"


def test_failure_missing_executable_and_trust_are_explicit(tmp_path):
    checks = {
        "bad": CheckConfig((sys.executable, "-c", "raise SystemExit(7)")),
        "missing": CheckConfig(("flow-no-such-executable-fixture",)),
    }
    result = runner(tmp_path, checks).run_all()
    assert result["status"] == "failed" and result["verified"] is False
    assert result["checks"][0]["returncode"] == 7
    assert result["checks"][1]["status"] == "unavailable"
    assert all(
        r["status"] == "unverified" for r in runner(tmp_path, checks, False).run_all()["checks"]
    )


def test_bounded_output_and_timeout(tmp_path):
    checks = {
        "noisy": CheckConfig((sys.executable, "-c", "print('x'*100000)")),
        "slow": CheckConfig((sys.executable, "-c", "import time; time.sleep(10)"), timeout=50),
    }
    start = time.monotonic()
    result = runner(tmp_path, checks).run_all()
    assert len(result["checks"][0]["stdout"]) <= 16000
    assert result["checks"][0]["stdout_truncated"]
    assert result["checks"][1]["status"] == "timeout"
    assert time.monotonic() - start < 3


def test_timeout_kills_check_process_group(tmp_path):
    child_code = (
        "import time; from pathlib import Path; time.sleep(1); Path('leak').write_text('bad')"
    )
    script = (
        f"import subprocess,time; subprocess.Popen([{sys.executable!r},'-c',"
        f"{child_code!r}]); time.sleep(20)"
    )
    checks = {"tree": CheckConfig((sys.executable, "-c", script), timeout=100)}
    assert runner(tmp_path, checks).run_all()["checks"][0]["status"] == "timeout"
    time.sleep(1.1)
    assert not (tmp_path / "leak").exists()
