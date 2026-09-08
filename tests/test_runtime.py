from __future__ import annotations

import json
import sys
import threading

from conftest import FakeBackend, message, tool, verdict

from flow.config import CheckConfig
from flow.runtime import Runtime, schema
from flow.tui.app import FlowApp


def static_check(script="pass"):
    return CheckConfig((sys.executable, "-c", script), kind="lint", filetypes=("python",))


def write_response(content="value = 42\n"):
    return message(
        calls=[tool("file_write", {"path": "edited.py", "content": content}, "write-id")]
    )


def test_real_tools_role_models_and_required_verification(cfg):
    cfg.models.planner, cfg.models.coder, cfg.models.reviewer = "tiny", "code", "review"
    cfg.checks = {
        "lint": static_check(
            "from pathlib import Path; assert Path('edited.py').read_text() == 'value = 42\\n'"
        )
    }
    backend = FakeBackend(
        [
            message(calls=[tool("file_list", {}, "plan-id")]),
            message("Implement a value"),
            write_response(),
            message("Implemented"),
            verdict(),
        ]
    )
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("Set value to 42")
    assert result["verified"] is True and result["status"] == "ok"
    assert result["changed_files"] == ["edited.py"]
    assert result["verification"]["coverage"][0]["source"] == "flow.check"
    assert result["checks"][0]["returncode"] == 0
    assert [c["model"] for c in backend.calls] == ["tiny", "tiny", "code", "code", "review"]
    assert backend.calls[1]["messages"][-1]["tool_call_id"] == "plan-id"
    assert backend.calls[3]["messages"][-1]["tool_call_id"] == "write-id"
    for index in [0, 4]:
        offered = {s["function"]["name"] for s in backend.calls[index]["tools"]}
        assert "file_write" not in offered
        assert "flow_check" not in offered
    assert "file_write" in {s["function"]["name"] for s in backend.calls[2]["tools"]}
    assert backend.closed


def test_missing_static_language_tools_never_become_verified(cfg):
    cfg.checks = {"tests": CheckConfig((sys.executable, "-c", "pass"), kind="test")}
    backend = FakeBackend(
        [message("plan"), write_response(), message("Everything passes"), verdict()]
    )
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("write source")
    assert result["verified"] is False
    assert result["status"] == "unverified"
    assert result["checks"][0]["status"] == "ok"
    assert "unavailable" in json.dumps(result["verification"]["coverage"])


def test_empty_checks_and_unsupported_language_remain_unverified(cfg):
    backend = FakeBackend([message("plan"), message("all done"), verdict()])
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("verify project")
    assert not result["verified"]
    assert result["verification"]["status"] == "unverified"


def test_failed_checks_feed_bounded_repair_context(cfg):
    cfg.agent.max_cycles = 2
    cfg.checks = {
        "lint": static_check(
            "from pathlib import Path; assert Path('edited.py').read_text() == 'value = 42\\n'"
        )
    }
    backend = FakeBackend(
        [
            message("plan"),
            write_response("bad"),
            message("attempt 1"),
            verdict(False),
            write_response(),
            message("fixed"),
            verdict(),
        ]
    )
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("set value")
    assert result["verified"] and result["cycles"] == 2
    assert result["verification_history"][0]["status"] == "failed"
    assert result["verification_history"][1]["status"] == "ok"
    repair = json.loads(backend.calls[4]["messages"][1]["content"])["context"]
    assert repair["verification"]["checks"][0]["returncode"] != 0
    assert repair["review"]["approved"] is False
    assert len([r for r in result["roles"] if r["role"] == "planner"]) == 1


def test_reviewer_rejection_does_not_pass_even_with_checks(cfg):
    cfg.checks = {"lint": static_check()}
    backend = FakeBackend([message("plan"), message("done"), verdict(False)])
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("review this")
    assert result["status"] == "failed"
    assert not result["verified"]
    assert result["verification"]["verified"]  # check result is separate from task approval


def test_unstructured_reviewer_never_approves(cfg):
    cfg.checks = {"lint": static_check()}
    backend = FakeBackend([message("plan"), message("done"), message("Looks great!")])
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("review")
    assert not result["verified"]
    assert result["roles"][-1]["status"] == "unverified"


def test_tool_denials_and_json_errors_return_matching_ids(cfg):
    backend = FakeBackend(
        [
            message(calls=[tool("file_write", {"path": "stolen", "content": "no"}, "plan-write")]),
            message("plan"),
            message(
                calls=[
                    tool("shell_exec", {"command": "python -c pass", "cwd": "/"}, "no-shell"),
                    {"id": "broken", "function": {"name": "file_read", "arguments": "{bad"}},
                    tool("file_read", {"path": "x", "cwd": "/"}, "bad-cwd"),
                ]
            ),
            message("done"),
            verdict(),
        ]
    )
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("malicious fixture")
    responses = backend.calls[3]["messages"][-3:]
    assert [r["tool_call_id"] for r in responses] == ["no-shell", "broken", "bad-cwd"]
    assert all(json.loads(r["content"])["status"] == "error" for r in responses)
    assert backend.calls[1]["messages"][-1]["tool_call_id"] == "plan-write"
    assert result["tool_calls"] == 4
    assert len(result["events"]) == 4
    assert not (runtime.workspace.root / "stolen").exists()


def test_tool_iteration_budget_is_finite_and_gate_still_runs(cfg):
    cfg.agent.max_iterations = 1
    cfg.checks = {"lint": static_check()}
    backend = FakeBackend([message("plan"), write_response(), verdict()])
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("write")
    assert not result["verified"]
    assert result["roles"][1]["error"] == "Role iteration budget exhausted"
    assert result["checks"][0]["status"] == "ok"
    assert result["model_calls"] == 3


def test_global_tool_budget_does_not_execute_oversized_batch(cfg):
    cfg.agent.max_tool_calls = 1
    backend = FakeBackend(
        [
            message(
                calls=[
                    tool("file_list", {}, "a"),
                    tool("file_list", {}, "b"),
                ]
            )
        ]
    )
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("inspect")
    assert result["tool_calls"] == 0
    assert result["status"] == "unverified"


def test_backend_errors_and_malformed_calls_fail_structurally(cfg):
    backend = FakeBackend([RuntimeError("fixture offline")])
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("inspect")
    assert result["status"] == "error" and "offline" in result["error"]
    backend = FakeBackend([message(calls=[{"id": "x"}])])
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("inspect")
    assert result["status"] == "error"


def test_invalid_task_and_context_budget_do_not_call_model(cfg):
    cfg.agent.max_context_chars = 4096
    backend = FakeBackend([])
    with Runtime(cfg, backend=backend) as runtime:
        assert runtime.run("")["status"] == "error"
        assert runtime.run("x" * 16001)["status"] == "error"
        assert runtime.run("x" * 5000)["status"] == "unverified"
    assert backend.calls == []


def test_single_writer_busy_and_config_snapshot(cfg):
    started, release = threading.Event(), threading.Event()

    def wait(**kwargs):
        started.set()
        assert release.wait(5)
        return message("plan")

    backend = FakeBackend([wait, message("done"), verdict()])
    with Runtime(cfg, backend=backend) as runtime:
        original = runtime.workspace.root
        cfg.workspace_dir = "/"
        cfg.trusted = False
        thread = threading.Thread(target=runtime.run, args=("first",))
        thread.start()
        assert started.wait(5)
        assert runtime.run("second")["status"] == "error"
        assert runtime.check()["status"] == "error"
        assert runtime.status()["busy"]
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert runtime.workspace.root == original and runtime.cfg.trusted


def test_tui_and_legacy_wrappers_share_runtime(cfg):
    cfg.checks = {"lint": static_check()}
    backend = FakeBackend([message("plan"), write_response(), message("done"), verdict()])
    with Runtime(cfg, backend=backend) as runtime:
        app = FlowApp(runtime=runtime)
        assert app.execute("/plugins")["status"] == "unavailable"
        assert app.execute("/status")["workspace"] == cfg.workspace_dir
        assert app.execute("set value")["verified"]


class FakeEditor:
    configured = True
    socket = None

    def __init__(self, workspace, *, modified=False, lint_verified=True):
        self.workspace, self.modified, self.lint_verified = workspace, modified, lint_verified
        self.calls = []
        self.snapshot = {}
        self.dirty = []

    def schemas(self):
        return [
            schema("editor_context", "context", {"path": {"type": "string"}}),
            schema("editor_diagnostics", "diagnostics", {"path": {"type": "string"}}),
            schema("editor_lint", "lint", {"path": {"type": "string"}}),
            schema(
                "editor_debug",
                "debug",
                {
                    "action": {"type": "string", "enum": ["status", "run"]},
                    "name": {"type": "string"},
                },
            ),
        ]

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "editor_context":
            return {
                "status": "ok",
                "workspace": self.workspace,
                "modified": self.modified,
                "dirty_buffers": self.dirty,
                "changedtick": 1,
                "workspace_snapshot_version": 1,
                "workspace_snapshot": self.snapshot,
            }
        if name == "editor_diagnostics":
            return {"status": "unavailable", "diagnostics": [], "verified": False}
        if name == "editor_lint":
            return {"status": "ok", "verified": self.lint_verified, "scope": "fixture linter"}
        if name == "file_write":
            from pathlib import Path

            if self.modified:
                return {"status": "error", "error": "unsaved buffer"}
            (Path(self.workspace) / args["path"]).write_text(args["content"])
            return {"status": "ok"}
        return {"status": "ok"}

    def status(self):
        return {"status": "ok", "tools": [s["function"]["name"] for s in self.schemas()]}

    def close(self):
        pass


def test_completed_native_lint_can_verify_without_claiming_lsp_support(cfg):
    cfg.checks = {"tests": CheckConfig((sys.executable, "-c", "pass"), kind="test")}
    editor = FakeEditor(cfg.workspace_dir)
    backend = FakeBackend([message("plan"), write_response(), message("done"), verdict()])
    with Runtime(cfg, backend=backend, editor=editor) as runtime:
        result = runtime.run("edit source")
    assert result["verified"]
    coverage = result["verification"]["coverage"][0]
    assert coverage["status"] == "ok"
    assert coverage["evidence"][1]["status"] == "unavailable"


def test_fake_ok_lint_without_completed_verified_flag_does_not_pass(cfg):
    cfg.checks = {"tests": CheckConfig((sys.executable, "-c", "pass"))}
    editor = FakeEditor(cfg.workspace_dir, lint_verified=False)
    backend = FakeBackend([message("plan"), write_response(), message("done"), verdict()])
    with Runtime(cfg, backend=backend, editor=editor) as runtime:
        assert not runtime.run("edit source")["verified"]


def test_reverse_root_mismatch_dirty_and_debug_denials(cfg):
    editor = FakeEditor("/")
    with Runtime(cfg, backend=FakeBackend([]), editor=editor) as runtime:
        assert runtime.call_tool("file_write", {"path": "x", "content": "no"})["status"] == "error"
        assert runtime.status()["editor"]["attached"] is False
        assert runtime.check()["status"] == "unverified"
        editor.workspace = cfg.workspace_dir
        editor.modified = True
        assert runtime.check()["status"] == "stale"
        assert runtime.call_tool("file_write", {"path": "x", "content": "no"})["status"] == "error"
        for role in ["planner", "reviewer", "coder"]:
            assert (
                runtime.call_tool("editor_debug", {"action": "run", "name": "probe"}, role=role)[
                    "status"
                ]
                == "error"
            )
        assert not any(n == "editor_debug" for n, _ in editor.calls)


def test_external_changes_during_review_invalidate_verification(cfg):
    cfg.checks = {"lint": static_check()}

    def external_change(**kwargs):
        from pathlib import Path

        (Path(cfg.workspace_dir) / "edited.py").write_text("external edit")
        return verdict()

    backend = FakeBackend([message("plan"), write_response(), message("done"), external_change])
    with Runtime(cfg, backend=backend) as runtime:
        result = runtime.run("edit source")
    assert not result["verified"]
    assert result["verification"]["status"] == "stale"


def test_noncurrent_dirty_buffer_and_saved_snapshot_changes_invalidate_checks(cfg):
    cfg.checks = {"lint": static_check()}
    editor = FakeEditor(cfg.workspace_dir)
    editor.dirty = [{"path": "other.py", "changedtick": 4}]
    with Runtime(cfg, backend=FakeBackend([]), editor=editor) as runtime:
        assert runtime.check()["status"] == "stale"
        editor.dirty = []
        original = runtime.checks.run_all

        def check_and_save(name=None):
            result = original(name)
            editor.snapshot = {
                "other.py": {"buffers": [{"changedtick": 5}], "disk": {"sha256": "new"}}
            }
            return result

        runtime.checks.run_all = check_and_save
        assert runtime.check()["status"] == "stale"


def test_noncurrent_saved_change_during_review_is_stale(cfg):
    cfg.checks = {"lint": static_check()}
    editor = FakeEditor(cfg.workspace_dir)
    editor.snapshot = {"other.py": {"buffers": [{"bufnr": 1, "changedtick": 1}]}}

    def change_other(**kwargs):
        editor.snapshot = {"other.py": {"buffers": [{"bufnr": 1, "changedtick": 2}]}}
        return verdict()

    backend = FakeBackend([message("plan"), write_response(), message("done"), change_other])
    with Runtime(cfg, backend=backend, editor=editor) as runtime:
        result = runtime.run("edit")
    assert not result["verified"]
    assert result["verification"]["status"] == "stale"


def test_missing_editor_freshness_fields_fail_closed(cfg):
    editor = FakeEditor(cfg.workspace_dir)
    original = editor.call

    def old_context(name, args):
        result = original(name, args)
        result.pop("workspace_snapshot_version", None)
        return result

    editor.call = old_context
    with Runtime(cfg, backend=FakeBackend([]), editor=editor) as runtime:
        assert runtime.check()["status"] == "unverified"
        assert runtime.call_tool("file_write", {"path": "a", "content": "no"})["status"] == "error"
