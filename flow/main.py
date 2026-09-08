"""Flow command line: all entrypoints use the same safe runtime."""

from __future__ import annotations

import argparse
import json
import signal
import sys

from flow import __version__
from flow.config import ConfigError, load_config
from flow.editor import TIMEOUT_S_DEFAULT, TIMEOUT_S_MAX, TIMEOUT_S_MIN, EditorBridge
from flow.runtime import Runtime


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # Suppressed defaults let options appear before OR after a subcommand.
    common.add_argument(
        "--config",
        "-c",
        default=argparse.SUPPRESS,
        help="Explicitly approved TOML (project-local config is never auto-loaded)",
    )
    common.add_argument(
        "--workspace",
        "-w",
        default=argparse.SUPPRESS,
        help="Workspace directory; resolved before any tool registration",
    )
    common.add_argument("--model", "-m", default=argparse.SUPPRESS, help="Override all role models")
    common.add_argument(
        "--trusted",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Allow workspace writes and configured named checks (not an OS sandbox)",
    )
    common.add_argument("--nvim", default=argparse.SUPPRESS, help="Explicit private Neovim socket")
    common.add_argument(
        "--editor-timeout",
        type=float,
        default=argparse.SUPPRESS,
        help=f"Bounded reverse editor request timeout in seconds (default {TIMEOUT_S_DEFAULT:g})",
    )
    root = argparse.ArgumentParser(
        description="Flow — bounded local multi-agent coding", parents=[common]
    )
    root.add_argument("--version", action="version", version=f"Flow {__version__}")
    commands = root.add_subparsers(dest="command")
    run = commands.add_parser("run", parents=[common], help="Run one task and print a JSON report")
    run.add_argument("task", nargs="+")
    commands.add_parser("serve", parents=[common], help="Serve newline-framed MCP over stdio")
    check = commands.add_parser("check", parents=[common], help="Run configured named checks")
    check.add_argument("name", nargs="?")
    check.add_argument("--name", dest="check_name", help="Run one configured named check")
    commands.add_parser(
        "status", parents=[common], help="Show local capabilities without model calls"
    )
    commands.add_parser("tui", parents=[common], help="Interactive terminal frontend")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    previous_term = None
    if hasattr(signal, "SIGTERM"):

        def terminate(_signum, _frame):
            # Unwind process-group check cleanup and editor connection ownership.
            raise KeyboardInterrupt

        previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        cfg = load_config(
            getattr(args, "config", None),
            workspace=getattr(args, "workspace", None),
            trusted=getattr(args, "trusted", False),
            model=getattr(args, "model", None),
        )
        editor_timeout_s = getattr(args, "editor_timeout", TIMEOUT_S_DEFAULT)
        # Operator input: reject out-of-range values as a usage error rather than asserting.
        if not TIMEOUT_S_MIN <= editor_timeout_s <= TIMEOUT_S_MAX:
            raise ConfigError(
                f"--editor-timeout must be between {TIMEOUT_S_MIN:g} and {TIMEOUT_S_MAX:g} seconds"
            )
        editor = EditorBridge(getattr(args, "nvim", None), timeout_s=editor_timeout_s)
        runtime = Runtime(cfg, editor=editor)
        with runtime:
            if args.command == "serve":
                from flow.mcp import MCPServer

                MCPServer(runtime).serve()
                return 0
            if args.command == "run":
                result = runtime.run(" ".join(args.task))
            elif args.command == "check":
                result = runtime.check(args.check_name or args.name)
            elif args.command == "status":
                result = runtime.status()
            else:
                from flow.tui.app import FlowApp

                FlowApp(runtime=runtime).run()
                return 0
            print(json.dumps(result, ensure_ascii=True, allow_nan=False))
            return 0 if result.get("status") == "ok" else 1
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Flow: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Flow interrupted; changes already written are not rolled back.", file=sys.stderr)
        return 130
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
