"""Flow CLI entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Flow — Local AI Workflow System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flow                          # Start interactive TUI
  flow --config /path/to/config.toml
  flow --workspace /my/project
  flow --model qwen2.5-coder:7b
        """
    )
    parser.add_argument("--config", "-c", help="Path to config TOML file")
    parser.add_argument("--workspace", "-w", help="Workspace directory")
    parser.add_argument("--model", "-m", help="Override Ollama model")
    parser.add_argument("--version", action="store_true", help="Show version")

    args = parser.parse_args()

    if args.version:
        print("Flow 0.1.0 — Amor Fati Labs")
        sys.exit(0)

    from flow.tui.app import FlowApp
    app = FlowApp(config_path=args.config)

    if args.workspace:
        app.workspace = Path(args.workspace).resolve()

    if args.model:
        app.cfg.ollama.model = args.model
        app.llm.cfg.model = args.model

    app.run()


if __name__ == "__main__":
    main()
