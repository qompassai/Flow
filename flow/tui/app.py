"""Rich + prompt_toolkit TUI for Flow."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

from flow.config import FlowConfig, load_config
from flow.llm.backend import OllamaClient
from flow.tools.registry import ToolRegistry
from flow.tools.web_search import make_web_search_tool, TOOL_SPEC as WEB_SPEC
from flow.tools.shell import make_shell_tool, TOOL_SPEC as SHELL_SPEC
from flow.tools.file_ops import make_file_ops_tools
from flow.tools.lsp_check import run_lsp_check, TOOL_SPEC as LSP_SPEC
from flow.agent.orchestrator import Orchestrator
from flow.agent.memory import MemoryStore
from flow.codegen.language_profiles import load_profiles, profile_summary
from flow.codegen.app_generator import AppGenerator
from flow.self_improve.feedback import FeedbackStore
from flow.self_improve.prompt_evolver import PromptEvolver

console = Console()

BANNER = """\
[bold cyan]
  ███████╗██╗      ██████╗ ██╗    ██╗
  ██╔════╝██║     ██╔═══██╗██║    ██║
  █████╗  ██║     ██║   ██║██║ █╗ ██║
  ██╔══╝  ██║     ██║   ██║██║███╗██║
  ██║     ███████╗╚██████╔╝╚███╔███╔╝
  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
[/bold cyan]
[dim]Local AI Workflow System — Amor Fati Labs[/dim]
[dim]Type /help for commands, /quit to exit[/dim]
"""

COMMANDS = {
    "/help": "Show available commands",
    "/build <language> <framework> <name>": "Generate a new app",
    "/models": "List available Ollama models",
    "/model <name>": "Switch active model",
    "/tools": "List loaded tools",
    "/plugins": "Reload user plugins",
    "/memory": "Show recent memories",
    "/clear": "Clear conversation context",
    "/feedback": "Rate the last session",
    "/evolve": "Trigger prompt evolution (if enough feedback)",
    "/quit": "Exit Flow",
}

PT_STYLE = Style.from_dict({
    "prompt": "bold cyan",
    "": "#cccccc",
})


class FlowApp:
    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(Path(config_path) if config_path else None)
        self.session_id = str(uuid.uuid4())[:8]
        self.workspace = Path(self.cfg.workspace_dir).resolve()

        # Init components
        self.llm = OllamaClient(self.cfg.ollama)
        self.memory = MemoryStore(self.cfg.memory_db)
        self.registry = ToolRegistry()
        self.profiles = load_profiles()
        self.feedback_store = FeedbackStore(
            str(Path(self.cfg.memory_db).parent / "feedback.db")
        )

        self._register_core_tools()
        self.registry.load_plugins(self.cfg.tools.plugins_dir)

        self.generator = AppGenerator(
            llm=self.llm,
            registry=self.registry,
            workspace=str(self.workspace),
        )

        self.orchestrator = Orchestrator(
            llm=self.llm,
            registry=self.registry,
            cfg=self.cfg.agent,
            memory=self.memory,
            on_pause=self._handle_pause,
            workspace=str(self.workspace),
        )

        self.evolver = PromptEvolver(
            llm=self.llm,
            feedback=self.feedback_store,
            skills_dir=Path("skills"),
        )

    def _register_core_tools(self) -> None:
        """Register built-in tools."""
        # Web search
        search_fn = make_web_search_tool(
            backend=self.cfg.tools.web_search_backend,
            searxng_url=self.cfg.tools.searxng_url,
        )
        self.registry.register(
            name=WEB_SPEC["name"],
            description=WEB_SPEC["description"],
            parameters=WEB_SPEC["parameters"],
            func=search_fn,
        )

        # Shell
        shell_fn = make_shell_tool(
            allowed_commands=self.cfg.tools.shell_allowed_commands,
            timeout=self.cfg.tools.shell_timeout,
            workspace=str(self.workspace),
        )
        self.registry.register(
            name=SHELL_SPEC["name"],
            description=SHELL_SPEC["description"],
            parameters=SHELL_SPEC["parameters"],
            func=shell_fn,
        )

        # File ops
        file_read, file_write, file_list, file_delete = make_file_ops_tools(str(self.workspace))
        for name, desc, fn, schema in [
            ("file_read", "Read a file from the workspace", file_read, {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative file path"}},
                "required": ["path"],
            }),
            ("file_write", "Write content to a file in the workspace", file_write, {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            }),
            ("file_list", "List files in a workspace directory", file_list, {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to list (default: .)"}
                },
            }),
        ]:
            self.registry.register(name=name, description=desc, parameters=schema, func=fn)

        # LSP check
        self.registry.register(
            name=LSP_SPEC["name"],
            description=LSP_SPEC["description"],
            parameters=LSP_SPEC["parameters"],
            func=lambda language, file_path=None: run_lsp_check(
                language, file_path, str(self.workspace)
            ),
        )

    def _handle_pause(self, reason: str) -> str:
        """Handle agent pause — prompt user for input."""
        return Prompt.ask("\n[yellow]Your input[/yellow]")

    def _check_ollama(self) -> bool:
        if not self.llm.is_available():
            console.print(Panel(
                "[red]Ollama is not running![/red]\n"
                f"Start it with: [bold]systemctl --user start ollama[/bold]\n"
                f"Or: [bold]ollama serve[/bold]\n"
                f"Then pull a model: [bold]ollama pull {self.cfg.ollama.model}[/bold]",
                title="Connection Error",
                border_style="red"
            ))
            return False
        return True

    def run(self) -> None:
        """Main TUI loop."""
        console.print(BANNER)

        if not self._check_ollama():
            sys.exit(1)

        console.print(f"[green]Connected to Ollama[/green] — Model: [bold]{self.cfg.ollama.model}[/bold]")
        console.print(f"[dim]Workspace: {self.workspace}[/dim]")
        console.print(f"[dim]Tools: {len(self.registry.all_tools())} loaded[/dim]")
        console.print()

        history_file = Path.home() / ".local" / "share" / "flow" / "history.txt"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            style=PT_STYLE,
        )

        while True:
            try:
                user_input = session.prompt(
                    f"[flow:{self.session_id}] > ",
                    multiline=False,
                ).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Use /quit to exit cleanly.[/dim]")
                continue

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # Run agent
            try:
                response = self.orchestrator.run(user_input)
                console.print()
                console.print(Panel(
                    Markdown(response),
                    title="[cyan]Flow[/cyan]",
                    border_style="cyan",
                ))
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")

    def _handle_command(self, cmd: str) -> None:
        parts = cmd.split()
        command = parts[0].lower()

        if command == "/help":
            table = Table(title="Flow Commands", box=box.ROUNDED, show_header=True)
            table.add_column("Command", style="cyan bold")
            table.add_column("Description")
            for k, v in COMMANDS.items():
                table.add_row(k, v)
            console.print(table)

        elif command == "/build":
            if len(parts) < 4:
                console.print("[red]Usage: /build <language> <framework> <project-name>[/red]")
                console.print(f"Languages: {', '.join(self.profiles.keys())}")
                return
            lang, framework, name = parts[1], parts[2], parts[3]
            if lang not in self.profiles:
                console.print(f"[red]Unknown language: {lang}[/red]")
                console.print(f"Available: {', '.join(self.profiles.keys())}")
                return
            profile = self.profiles[lang]
            request = Prompt.ask("Describe what you want to build")
            self.generator.generate(
                request=request,
                language=lang,
                framework=framework,
                project_name=name,
                profile=profile,
            )

        elif command == "/models":
            try:
                models = self.llm.list_models()
                table = Table(title="Available Ollama Models", box=box.SIMPLE)
                table.add_column("Model", style="cyan")
                for m in models:
                    marker = " [green]← active[/green]" if m == self.cfg.ollama.model else ""
                    table.add_row(m + marker)
                console.print(table)
            except Exception as e:
                console.print(f"[red]Error listing models:[/red] {e}")

        elif command == "/model" and len(parts) > 1:
            self.cfg.ollama.model = parts[1]
            self.llm.cfg.model = parts[1]
            console.print(f"[green]Switched to model:[/green] {parts[1]}")

        elif command == "/tools":
            table = Table(title="Loaded Tools", box=box.SIMPLE)
            table.add_column("Name", style="cyan bold")
            table.add_column("Description")
            for tool in self.registry.all_tools():
                table.add_row(tool.name, tool.description[:60])
            console.print(table)

        elif command == "/plugins":
            self.registry.load_plugins(self.cfg.tools.plugins_dir)
            console.print("[green]Plugins reloaded.[/green]")

        elif command == "/memory":
            recent = self.memory.recent(10)
            if not recent:
                console.print("[dim]No memories yet.[/dim]")
            else:
                table = Table(title="Recent Memories", box=box.SIMPLE)
                table.add_column("When", style="dim")
                table.add_column("Query")
                for m in recent:
                    table.add_row(m["created_at"][:16], m["query"][:60])
                console.print(table)

        elif command == "/clear":
            self.orchestrator.context.clear()
            console.print("[green]Conversation context cleared.[/green]")

        elif command == "/feedback":
            rating = IntPrompt.ask("Rate this session (1-5)", choices=["1", "2", "3", "4", "5"])
            comment = Prompt.ask("Comment (optional)", default="")
            self.feedback_store.record(
                session_id=self.session_id,
                rating=rating,
                comment=comment,
            )
            console.print("[green]Feedback recorded. Thank you.[/green]")

        elif command == "/evolve":
            if not self.cfg.self_improve.enabled:
                console.print("[yellow]Self-improvement is disabled in config.[/yellow]")
                return
            console.print("[dim]Analyzing feedback and evolving prompt...[/dim]")
            improved = self.evolver.evolve()
            if improved:
                console.print("[green]System prompt evolved and committed to git.[/green]")
                console.print(Panel(improved[:500] + "...", title="Evolved Prompt Preview"))
            else:
                console.print("[yellow]Not enough feedback to evolve yet.[/yellow]")

        elif command == "/quit":
            avg = self.feedback_store.average_rating()
            if avg > 0:
                console.print(f"[dim]Session avg rating: {avg:.1f}/5[/dim]")
            console.print("[cyan]Goodbye.[/cyan]")
            sys.exit(0)

        else:
            console.print(f"[red]Unknown command:[/red] {command}. Type /help.")
