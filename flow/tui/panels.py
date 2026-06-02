"""Reusable TUI panel components for Flow."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich import box

console = Console()


def tool_call_panel(tool_name: str, args: dict) -> Panel:
    """Render a tool call as a panel."""
    args_text = "\n".join(f"  {k}: {repr(v)}" for k, v in args.items())
    return Panel(
        f"[bold]{tool_name}[/bold]\n{args_text}",
        title="[cyan]Tool Call[/cyan]",
        border_style="cyan",
        expand=False,
    )


def tool_result_panel(tool_name: str, result: str, max_len: int = 500) -> Panel:
    """Render a tool result as a panel."""
    display = result[:max_len] + ("..." if len(result) > max_len else "")
    return Panel(
        display,
        title=f"[dim]Result: {tool_name}[/dim]",
        border_style="dim",
        expand=False,
    )


def error_panel(message: str, title: str = "Error") -> Panel:
    """Render an error as a red panel."""
    return Panel(
        f"[red]{message}[/red]",
        title=f"[bold red]{title}[/bold red]",
        border_style="red",
    )


def success_panel(message: str, title: str = "Success") -> Panel:
    """Render a success message as a green panel."""
    return Panel(
        f"[green]{message}[/green]",
        title=f"[bold green]{title}[/bold green]",
        border_style="green",
    )


def warning_panel(message: str, title: str = "Warning") -> Panel:
    """Render a warning as a yellow panel."""
    return Panel(
        f"[yellow]{message}[/yellow]",
        title=f"[bold yellow]{title}[/bold yellow]",
        border_style="yellow",
    )


def code_panel(code: str, language: str = "python", title: str = "") -> Panel:
    """Render a code block in a syntax-highlighted panel."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    return Panel(syntax, title=title or f"[cyan]{language}[/cyan]", border_style="cyan")


def status_table(items: dict[str, str], title: str = "Status") -> Table:
    """Render a key-value status table."""
    table = Table(title=title, box=box.ROUNDED, show_header=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    for k, v in items.items():
        table.add_row(k, v)
    return table


def memory_table(memories: list[dict]) -> Table:
    """Render memory entries as a table."""
    table = Table(title="Recent Memories", box=box.SIMPLE)
    table.add_column("When", style="dim", no_wrap=True)
    table.add_column("Query", style="cyan")
    table.add_column("Response preview", style="dim")
    for m in memories:
        table.add_row(
            m.get("created_at", "")[:16],
            m.get("query", "")[:50],
            m.get("response", "")[:60],
        )
    return table


def tools_table(tools: list) -> Table:
    """Render loaded tools as a table."""
    table = Table(title="Loaded Tools", box=box.SIMPLE)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    for tool in tools:
        table.add_row(tool.name, tool.description[:70])
    return table
