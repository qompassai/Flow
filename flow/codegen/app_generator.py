"""Application code generator with iterative LSP validation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from flow.codegen.language_profiles import LanguageProfile, profile_summary
from flow.llm.prompts import build_codegen_prompt, build_error_fix_prompt

if TYPE_CHECKING:
    from flow.llm.backend import OllamaClient
    from flow.tools.registry import ToolRegistry

console = Console()


def parse_files_from_response(response: str) -> dict[str, str]:
    """Parse FILE: path followed by code block from LLM response."""
    files = {}
    pattern = r'FILE:\s*([^\n]+)\n```[a-z]*\n(.*?)```'
    matches = re.finditer(pattern, response, re.DOTALL)
    for match in matches:
        path = match.group(1).strip()
        content = match.group(2)
        files[path] = content
    return files


class AppGenerator:
    """Generates application scaffolds with LSP validation loop."""

    def __init__(
        self,
        llm: "OllamaClient",
        registry: "ToolRegistry",
        workspace: str = ".",
        max_fix_iterations: int = 3,
    ):
        self.llm = llm
        self.registry = registry
        self.workspace = Path(workspace)
        self.max_fix_iterations = max_fix_iterations

    def generate(
        self,
        request: str,
        language: str,
        framework: str,
        project_name: str,
        profile: LanguageProfile,
    ) -> dict[str, str]:
        """Generate an application. Returns dict of {path: content}."""

        console.print(Panel(
            f"[bold cyan]Generating {language}/{framework} app: {project_name}[/bold cyan]\n"
            f"Request: {request}",
            title="Flow CodeGen",
            border_style="cyan"
        ))

        profile_ctx = profile_summary(profile)
        prompt = build_codegen_prompt(
            request=request,
            language=language,
            framework=framework,
            project_name=project_name,
        )

        messages = [
            {"role": "system", "content": f"You are an expert {language} developer.\n\nLanguage Profile:\n{profile_ctx}"},
            {"role": "user", "content": prompt},
        ]

        # Generate initial code
        console.print("[dim]Generating initial code...[/dim]")
        resp = self.llm.chat(messages=messages)
        content = resp["choices"][0]["message"]["content"]
        files = parse_files_from_response(content)

        if not files:
            console.print("[yellow]No files parsed from response, returning raw content.[/yellow]")
            return {"main": content}

        # Write files
        self._write_files(files, project_name)

        # LSP validation loop
        files = self._validate_and_fix(files, language, project_name, messages, profile)

        return files

    def _write_files(self, files: dict[str, str], project_name: str) -> None:
        project_dir = self.workspace / project_name
        for rel_path, content in files.items():
            full_path = project_dir / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            console.print(f"[green]✓[/green] Wrote: {rel_path}")

    def _validate_and_fix(
        self,
        files: dict[str, str],
        language: str,
        project_name: str,
        messages: list[dict],
        profile: LanguageProfile,
    ) -> dict[str, str]:
        """Run LSP check and fix errors iteratively."""
        import json

        for iteration in range(self.max_fix_iterations):
            console.print(f"\n[dim]Running LSP check (iteration {iteration + 1}/{self.max_fix_iterations})...[/dim]")

            result_str = self.registry.dispatch(
                "lsp_check",
                {"language": language, "file_path": None}
            )

            try:
                result = json.loads(result_str) if isinstance(result_str, str) else result_str
            except Exception:
                break

            if result.get("error"):
                console.print(f"[yellow]LSP tool error:[/yellow] {result['error']}")
                break

            if result.get("clean", True):
                console.print("[bold green]✓ Code passes LSP check — no errors.[/bold green]")
                break

            # Feed errors back to LLM
            console.print("[yellow]LSP found issues, asking LLM to fix...[/yellow]")
            messages.append({"role": "assistant", "content": f"LSP check result:\n{result_str}"})
            messages.append({
                "role": "user",
                "content": "Fix all LSP errors above. Output only corrected files in FILE: path format."
            })

            resp = self.llm.chat(messages=messages)
            fix_content = resp["choices"][0]["message"]["content"]
            new_files = parse_files_from_response(fix_content)

            if new_files:
                files.update(new_files)
                self._write_files(new_files, project_name)

        return files
