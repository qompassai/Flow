"""LSP diagnostics tool — runs language-appropriate linters/checkers."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


LSP_COMMANDS: dict[str, list[str]] = {
    "python": ["ruff", "check", "--output-format=json", "{file}"],
    "rust": ["cargo", "check", "--message-format=json"],
    "typescript": ["npx", "tsc", "--noEmit", "--pretty", "false"],
    "javascript": ["npx", "eslint", "--format=json", "{file}"],
    "go": ["go", "vet", "./..."],
    "lua": ["luacheck", "{file}", "--formatter", "plain"],
    "bash": ["shellcheck", "-f", "json", "{file}"],
    "c": ["clang", "-fsyntax-only", "{file}"],
    "cpp": ["clang++", "-fsyntax-only", "{file}"],
}


def run_lsp_check(language: str, file_path: str | None = None, workspace: str = ".") -> str:
    """Run the appropriate linter/checker for a language and return JSON results."""
    lang = language.lower()
    if lang not in LSP_COMMANDS:
        return json.dumps({"error": f"No LSP check configured for {language}"})

    cmd_template = LSP_COMMANDS[lang]
    cmd = [
        c.replace("{file}", file_path or ".") for c in cmd_template
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=workspace,
        )

        # Try to parse JSON output
        output = result.stdout or result.stderr
        try:
            parsed = json.loads(output)
            return json.dumps({
                "language": language,
                "returncode": result.returncode,
                "diagnostics": parsed,
                "clean": result.returncode == 0,
            })
        except json.JSONDecodeError:
            # Return raw output if not JSON
            return json.dumps({
                "language": language,
                "returncode": result.returncode,
                "raw_output": output[:3000],
                "clean": result.returncode == 0,
            })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "LSP check timed out"})
    except FileNotFoundError:
        return json.dumps({
            "error": f"LSP tool not found for {language}. Install: {cmd[0]}",
            "hint": f"Try: pacman -S {_install_hint(lang)}"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _install_hint(lang: str) -> str:
    hints = {
        "python": "python-ruff",
        "rust": "rust",
        "typescript": "npm (then: npm i -g typescript)",
        "javascript": "npm (then: npm i -g eslint)",
        "go": "go",
        "lua": "lua-check",
        "bash": "shellcheck",
        "c": "clang",
        "cpp": "clang",
    }
    return hints.get(lang, lang)


TOOL_SPEC = {
    "name": "lsp_check",
    "description": "Run LSP diagnostics/linting on code files. Returns errors and warnings.",
    "parameters": {
        "type": "object",
        "properties": {
            "language": {"type": "string", "description": "Programming language (python, rust, typescript, go, lua, bash, c, cpp)"},
            "file_path": {"type": "string", "description": "Specific file to check (optional, defaults to whole project)"},
        },
        "required": ["language"],
    },
}
