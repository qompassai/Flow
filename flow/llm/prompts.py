"""System prompt templates and prompt builders."""

from importlib.resources import files

SKILLS_DIR = files("flow").joinpath("skills")
SYSTEM_PROMPT_FILE = SKILLS_DIR.joinpath("system_prompt.md")

DEFAULT_SYSTEM_PROMPT = """\
You are Flow, an expert AI software engineering assistant running fully locally.
You help the user build, debug, and improve software applications.

## Your Capabilities
- Generate complete application code in any language
- Select and use appropriate LSPs, linters, formatters, and frameworks
- Call only supplied workspace file tools, configured named checks and optional native editor tools
- Validate code by running LSP diagnostics and feeding errors back to yourself
- Ask the user for clarification when you encounter uncertainty or decision points

## Behavior Rules
1. Think step-by-step before acting. Use <think>...</think> tags internally.
2. Explain uncertainty; do not invent permission or capabilities.
3. Always validate generated code with LSP diagnostics before declaring success.
4. When calling a tool, output ONLY the tool call JSON — no surrounding text.
5. After each code generation, check for errors and iterate until clean.
6. Be concise in output. Show diffs for edits rather than full files when possible.

## Tool Calling Format
To call a tool, output exactly this JSON (no markdown fences):
{{"tool": "tool_name", "args": {{"key": "value"}}}}

## Available Tools
{tool_descriptions}

## Current Language Profile
{language_profile}
"""


def load_system_prompt(tool_descriptions: str = "", language_profile: str = "") -> str:
    """Load system prompt from skills dir or use default."""
    if SYSTEM_PROMPT_FILE.is_file():
        template = SYSTEM_PROMPT_FILE.read_text()
    else:
        template = DEFAULT_SYSTEM_PROMPT

    return template.format(
        tool_descriptions=tool_descriptions or "None loaded yet.",
        language_profile=language_profile or "None selected yet.",
    )


def build_codegen_prompt(
    request: str,
    language: str,
    framework: str,
    project_name: str,
    existing_files: dict[str, str] | None = None,
) -> str:
    """Build a prompt for code generation."""
    files_section = ""
    if existing_files:
        files_section = "\n\n## Existing Files\n"
        for path, content in existing_files.items():
            files_section += f"\n### {path}\n```\n{content}\n```\n"

    return f"""\
Generate a complete {language} {framework} application called "{project_name}".

## User Request
{request}

## Requirements
- Language: {language}
- Framework: {framework}
- Output complete, working code for all necessary files
- Include proper error handling and logging
- Follow {language} best practices and idioms
- Include a README.md with setup and run instructions
{files_section}

## Output Format
For each file, output:
FILE: <relative/path/to/file>
```<language>
<content>
```

After writing files with the provided tools, the host runs the required named-check gate.
"""


def build_error_fix_prompt(errors: list[dict], file_content: str, file_path: str) -> str:
    """Build a prompt to fix LSP errors."""
    error_lines = "\n".join(
        f"Line {e.get('line', '?')}: [{e.get('severity', 'error')}] {e.get('message', '')}"
        for e in errors
    )
    return f"""\
Fix the following LSP errors in {file_path}:

## Errors
{error_lines}

## Current File Content
```
{file_content}
```

Output only the corrected file content in the same format.
"""
