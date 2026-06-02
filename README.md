# Flow — Local AI Workflow System

> Built by Amor Fati Labs for phaedrus

A fully local, on-prem AI agentic workflow system for Arch Linux and NixOS. No cloud. No API keys. Just you, your models, and your code.

## What Flow Does

Flow is a terminal-based AI coding agent that:

- **Builds apps** in any language using the correct LSP, linter, formatter, and framework
- **Self-validates** generated code via LSP diagnostics and iterates until clean
- **Calls tools** — web search, shell execution, file I/O, and your own custom plugins
- **Pauses and cues** you for guidance at decision points or when uncertain
- **Improves itself** by collecting session feedback and evolving its own system prompts

## Requirements

- **Ollama** running locally (`ollama serve`)
- A capable code model pulled: `ollama pull qwen2.5-coder:14b`
- **Python 3.11+**
- Language tools for whatever you're building:
  - Python: `pip install ruff python-lsp-server`
  - Rust: `pacman -S rust rust-analyzer`
  - Go: `pacman -S go gopls`
  - TypeScript: `npm i -g typescript typescript-language-server`
  - Lua: `pacman -S lua-language-server stylua`
  - Bash: `pacman -S shellcheck shfmt`

## Installation

### Option 1: pip (recommended for development)
```bash
git clone https://github.com/qompassai/Flow.git
cd Flow
pip install -e .
flow
```

### Option 2: Nix flake
```bash
git clone https://github.com/qompassai/Flow.git
cd Flow
nix develop         # Enter dev shell with all tools
pip install -e .
flow
```

### Option 3: systemd user service
```bash
cp systemd/flow.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now flow
```

## Configuration

Copy and edit `config.toml` (or place at `~/.config/flow/config.toml`):

```toml
[ollama]
model = "qwen2.5-coder:14b"   # Change to your model
base_url = "http://localhost:11434"

[tools]
web_search_backend = "duckduckgo"  # or "searxng" if you self-host
shell_timeout = 30

[self_improve]
enabled = true
```

## Usage

### Interactive TUI
```
flow
```

### TUI Commands

| Command | Description |
|---------|-------------|
| `/build python fastapi myapp` | Generate a FastAPI app |
| `/build rust clap mycli` | Generate a Rust CLI with clap |
| `/build go cobra myctl` | Generate a Go CLI with cobra |
| `/build typescript hono myapi` | Generate a TypeScript/Hono API |
| `/models` | List available Ollama models |
| `/model qwen2.5-coder:7b` | Switch model |
| `/tools` | List loaded tools |
| `/plugins` | Reload custom plugins |
| `/memory` | Show recent session memory |
| `/feedback` | Rate the current session |
| `/evolve` | Trigger self-improvement from feedback |
| `/clear` | Clear conversation context |
| `/quit` | Exit |

### CLI Flags

```bash
flow --config /path/to/config.toml   # Custom config
flow --workspace /my/project          # Set workspace dir
flow --model qwen2.5-coder:7b         # Override model
flow --version                        # Print version
```

### Natural Language Workflow

```
[flow:a1b2] > build me a REST API in Python with FastAPI that manages a list of patients with CRUD operations

[flow:a1b2] > now add JWT authentication to it

[flow:a1b2] > write unit tests for all the endpoints

[flow:a1b2] > scaffold a Rust CLI tool that wraps ffmpeg with a friendlier interface
```

### Custom Tool Plugins

Drop a `.py` file in `plugins/` with this shape:

```python
TOOL_SPEC = {
    "name": "my_tool",
    "description": "What this tool does",
    "parameters": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input data"}
        },
        "required": ["input"],
    },
}

def run(input: str) -> str:
    # Your logic here
    return "result"
```

Then run `/plugins` to load it. Flow's agent can now call your tool.

See `plugins/example_tool.py` for a full template.

## Supported Languages

| Language | LSP | Linter | Frameworks |
|----------|-----|--------|------------|
| Python | pylsp/pyright | ruff | FastAPI, Flask, Django, Click, Typer |
| Rust | rust-analyzer | clippy | axum, actix-web, clap, tokio, bevy |
| TypeScript | tsserver | eslint | Next.js, React, Express, Hono, NestJS |
| Go | gopls | golangci-lint | gin, echo, fiber, cobra |
| Lua | lua-language-server | luacheck | Neovim plugin, LÖVE2D |
| Bash | bash-language-server | shellcheck | systemd, CLI, deployment |
| C | clangd | clang-tidy | CMake, meson |
| Nix | nil/nixd | statix | flake, home-manager, NixOS module |

## Self-Improvement

Flow tracks session feedback and can evolve its own system prompt:

1. After a session, run `/feedback` and rate 1–5
2. When enough low-rated sessions accumulate, run `/evolve`
3. Flow uses the LLM to rewrite its system prompt to address failures
4. The new prompt is committed to `skills/system_prompt.md` via git

All prompts and language profiles live in `skills/` — edit them directly anytime.
They are git-tracked so you can revert changes.

## Adding a SearXNG Backend (optional)

For a fully offline web search (no DuckDuckGo DDG API calls):

```bash
docker run -d -p 8080:8080 searxng/searxng
# or with podman:
podman run -d -p 8080:8080 searxng/searxng
```

Then in `config.toml`:
```toml
[tools]
web_search_backend = "searxng"
searxng_url = "http://localhost:8080"
```

## Architecture

```
User Input → Orchestrator (ReAct loop)
               ├── LLM (Ollama local — OpenAI-compat API)
               ├── Tool Registry → [web_search, shell_exec, file_ops, lsp_check, plugins]
               ├── Memory (SQLite FTS5 — persistent across sessions)
               └── CodeGen → Language Profile → LSP Validation Loop
                                                   └── Fix Loop (max 3x)

Self-Improve: Feedback (SQLite) → Prompt Evolver (LLM) → Git-versioned skill store
```

## Memory

Flow stores every conversation turn in a local SQLite database at
`~/.local/share/flow/memory.db` using FTS5 full-text search. Relevant past
conversations are automatically retrieved and injected into the system prompt.

Use `/memory` to see recent entries. The database is yours — never leaves your machine.

## Model Recommendations

| Use Case | Model |
|----------|-------|
| Best code quality | `qwen2.5-coder:14b` |
| Faster iteration | `qwen2.5-coder:7b` |
| General + code | `qwen3:14b` |
| Low VRAM | `qwen2.5-coder:3b` |
| Reasoning-heavy | `deepseek-r1:14b` |

Pull any model: `ollama pull <model-name>`

## File Structure

```
flow/
├── flow/                    # Python package
│   ├── agent/               # ReAct orchestrator, context, SQLite memory
│   ├── llm/                 # Ollama client, prompt templates
│   ├── tools/               # Tool registry + built-in tools
│   ├── codegen/             # Language profiles, app generator, LSP validator
│   ├── self_improve/        # Feedback store, prompt evolver, skill store
│   └── tui/                 # Rich + prompt_toolkit TUI
├── plugins/                 # Drop your .py tool plugins here
├── skills/                  # Git-versioned prompts and language profiles
│   └── language_profiles/   # Per-language TOML config
├── systemd/                 # systemd user service unit
├── config.toml              # User configuration
├── pyproject.toml           # Python package definition
└── flake.nix                # Nix flake for reproducible environment
```

## License

MIT — Amor Fati Labs
