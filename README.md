# Flow

Local, bounded planner → coder → reviewer workflows, with real file tools and a
host-enforced verification gate. Ollama is the default backend. CLI, terminal UI
and the MCP server share **one safe runtime**; none uses the old unrestricted shell
or auto-loaded plugin path.

Flow can work alone or bidirectionally with [Rose](https://github.com/qompassai/rose.nvim):
Rose calls Flow over MCP stdio; Flow calls Rose's native editor tools over a private
Neovim socket.

## Get started

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Ollama with an
installed tool-capable model. POSIX is required for Flow's secure local file I/O;
the primary targets are Linux and macOS.

From this checkout:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[editor,dev]'

# In a separate terminal, if not already running:
ollama serve
ollama pull qwen2.5-coder:7b

# No model request or command execution; this also works without Ollama running:
.venv/bin/flow status --workspace /absolute/path/to/project
```

The `editor` extra installs optional pynvim. `dev` installs pytest, Ruff and the
wheel builder. Omit extras for the basic CLI/TUI runtime. Dependencies belong in
the project environment, not a global Python installation.

### Approve checks, then run

Create an **operator-reviewed** TOML file, for example `flow-operator.toml`.
Workspace values must be at the root, **before any TOML table header**.

```toml
workspace_dir = "/absolute/path/to/project"

[ollama]
base_url = "http://127.0.0.1:11434"
model = "qwen2.5-coder:7b"
timeout = 120 # seconds

[models]
planner = "qwen2.5-coder:7b"
coder = "qwen2.5-coder:7b"
reviewer = "qwen2.5-coder:7b"

[agent]
max_iterations = 6      # model turns per role invocation
max_cycles = 2          # coder → verification → reviewer cycles
max_tool_calls = 64     # total across the entire run
max_context_chars = 100000

[checks.tests]
cmd = [".venv/bin/python", "-m", "pytest", "-q"]
timeout = 60000         # milliseconds, as in Rose
required = true
kind = "test"
filetypes = ["python"]

[checks.lint]
cmd = [".venv/bin/ruff", "check", "."]
timeout = 30000
required = true
kind = "lint"
filetypes = ["python"]
```

Those executables must exist **in the target project's environment**; Flow does
not install project dependencies or select commands for you. For this Flow checkout,
the included `config.toml` supplies its own test and lint checks.

```sh
.venv/bin/flow check --config /absolute/path/to/flow-operator.toml --trusted
.venv/bin/flow run --config /absolute/path/to/flow-operator.toml --trusted \
  "Fix the failing parser tests and explain the changes"
.venv/bin/flow status --workspace /absolute/path/to/project
.venv/bin/flow tui --config /absolute/path/to/flow-operator.toml --trusted
```

`run`, `check`, and `status` print one JSON object. Exit codes: `0` for an `ok`
result, `1` for failed/unverified results, `2` for configuration/usage errors, and
`130` for an interrupted session. Common flags work before or after the command.
`--workspace` overrides configuration **before** runtime/tool construction;
`--model` overrides every role. With no subcommand, `flow` starts the TUI.

TUI commands: `/help`, `/status`, `/check [name]`, `/tools`, `/models`,
`/model <installed-model>`, `/build <request>`, `/clear`, `/quit`. Ctrl-D exits.
Natural-language tasks and `/build` use the same verified runtime. Every task has
fresh, separate role contexts. Legacy `/plugins`, `/evolve`, `/memory`, and
`/feedback` are explicitly unavailable, not silently wired to unsafe helpers.

## What “verified” means

A run is `verified: true` only when:

1. The planner/coder complete within their budgets.
2. At least one required named check runs, and **every required check passes**.
3. Every source file changed through Flow has successful relevant static analysis.
4. A separate read-only reviewer returns a structured approval.
5. Edited files have not changed during verification/review and attached editor
   buffers are not dirty.

Static evidence is an explicitly configured required check with
`kind = "lint"`, `"typecheck"`, or `"diagnostics"` and matching `filetypes`, or a
completed native Rose linter returning `status="ok", verified=true`.
`kind="test"`, `"build"`, and `"check"` do not by themselves establish static
language coverage. Use language names (`"python"`), extensions (`".py"`), or an
explicit `"*"` only when a check really covers every language you edit.
Unknown source extensions can use their extension or suffix as the coverage key.
Documentation/data extensions such as `.md`, `.json`, and `.toml` still require
the named gate but do not require a source-language linter.

All required checks run: `filetypes` describes coverage, never silently skips a
required command. Optional checks are reported but cannot replace a required gate.
Checks are language-agnostic exact argv: Rust, Go, TypeScript, C, Nix, or another
language can use the same mechanism. There are no inferred `npx` downloads or
automatic package-manager installs.

Missing executables/checks, timeout, invalid reviewer output, absent static
coverage and exhausted budgets are **not success**. Failed checks and review
issues feed the next bounded repair cycle. Reports retain `checks`,
`verification.coverage`, `verification_history`, changed files, role/model/turn
counts and tool-call IDs/errors. Empty cached LSP diagnostics are only a snapshot,
not verification; a missing LSP remains explicitly `unavailable` even when a
real linter independently verifies its configured scope. Actual diagnostic
errors or stale results veto native coverage.

This is evidence within configured checks' scope, not a proof that all behavior,
all languages, or an entire application are correct.

## Trust and security

- Default mode is **read-only**. `--trusted` allows project edits and explicitly
  configured named checks. It does not approve new model-selected commands.
- `--config FILE` explicitly approves loading that operator file.
  Workspace `config.toml`/`.flow.toml` are never auto-loaded, even with `--trusted`.
  Only `$XDG_CONFIG_HOME/flow/config.toml` (default `~/.config/flow/config.toml`)
  is loaded automatically as user configuration.
- No model-controlled shell, Python/Bash code execution, executable arguments,
  cwd, environment overrides, deletion tool, plugins, Lua, or Ex commands.
  A named check may invoke an interpreter, but its **entire argv** is fixed by
  operator configuration and snapshotted before the run.
- File tools require relative paths. Parent traversal, absolute/Windows-style
  paths, `.git`, symlinks, multiply-linked files and outside-root paths are
  rejected. Writes are atomic, size-limited, descriptor-relative and no-follow.
  Root replacement is detected as stale. Operator configuration cannot be edited
  through the model file tools.
- File reads/writes are limited to 256 KiB. Listings are capped and exclude
  generated directories. Tool output and model response sizes are bounded.
- The default Ollama URL is loopback. HTTP redirects and environment proxies are
  disabled. Non-loopback endpoints require explicit `ollama.allow_remote=true`;
  no cloud fallback, API key lookup or automatic external search exists.
- Trusted project **tests/builds/linters execute code** and can access the OS,
  network and files as your user. Named commands can run project configuration,
  plugins or hooks. This is **not an OS sandbox**: review the project and the
  exact check command, use a container/VM for untrusted projects, and do not run
  Flow with elevated privileges. Concurrent hostile same-user processes are
  outside the security guarantee.

Legacy executable factories now fail closed. The former application generator
and orchestrator names require an explicit `Runtime`; their raw-write/text-to-shell
paths are gone. Bundled prompts/profiles are installed inside `flow/skills` and
read as package resources, never implicitly from a project's `skills/` directory.

## Rose integration

Configure Rose's native entrypoint with the same absolute workspace and trust:

```lua
require('rose').setup({
  workspace = '/absolute/path/to/project',
  trusted = true,
  ollama = { base_url = 'http://127.0.0.1:11434', model = 'qwen2.5-coder:7b' },
  flow = {
    cmd = { '/absolute/path/to/flow/.venv/bin/flow', 'serve',
            '--config', '/absolute/path/to/flow-operator.toml' },
    bridge = true,
    timeout = 300000, -- milliseconds; budget for the complete bounded run
  },
})
```

Rose appends the workspace/trust flags and starts/passes a private socket for the
reverse bridge. Manual invocation has the same interface:

```sh
flow serve --workspace /absolute/root --trusted \
  --config /absolute/operator.toml --nvim /private/directory/nvim.sock \
  --editor-timeout 120
```

The socket must be explicitly supplied and same-user/private: on POSIX use a
`0700` parent directory or a socket inaccessible to group/others. TCP is rejected.
**A Neovim socket grants arbitrary code execution to other same-user clients**;
keep it private. Flow itself only sends the fixed expressions
`return require('rose.tools').schemas()` and
`return require('rose.tools').call(...)`, with names/args as parameters.

Editor schemas are discovered dynamically and filtered to `editor_context`,
`editor_diagnostics`, `editor_symbols`, `editor_references`, `editor_lint`,
`editor_check`, `editor_scip`, and `editor_debug`. Debug launch/probes remain
manual: the agent sees only `editor_debug` status. Planner/reviewer receive
read-only tools, not checks/lint or writes.

For buffer safety the reverse bridge also allowlists fixed `file_read` and
`file_write` calls. Flow validates its paths, trust, role, size and protected
configuration first, verifies Rose's resolved workspace matches, then uses
Rose's buffer-aware I/O. Unsaved/read-only buffers are not overwritten; loaded
buffers are synchronized with disk writes. An unavailable/mismatched bridge
does **not** fall back to unsafe disk writes. Named checks refuse unsaved buffers.
Rose's editor-context freshness API v1 is required: Flow checks the global dirty
buffer list and compares loaded/observed workspace snapshots across checks and
review, including noncurrent buffers. Missing freshness fields fail closed.

pynvim is imported on the main thread before attachment (pynvim records the
signal-owner thread at import time); connection creation, requests and close stay
on one owning worker. Editor request timeout defaults to 120 seconds and is
configurable from 0.1–660 seconds. Model and check timeouts have separate units.
SCIP support is decoded JSON supplied by Rose, not raw protobuf decoding.
No reverse `flow_run` calls are exposed.

### MCP wire contract

One UTF-8 JSON-RPC 2.0 object per newline; not Content-Length framing.
Protocol `2025-11-25`; compatible dates `2025-06-18` and `2025-03-26` are accepted.
Send `initialize`, then `notifications/initialized`, before `tools/list` or
`tools/call`. `ping` is available throughout.

| Tool | Arguments |
|---|---|
| `flow_status` | `{}` |
| `flow_run` | `{"task":"..."}` |
| `flow_check` | `{}` or `{"name":"configured-name"}` |

Tool responses contain `content=[{type:"text",text:"<JSON report>"}]`,
`structuredContent=report`, and `isError` for non-ok outcomes. The MCP tool list
is intentionally only these three tools. Stdout is **protocol only**; stderr
is for diagnostics. Parse/method/parameter errors preserve JSON-RPC IDs where
valid. Oversized frames close the session after an error.

The server is deliberately sequential to fit on device. It does not process
`notifications/cancelled` cooperatively during a running request. To cancel,
terminate the Flow session (RoseStop): SIGTERM/KeyboardInterrupt unwind active
check process-group cleanup and close the editor client; reconnect afterward.
Already saved edits are not rolled back. EOF closes an idle session; a busy
request finishes within its bounds unless the supervisor terminates it.
This stdio transport needs a client supervisor, not stdout redirected to a
systemd journal as a pretend network daemon.

## Development and verification

```sh
uv pip install --python .venv/bin/python -e '.[editor,dev]'
.venv/bin/pytest -q
.venv/bin/ruff check flow tests
.venv/bin/python -m compileall -q flow
.venv/bin/python -m build --wheel --no-isolation

# Include real headless editor tests when Neovim is not on PATH:
NVIM=/absolute/path/to/nvim .venv/bin/pytest -q
```

Tests use deterministic fake backends, a local fake HTTP Ollama endpoint, real
CLI/MCP subprocesses, and optional real headless Neovim. They cover role separation,
repair bounds, verification, tool IDs/errors, trust/config/path regressions,
timeouts/process cleanup, buffer bridge and wheel resources loaded outside the
checkout. Neovim-specific tests report skips if no executable is available.
No live Ollama inference/GPU quality test is claimed. A real model may choose poor
edits or exhaust its budget; inspect the resulting report and diff.

The Nix expression is retained but was not built in this environment.

MIT — Amor Fati Labs.
