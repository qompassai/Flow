# Safety model

phlow is **fail-closed**: anything not explicitly allowed is denied, and
every denial is loud. The model's job is to be clever inside the walls;
the runtime's job is to be the walls.

## The walls

- **No shell.** There is no tool that runs a shell command, and no
  model-selected command execution. This is not a missing feature; it is
  the load-bearing wall. Restoring a shell tool would be a security
  redesign, not an enhancement.
- **No raw writes.** All file writes go through the workspace layer:
  contained to the workspace root, no symlink following, descriptor-
  relative operations, atomic writes, root-freshness checks, size caps.
- **No executable plugins.** Plugins are data (profiles, prompts), never
  code that the runtime loads and executes.
- **No model-selected network.** The only network peer is the configured
  Ollama server, loopback by default; remote backends need explicit
  opt-in. There is no silent cloud fallback.
- **Read-only by default.** Workspace writes and named checks require
  `--trusted`. `--trusted` is *not* an OS sandbox and does not invent
  new commands — it only unlocks the two things the operator configured.

## Operator-approved checks

Named checks are the one place phlow executes arbitrary host commands —
so they are the most tightly held:

- Checks are **configured explicitly** in TOML: a name and an exact
  argv. No globbing, no shell interpolation, no PATH search surprises.
- `phlow check [name]` / `flow_check` runs the named check, or all of
  them. A missing check is an error, never a silent skip.
- Checks run **on the host**, and a project test executes code: the
  operator who configures a check is asserting they trust that argv.
  Reviewer approval inside a run does not substitute for host-run
  checks.

## Disabled paths (deliberate, permanent)

These Python features are **disabled in the Rust port by design** and
stay disabled:

- **Shell tool** — see above.
- **LSP integration** — the port does not speak LSP; editor intelligence
  arrives through the Neovim socket's tool surface instead.
- **Prompt evolution** (`phlow-self-improve`'s evolver) — the crate holds
  the data model and validation, but the evolution loop does not run
  unprompted. Self-modifying prompts are an operator decision, not a
  background process.

## Fail-closed, mechanically

- Unknown MCP methods → JSON-RPC errors, not panics.
- Backend down → the report says `status: "error"` with an explanation;
  the process exits 1. Never a bare traceback, never a hang.
- Oversize input (1 MiB MCP frames, 1 MiB TUI lines, bounded task
  lengths, bounded feedback fields) → rejected before allocation, never
  truncated silently.
- SIGTERM → the interruption message on stderr, exit 130. Already-
  written changes are not rolled back — the message says so, honestly.
