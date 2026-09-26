# Architecture (ELI5)

Imagine a workshop with very strict house rules. A customer (you, or your
editor) hands a work order through a little window. Inside, three
specialists — a **planner**, a **coder**, and a **reviewer** — pass the job
along a conveyor belt. A foreman (the **runtime**) watches everything:
nobody works overtime (bounded iterations), nobody leaves the workshop
(no shell, no network except the local model server), and nobody touches
the customer's shelves unless the customer signed a permission slip
(`--trusted`, explicit config).

That is phlow. The work order is a *task string*. The window is the CLI,
the TUI, or MCP-over-stdio. The conveyor belt is the agent loop. The
permission slip is the config.

## The 14 port crates

The Rust port is 14 crates, one per responsibility. (A 15th workspace
member, `phlow-inference`, predates the port and is not part of it — it
is an existing supporting crate, documented separately below.)

| Crate | Job, in one line |
|---|---|
| `phlow-json` | Python-compatible JSON values and `json.dumps` byte-parity |
| `phlow-config` | Validated config: every struct is born checked, fields are private |
| `phlow-workspace` | The channel bed: contained, descriptor-relative file operations |
| `phlow-checks` | Operator-approved named checks, exact argv, run on the host |
| `phlow-mcp` | The MCP window: newline JSON-RPC frames over stdin/stdout |
| `phlow-editor` | The private Neovim socket: reverse requests into the editor |
| `phlow-llm` | The Ollama backend: local model calls, loopback only |
| `phlow-runtime` | The foreman: owns config, agents, tools, reports, shutdown |
| `phlow-agent` | The conveyor belt: planner → coder → reviewer loop, bounded |
| `phlow-tools` | Tool implementations + Python-compatible JSON serialization |
| `phlow-codegen` | Code generation profiles (8 effective profiles, 256-file cap) |
| `phlow-self-improve` | Feedback records and prompt evolution (disabled by default) |
| `phlow-tui` | The terminal frontend: ratatui when there's a TTY, line mode otherwise |
| `phlow-cli` | The `phlow`/`flow` binaries: flags, exit codes, signals, wiring |

`phlow-inference` is a pre-existing workspace crate from before the port
began. It is not one of the 14 port crates and is not covered by the
Python-parity gates; it stays in the workspace as a supporting library.

## How a `run` flows

1. **Parse.** `phlow run "fix the typo"` → clap parses flags (accepted
   before *or* after the subcommand, like argparse's `parents=[common]`).
2. **Load config.** Explicit `--config` TOML only; project-local config
   is never auto-loaded. Invalid config → `Flow: <reason>`, exit 2.
3. **Build the runtime.** One `Runtime` owns the Ollama transport, the
   editor bridge, the check runner, and the workspace. Exactly one owner
   for every handle; everything closes exactly once.
4. **Agent loop.** Planner drafts, coder edits (workspace-contained,
   atomic writes), reviewer verdicts. Budgets bound every turn; the loop
   cannot run forever.
5. **Report.** One ASCII-escaped JSON line on stdout, byte-compatible
   with Python's `json.dumps(..., ensure_ascii=True)`. Exit 0 iff
   `status == "ok"`, else 1.

`serve` replaces steps 4–5 with a frame loop: read one JSON-RPC line,
dispatch (`initialize`, `tools/list`, `tools/call`), write one response
line, flush. EOF ends the session; the runtime closes on every exit
path. `status` and `check` skip the model entirely.

## The one-owner rule

Every resource in phlow — files, sockets, threads, the runtime itself —
has exactly one owner and is released exactly once on every path,
including errors and SIGTERM. This is the mechanical reason the
fail-closed model holds: there is no path where a half-built state is
left for someone else to trip over.
