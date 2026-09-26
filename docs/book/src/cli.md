# The CLI

Two binaries, one `main.rs`: `phlow` (primary) and `flow` (compatibility
alias). They differ only in the argv[0]-derived name on the `--help`
usage line; `--version` output is byte-identical.

## Commands

| Command | Does |
|---|---|
| `phlow run TASK...` | Run one task; print one JSON report line |
| `phlow serve` | MCP over stdio (see [MCP](mcp.md)) |
| `phlow check [NAME] [--name NAME]` | Run configured named checks |
| `phlow status` | Local capabilities; no model call |
| `phlow tui` | Interactive terminal frontend |
| `phlow` (no command) | Drops into the TUI, like Python |

Global flags (`--config/-c`, `--workspace/-w`, `--model/-m`,
`--trusted`, `--nvim`, `--editor-timeout`) are accepted **before or
after** the subcommand, exactly like argparse's `parents=[common]`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (`status == "ok"`, or the TUI/MCP session ended cleanly) |
| 1 | The command ran but the report status is not `"ok"` |
| 2 | Usage or config error (`Flow: <reason>` on stderr) |
| 130 | Interrupted (SIGTERM → the interruption message, exit 130) |

Notes that bite:

- `--version` prints `Flow 0.2.0` and exits 0. It is root-only:
  `phlow run --version` is a usage error (exit 2), like argparse.
- `check --name X` wins over a positional name (`args.check_name or
  args.name`).
- `--editor-timeout` must be within 0.1–660 s; anything else (including
  `nan`, `inf`, and negative numbers) is exit 2 with the exact message
  `Flow: --editor-timeout must be between 0.1 and 660 seconds`.
- Reports are one ASCII-escaped JSON line, byte-compatible with Python's
  `json.dumps(result, ensure_ascii=True)` — including surrogate-pair
  escaping and nested empty containers.

## The TUI

With a TTY on stdout, `phlow tui` runs the ratatui frontend (banner,
`/help`, `/models`, `/model`, `/status`, `/check`, `/quit`). With stdout
piped, the same line loop runs **headless** — plain prompts, results as
compact JSON — so `phlow < /dev/null` and CI pipelines work. Either way,
Ctrl-D exits 0.

`/model <name>` switches the model for all roles: it assigns
`ollama.model` and resets per-role overrides, exactly like Python's
`/model` handler. `/models` lists installed models from the Ollama
backend; if Ollama is unreachable the TUI warns on stderr and continues
with an empty list (Python fetched it lazily; the difference is
documented, the capability is identical).

One deliberate deviation: **Ctrl-C terminates the process** instead of
continuing the loop as Python does. See [Porting notes](porting-notes.md).

## Signals

SIGTERM is watched on a dedicated thread (unix only, mirroring Python's
`hasattr(signal, "SIGTERM")` guard). Delivery prints
`Flow interrupted; changes already written are not rolled back.` to
stderr and exits 130 — the same observable contract as Python's
SIGTERM→KeyboardInterrupt handler, including during a blocked `serve`
(the watcher owns signal delivery; the serve loop owns stdin; neither
waits on the other).
