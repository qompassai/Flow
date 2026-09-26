# Introduction: what phlow is

**phlow** — "physics-informed flow" — is a small, bounded, local
multi-agent coding runtime. The name is the design brief in one word:

- **flow**: work moves through a pipeline the way a fluid moves through
  channels — planner, coder, reviewer, each stage feeding the next, with
  the workspace as the channel bed.
- **physics-informed**: the channels have walls. Like a physical system
  with conservation laws, phlow has invariants that no prompt, plugin, or
  model output can bend: bounded iterations, bounded context, explicit
  operator trust, no shell, no network except loopback Ollama. The
  physics don't negotiate.

In practice phlow is one shared safe runtime with three faces:

- a **CLI** (`phlow` / `flow`): `run`, `serve`, `check`, `status`, `tui`;
- a **TUI**: an interactive terminal frontend over the same runtime;
- an **MCP server** (`phlow serve`): newline-delimited JSON-RPC over
  stdin/stdout, so editors (rose.nvim) and other agents can drive it.

It was born as Python (`flow/`) and has been ported crate-by-crate to
Rust on the `rust` branch. This book documents the Rust port the way
you'd explain it to a smart five-year-old — with the exact contracts an
operator or integrator needs, and an honest list of where the port
deliberately differs from the Python.

## Who this book is for

- **Operators** who run `phlow serve` under systemd and want the safety
  model in plain language.
- **Integrators** (rose.nvim, other MCP clients) who need the wire
  contracts byte-exact.
- **Contributors** who need to know which differences from Python are
  deliberate and which are bugs (spoiler: the deliberate ones are all in
  [Porting notes](porting-notes.md); anything else is a bug — file it).
