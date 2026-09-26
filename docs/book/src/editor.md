# The private Neovim socket

phlow can ask *back* into the editor. While the MCP window lets the
editor drive phlow, the **reverse channel** lets phlow's agents ask the
editor to do editor things — via a private msgpack-RPC socket that
rose.nvim opens.

## The contract

- The socket path comes from `--nvim` (explicit) — never guessed, never
  scanned. Without `--nvim`, the bridge's worker thread idles until the
  process exits; no connection is attempted.
- Requests are **msgpack-RPC** with a bounded round-trip timeout
  (`--editor-timeout`, default 120 s, hard window 0.1–660 s). A timeout
  or transport failure **poisons** the bridge: later calls fail fast
  instead of hanging.
- The model never sees raw editor access. It sees **tool schemas**; the
  runtime validates every call against an allow-list before anything
  touches the socket.

## The Lua surface (exact)

rose.nvim exposes the tool surface through two Lua expressions. The
model-facing schema list is produced by:

```lua
return require('rose.tools').schemas()
```

and a tool call is executed by:

```lua
return require('rose.tools').call(...)
```

phlow sends exactly these expressions over the socket. `schemas()`
advertises only what the runtime's validation gate will accept.

## The `editor_debug` gate: `status` only

The runtime's dispatch gate accepts **one** `editor_debug` action:
`status`. The advertised schema is `["status"]`, and validation runs
*before* the gate — so the model-facing surface is `status`-only, with
the 4-action gate kept behind it as defense-in-depth. This mirrors the
Python exactly (empirically probed, not just code-read: Python's
`schemas()` advertises `["status"]` while its `DISPATCH_GATE` lists four
actions, and validation rejects anything outside the schema first).

### ⚠️ Unresolved mismatch (rose.nvim side, not phlow)

`rose.nvim/lua/rose/tools.lua` advertises `editor_debug` with
`enum = {"status", "run"}` — but the runtime accepts `status` only and
rejects `run`. The runtime is correct and fail-closed; Rose's own schema
is misleading. Fixing it means editing the rose.nvim repo, which is **not
authorized** in this port — it is flagged for the rose/diver integration
pass with Matt's go-ahead. Do not "fix" it from the phlow side by
accepting `run`: that would widen the model-facing surface beyond what
Python allows.

## Timeouts

`--editor-timeout` is validated twice: the CLI rejects anything outside
0.1–660 s with `Flow: --editor-timeout must be between 0.1 and 660
seconds` (exit 2), and the bridge constructor enforces the same window.
The bridge also enforces a worker-join grace period on shutdown so a
stuck reverse request cannot hold the process open.
