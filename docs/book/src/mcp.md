# The MCP stdio contract

`phlow serve` is an MCP server that talks **newline-delimited JSON-RPC
2.0**: one JSON object per line on stdin, one JSON object per line on
stdout. There is no HTTP, no Content-Length framing, no websockets.

## The rules of the wire

- **stdout carries protocol JSON exclusively.** Logs, warnings, and the
  interruption message go to stderr. A client may parse every stdout
  line as JSON, unconditionally.
- **Every frame is flushed** before the next is read. No batching.
- **Blank lines are skipped** (but counted, like the Python server).
- **A frame over 1 MiB** gets one `-32700` parse-error response and the
  connection closes. The over-long line's remainder is deliberately left
  unread.
- **EOF ends the session cleanly** (exit 0). The runtime is closed on
  every exit path — EOF, oversize frame, I/O error, SIGTERM.
- **Malformed JSON** gets a `-32700` parse error; the session continues.
- **Unknown methods** get `-32601`; **bad params** get `-32602`.

## Lifecycle

1. Client sends `initialize` with `params.protocolVersion` (nonempty
   string), `capabilities`, and `clientInfo`. The server negotiates and
   answers with its own `protocolVersion`, capabilities, and server info.
2. Client sends the `notifications/initialized` notification (no `id`, no
   answer). The server is now *ready*.
3. Before step 2 completes, `tools/list` and `tools/call` are rejected
   with `-32002`: *"Initialize and send notifications/initialized
   first"* — the same gate the Python server enforces.

Minimal session:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"rose","version":"1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
```

## Tools

`tools/list` advertises three tools; `tools/call` dispatches by name:

| Tool | Does |
|---|---|
| `flow_run` | Run one task; the result is the report object |
| `flow_status` | Local capabilities; no model call |
| `flow_check` | Run configured named checks (`name` argument optional) |

`run`/`status`/`check` never fail as Rust operations — failure is encoded
*in the report value* (`status` ≠ `"ok"`), exactly like the Python
runtime. The MCP error channel is reserved for protocol-level problems.

## Systemd shape

Because the protocol is stdio, the systemd units (`packaging/`) use
**socket activation**: each accepted connection spawns one `phlow serve`
whose stdin/stdout are the connection. See [the CLI chapter](cli.md) and
`packaging/README.md` for the full story.
