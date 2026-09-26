# Porting notes: deliberate deviations

The port's rule is *byte-parity where it matters, honesty where it
can't*. Everything below is a place where the Rust port knowingly differs
from the Python — each was a conscious decision, each is tested, and none
is silent. If you find a difference not listed here, it is a bug: file
it.

## 256-profile cap: error, not truncation

Python's codegen silently truncates the profile file list at
`PROFILE_FILES_MAX = 256`. The Rust port **errors** instead
(`phlow-codegen`, `TooManyProfiles`). Silent truncation drops user data
without a trace; a loud error names the limit and stops. The eight
effective profiles (including inherited `project_templates`) match
Python exactly — only the overflow behavior differs.

## The `RuntimeFacade` seam (closed in Phase 6)

Phase 5's TUI drove a `RuntimeFacade` trait because the real runtime
exposed no model listing or switching. Phase 6 closed the seam:
`phlow-cli`'s `CliRuntime` newtype implements both `McpRuntime` (for
`serve`) and `RuntimeFacade` (for the TUI) over the real
`Runtime<ReqwestTransport, MsgpackTransport>`, backed by three small
additive methods — `select_model`, `set_editor_timeout`, `list_models`.
No facade behavior is faked: `/model` really switches the model the next
run uses, because it assigns `ollama.model` and resets role overrides
exactly like Python's `/model` handler.

## Ctrl-C terminates; it doesn't continue

Python's TUI catches `KeyboardInterrupt` and continues the loop. The
Rust TUI (both the ratatui frontend and the headless line loop) lets
Ctrl-C terminate the process. Rationale: prompt_toolkit's resume-after-
interrupt depends on its own signal machinery; reproducing "continue"
around a half-read line in the Rust line loop would risk swallowing the
operator's real intent to stop. Termination is the safer default, and it
is the documented one.

## Feedback character limits

`phlow-self-improve` validates feedback records with named character
bounds (`SESSION_ID_CHARS_MAX = 256`, `COMMENT_CHARS_MAX = 10_000`,
`PROMPT_USED_CHARS_MAX = 100_000`, `OUTCOME_CHARS_MAX = 10_000`,
`LOW_RATED_LIMIT_MAX = 1000`). Over-limit fields are rejected with a
typed error, never truncated — the same philosophy as the profile cap:
bounds are contracts, and contracts fail loudly.

## Python-compatible JSON serialization

Reports and MCP frames use `python_json_dumps`, which reproduces
`json.dumps(..., ensure_ascii=True)` byte-for-byte: ASCII escaping of
non-ASCII characters, surrogate pairs for astral-plane codepoints, no
spaces in separators, and nested empty containers preserved (`{}`,
`[]` stay as-is rather than collapsing). Verified differentially
against CPython in the test suite.

## The exact stdin boundary

The TUI's line reader enforces `INPUT_LINE_BYTES_MAX = 1_000_000` bytes
**including the line's single trailing newline**: a 1,000,000-byte line
plus its newline is 1,000,001 bytes and is rejected before the strip —
exactly like the Python's accounting. The returned line has the one
trailing newline removed, like Python's `input()`. The boundary is
pinned by tests on both sides of it.

## Non-TTY TUI runs headless

Python's prompt_toolkit degrades on pipes; the Rust ratatui frontend
cannot draw without a TTY at all. So when stdout is not a terminal,
`phlow tui` runs the same `read_line`/`classify_line` loop headless:
plain-text banner, results as compact JSON lines. The line loop — the
part Phase 5 verified — is identical; only the widgets are gone.

## Unresolved: the Rose `editor_debug` mismatch

Recorded here so it is not lost: `rose.nvim/lua/rose/tools.lua`
advertises `editor_debug` with `enum = {"status", "run"}`, but the
runtime (Python and Rust alike) accepts `status` only. The runtime is
correct; Rose's schema is misleading. Fixing it requires editing the
rose.nvim repo, which is outside this port's authorization — it waits on
the rose/diver integration pass. The port must not "fix" it by accepting
`run`.
