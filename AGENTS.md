# Flow Agent Guidelines

Read this file before editing and the applicable procedure in `SKILLS.md` before running
checks. Follow scoped instructions and the current task; this file grants no extra trust.
These rules govern development of Flow, not automatic loading into its model runtime.

## Think before coding

- Inspect relevant code, branch, dirty files and tooling. State material assumptions,
  uncertainties and tradeoffs. Ask before choosing between materially different behaviors.
- Define observable acceptance criteria and non-goals. Map a short plan to actual checks.
  For a bug, first add a reproducer; for a refactor, verify before and after.
- Prefer a simpler approach and say when the requested design is overcomplicated.
  Keep trivial tasks lightweight rather than manufacturing a planning pipeline.

## Simplicity and surgical changes

- Implement only the requested behavior, with minimum code. No speculative features,
  configurability, dependencies, single-use frameworks or impossible-case handling.
- Preserve public APIs, JSON/MCP contracts, configuration and unrelated user work.
  Every changed line must serve the request. Do not reformat or refactor adjacent code.
- Remove only dead code your changes create. Flag existing debt without deleting it.

## Tiger Style and performance

- Prioritize safety/correctness, performance, then convenience. Use simple control flow,
  small scopes and explicit units; introduce no recursion.
- Bound input, context, tool calls, iterations, subprocess output, queues and retries.
  Use named limits, reject overflow before unbounded allocation and retain cancellation.
- Assert meaningful internal invariants. Validate external data with always-on checks;
  handle real I/O/missing-tool failures as errors, never fabricated success.
- Own and release files/processes/connections exactly once on every termination path.
  Revalidate ownership and freshness after asynchronous work.
- Target changed functions at most 70 physical lines. Keep Python at the repository's
  100-column Ruff formatting. Split by responsibility, not minification; report exceptions.
- Prefer native APIs, bounded batching and minimal copies. Managed runtimes are not
  allocation-free. Benchmark before claiming speedups.
- Use targeted reads/tests and disjoint file ownership. Avoid duplicate scans and competing
  writers. After two failed attempts at one hypothesis, investigate or escalate, not retry blindly.

## Flow-specific invariants

- Keep the shared safe runtime authoritative for CLI, TUI and MCP. Do not restore legacy
  unrestricted shell, raw writes, executable plugins or model-selected command execution.
- Preserve read-only defaults, explicit operator configuration and exact approved check argv.
  `--trusted` is not permission to invent commands or automatically load project instructions.
- Preserve containment, no-follow/descriptor-relative file operations, atomic writes,
  root-freshness checks and size caps. A project test executes code; Flow is not an OS sandbox.
- Preserve loopback defaults, explicit remote opt-in and no silent cloud fallback.
  Do not log secrets or download models during routine tests.
- Missing checks/tools, stale content, exhausted budgets and empty cached diagnostics are
  not verification. Reviewer approval does not substitute for host-run checks.
- Python is governed by `pyproject.toml`. If adding/editing Lua, read the strict Lua
  contract in `SKILLS.md`; do not weaken nil/union/shape checks.

## Verify and hand off

Run focused tests first, then affected regression/static gates on the final diff.
Report exact commands, scope, exit status, skips and remaining failures. Separate behavior,
style, type coverage and measured performance. Re-run affected gates after edits.
Never suppress diagnostics or remove tests to create a green result.

For substantive work using explicitly selected Astra6/Fable5.1, also provide a task
`HANDOFF.md` and reusable `AGENTS.md`/`SKILL.md` text. Include exact inspected files/APIs,
contracts, limits, ordered edits, failure cases, checks and stop rules. Do not infer model
identity or promise model parity. Do not commit handoff artifacts unless requested.

This policy adapts the user's
[Karpathy-inspired guidelines](https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md)
and [Tiger Style](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md).
