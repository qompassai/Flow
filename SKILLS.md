# Flow Engineering Playbook

This is a repository playbook explicitly referenced by `AGENTS.md` and `CLAUDE.md`.
It is not an automatically loaded Agent Skill or a new runtime plugin. Read only the
section relevant to the task; resolve executable paths on the current machine.

## Python implementation or bug fix

1. Read the affected code and tests plus `pyproject.toml`. Define behavior and non-goals.
2. Reproduce the failure with a focused test. Include real invalid-input or I/O cases.
3. Apply one minimal patch preserving runtime trust, limits and API contracts.
4. Run the focused test, then the applicable gates below from the repository root.
5. Review the final diff and report exact outcomes, including unavailable tools.

With an existing development environment:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
git diff --check
```

For a focused run, replace the pytest argument with the actual affected test path.
Dependencies belong in a project environment; if it is missing, report that or obtain
approval to set it up. Do not pull an Ollama model or run a paid API to test offline code.

## Trust, checks or editor/MCP changes

Use the relevant tests in `tests/test_security_config.py`, `tests/test_limits.py`,
`tests/test_checks.py`, `tests/test_runtime.py`, `tests/test_editor.py` and
`tests/test_mcp_cli.py`. Inspect assertions rather than assuming a filename covers the change.

Preserve explicit operator configuration, fixed approved argv, containment, atomic writes,
freshness and output bounds. Test missing tools, permission/containment rejection, timeout,
stale buffers and invalid completion where relevant. Use fakes for unit tests; label actual
Neovim integration separately. Do not require Rose or Diver for unrelated Flow development.

## Lua additions or shared Lua review

Flow's Python checks do not validate Lua. If Lua is changed, use LuaJIT parsing and the
strict settings in
[Diver's LuaLS profile](https://github.com/qompassai/Diver/blob/main/lsp/lua_ls.lua):

```text
type.weakNilCheck = false
type.weakUnionCheck = false
type.checkTableShape = true
type.castNumberToInteger = false
type.inferParamType = true
diagnostics.groupSeverity["type-check"] = "Error"
diagnostics.severity["undefined-field"] = "Error"
```

For batch checks, set `diagnostics.groupFileStatus["type-check"] = "Any"` instead of
`Opened`, preserve diagnostic severity/disable settings and record the exact file coverage.
Resolve runtime libraries locally. Narrow nullable I/O/module/uv results before use;
prefer precise types and guards to blanket `any`, blind casts or suppressions.

Follow LuaJIT syntax behavior from
[Diver's StyLua profile](https://github.com/qompassai/Diver/blob/main/lsp/stylua_ls.lua),
but preserve the target repository's formatting configuration. Do not mistake formatting
for semantic diagnostics or claim Lua passed based on Python checks.

## Model-transfer packet

For substantive Astra6/Fable5.1 tasks or requested delegation, provide exact revision,
allowed files, verified APIs, goal/non-goals, input/output examples, limits, failure cases,
ordered steps and commands with expected behavior. Separate observed results from expected
results. Add concise reusable lessons with trigger, procedure, pitfalls and acceptance gates.

The receiving model verifies the revision, executes one bounded step and runs its check.
It stops for stale context, conflicting contracts, missing permissions or an unknown API
instead of guessing, widening scope or treating a model review as test evidence.
