# Flow Claude Guidelines

Before editing, read `AGENTS.md` completely and the relevant section of `SKILLS.md`.
Use that repository policy as the detailed contract; do not assume either file was
automatically loaded. Report conflicts instead of silently choosing a weaker rule.

## Working habits

- **Think first:** state material assumptions; surface ambiguity and simpler alternatives.
  Ask when a decision would change the implementation.
- **Keep it simple:** minimum code for the request, no speculative features or abstractions.
- **Edit surgically:** preserve APIs, formatting and unrelated work; remove only dead code
  introduced by your edits. Every changed line must serve the task.
- **Verify goals:** define pass/fail checks, reproduce bugs, implement, test and review the
  final diff. Unrun, unavailable and skipped checks are not passes.

Keep Flow's safe-runtime and operator-trust boundaries intact. Use Python 3.11-compatible
code and the existing Ruff settings. For substantive Astra6/Fable5.1 work, provide the
bounded handoff and reusable instructions required by `AGENTS.md`, without capability claims.
