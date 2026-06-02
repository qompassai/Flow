"""Main ReAct agent orchestration loop."""
from __future__ import annotations

import json
import re
from typing import Callable, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from flow.llm.prompts import load_system_prompt
from flow.agent.context import ConversationContext
from flow.agent.memory import MemoryStore

if TYPE_CHECKING:
    from flow.llm.backend import OllamaClient
    from flow.tools.registry import ToolRegistry
    from flow.config import AgentConfig

console = Console()

PAUSE_PATTERN = re.compile(r"PAUSE:\s*(.+?)(?:\n|$)", re.IGNORECASE)


class Orchestrator:
    """
    ReAct-pattern agent that:
    1. Receives user input
    2. Thinks and plans (internal)
    3. Calls tools as needed
    4. Pauses for user input when uncertain
    5. Returns final response
    """

    def __init__(
        self,
        llm: "OllamaClient",
        registry: "ToolRegistry",
        cfg: "AgentConfig",
        memory: MemoryStore,
        on_pause: Callable[[str], str] | None = None,
        workspace: str = ".",
    ):
        self.llm = llm
        self.registry = registry
        self.cfg = cfg
        self.memory = memory
        self.on_pause = on_pause  # Callback when agent needs user input
        self.workspace = workspace
        self.context = ConversationContext()

    def run(self, user_message: str) -> str:
        """Run the agent loop for a user message."""

        # Build system prompt with current tool descriptions
        system_prompt = load_system_prompt(
            tool_descriptions=self.registry.tool_descriptions(),
        )

        # Add user message to context
        self.context.add_message("user", user_message)

        # Fetch relevant memory
        memories = self.memory.search(user_message, top_k=3)
        if memories:
            memory_ctx = "\n".join(f"- {m}" for m in memories)
            system_prompt += f"\n\n## Relevant Context from Memory\n{memory_ctx}"

        messages = [
            {"role": "system", "content": system_prompt},
            *self.context.get_messages(),
        ]

        iteration = 0
        final_response = ""

        while iteration < self.cfg.max_iterations:
            iteration += 1

            console.print(f"\n[dim]Agent iteration {iteration}/{self.cfg.max_iterations}[/dim]")

            # Call LLM
            resp = self.llm.chat(
                messages=messages,
                tools=self.registry.openai_tools_schema(),
            )

            choice = resp["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "stop")

            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])

            # Check for PAUSE directive
            if self.cfg.pause_on_uncertainty and content:
                pause_match = PAUSE_PATTERN.search(content)
                if pause_match:
                    pause_reason = pause_match.group(1).strip()
                    console.print(Panel(
                        f"[bold yellow]{pause_reason}[/bold yellow]",
                        title="[yellow]Flow needs your input[/yellow]",
                        border_style="yellow"
                    ))

                    if self.on_pause:
                        user_answer = self.on_pause(pause_reason)
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": user_answer})
                        self.context.add_message("assistant", content)
                        self.context.add_message("user", user_answer)
                        continue

            # Handle tool calls (OpenAI-format)
            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                    console.print(
                        f"[cyan]→ Tool:[/cyan] [bold]{tool_name}[/bold]"
                        f"({', '.join(f'{k}={repr(v)}' for k, v in args.items())})"
                    )

                    result = self.registry.dispatch(tool_name, args)
                    result_str = json.dumps(result) if not isinstance(result, str) else result

                    console.print(
                        f"[dim]  Result: {result_str[:200]}{'...' if len(result_str) > 200 else ''}[/dim]"
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", tool_name),
                        "content": result_str,
                    })

                continue

            # Check for inline JSON tool call in content
            # (fallback for models without native tool calling)
            if content and "{" in content:
                parsed = self.registry.parse_tool_call(content)
                if parsed:
                    tool_name, args = parsed
                    console.print(f"[cyan]→ Tool (inline):[/cyan] [bold]{tool_name}[/bold]")
                    result = self.registry.dispatch(tool_name, args)
                    result_str = json.dumps(result) if not isinstance(result, str) else result

                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tool_name}:\n{result_str}\n\nContinue."
                    })
                    continue

            # Final response
            if finish_reason in ("stop", "end_turn") or not tool_calls:
                final_response = content
                self.context.add_message("assistant", content)
                break

        # Store to memory
        self.memory.store(user_message, final_response)

        return final_response
