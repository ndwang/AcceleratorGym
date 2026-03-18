"""Claude SDK adapter (callback mode) using the Anthropic API."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an accelerator operations agent. You have access to tools for monitoring \
and controlling a particle accelerator. Use the tools to complete the task.

IMPORTANT: When you have your final answer, output it as a JSON object inside a \
```json fenced code block. Include all requested values with the exact keys \
specified in the task prompt."""


class ClaudeSDKAdapter:
    """Reference adapter using the Anthropic Python SDK (callback mode)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def run(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        call_tool: Callable[[str, dict[str, Any]], str],
    ) -> str:
        import anthropic

        client = anthropic.Anthropic()

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        while True:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            # Collect text and tool_use blocks
            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            if response.stop_reason != "tool_use":
                # Agent is done
                return "\n".join(text_parts)

            # Execute tool calls and feed results back
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                logger.debug(f"Tool call: {tu['name']}({tu['input']})")
                result = call_tool(tu["name"], tu["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
