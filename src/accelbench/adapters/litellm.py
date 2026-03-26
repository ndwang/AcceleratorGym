"""LiteLLM adapter — supports OpenAI, Gemini, Mistral, Llama, and 100+ providers."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from accelbench.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LiteLLMAdapter:
    """Adapter using LiteLLM for provider-agnostic model access.

    Supports any model that LiteLLM supports:
      - OpenAI:   model="gpt-4o"
      - Gemini:   model="gemini/gemini-2.5-pro"
      - Mistral:  model="mistral/mistral-large-latest"
      - Ollama:   model="ollama/llama3"
      - etc.

    See https://docs.litellm.ai/docs/providers for the full list.
    Set the appropriate API key env var for your provider
    (OPENAI_API_KEY, GEMINI_API_KEY, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_base = api_base
        self._extra_kwargs = kwargs
        # Reset per run
        self._last_usage: dict[str, int] = {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_usage(self) -> dict[str, int]:
        """Token usage from the most recent run() call."""
        return dict(self._last_usage)

    def run(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        call_tool: Callable[[str, dict[str, Any]], str],
    ) -> str:
        import litellm

        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        while True:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "max_tokens": self._max_tokens,
                **self._extra_kwargs,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = litellm.completion(**kwargs)

            # Accumulate token usage
            if response.usage:
                self._last_usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                self._last_usage["completion_tokens"] += response.usage.completion_tokens or 0
                self._last_usage["total_tokens"] += response.usage.total_tokens or 0

            choice = response.choices[0]
            message = choice.message

            # Append assistant message to history
            messages.append(message.model_dump(exclude_none=True))

            # Check for tool calls
            tool_calls = message.tool_calls
            if not tool_calls:
                # Agent is done — return text content
                return message.content or ""

            # Execute each tool call and feed results back
            budget_exceeded = False
            for tc in tool_calls:
                fn = tc.function
                try:
                    arguments = json.loads(fn.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                logger.debug(f"Tool call: {fn.name}({arguments})")
                result = call_tool(fn.name, arguments)

                if "budget exceeded" in result:
                    budget_exceeded = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            if budget_exceeded:
                # One final turn to let the model produce an answer
                kwargs_final: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": self._max_tokens,
                    **self._extra_kwargs,
                }
                if self._api_base:
                    kwargs_final["api_base"] = self._api_base

                response = litellm.completion(**kwargs_final)
                if response.usage:
                    self._last_usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                    self._last_usage["completion_tokens"] += response.usage.completion_tokens or 0
                    self._last_usage["total_tokens"] += response.usage.total_tokens or 0

                return response.choices[0].message.content or ""
