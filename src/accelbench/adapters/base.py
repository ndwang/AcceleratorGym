"""Base adapter protocol for benchmark agents."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class AgentAdapter(Protocol):
    """Protocol that agent adapters must implement."""

    def run(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        call_tool: Callable[[str, dict[str, Any]], str],
    ) -> str:
        """Run the agent on a task.

        Args:
            prompt: The task prompt for the agent.
            tools: JSON schemas for the 5 MCP tools.
            call_tool: Callable to execute a tool — call_tool("get_variables", {"names": ["QF:K1"]}) -> str.

        Returns:
            The agent's final text response containing a JSON answer.
        """
        ...
