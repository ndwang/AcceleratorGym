"""Claude Code adapter (MCP server mode).

Launches an instrumented MCP server as a subprocess and connects
Claude Code to it via --mcp-config. Tool call counting is handled
by the instrumented server.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ClaudeCodeAdapter:
    """Adapter that runs Claude Code CLI against an MCP server.

    This adapter starts an instrumented accelerator-gym MCP server,
    writes a temporary MCP config, then invokes the `claude` CLI
    with the prompt piped via stdin.

    Note: Tool call counting is done via the InstrumentedMachine in
    the runner, not by this adapter. The MCP server subprocess uses
    the same Machine instance's state.
    """

    def __init__(
        self,
        config_path: str,
        claude_cmd: str = "claude",
        model: str | None = None,
    ) -> None:
        self._config_path = config_path
        self._claude_cmd = claude_cmd
        self._model = model

    def run(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        call_tool: Callable[[str, dict[str, Any]], str],
    ) -> str:
        # Write a temporary MCP config pointing to the accelerator-gym server
        mcp_config = {
            "mcpServers": {
                "accelerator-gym": {
                    "command": sys.executable,
                    "args": ["-m", "accelerator_gym.server", "--config", self._config_path],
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="accelbench_mcp_"
        ) as f:
            json.dump(mcp_config, f)
            mcp_config_path = f.name

        try:
            cmd = [self._claude_cmd, "--print", "--mcp-config", mcp_config_path]
            if self._model:
                cmd.extend(["--model", self._model])

            logger.info(f"Launching Claude Code: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"Claude Code stderr: {result.stderr}")
                return result.stdout or f"Error: Claude Code exited with code {result.returncode}"

            return result.stdout

        finally:
            Path(mcp_config_path).unlink(missing_ok=True)
