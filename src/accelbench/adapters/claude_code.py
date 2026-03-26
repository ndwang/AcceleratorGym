"""Claude Code adapter — launches bench_server as an instrumented MCP server.

Claude CLI connects to the bench_server subprocess, which handles task setup,
budget enforcement, and trace recording. After Claude CLI exits, the adapter
reads back the trace file for the runner to use in replay-based verification.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ClaudeCodeAdapter:
    """Adapter that runs Claude Code CLI against a bench_server.

    The bench_server subprocess creates its own Machine, runs task setup
    with a deterministic seed, wraps it in InstrumentedMachine, and serves
    the 5 accelerator-gym tools. After Claude CLI disconnects, the server
    dumps its trace to a file which this adapter reads back.

    The runner replays the agent's set_variables/reset calls from the trace
    onto its own Machine for verification.
    """

    def __init__(
        self,
        config_path: str,
        claude_cmd: str = "claude",
        model: str | None = None,
        timeout: int = 300,
    ) -> None:
        self._config_path = str(Path(config_path).resolve())
        self._claude_cmd = claude_cmd
        self._model = model
        self._timeout = timeout

        # Set per-task by the harness before each run()
        self._task_id: str = ""
        self._seed: int = 0
        self._budget: int = 0

        # Populated after run() completes
        self._last_trace: dict[str, Any] | None = None

    @property
    def model(self) -> str:
        return self._model or ""

    @property
    def last_trace(self) -> dict[str, Any] | None:
        return self._last_trace

    def set_task_context(self, task_id: str, seed: int, budget: int) -> None:
        """Set task-specific parameters before calling run()."""
        self._task_id = task_id
        self._seed = seed
        self._budget = budget

    def run(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        call_tool: Callable[[str, dict[str, Any]], str],
    ) -> str:
        if not self._task_id:
            raise RuntimeError("set_task_context() must be called before run()")

        self._last_trace = None

        # Create temp file for trace output
        trace_fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="accelbench_trace_"
        )
        trace_path = trace_fd.name
        trace_fd.close()

        # Write MCP config pointing to bench_server
        mcp_config = {
            "mcpServers": {
                "accelerator-gym": {
                    "command": sys.executable,
                    "args": [
                        "-m", "accelbench.bench_server",
                        "--config", self._config_path,
                        "--task-id", self._task_id,
                        "--seed", str(self._seed),
                        "--budget", str(self._budget),
                        "--trace-file", trace_path,
                    ],
                }
            }
        }

        mcp_config_fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="accelbench_mcp_"
        )
        json.dump(mcp_config, mcp_config_fd)
        mcp_config_path = mcp_config_fd.name
        mcp_config_fd.close()

        # Run from a temp directory so Claude Code can't read the repo
        sandbox_dir = tempfile.mkdtemp(prefix="accelbench_sandbox_")

        try:
            cmd = [
                self._claude_cmd, "--print",
                "--mcp-config", mcp_config_path,
                "--allowedTools", "mcp__accelerator-gym__*",
            ]
            if self._model:
                cmd.extend(["--model", self._model])

            logger.info(f"Launching Claude Code: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=sandbox_dir,
            )

            # Claude Code --print may write to stdout or stderr depending
            # on version/environment.  Prefer stdout, fall back to stderr.
            response = result.stdout or result.stderr or ""

            if result.returncode != 0:
                logger.error(f"Claude Code failed (rc={result.returncode})")
                logger.error(f"  stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
                logger.error(f"  stderr: {result.stderr[:500] if result.stderr else '(empty)'}")

            # Read trace file written by bench_server on exit
            try:
                with open(trace_path) as f:
                    self._last_trace = json.load(f)
                logger.info(
                    f"Trace loaded: {self._last_trace['call_count']} tool calls"
                )
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to read trace file: {e}")

            return response or f"Error: Claude Code exited with code {result.returncode}"

        finally:
            Path(mcp_config_path).unlink(missing_ok=True)
            Path(trace_path).unlink(missing_ok=True)
            shutil.rmtree(sandbox_dir, ignore_errors=True)
