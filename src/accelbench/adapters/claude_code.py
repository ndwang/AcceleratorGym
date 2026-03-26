"""Claude Code adapter — launches bench_server as an instrumented MCP server.

Claude CLI connects to the bench_server subprocess, which handles task setup,
budget enforcement, and trace recording. After Claude CLI exits, the adapter
reads back the trace file for the runner to use in replay-based verification.

The adapter uses ``--output-format stream-json --verbose`` so that Claude's
intermediate reasoning (assistant text between tool calls) is captured and
merged into the trace alongside the tool call entries from bench_server.
"""

from __future__ import annotations

import json
import logging
import os
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
    ) -> None:
        self._config_path = str(Path(config_path).resolve())
        self._claude_cmd = claude_cmd
        self._model = model

        # Set per-task by the harness before each run()
        self._task_id: str = ""
        self._seed: int = 0
        self._budget: int = 0

        # Populated after run() completes
        self._last_trace: dict[str, Any] | None = None

        # Set during run() so the runner can kill on timeout
        self._proc: subprocess.Popen | None = None
        self._trace_path: str | None = None

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
                "--output-format", "stream-json",
                "--verbose",
                "--mcp-config", mcp_config_path,
                "--allowedTools", "mcp__accelerator-gym__*",
            ]
            if self._model:
                cmd.extend(["--model", self._model])

            logger.info(f"Launching Claude Code: {' '.join(cmd)}")

            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=sandbox_dir,
            )
            self._trace_path = trace_path

            # No timeout here — the runner handles the timeout externally.
            stdout, stderr = self._proc.communicate(input=prompt)

            if stderr:
                logger.info(f"Claude Code stderr:\n{stderr[:2000]}")

            # Parse stream-json output for response and reasoning
            response, reasoning = _parse_stream_json(stdout or "")

            if not response:
                logger.warning("No result event in stream-json output")
                response = ""

            if self._proc.returncode != 0:
                logger.error(f"Claude Code failed (rc={self._proc.returncode})")
                logger.error(f"  stdout lines: {len((stdout or '').splitlines())}")
                logger.error(f"  stderr: {stderr[:500] if stderr else '(empty)'}")

            self._load_trace(trace_path)

            # Merge reasoning into the tool call trace
            if self._last_trace and reasoning:
                self._last_trace["trace"] = _merge_reasoning(
                    self._last_trace["trace"], reasoning
                )

            return response or f"Error: Claude Code exited with code {self._proc.returncode}"

        finally:
            self._proc = None
            self._trace_path = None
            Path(mcp_config_path).unlink(missing_ok=True)
            Path(trace_path).unlink(missing_ok=True)
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    def _load_trace(self, trace_path: str) -> None:
        """Read the bench_server trace file."""
        trace_size = 0
        try:
            trace_size = os.path.getsize(trace_path)
        except OSError:
            pass
        logger.info(f"Trace file size: {trace_size} bytes")

        if trace_size == 0:
            logger.error(
                "Trace file is empty — bench_server likely did not "
                "start or was killed before writing trace"
            )
            self._last_trace = None
        else:
            try:
                with open(trace_path) as f:
                    self._last_trace = json.load(f)
                logger.info(
                    f"Trace loaded: {self._last_trace['call_count']} tool calls"
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse trace file: {e}")

    def stop(self) -> None:
        """Kill the subprocess and load whatever trace was written.

        Called by the runner on timeout.
        """
        proc = self._proc
        trace_path = self._trace_path
        if proc is None:
            return

        logger.info(f"Stopping Claude Code for task {self._task_id}")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown failed, sending SIGKILL")
            proc.kill()
            proc.wait()

        if trace_path:
            self._load_trace(trace_path)


def _parse_stream_json(raw: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse stream-json NDJSON output into (response, reasoning_entries).

    Returns:
        response: The final text response from the ``result`` event.
        reasoning_entries: List of ``{"role": "assistant", "content": str,
            "tool_call_index": int}`` dicts — one per assistant turn that
            contained text alongside tool calls.
    """
    reasoning: list[dict[str, Any]] = []
    response = ""
    tool_call_index = 0

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")

        if etype == "assistant":
            message = event.get("message", {})
            content_blocks = message.get("content", [])

            text_parts = []
            num_tool_uses = 0
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    num_tool_uses += 1

            text = "\n".join(text_parts)
            if text:
                reasoning.append({
                    "role": "assistant",
                    "content": text,
                    "tool_call_index": tool_call_index,
                })
            tool_call_index += num_tool_uses

        elif etype == "result":
            response = event.get("result", "")

    return response, reasoning


def _merge_reasoning(
    tool_trace: list[dict[str, Any]],
    reasoning: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert reasoning entries into the tool call trace at the right positions.

    Each reasoning entry has a ``tool_call_index`` indicating which tool call
    it preceded.  Reasoning entries are inserted just before that index.
    """
    before: dict[int, list[dict[str, Any]]] = {}
    trailing: list[dict[str, Any]] = []
    for r in reasoning:
        idx = r.get("tool_call_index")
        entry = {"role": "assistant", "content": r["content"]}
        if idx is not None and idx < len(tool_trace):
            before.setdefault(idx, []).append(entry)
        else:
            trailing.append(entry)

    merged: list[dict[str, Any]] = []
    for i, tool_entry in enumerate(tool_trace):
        for r_entry in before.get(i, []):
            merged.append(r_entry)
        merged.append(tool_entry)
    merged.extend(trailing)
    return merged
