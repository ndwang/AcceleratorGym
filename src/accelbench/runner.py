"""Single task lifecycle runner."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Callable

from accelbench.extract import extract_json_answer
from accelbench.instrument import InstrumentedMachine, TOOL_SCHEMAS, make_call_tool
from accelbench.prompts import ANSWER_INSTRUCTION
from accelbench.types import Env, TaskDef, TaskResult

from accelerator_gym.core.machine import Machine

# Failure reason constants
REASON_SETUP_ERROR = "setup_error"
REASON_TIMEOUT = "timeout"
REASON_CRASH = "crash"
REASON_NO_ANSWER = "no_answer"
REASON_BUDGET_EXCEEDED = "budget_exceeded"
REASON_WRONG_ANSWER = "wrong_answer"

logger = logging.getLogger(__name__)


def _replay_trace(machine: Machine, trace: list[dict[str, Any]]) -> None:
    """Replay set_variables and reset calls from a trace onto a machine.

    This reconstructs the post-agent machine state so that verification
    functions can read the same values the agent left behind.
    """
    for entry in trace:
        tool = entry.get("tool")
        if not tool:
            continue  # skip reasoning entries
        if tool == "set_variables":
            args = entry["arguments"]
            # Skip replaying calls that failed (budget exceeded or validation error)
            result = entry.get("result", "")
            if isinstance(result, str) and result.startswith("Error"):
                continue
            machine.set_many(args["values"])
        elif tool == "reset":
            result = entry.get("result", "")
            if isinstance(result, str) and result.startswith("Error"):
                continue
            machine.reset()


def _merge_reasoning_trace(
    tool_trace: list[dict[str, Any]],
    reasoning_trace: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Interleave reasoning entries into the tool call trace.

    Each reasoning entry has a ``tool_call_index`` indicating which tool call
    it preceded.  This function inserts each reasoning entry just before the
    tool call at that index, preserving correct ordering even when one
    assistant turn triggers multiple parallel tool calls.
    """
    if not reasoning_trace:
        return tool_trace

    # Build a map: tool_call_index -> list of reasoning entries before it
    before: dict[int, list[dict[str, Any]]] = {}
    trailing: list[dict[str, Any]] = []
    for r in reasoning_trace:
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


def run_task(
    task: TaskDef,
    machine: Machine,
    adapter: Any,
    rng: Any,
    timeout: int = 600,
) -> TaskResult:
    """Run a single benchmark task through the full lifecycle.

    1. Wrap machine in InstrumentedMachine
    2. Create Env, call task.setup
    3. Format prompt, build call_tool closure
    4. Call adapter.run with timeout
    5. Extract JSON answer
    6. If adapter provides trace (e.g. ClaudeCodeAdapter), replay mutations
    7. Call task.verify
    8. Return TaskResult
    """
    instrumented = InstrumentedMachine(machine, task.budget)
    env = Env(machine=machine, rng=rng)

    # Setup
    try:
        setup_data = task.setup(env)
    except Exception as e:
        logger.exception(f"Task {task.id} setup failed")
        return TaskResult(
            task_id=task.id,
            passed=False,
            tool_calls=0,
            budget=task.budget,
            wall_time=0.0,
            extracted_answer=None,
            error=f"Setup failed: {e}",
            failure_reason=REASON_SETUP_ERROR,
        )

    # Format prompt
    prompt = task.prompt_template.format(**setup_data) + "\n\n" + ANSWER_INSTRUCTION

    # Build tool interface
    call_tool = make_call_tool(instrumented)

    def _adapter_meta() -> dict[str, Any]:
        """Extract model and usage from adapter if available."""
        model = getattr(adapter, "model", "") or ""
        usage = getattr(adapter, "last_usage", None) or {}
        return {"model": str(model), "usage": dict(usage)}

    # Run agent
    start = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(adapter.run, prompt, TOOL_SCHEMAS, call_tool)
    # Shut down the pool without waiting so it won't block if the
    # thread is still running after a timeout.
    pool.shutdown(wait=False)
    try:
        response = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        elapsed = time.monotonic() - start
        logger.error(f"Task {task.id} timed out after {timeout}s")

        # Kill the subprocess and recover the trace
        stop = getattr(adapter, "stop", None)
        if stop:
            stop()

        meta = _adapter_meta()
        adapter_trace = getattr(adapter, "last_trace", None)
        if adapter_trace is not None:
            trace_dicts = adapter_trace["trace"]
            tool_calls = adapter_trace["call_count"]
        else:
            trace_dicts = [tc.to_dict() for tc in instrumented.trace]
            tool_calls = instrumented.call_count
        return TaskResult(
            task_id=task.id,
            passed=False,
            tool_calls=tool_calls,
            budget=task.budget,
            wall_time=elapsed,
            extracted_answer=None,
            error=f"Agent timed out after {timeout}s",
            failure_reason=REASON_TIMEOUT,
            setup_data=setup_data,
            prompt=prompt,
            response="",
            trace=trace_dicts,
            **meta,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        meta = _adapter_meta()
        logger.exception(f"Task {task.id} agent run failed")
        return TaskResult(
            task_id=task.id,
            passed=False,
            tool_calls=instrumented.call_count,
            budget=task.budget,
            wall_time=elapsed,
            extracted_answer=None,
            error=f"Agent error: {e}",
            failure_reason=REASON_CRASH,
            setup_data=setup_data,
            prompt=prompt,
            response="",
            trace=[],
            **meta,
        )
    elapsed = time.monotonic() - start
    meta = _adapter_meta()

    # Resolve trace source: adapter-provided (ClaudeCode) or local instrumented
    adapter_trace = getattr(adapter, "last_trace", None)
    if adapter_trace is not None:
        tool_calls = adapter_trace["call_count"]
        trace_dicts = adapter_trace["trace"]
        # Replay agent mutations onto the harness machine for verification
        _replay_trace(machine, trace_dicts)
    else:
        tool_calls = instrumented.call_count
        trace_dicts = _merge_reasoning_trace(
            [tc.to_dict() for tc in instrumented.trace],
            getattr(adapter, "reasoning_trace", None),
        )

    # Extract answer
    answer = extract_json_answer(response) if response else None

    # Verify
    if answer is None:
        return TaskResult(
            task_id=task.id,
            passed=False,
            tool_calls=tool_calls,
            budget=task.budget,
            wall_time=elapsed,
            extracted_answer=None,
            error="No JSON answer extracted from response",
            failure_reason=REASON_NO_ANSWER,
            setup_data=setup_data,
            prompt=prompt,
            response=response or "",
            trace=trace_dicts,
            **meta,
        )

    try:
        passed = task.verify(answer, env, setup_data)
    except Exception as e:
        logger.exception(f"Task {task.id} verification failed")
        passed = False

    # Determine failure reason
    failure_reason: str | None = None
    if not passed:
        if tool_calls >= task.budget:
            failure_reason = REASON_BUDGET_EXCEEDED
        else:
            failure_reason = REASON_WRONG_ANSWER

    return TaskResult(
        task_id=task.id,
        passed=passed,
        tool_calls=tool_calls,
        budget=task.budget,
        wall_time=elapsed,
        extracted_answer=answer,
        failure_reason=failure_reason,
        setup_data=setup_data,
        prompt=prompt,
        response=response or "",
        trace=trace_dicts,
        **meta,
    )
