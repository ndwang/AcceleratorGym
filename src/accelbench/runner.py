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

logger = logging.getLogger(__name__)


def _replay_trace(machine: Machine, trace: list[dict[str, Any]]) -> None:
    """Replay set_variables and reset calls from a trace onto a machine.

    This reconstructs the post-agent machine state so that verification
    functions can read the same values the agent left behind.
    """
    for entry in trace:
        tool = entry["tool"]
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
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(adapter.run, prompt, TOOL_SCHEMAS, call_tool)
            response = future.result(timeout=timeout)
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
        trace_dicts = [tc.to_dict() for tc in instrumented.trace]

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

    return TaskResult(
        task_id=task.id,
        passed=passed,
        tool_calls=tool_calls,
        budget=task.budget,
        wall_time=elapsed,
        extracted_answer=answer,
        setup_data=setup_data,
        prompt=prompt,
        response=response or "",
        trace=trace_dicts,
        **meta,
    )
