"""Single task lifecycle runner."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from accelbench.extract import extract_json_answer
from accelbench.instrument import InstrumentedMachine, TOOL_SCHEMAS, make_call_tool
from accelbench.types import Env, TaskDef, TaskResult

from accelerator_gym.core.machine import Machine

logger = logging.getLogger(__name__)

_ANSWER_INSTRUCTION = (
    "\n\nIMPORTANT: When you have your final answer, output it as a JSON object "
    "inside a ```json fenced code block."
)


def run_task(
    task: TaskDef,
    machine: Machine,
    adapter: Any,
    rng: Any,
) -> TaskResult:
    """Run a single benchmark task through the full lifecycle.

    1. Wrap machine in InstrumentedMachine
    2. Create Env, call task.setup
    3. Format prompt, build call_tool closure
    4. Call adapter.run with timeout
    5. Extract JSON answer
    6. Call task.verify
    7. Return TaskResult
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
    prompt = task.prompt_template.format(**setup_data) + _ANSWER_INSTRUCTION

    # Build tool interface
    call_tool = make_call_tool(instrumented)

    # Run agent
    start = time.monotonic()
    try:
        response = adapter.run(prompt, TOOL_SCHEMAS, call_tool)
    except Exception as e:
        elapsed = time.monotonic() - start
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
        )
    elapsed = time.monotonic() - start

    # Extract answer
    answer = extract_json_answer(response) if response else None

    # Verify
    if answer is None:
        return TaskResult(
            task_id=task.id,
            passed=False,
            tool_calls=instrumented.call_count,
            budget=task.budget,
            wall_time=elapsed,
            extracted_answer=None,
            error="No JSON answer extracted from response",
            setup_data=setup_data,
        )

    try:
        passed = task.verify(answer, env, setup_data)
    except Exception as e:
        logger.exception(f"Task {task.id} verification failed")
        passed = False

    return TaskResult(
        task_id=task.id,
        passed=passed,
        tool_calls=instrumented.call_count,
        budget=task.budget,
        wall_time=elapsed,
        extracted_answer=answer,
        setup_data=setup_data,
    )
