"""Orchestrator for full benchmark runs."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from accelerator_gym.core.machine import Machine
from accelbench.runner import run_task
from accelbench.tasks import ALL_TASKS, TASKS_BY_ID
from accelbench.types import RunRecord, TaskDef, TaskResult

logger = logging.getLogger(__name__)


def run_benchmark(
    config_path: str,
    adapter: Any,
    seed: int = 42,
    task_ids: list[str] | None = None,
    tier: int | None = None,
) -> RunRecord:
    """Run the full benchmark (or a subset) and return results.

    Args:
        config_path: Path to accelerator-gym YAML config.
        adapter: An AgentAdapter instance.
        seed: Random seed for reproducibility.
        task_ids: If given, only run these task IDs.
        tier: If given, only run tasks from this tier.

    Returns:
        A RunRecord with all results.
    """
    # Select tasks
    tasks = _select_tasks(task_ids, tier)

    record = RunRecord(
        seed=seed,
        config_path=config_path,
        adapter_name=type(adapter).__name__,
    )

    for task in tasks:
        logger.info(f"Running task {task.id}: {task.name}")

        # Create fresh machine for each task
        machine = Machine.from_config(config_path)
        rng = np.random.default_rng(seed + hash(task.id))

        try:
            result = run_task(task, machine, adapter, rng)
            record.results.append(result)

            status = "PASS" if result.passed else "FAIL"
            logger.info(
                f"Task {task.id}: {status} "
                f"(tools: {result.tool_calls}/{task.budget}, "
                f"time: {result.wall_time:.1f}s)"
            )
            if result.error:
                logger.warning(f"Task {task.id} error: {result.error}")
        except Exception as e:
            logger.exception(f"Task {task.id} crashed")
            record.results.append(TaskResult(
                task_id=task.id,
                passed=False,
                tool_calls=0,
                budget=task.budget,
                wall_time=0.0,
                extracted_answer=None,
                error=f"Crash: {e}",
            ))
        finally:
            machine.close()

    return record


def _select_tasks(
    task_ids: list[str] | None, tier: int | None
) -> list[TaskDef]:
    """Filter tasks by ID list or tier."""
    if task_ids:
        tasks = []
        for tid in task_ids:
            if tid not in TASKS_BY_ID:
                raise ValueError(f"Unknown task ID: {tid}")
            tasks.append(TASKS_BY_ID[tid])
        return tasks

    if tier is not None:
        return [t for t in ALL_TASKS if t.tier == tier]

    return list(ALL_TASKS)
