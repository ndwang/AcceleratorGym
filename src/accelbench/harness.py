"""Orchestrator for full benchmark runs."""

from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from accelerator_gym.core.machine import Machine
from accelbench.runner import run_task
from accelbench.tasks import ALL_TASKS, TASKS_BY_ID
from accelbench.types import RunRecord, TaskDef, TaskResult

logger = logging.getLogger(__name__)


def task_seed(seed: int, task_id: str) -> int:
    """Deterministic per-task seed derivation (stable across processes)."""
    h = hashlib.sha256(task_id.encode()).hexdigest()
    return seed + int(h, 16) % (2**31)


def _run_single_task(
    task: TaskDef,
    config_path: str,
    adapter: Any,
    seed: int,
    timeout: int,
    traces_dir: Path | None,
) -> TaskResult:
    """Run a single task end-to-end with its own machine and adapter copy."""
    logger.info(f"Running task {task.id}: {task.name}")

    machine = Machine.from_config(config_path)
    rng = np.random.default_rng(task_seed(seed, task.id))

    # Deep-copy adapter so each task has isolated mutable state
    task_adapter = copy.deepcopy(adapter)

    if hasattr(task_adapter, "set_task_context"):
        task_adapter.set_task_context(task.id, seed, task.budget)

    try:
        task_timeout = task.timeout if task.timeout is not None else timeout
        result = run_task(task, machine, task_adapter, rng, timeout=task_timeout)

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
        result = TaskResult(
            task_id=task.id,
            passed=False,
            tool_calls=0,
            budget=task.budget,
            wall_time=0.0,
            extracted_answer=None,
            error=f"Crash: {e}",
        )
    finally:
        machine.close()

    if traces_dir:
        _save_trajectory(result, task, traces_dir)

    return result


def run_benchmark(
    config_path: str,
    adapter: Any,
    seed: int = 42,
    task_ids: list[str] | None = None,
    tier: int | None = None,
    output_dir: str | None = None,
    timeout: int = 600,
    max_workers: int = 1,
) -> RunRecord:
    """Run the full benchmark (or a subset) and return results.

    Args:
        config_path: Path to accelerator-gym YAML config.
        adapter: An AgentAdapter instance.
        seed: Random seed for reproducibility.
        task_ids: If given, only run these task IDs.
        tier: If given, only run tasks from this tier.
        output_dir: If given, save per-task trajectory files here.
        timeout: Wall-clock timeout in seconds per task (default: 600).
        max_workers: Number of tasks to run in parallel (default: 1).

    Returns:
        A RunRecord with all results.
    """
    tasks = _select_tasks(task_ids, tier)

    traces_dir = None
    if output_dir:
        traces_dir = Path(output_dir) / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

    model = str(getattr(adapter, "model", "")) or ""
    record = RunRecord(
        seed=seed,
        config_path=config_path,
        adapter_name=type(adapter).__name__,
        model=model,
    )

    if max_workers <= 1:
        # Serial execution (original behavior)
        for task in tasks:
            result = _run_single_task(
                task, config_path, adapter, seed, timeout, traces_dir
            )
            record.results.append(result)
    else:
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _run_single_task,
                    task, config_path, adapter, seed, timeout, traces_dir,
                ): task
                for task in tasks
            }
            # Collect results in submission order
            task_to_future = {task.id: f for f, task in futures.items()}
            for task in tasks:
                future = task_to_future[task.id]
                result = future.result()
                record.results.append(result)

    return record


def _save_trajectory(
    result: TaskResult, task: TaskDef, traces_dir: Path
) -> None:
    """Save a per-task trajectory file with the full trace."""
    trajectory = {
        "task_id": result.task_id,
        "task_name": task.name,
        "tier": task.tier,
        "passed": result.passed,
        "tool_calls": result.tool_calls,
        "budget": result.budget,
        "efficiency": round(result.efficiency, 3),
        "wall_time": round(result.wall_time, 2),
        "error": result.error,
        "model": result.model,
        "usage": result.usage,
        "prompt": result.prompt,
        "response": result.response,
        "extracted_answer": result.extracted_answer,
        "setup_data": _safe_serialize(result.setup_data),
        "trace": result.trace,
    }
    path = traces_dir / f"task_{result.task_id.replace('.', '_')}.json"
    with open(path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    logger.debug(f"Trajectory saved: {path}")


def _safe_serialize(obj: Any) -> Any:
    """Convert an object to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
