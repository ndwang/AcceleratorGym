"""Core dataclasses for the AccelBench harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from accelerator_gym.core.machine import Machine


@dataclass
class Env:
    """Environment passed to task setup and verify functions."""

    machine: Machine
    rng: np.random.Generator

    def get_design(self, name: str) -> float:
        """Read the design (unperturbed) value of a variable."""
        return self.machine.get_design(name)


@dataclass
class TaskDef:
    """Definition of a single benchmark task."""

    id: str
    name: str
    tier: int
    prompt_template: str
    budget: int
    abilities: list[str]
    setup: Callable[[Env], dict[str, Any]]
    verify: Callable[[dict[str, Any], Env, dict[str, Any]], bool]
    timeout: int = 300


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""

    task_id: str
    passed: bool
    tool_calls: int
    budget: int
    wall_time: float
    extracted_answer: dict[str, Any] | None
    error: str | None = None
    setup_data: dict[str, Any] | None = None
    prompt: str = ""
    response: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """Efficiency score: fraction of budget remaining."""
        return max(0.0, 1.0 - self.tool_calls / self.budget)


@dataclass
class RunRecord:
    """Record of a complete benchmark run."""

    seed: int
    results: list[TaskResult] = field(default_factory=list)
    config_path: str = ""
    adapter_name: str = ""
    model: str = ""
