"""Shared utilities for benchmark task setup and verification."""

from __future__ import annotations

import math
from typing import Any

from accelbench.types import Env


def query_variables(
    env: Env, device_type: str, attribute_name: str
) -> list[dict[str, Any]]:
    """Return [{device_id, variable, s_position}, ...] for a device type + attribute."""
    return env.machine.catalog.query(
        "SELECT d.device_id, a.variable, d.s_position "
        "FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type = ? AND a.attribute_name = ?",
        (device_type, attribute_name),
    )


def global_variable(env: Env, attribute_name: str) -> str:
    """Return the variable name for a global lattice parameter."""
    rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a "
        "JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.system = 'global' AND a.attribute_name = ?",
        (attribute_name,),
    )
    return rows[0]["variable"]


def close(
    actual: float, expected: float, rtol: float = 0.05, atol: float = 1e-6
) -> bool:
    """Check if two values are close (generous tolerance for benchmarks)."""
    return math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)
