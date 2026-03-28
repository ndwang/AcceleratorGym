"""Shared utilities for benchmark task setup and verification."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from accelbench.types import Env

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Random lattice errors (persistent across reset via Tao startup file)
# ---------------------------------------------------------------------------


@dataclass
class ErrorSpec:
    """Specification for one type of random lattice error.

    Attributes:
        attribute: Bmad element attribute to perturb (e.g. ``"x_offset"``).
        rms: RMS of the normal distribution in natural units.
        elements: Explicit list of element names to perturb.
    """

    attribute: str
    rms: float
    elements: list[str] = field(default_factory=list)


def apply_random_errors(
    env: Env,
    specs: list[ErrorSpec],
) -> dict[str, dict[str, float]]:
    """Apply normally-distributed random errors and persist via Tao startup file.

    The caller decides *which* elements to perturb (by querying names,
    filtering by s-range, etc.) and passes them in each :class:`ErrorSpec`.
    This function draws ``N(0, rms)`` for every element/attribute pair,
    writes a Tao command file, then reconnects with that file as the
    ``startup_file`` so errors survive ``reset`` (``reinit tao``).

    Examples::

        quads = [r["device_id"] for r in query_variables(env, "quadrupole", "K1")]
        apply_random_errors(env, [
            ErrorSpec("x_offset", 0.5e-3, quads),   # 0.5 mm RMS
            ErrorSpec("y_offset", 0.5e-3, quads),
        ])

        # Sector-A sextupoles only
        sexts_a = [
            r["device_id"] for r in env.machine.catalog.query(
                "SELECT device_id FROM devices "
                "WHERE device_type='sextupole' AND s_position < 25"
            )
        ]
        apply_random_errors(env, [ErrorSpec("y_offset", 3e-3, sexts_a)])

    Returns:
        ``{element_name: {attribute: value, ...}, ...}``
    """
    rng = env.rng

    # Generate errors
    errors: dict[str, dict[str, float]] = {}
    for spec in specs:
        for name in spec.elements:
            val = float(rng.normal(0, spec.rms))
            errors.setdefault(name, {})[spec.attribute] = val

    # Write Tao command file
    init_dir: Path = env.machine._backend._init_dir
    startup_path = init_dir / "bench_errors.tao"
    with open(startup_path, "w") as f:
        f.write("set global lattice_calc_on = F\n")
        for name, attrs in errors.items():
            for attr, val in attrs.items():
                f.write(f"set element {name} {attr} = {val}\n")
        f.write("set global lattice_calc_on = T\n")

    # Reconnect with startup_file so errors persist across reset (reinit tao)
    backend = env.machine._backend
    backend.disconnect()
    backend._settings["startup_file"] = str(startup_path)
    backend.connect()

    logger.info(
        "Applied random errors to %d elements, startup_file=%s",
        len(errors),
        startup_path,
    )
    return errors
