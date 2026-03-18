"""Shared helper utilities for benchmark tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from accelbench.types import Env


def get_bpm_vars(env: Env, attr: str) -> list[str]:
    """Query all monitor variables for a given attribute (e.g. 'orbit.x')."""
    rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        f"WHERE d.device_type='monitor' AND a.attribute_name='{attr}' ORDER BY d.s_position"
    )
    return [r["variable"] for r in rows]


def get_bpm_devices(env: Env) -> list[dict[str, Any]]:
    """Get all BPM device info (device_id, s_position)."""
    return env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices WHERE device_type='monitor' ORDER BY s_position"
    )


def get_corrector_vars(env: Env, plane: str = "h") -> list[str]:
    """Get all corrector kick variables for a given plane."""
    dtype = "hkicker" if plane == "h" else "vkicker"
    rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        f"WHERE d.device_type='{dtype}' AND a.attribute_name='kick' ORDER BY d.s_position"
    )
    return [r["variable"] for r in rows]


def get_quad_vars(env: Env) -> list[str]:
    """Get all quadrupole K1 variables."""
    rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type='quadrupole' AND a.attribute_name='K1' ORDER BY d.s_position"
    )
    return [r["variable"] for r in rows]


def random_element(env: Env, device_type: str, attribute: str) -> str:
    """Pick a random device of a given type and return its variable name."""
    rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        f"WHERE d.device_type='{device_type}' AND a.attribute_name='{attribute}'"
    )
    idx = env.rng.integers(0, len(rows))
    return rows[idx]["variable"]


def compute_orm(env: Env, plane: str = "x", kick: float = 1e-4) -> np.ndarray:
    """Compute the orbit response matrix from the current machine state.

    Returns an (n_bpms, n_correctors) array.
    """
    orbit_attr = f"orbit.{plane}"
    cor_type = "h" if plane == "x" else "v"

    bpm_vars = get_bpm_vars(env, orbit_attr)
    cor_vars = get_corrector_vars(env, cor_type)

    initial_orbit = env.machine.get_many(bpm_vars)
    orm = np.zeros((len(bpm_vars), len(cor_vars)))

    for j, cvar in enumerate(cor_vars):
        initial_kick = env.machine.get(cvar)
        env.machine.set(cvar, initial_kick + kick)
        kicked_orbit = env.machine.get_many(bpm_vars)
        env.machine.set(cvar, initial_kick)
        for i, bvar in enumerate(bpm_vars):
            orm[i, j] = (kicked_orbit[bvar] - initial_orbit[bvar]) / kick

    return orm
