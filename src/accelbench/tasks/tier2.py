"""Tier 2: Procedural tasks (budget: 5–15). Known multi-step procedures."""

from __future__ import annotations

from typing import Any

import numpy as np

from accelbench.tasks.helpers import get_bpm_vars, get_corrector_vars
from accelbench.types import Env, TaskDef


# --- 2.1: Orbit RMS ---

def _setup_2_1(env: Env) -> dict[str, Any]:
    return {}


def _verify_2_1(result: dict, env: Env, setup: dict) -> bool:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    truth = np.sqrt(np.mean([v**2 for v in values.values()]))
    if truth < 1e-15:
        return abs(result["rms"]) < 1e-12
    return abs(result["rms"] - truth) / truth < 0.01


# --- 2.2: Single Corrector Orbit Response ---

def _setup_2_2(env: Env) -> dict[str, Any]:
    # Pick a specific corrector — use first hkicker
    cor_vars = get_corrector_vars(env, "h")
    var = cor_vars[env.rng.integers(0, len(cor_vars))]
    name = var.split(":")[0]
    return {"corrector": name, "corrector_var": var}


def _verify_2_2(result: dict, env: Env, setup: dict) -> bool:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    cor_var = setup["corrector_var"]

    initial = env.machine.get_many(bpm_vars)
    initial_kick = env.machine.get(cor_var)
    env.machine.set(cor_var, initial_kick + 0.1e-3)
    kicked = env.machine.get_many(bpm_vars)
    env.machine.set(cor_var, initial_kick)

    truth = np.array([kicked[v] - initial[v] for v in bpm_vars])
    agent = np.array(result["response"])
    if np.linalg.norm(truth) < 1e-15:
        return np.linalg.norm(agent) < 1e-12
    return np.linalg.norm(agent - truth) / np.linalg.norm(truth) < 0.01


# --- 2.3: Maximum Beta Function ---

def _setup_2_3(env: Env) -> dict[str, Any]:
    return {}


def _verify_2_3(result: dict, env: Env, setup: dict) -> bool:
    rows = env.machine.catalog.query(
        "SELECT d.device_id, a.variable FROM attributes a "
        "JOIN devices d ON a.device_id = d.device_id "
        "WHERE a.attribute_name='beta.a' AND d.device_type != 'global'"
    )
    values = {r["device_id"]: env.machine.get(r["variable"]) for r in rows}
    max_dev = max(values, key=values.get)
    truth_val = values[max_dev]
    return (
        abs(result["value"] - truth_val) / truth_val < 0.02
        and result["element"] == max_dev
    )


# --- 2.4: Phase Advance Between Two Points ---

def _setup_2_4(env: Env) -> dict[str, Any]:
    bpms = env.machine.catalog.query(
        "SELECT device_id FROM devices WHERE device_type='monitor' ORDER BY s_position"
    )
    # Pick two BPMs separated by some distance
    n = len(bpms)
    i = env.rng.integers(0, n // 2)
    j = env.rng.integers(n // 2, n)
    return {"bpm1": bpms[i]["device_id"], "bpm2": bpms[j]["device_id"]}


def _verify_2_4(result: dict, env: Env, setup: dict) -> bool:
    bpm1, bpm2 = setup["bpm1"], setup["bpm2"]
    phi_1 = env.machine.get(f"{bpm1}:phi.a")
    phi_2 = env.machine.get(f"{bpm2}:phi.a")
    truth = phi_2 - phi_1
    return abs(result["value"] - truth) < 0.01


# --- 2.5: Total Bending Angle ---

def _setup_2_5(env: Env) -> dict[str, Any]:
    return {}


def _verify_2_5(result: dict, env: Env, setup: dict) -> bool:
    rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type='sbend' AND a.attribute_name='ANGLE'"
    )
    values = env.machine.get_many([r["variable"] for r in rows])
    truth = sum(abs(v) for v in values.values())
    angle_ok = abs(result["total_angle"] - truth) / truth < 0.001
    consistent_ok = result["consistent"] == (abs(truth - 2 * np.pi) < 0.01)
    return angle_ok and consistent_ok


# --- 2.6: Dispersion Measurement ---

def _setup_2_6(env: Env) -> dict[str, Any]:
    return {}


def _verify_2_6(result: dict, env: Env, setup: dict) -> bool:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    # Get dispersion via eta.x at same elements
    truth = np.array([
        env.machine.get(v.replace("orbit.x", "eta.x")) for v in bpm_vars
    ])
    agent = np.array(result["dispersion"])
    if np.linalg.norm(truth) < 1e-15:
        return np.linalg.norm(agent) < 1e-12
    return np.linalg.norm(agent - truth) / np.linalg.norm(truth) < 0.05


# --- 2.7: Chromaticity Measurement ---

def _setup_2_7(env: Env) -> dict[str, Any]:
    return {}


def _verify_2_7(result: dict, env: Env, setup: dict) -> bool:
    truth = env.machine.get("params:chrom.a")
    if abs(truth) < 1e-10:
        return abs(result["chromaticity"]) < 1e-6
    return abs(result["chromaticity"] - truth) / abs(truth) < 0.05


# --- Aggregation ---

TIER2_TASKS: list[TaskDef] = [
    TaskDef(
        id="2.1",
        name="Orbit RMS",
        tier=2,
        prompt_template=(
            "Measure the horizontal orbit at all BPMs and report the RMS.\n"
            'Answer: {{"rms": <number in meters>}}'
        ),
        budget=5,
        abilities=["io", "analysis"],
        setup=_setup_2_1,
        verify=_verify_2_1,
    ),
    TaskDef(
        id="2.2",
        name="Single Corrector Orbit Response",
        tier=2,
        prompt_template=(
            "Measure the horizontal orbit response to corrector {corrector}. "
            "Apply a kick of 0.1 mrad, record the orbit change at all BPMs, "
            "then restore the corrector.\n"
            'Answer: {{"response": [<orbit change at each BPM in meters, ordered by s-position>]}}'
        ),
        budget=10,
        abilities=["measurement"],
        setup=_setup_2_2,
        verify=_verify_2_2,
    ),
    TaskDef(
        id="2.3",
        name="Maximum Beta Function",
        tier=2,
        prompt_template=(
            "What is the maximum horizontal beta function in the ring and "
            "where does it occur?\n"
            'Answer: {{"element": "<device_id>", "value": <number in meters>}}'
        ),
        budget=8,
        abilities=["io", "analysis"],
        setup=_setup_2_3,
        verify=_verify_2_3,
    ),
    TaskDef(
        id="2.4",
        name="Phase Advance Between Two Points",
        tier=2,
        prompt_template=(
            "What is the horizontal phase advance between {bpm1} and {bpm2}?\n"
            'Answer: {{"value": <number in radians>}}'
        ),
        budget=5,
        abilities=["analysis"],
        setup=_setup_2_4,
        verify=_verify_2_4,
    ),
    TaskDef(
        id="2.5",
        name="Total Bending Angle",
        tier=2,
        prompt_template=(
            "What is the total bending angle from all dipoles? "
            "Is it consistent with a full ring (2\u03c0)?\n"
            'Answer: {{"total_angle": <number in radians>, "consistent": <true/false>}}'
        ),
        budget=10,
        abilities=["discovery", "analysis"],
        setup=_setup_2_5,
        verify=_verify_2_5,
    ),
    TaskDef(
        id="2.6",
        name="Dispersion Measurement",
        tier=2,
        prompt_template=(
            "Measure the horizontal dispersion at all BPMs by changing the beam energy "
            "by \u00b10.01% and observing the orbit shift.\n"
            'Answer: {{"dispersion": [<dispersion at each BPM in meters, ordered by s-position>]}}'
        ),
        budget=12,
        abilities=["measurement"],
        setup=_setup_2_6,
        verify=_verify_2_6,
    ),
    TaskDef(
        id="2.7",
        name="Chromaticity Measurement",
        tier=2,
        prompt_template=(
            "Measure the horizontal chromaticity by shifting the beam energy "
            "by \u00b10.01% and observing the tune change.\n"
            'Answer: {{"chromaticity": <number>}}'
        ),
        budget=10,
        abilities=["measurement"],
        setup=_setup_2_7,
        verify=_verify_2_7,
    ),
]
