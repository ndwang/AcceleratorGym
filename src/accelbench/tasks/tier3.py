"""Tier 3: Adaptive tasks (budget: 10–40). Next action depends on observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from accelbench.tasks.helpers import (
    get_bpm_devices,
    get_bpm_vars,
    get_corrector_vars,
    get_quad_vars,
)
from accelbench.types import Env, TaskDef


# --- 3.1: Find Most Effective Corrector ---

def _setup_3_1(env: Env) -> dict[str, Any]:
    return {}


def _verify_3_1(result: dict, env: Env, setup: dict) -> bool:
    # Find closest BPM to s=45
    bpms = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices WHERE device_type='monitor' "
        "ORDER BY ABS(s_position - 45) LIMIT 1"
    )
    target_bpm = bpms[0]["device_id"]
    bpm_var = f"{target_bpm}:orbit.x"

    hcors = get_corrector_vars(env, "h")
    best_cor, best_resp = None, 0.0
    for cvar in hcors:
        name = cvar.split(":")[0]
        initial_orbit = env.machine.get(bpm_var)
        initial_kick = env.machine.get(cvar)
        env.machine.set(cvar, initial_kick + 1e-4)
        new_orbit = env.machine.get(bpm_var)
        env.machine.set(cvar, initial_kick)
        resp = abs(new_orbit - initial_orbit)
        if resp > best_resp:
            best_resp = resp
            best_cor = name
    return result["corrector"] == best_cor


# --- 3.2: Find a Nonzero Corrector ---

def _setup_3_2(env: Env) -> dict[str, Any]:
    cor_vars = get_corrector_vars(env, "h")
    var = cor_vars[env.rng.integers(0, len(cor_vars))]
    name = var.split(":")[0]
    kick = float(env.rng.uniform(0.5e-3, 2e-3)) * env.rng.choice([-1, 1])
    env.machine.set(var, kick)
    return {"corrector_name": name, "corrector_var": var, "kick_value": kick}


def _verify_3_2(result: dict, env: Env, setup: dict) -> bool:
    return (
        result["corrector"] == setup["corrector_name"]
        and abs(result["kick"] - setup["kick_value"]) / abs(setup["kick_value"]) < 0.05
    )


# --- 3.3: Gradient Error Detection ---

def _setup_3_3(env: Env) -> dict[str, Any]:
    quad_vars = get_quad_vars(env)
    # Find focusing quads (K1 > 0)
    focusing = []
    for qv in quad_vars:
        val = env.machine.get(qv)
        if val > 0:
            focusing.append(qv)
    var = focusing[env.rng.integers(0, len(focusing))]
    name = var.split(":")[0]
    original = env.machine.get(var)
    env.machine.set(var, original * 1.05)
    return {"perturbed_quad": name, "perturbed_var": var, "original_k1": original}


def _verify_3_3(result: dict, env: Env, setup: dict) -> bool:
    return result["element"] == setup["perturbed_quad"]


# --- 3.4: Beta Function Outlier ---

def _setup_3_4(env: Env) -> dict[str, Any]:
    # Perturb a quad near a BPM to create a beta outlier
    quad_vars = get_quad_vars(env)
    var = quad_vars[env.rng.integers(0, len(quad_vars))]
    original = env.machine.get(var)
    env.machine.set(var, original * 1.3)
    return {"perturbed_var": var, "original_k1": original}


def _verify_3_4(result: dict, env: Env, setup: dict) -> bool:
    # Check if the reported BPM actually has beta > 1.5 * average
    bpm_devices = get_bpm_devices(env)
    bpm_ids = [d["device_id"] for d in bpm_devices]
    rows = env.machine.catalog.query(
        "SELECT d.device_id, a.variable FROM attributes a "
        "JOIN devices d ON a.device_id = d.device_id "
        "WHERE a.attribute_name='beta.a' AND d.device_id IN "
        f"({','.join(repr(b) for b in bpm_ids)})"
    )
    betas = {r["device_id"]: env.machine.get(r["variable"]) for r in rows}
    avg = np.mean(list(betas.values()))
    outliers = {name for name, b in betas.items() if b > 1.5 * avg}
    return result["bpm"] in outliers


# --- 3.5: Tune Adjustment ---

def _setup_3_5(env: Env) -> dict[str, Any]:
    # Detune by perturbing focusing quads
    quad_vars = get_quad_vars(env)
    for qv in quad_vars:
        val = env.machine.get(qv)
        if val > 0:
            env.machine.set(qv, val * 0.98)
            break
    return {}


def _verify_3_5(result: dict, env: Env, setup: dict) -> bool:
    tune = env.machine.get("ring:tune.a")
    return abs(tune - 4.82) < 0.005


# --- 3.6: Chromaticity Adjustment ---

def _setup_3_6(env: Env) -> dict[str, Any]:
    # No specific perturbation needed — natural chromaticity is usually far from +1.0
    return {}


def _verify_3_6(result: dict, env: Env, setup: dict) -> bool:
    chrom = env.machine.get("ring:chrom.a")
    return abs(chrom - 1.0) < 0.3


# --- 3.7: Single-BPM Orbit Correction ---

def _setup_3_7(env: Env) -> dict[str, Any]:
    # Distort orbit with a random corrector kick
    cor_vars = get_corrector_vars(env, "h")
    var = cor_vars[env.rng.integers(0, len(cor_vars))]
    kick = float(env.rng.uniform(1e-3, 3e-3)) * env.rng.choice([-1, 1])
    env.machine.set(var, kick)
    return {"distortion_var": var, "distortion_kick": kick}


def _verify_3_7(result: dict, env: Env, setup: dict) -> bool:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    worst_var = max(values, key=lambda v: abs(values[v]))
    return abs(env.machine.get(worst_var)) < 0.1e-3


# --- 3.8: Local Orbit Bump ---

def _setup_3_8(env: Env) -> dict[str, Any]:
    # Record initial orbits for comparison
    bpm_vars = get_bpm_vars(env, "orbit.x")
    initial_orbits = env.machine.get_many(bpm_vars)
    return {"initial_orbits": initial_orbits}


def _verify_3_8(result: dict, env: Env, setup: dict) -> bool:
    bpm_rows = get_bpm_devices(env)
    target_bpm = min(bpm_rows, key=lambda r: abs(r["s_position"] - 30))
    target_var = f"{target_bpm['device_id']}:orbit.x"
    initial = setup["initial_orbits"]

    bump_ok = abs(
        env.machine.get(target_var) - initial[target_var] - 2e-3
    ) < 0.3e-3

    far_bpms_ok = all(
        abs(env.machine.get(f"{r['device_id']}:orbit.x") - initial[f"{r['device_id']}:orbit.x"]) < 0.1e-3
        for r in bpm_rows
        if abs(r["s_position"] - 30) > 5
    )
    return bump_ok and far_bpms_ok


# --- Aggregation ---

TIER3_TASKS: list[TaskDef] = [
    TaskDef(
        id="3.1",
        name="Find Most Effective Corrector",
        tier=3,
        prompt_template="Which horizontal corrector has the largest effect on the orbit near s = 45 m?",
        budget=30,
        abilities=["discovery", "analysis"],
        setup=_setup_3_1,
        verify=_verify_3_1,
    ),
    TaskDef(
        id="3.2",
        name="Find a Nonzero Corrector",
        tier=3,
        prompt_template=(
            "Something is applying a horizontal kick somewhere in the ring. Find the source."
        ),
        budget=25,
        abilities=["discovery", "io"],
        setup=_setup_3_2,
        verify=_verify_3_2,
    ),
    TaskDef(
        id="3.3",
        name="Gradient Error Detection",
        tier=3,
        prompt_template="One of the focusing quadrupoles has an anomalous gradient. Which one?",
        budget=40,
        abilities=["discovery", "diagnosis"],
        setup=_setup_3_3,
        verify=_verify_3_3,
    ),
    TaskDef(
        id="3.4",
        name="Beta Function Outlier",
        tier=3,
        prompt_template="Find any BPM where the horizontal beta function is more than 50% above average.",
        budget=15,
        abilities=["analysis", "diagnosis"],
        setup=_setup_3_4,
        verify=_verify_3_4,
    ),
    TaskDef(
        id="3.5",
        name="Tune Adjustment",
        tier=3,
        prompt_template="Adjust the horizontal tune to 4.82 \u00b1 0.005.",
        budget=30,
        abilities=["physics", "optimization"],
        setup=_setup_3_5,
        verify=_verify_3_5,
    ),
    TaskDef(
        id="3.6",
        name="Chromaticity Adjustment",
        tier=3,
        prompt_template="Adjust the horizontal chromaticity to +1.0 \u00b1 0.3.",
        budget=30,
        abilities=["physics", "optimization"],
        setup=_setup_3_6,
        verify=_verify_3_6,
    ),
    TaskDef(
        id="3.7",
        name="Single-BPM Orbit Correction",
        tier=3,
        prompt_template=(
            "Correct the horizontal orbit at the BPM with the worst reading to within 0.1 mm."
        ),
        budget=25,
        abilities=["discovery", "optimization"],
        setup=_setup_3_7,
        verify=_verify_3_7,
    ),
    TaskDef(
        id="3.8",
        name="Local Orbit Bump",
        tier=3,
        prompt_template=(
            "Create a +2 mm horizontal orbit bump near s = 30 m that doesn't affect "
            "BPMs more than 5 m away by more than 0.1 mm."
        ),
        budget=40,
        abilities=["measurement", "physics"],
        setup=_setup_3_8,
        verify=_verify_3_8,
    ),
]
