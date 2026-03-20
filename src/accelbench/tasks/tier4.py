"""Tier 4: Complex tasks (budget: 50–200). Multi-procedure compositions."""

from __future__ import annotations

from typing import Any

import numpy as np

from accelbench.tasks.helpers import (
    compute_orm,
    get_bpm_devices,
    get_bpm_vars,
    get_corrector_vars,
    get_quad_vars,
)
from accelbench.types import Env, TaskDef


# --- 4.1: Full ORM Measurement ---

def _setup_4_1(env: Env) -> dict[str, Any]:
    return {}


def _verify_4_1(result: dict, env: Env, setup: dict) -> bool:
    n_bpms = env.machine.catalog.query(
        "SELECT COUNT(*) as n FROM devices WHERE device_type='monitor'"
    )[0]["n"]
    n_hcors = env.machine.catalog.query(
        "SELECT COUNT(*) as n FROM devices WHERE device_type='hkicker'"
    )[0]["n"]
    shape_ok = result["rows"] == n_bpms and result["cols"] == n_hcors
    agent_orm = np.array(result["matrix"])
    truth_orm = compute_orm(env, plane="x", kick=1e-4)
    norm_err = np.linalg.norm(agent_orm - truth_orm, "fro") / np.linalg.norm(truth_orm, "fro")
    return shape_ok and norm_err < 0.05


# --- 4.2: Global Orbit Correction ---

def _setup_4_2(env: Env) -> dict[str, Any]:
    # Distort orbit with multiple corrector kicks
    cor_vars = get_corrector_vars(env, "h")
    n_kicks = min(3, len(cor_vars))
    indices = env.rng.choice(len(cor_vars), size=n_kicks, replace=False)
    for idx in indices:
        kick = float(env.rng.uniform(0.5e-3, 2e-3)) * env.rng.choice([-1, 1])
        env.machine.set(cor_vars[idx], kick)

    # Record initial RMS for verification
    bpm_vars = get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    initial_rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    return {"initial_rms": float(initial_rms)}


def _verify_4_2(result: dict, env: Env, setup: dict) -> bool:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    final_rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    return final_rms < setup["initial_rms"] * 0.2


# --- 4.3: Orbit Correction with Dead Corrector ---

def _setup_4_3(env: Env) -> dict[str, Any]:
    cor_vars = get_corrector_vars(env, "h")

    # Pick one corrector to "lock" by setting its value to 0 and noting it
    stuck_idx = int(env.rng.integers(0, len(cor_vars)))
    stuck_var = cor_vars[stuck_idx]
    stuck_name = stuck_var.split(":")[0]

    # Distort orbit with other correctors
    available = [v for i, v in enumerate(cor_vars) if i != stuck_idx]
    n_kicks = min(3, len(available))
    indices = env.rng.choice(len(available), size=n_kicks, replace=False)
    for idx in indices:
        kick = float(env.rng.uniform(0.5e-3, 2e-3)) * env.rng.choice([-1, 1])
        env.machine.set(available[idx], kick)

    return {
        "stuck_corrector_var": stuck_var,
        "stuck_corrector_name": stuck_name,
    }


def _verify_4_3(result: dict, env: Env, setup: dict) -> bool:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    stuck_ok = abs(env.machine.get(setup["stuck_corrector_var"])) < 1e-12
    return rms < 1.0e-3 and stuck_ok


# --- 4.4: Combined Orbit and Tune Correction ---

def _setup_4_4(env: Env) -> dict[str, Any]:
    # Record initial vertical tune
    initial_tune_b = env.machine.get("params:tune.b")

    # Distort orbit
    cor_vars = get_corrector_vars(env, "h")
    n_kicks = min(3, len(cor_vars))
    indices = env.rng.choice(len(cor_vars), size=n_kicks, replace=False)
    for idx in indices:
        kick = float(env.rng.uniform(0.5e-3, 2e-3)) * env.rng.choice([-1, 1])
        env.machine.set(cor_vars[idx], kick)

    # Detune horizontal tune
    quad_vars = get_quad_vars(env)
    for qv in quad_vars:
        val = env.machine.get(qv)
        if val > 0:
            env.machine.set(qv, val * 0.97)
            break

    return {"initial_tune_b": float(initial_tune_b)}


def _verify_4_4(result: dict, env: Env, setup: dict) -> bool:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    tune_a = env.machine.get("params:tune.a")
    tune_b = env.machine.get("params:tune.b")
    return (
        rms < 0.5e-3
        and abs(tune_a - 4.82) < 0.01
        and abs(tune_b - setup["initial_tune_b"]) < 0.01
    )


# --- 4.5: Three-Corrector Closed Bump ---

def _setup_4_5(env: Env) -> dict[str, Any]:
    bpm_vars = get_bpm_vars(env, "orbit.x")
    initial_orbits = env.machine.get_many(bpm_vars)
    return {"initial_orbits": initial_orbits}


def _verify_4_5(result: dict, env: Env, setup: dict) -> bool:
    bpm_rows = get_bpm_devices(env)
    target = min(bpm_rows, key=lambda r: abs(r["s_position"] - 30))
    target_var = f"{target['device_id']}:orbit.x"
    initial = setup["initial_orbits"]

    bump_ok = abs(
        env.machine.get(target_var) - initial[target_var] - 3e-3
    ) < 0.3e-3

    others_ok = all(
        abs(env.machine.get(f"{r['device_id']}:orbit.x") - initial[f"{r['device_id']}:orbit.x"]) < 0.2e-3
        for r in bpm_rows
        if r["device_id"] != target["device_id"]
    )
    return bump_ok and others_ok


# --- 4.6: End-to-End Diagnosis and Correction ---

def _setup_4_6(env: Env) -> dict[str, Any]:
    design_tune = env.machine.get("params:tune.a")

    # Perturb a quadrupole
    quad_vars = get_quad_vars(env)
    focusing = [qv for qv in quad_vars if env.machine.get(qv) > 0]
    var = focusing[env.rng.integers(0, len(focusing))]
    name = var.split(":")[0]
    original = env.machine.get(var)
    env.machine.set(var, original * 1.08)

    # Also distort orbit
    cor_vars = get_corrector_vars(env, "h")
    kick_var = cor_vars[env.rng.integers(0, len(cor_vars))]
    kick = float(env.rng.uniform(0.5e-3, 1.5e-3)) * env.rng.choice([-1, 1])
    env.machine.set(kick_var, kick)

    return {
        "perturbed_quad": name,
        "perturbed_var": var,
        "original_k1": float(original),
        "design_tune": float(design_tune),
    }


def _verify_4_6(result: dict, env: Env, setup: dict) -> bool:
    tune = env.machine.get("params:tune.a")
    bpm_vars = get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    tune_ok = abs(tune - setup["design_tune"]) < 0.02
    orbit_ok = rms < 1.0e-3
    diagnosis_ok = result.get("faulty_element") == setup["perturbed_quad"]
    return tune_ok and orbit_ok and diagnosis_ok


# --- Aggregation ---

TIER4_TASKS: list[TaskDef] = [
    TaskDef(
        id="4.1",
        name="Full ORM Measurement",
        tier=4,
        prompt_template=(
            "Measure the full horizontal orbit response matrix. "
            "For each corrector, apply a small kick, record the orbit change at all BPMs, "
            "then restore the corrector.\n"
            'Answer: {{"rows": <n_bpms>, "cols": <n_correctors>, '
            '"matrix": [[<row for BPM 1>], [<row for BPM 2>], ...]}}'
        ),
        budget=80,
        abilities=["discovery", "measurement", "analysis"],
        setup=_setup_4_1,
        verify=_verify_4_1,
    ),
    TaskDef(
        id="4.2",
        name="Global Orbit Correction",
        tier=4,
        prompt_template=(
            "Correct the horizontal orbit to minimize the RMS across all BPMs.\n"
            'Answer: {{"status": "done"}}'
        ),
        budget=100,
        abilities=["measurement", "physics", "optimization"],
        setup=_setup_4_2,
        verify=_verify_4_2,
    ),
    TaskDef(
        id="4.3",
        name="Orbit Correction with Dead Corrector",
        tier=4,
        prompt_template=(
            "Correct the horizontal orbit to below 1.0 mm RMS. "
            "One of the correctors appears to be stuck \u2014 find it and work around it.\n"
            'Answer: {{"status": "done"}}'
        ),
        budget=120,
        abilities=["diagnosis", "optimization"],
        setup=_setup_4_3,
        verify=_verify_4_3,
    ),
    TaskDef(
        id="4.4",
        name="Combined Orbit and Tune Correction",
        tier=4,
        prompt_template=(
            "Get the horizontal orbit below 0.5 mm RMS and the horizontal tune to "
            "4.82 \u00b1 0.01 without changing the vertical tune by more than 0.01.\n"
            'Answer: {{"status": "done"}}'
        ),
        budget=150,
        abilities=["physics", "optimization"],
        setup=_setup_4_4,
        verify=_verify_4_4,
    ),
    TaskDef(
        id="4.5",
        name="Three-Corrector Closed Bump",
        tier=4,
        prompt_template=(
            "Create a closed orbit bump producing +3 mm horizontal displacement at the "
            "BPM closest to s = 30 m, closing within 0.2 mm at all other BPMs.\n"
            'Answer: {{"status": "done"}}'
        ),
        budget=100,
        abilities=["physics", "measurement", "optimization"],
        setup=_setup_4_5,
        verify=_verify_4_5,
    ),
    TaskDef(
        id="4.6",
        name="End-to-End Diagnosis and Correction",
        tier=4,
        prompt_template=(
            "The beam orbit is distorted and the tune is wrong. "
            "Diagnose the problem and fix it.\n"
            'Answer: {{"faulty_element": "<device_id>"}}'
        ),
        budget=200,
        abilities=["diagnosis", "physics", "optimization"],
        setup=_setup_4_6,
        verify=_verify_4_6,
    ),
]
