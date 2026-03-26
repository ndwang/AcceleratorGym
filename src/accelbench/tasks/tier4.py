"""Tier 4: Complex — multi-procedure compositions (budget: 80–200)."""

from __future__ import annotations

from typing import Any

from accelbench.prompts import TASK_PROMPTS
from accelbench.types import TaskDef, Env
from ._helpers import query_variables, close


# --- 4.1 Full ORM Measurement ---


def _setup_4_1(env: Env) -> dict[str, Any]:
    return {"kick": 1e-4}


def _verify_4_1(result: dict, env: Env, setup_data: dict) -> bool:
    orm = result.get("orm")
    if not isinstance(orm, list):
        return False

    kick = setup_data["kick"]
    hkickers = query_variables(env, "hkicker", "kick")
    bpms = query_variables(env, "monitor", "orbit.x")
    n_corr, n_bpm = len(hkickers), len(bpms)

    # Check dimensions
    if len(orm) != n_corr:
        return False
    for row in orm:
        if not isinstance(row, list) or len(row) != n_bpm:
            return False

    # Spot-check first row against ground truth
    bpm_vars = [r["variable"] for r in bpms]
    corr = hkickers[0]
    initial = env.machine.get_many(bpm_vars)
    env.machine.set(corr["variable"], kick)
    kicked = env.machine.get_many(bpm_vars)
    env.machine.set(corr["variable"], env.get_design(corr["variable"]))

    expected_row = [(kicked[b] - initial[b]) / kick for b in bpm_vars]
    for j in range(n_bpm):
        if not close(float(orm[0][j]), expected_row[j], rtol=0.2, atol=1e-3):
            return False

    return True


# --- 4.2 Local Closed Bump ---


def _setup_4_2(env: Env) -> dict[str, Any]:
    rng = env.rng
    bpms = query_variables(env, "monitor", "orbit.x")
    pick = bpms[int(rng.integers(len(bpms)))]

    # Record initial orbit (clean lattice — should be near zero)
    initial = {r["variable"]: env.machine.get(r["variable"]) for r in bpms}

    return {
        "bpm": pick["device_id"],
        "target_mm": 3.0,
        "closure_mm": 0.2,
        "_bpm_var": pick["variable"],
        "_initial_orbit": initial,
    }


def _verify_4_2(result: dict, env: Env, setup_data: dict) -> bool:
    bpm_var = setup_data["_bpm_var"]
    initial = setup_data["_initial_orbit"]
    target_m = setup_data["target_mm"] * 1e-3
    closure_m = setup_data["closure_mm"] * 1e-3

    bpms = query_variables(env, "monitor", "orbit.x")

    # Check target BPM: displacement should match target
    change = env.machine.get(bpm_var) - initial[bpm_var]
    if abs(change - target_m) > closure_m:
        return False

    # Check closure: all other BPMs within tolerance
    for r in bpms:
        if r["variable"] == bpm_var:
            continue
        change = abs(env.machine.get(r["variable"]) - initial[r["variable"]])
        if change > closure_m:
            return False

    return True


# --- Task list ---

TIER4_TASKS = [
    TaskDef(
        id="4.1",
        name="Full ORM Measurement",
        tier=4,
        budget=80,
        prompt_template=TASK_PROMPTS["4.1"],
        setup=_setup_4_1,
        verify=_verify_4_1,
    ),
    TaskDef(
        id="4.2",
        name="Local Closed Bump",
        tier=4,
        budget=200,
        prompt_template=TASK_PROMPTS["4.2"],
        setup=_setup_4_2,
        verify=_verify_4_2,
    ),
]
