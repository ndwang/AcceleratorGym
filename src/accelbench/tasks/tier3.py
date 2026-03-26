"""Tier 3: Adaptive — discovery and iterative optimization (budget: 15–40)."""

from __future__ import annotations

from typing import Any

from accelbench.prompts import TASK_PROMPTS
from accelbench.types import TaskDef, Env
from ._helpers import query_variables, global_variable, close


# --- 3.1 Find the Tune ---


def _setup_3_1(env: Env) -> dict[str, Any]:
    return {}


def _verify_3_1(result: dict, env: Env, setup_data: dict) -> bool:
    # Accept several possible key names
    tune = (
        result.get("tune")
        or result.get("tune_a")
        or result.get("horizontal_tune")
    )
    if tune is None:
        return False
    expected = env.machine.get(global_variable(env, "tune.a"))
    return close(float(tune), expected, rtol=0.01)


# --- 3.2 List Elements and Settings ---


def _setup_3_2(env: Env) -> dict[str, Any]:
    rng = env.rng
    hkickers = query_variables(env, "hkicker", "kick")
    if hkickers:
        kicks = {r["variable"]: float(rng.normal(0, 2e-4)) for r in hkickers}
        env.machine.set_many(kicks)
    return {}


def _verify_3_2(result: dict, env: Env, setup_data: dict) -> bool:
    correctors = result.get("correctors")
    if not isinstance(correctors, dict):
        return False

    hkickers = query_variables(env, "hkicker", "kick")
    if len(correctors) != len(hkickers):
        return False

    # Build lookup accepting both device_id and variable name
    expected: dict[str, float] = {}
    for r in hkickers:
        val = env.machine.get(r["variable"])
        expected[r["device_id"]] = val
        expected[r["variable"]] = val

    matched = sum(
        1
        for k, v in correctors.items()
        if k in expected and close(float(v), expected[k], rtol=0.05)
    )
    return matched >= len(hkickers)


# --- 3.3 Nearest Quad to BPM ---


def _setup_3_3(env: Env) -> dict[str, Any]:
    rng = env.rng
    bpms = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices WHERE device_type = 'monitor'"
    )
    pick = bpms[int(rng.integers(len(bpms)))]
    return {"bpm": pick["device_id"], "_bpm_s": pick["s_position"]}


def _verify_3_3(result: dict, env: Env, setup_data: dict) -> bool:
    quad = result.get("quadrupole") or result.get("element") or result.get("quad")
    if quad is None:
        return False
    quads = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices "
        "WHERE device_type = 'quadrupole'"
    )
    nearest = min(quads, key=lambda q: abs(q["s_position"] - setup_data["_bpm_s"]))
    return quad == nearest["device_id"]


# --- 3.4 Elements in a Range ---


def _setup_3_4(env: Env) -> dict[str, Any]:
    rng = env.rng
    bpms = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices "
        "WHERE device_type = 'monitor' ORDER BY s_position"
    )
    if len(bpms) < 2:
        raise ValueError("Need at least 2 BPMs for range query")
    idx1, idx2 = sorted(rng.choice(len(bpms), size=2, replace=False))
    return {
        "bpm1": bpms[int(idx1)]["device_id"],
        "bpm2": bpms[int(idx2)]["device_id"],
        "_s1": bpms[int(idx1)]["s_position"],
        "_s2": bpms[int(idx2)]["s_position"],
    }


def _verify_3_4(result: dict, env: Env, setup_data: dict) -> bool:
    magnets = result.get("magnets")
    if not isinstance(magnets, list):
        return False
    lo = min(setup_data["_s1"], setup_data["_s2"])
    hi = max(setup_data["_s1"], setup_data["_s2"])
    expected = env.machine.catalog.query(
        "SELECT device_id FROM devices "
        "WHERE system = 'magnets' AND s_position > ? AND s_position < ?",
        (lo, hi),
    )
    return sorted(magnets) == sorted(r["device_id"] for r in expected)


# --- 3.5 Tune Adjustment ---


def _setup_3_5(env: Env) -> dict[str, Any]:
    rng = env.rng
    quads = query_variables(env, "quadrupole", "K1")

    qf_list, qd_list = [], []
    for r in quads:
        k1 = env.machine.get(r["variable"])
        if k1 > 0:
            qf_list.append(r)
        elif k1 < 0:
            qd_list.append(r)
    if not qf_list or not qd_list:
        raise ValueError("Need both focusing and defocusing quadrupoles")

    qf = qf_list[int(rng.integers(len(qf_list)))]
    qd = qd_list[int(rng.integers(len(qd_list)))]

    tune_var = global_variable(env, "tune.a")
    current_tune = env.machine.get(tune_var)
    target = round(current_tune + float(rng.uniform(-0.03, 0.03)), 4)

    return {
        "target": target,
        "tol": 0.005,
        "qf_var": qf["variable"],
        "qd_var": qd["variable"],
    }


def _verify_3_5(result: dict, env: Env, setup_data: dict) -> bool:
    tune_var = global_variable(env, "tune.a")
    current = env.machine.get(tune_var)
    return abs(current - setup_data["target"]) <= setup_data["tol"]


# --- 3.6 Chromaticity Adjustment ---


def _setup_3_6(env: Env) -> dict[str, Any]:
    sextupoles = query_variables(env, "sextupole", "K2")
    if not sextupoles:
        raise ValueError("No sextupoles found in lattice")
    return {"target": 1.0, "tol": 0.3}


def _verify_3_6(result: dict, env: Env, setup_data: dict) -> bool:
    chrom_var = global_variable(env, "chrom.a")
    current = env.machine.get(chrom_var)
    return abs(current - setup_data["target"]) <= setup_data["tol"]


# --- 3.7 Local Orbit Correction ---


def _setup_3_7(env: Env) -> dict[str, Any]:
    rng = env.rng
    hkickers = query_variables(env, "hkicker", "kick")
    bpms = query_variables(env, "monitor", "orbit.x")

    # Apply corrector kicks to create orbit distortion
    if hkickers:
        kicks = {r["variable"]: float(rng.normal(0, 3e-4)) for r in hkickers}
        env.machine.set_many(kicks)

    # Pick BPM with orbit above threshold
    threshold_m = 1e-4  # 0.1 mm
    orbits = [(r, abs(env.machine.get(r["variable"]))) for r in bpms]
    large = [(r, o) for r, o in orbits if o > threshold_m * 2]

    if not large:
        # Increase kicks to create larger orbit
        if hkickers:
            kicks = {r["variable"]: float(rng.normal(0, 1e-3)) for r in hkickers}
            env.machine.set_many(kicks)
            orbits = [(r, abs(env.machine.get(r["variable"]))) for r in bpms]
            large = [(r, o) for r, o in orbits if o > threshold_m * 2]

    if large:
        bpm, _ = large[int(rng.integers(len(large)))]
    else:
        bpm, _ = max(orbits, key=lambda x: x[1])

    return {
        "bpm": bpm["device_id"],
        "threshold_mm": 0.1,
        "_bpm_var": bpm["variable"],
    }


def _verify_3_7(result: dict, env: Env, setup_data: dict) -> bool:
    orbit = abs(env.machine.get(setup_data["_bpm_var"]))
    return orbit < setup_data["threshold_mm"] * 1e-3


# --- Task list ---

TIER3_TASKS = [
    TaskDef(
        id="3.1",
        name="Find the Tune",
        tier=3,
        budget=15,
        prompt_template=TASK_PROMPTS["3.1"],
        setup=_setup_3_1,
        verify=_verify_3_1,
    ),
    TaskDef(
        id="3.2",
        name="List Elements and Settings",
        tier=3,
        budget=15,
        prompt_template=TASK_PROMPTS["3.2"],
        setup=_setup_3_2,
        verify=_verify_3_2,
    ),
    TaskDef(
        id="3.3",
        name="Nearest Quad to BPM",
        tier=3,
        budget=15,
        prompt_template=TASK_PROMPTS["3.3"],
        setup=_setup_3_3,
        verify=_verify_3_3,
    ),
    TaskDef(
        id="3.4",
        name="Elements in a Range",
        tier=3,
        budget=15,
        prompt_template=TASK_PROMPTS["3.4"],
        setup=_setup_3_4,
        verify=_verify_3_4,
    ),
    TaskDef(
        id="3.5",
        name="Tune Adjustment",
        tier=3,
        budget=40,
        prompt_template=TASK_PROMPTS["3.5"],
        setup=_setup_3_5,
        verify=_verify_3_5,
    ),
    TaskDef(
        id="3.6",
        name="Chromaticity Adjustment",
        tier=3,
        budget=40,
        prompt_template=TASK_PROMPTS["3.6"],
        setup=_setup_3_6,
        verify=_verify_3_6,
    ),
    TaskDef(
        id="3.7",
        name="Local Orbit Correction",
        tier=3,
        budget=25,
        prompt_template=TASK_PROMPTS["3.7"],
        setup=_setup_3_7,
        verify=_verify_3_7,
    ),
]
