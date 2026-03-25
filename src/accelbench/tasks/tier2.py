"""Tier 2: Procedural — deterministic multi-step procedures (budget: 5–12)."""

from __future__ import annotations

import math
from typing import Any

from accelbench.types import TaskDef, Env
from ._helpers import query_variables, global_variable, close


# --- 2.1 Maximum Beta Function ---


def _setup_2_1(env: Env) -> dict[str, Any]:
    all_beta = env.machine.catalog.query(
        "SELECT d.device_id, a.variable "
        "FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE a.attribute_name = 'beta.a'"
    )
    if not all_beta:
        raise ValueError("No elements with beta.a found")
    variables = [r["variable"] for r in all_beta]
    return {"variables": ", ".join(f"`{v}`" for v in variables)}


def _verify_2_1(result: dict, env: Env, setup_data: dict) -> bool:
    max_beta = result.get("max_beta")
    element = result.get("element")
    if max_beta is None or element is None:
        return False
    all_beta = env.machine.catalog.query(
        "SELECT d.device_id, a.variable "
        "FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE a.attribute_name = 'beta.a'"
    )
    values = {r["device_id"]: env.machine.get(r["variable"]) for r in all_beta}
    best = max(values, key=values.get)
    return close(float(max_beta), values[best], rtol=0.05) and element == best


# --- 2.2 Orbit RMS ---


def _setup_2_2(env: Env) -> dict[str, Any]:
    rng = env.rng
    hkickers = query_variables(env, "hkicker", "kick")
    if hkickers:
        kicks = {r["variable"]: float(rng.normal(0, 1e-4)) for r in hkickers}
        env.machine.set_many(kicks)
    bpms = query_variables(env, "monitor", "orbit.x")
    if not bpms:
        raise ValueError("No BPMs found in lattice")
    bpm_vars = [r["variable"] for r in bpms]
    return {"bpm_variables": ", ".join(f"`{v}`" for v in bpm_vars)}


def _verify_2_2(result: dict, env: Env, setup_data: dict) -> bool:
    rms = result.get("rms")
    if rms is None:
        return False
    bpms = query_variables(env, "monitor", "orbit.x")
    vals = [env.machine.get(r["variable"]) for r in bpms]
    expected = math.sqrt(sum(v**2 for v in vals) / len(vals))
    return close(float(rms), expected, rtol=0.1)


# --- 2.3 Corrector Orbit Response ---


def _setup_2_3(env: Env) -> dict[str, Any]:
    rng = env.rng
    hkickers = query_variables(env, "hkicker", "kick")
    if not hkickers:
        raise ValueError("No horizontal correctors found in lattice")
    pick = hkickers[int(rng.integers(len(hkickers)))]
    bpms = query_variables(env, "monitor", "orbit.x")
    if not bpms:
        raise ValueError("No BPMs found in lattice")
    bpm_vars = [r["variable"] for r in bpms]
    return {
        "corrector_var": pick["variable"],
        "kick": 1e-4,
        "bpm_variables": ", ".join(f"`{v}`" for v in bpm_vars),
    }


def _verify_2_3(result: dict, env: Env, setup_data: dict) -> bool:
    changes = result.get("orbit_change")
    if not isinstance(changes, dict):
        return False

    corrector = setup_data["corrector_var"]
    kick = setup_data["kick"]
    bpms = query_variables(env, "monitor", "orbit.x")

    # Compute expected orbit change from design corrector state
    env.machine.set(corrector, env.get_design(corrector))
    initial = {r["variable"]: env.machine.get(r["variable"]) for r in bpms}
    env.machine.set(corrector, kick)
    kicked = {r["variable"]: env.machine.get(r["variable"]) for r in bpms}
    env.machine.set(corrector, env.get_design(corrector))

    # Accept both device_id and variable name as keys
    expected: dict[str, float] = {}
    for r in bpms:
        delta = kicked[r["variable"]] - initial[r["variable"]]
        expected[r["device_id"]] = delta
        expected[r["variable"]] = delta

    if len(changes) < len(bpms):
        return False
    matched = sum(
        1
        for k, v in changes.items()
        if k in expected and close(float(v), expected[k], rtol=0.1, atol=1e-8)
    )
    return matched >= len(bpms)


# --- 2.4 Quad Tune Shift ---


def _setup_2_4(env: Env) -> dict[str, Any]:
    rng = env.rng
    quads = query_variables(env, "quadrupole", "K1")
    pick = quads[int(rng.integers(len(quads)))]
    tune_var = global_variable(env, "tune.a")
    return {"quad_var": pick["variable"], "tune_var": tune_var}


def _verify_2_4(result: dict, env: Env, setup_data: dict) -> bool:
    tune_change = result.get("tune_change")
    if tune_change is None:
        return False

    quad_var = setup_data["quad_var"]
    tune_var = global_variable(env, "tune.a")

    # Compute expected tune shift from current state
    k1 = env.machine.get(quad_var)
    tune_before = env.machine.get(tune_var)
    env.machine.set(quad_var, k1 * 1.01)
    tune_after = env.machine.get(tune_var)
    env.machine.set(quad_var, k1)

    expected = tune_after - tune_before
    return close(float(tune_change), expected, rtol=0.15, atol=1e-6)


# --- 2.5 Increase Magnet Strength ---


def _setup_2_5(env: Env) -> dict[str, Any]:
    rng = env.rng
    quads = env.machine.catalog.query(
        "SELECT a.variable, a.lower_limit, a.upper_limit "
        "FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type = 'quadrupole' AND a.attribute_name = 'K1' "
        "AND a.writable = 1"
    )
    pick = quads[int(rng.integers(len(quads)))]
    pct = int(rng.integers(1, 10))
    current = env.machine.get(pick["variable"])
    new_val = current * (1 + pct / 100)
    # Ensure within limits
    lo, hi = pick["lower_limit"], pick["upper_limit"]
    if lo is not None and new_val < lo:
        pct = -pct
        new_val = current * (1 + pct / 100)
    if hi is not None and new_val > hi:
        pct = -abs(pct)
        new_val = current * (1 + pct / 100)
    return {"variable": pick["variable"], "pct": pct, "_expected": new_val}


def _verify_2_5(result: dict, env: Env, setup_data: dict) -> bool:
    return close(
        env.machine.get(setup_data["variable"]), setup_data["_expected"], rtol=0.01
    )


# --- 2.6 Zero Correctors and Read Orbit ---


def _setup_2_6(env: Env) -> dict[str, Any]:
    rng = env.rng
    hkickers = query_variables(env, "hkicker", "kick")
    if not hkickers:
        raise ValueError("No horizontal correctors found in lattice")
    kicks = {r["variable"]: float(rng.normal(0, 1e-4)) for r in hkickers}
    env.machine.set_many(kicks)
    corr_vars = [r["variable"] for r in hkickers]
    bpms = query_variables(env, "monitor", "orbit.x")
    if not bpms:
        raise ValueError("No BPMs found in lattice")
    bpm_vars = [r["variable"] for r in bpms]
    return {
        "corrector_variables": ", ".join(f"`{v}`" for v in corr_vars),
        "bpm_variables": ", ".join(f"`{v}`" for v in bpm_vars),
    }


def _verify_2_6(result: dict, env: Env, setup_data: dict) -> bool:
    # All correctors must be zero
    hkickers = query_variables(env, "hkicker", "kick")
    for r in hkickers:
        if abs(env.machine.get(r["variable"])) > 1e-10:
            return False
    # Orbit dict must have correct number of entries
    orbit = result.get("orbit")
    if not isinstance(orbit, dict):
        return False
    bpms = query_variables(env, "monitor", "orbit.x")
    return len(orbit) >= len(bpms)


# --- 2.7 Corrector Scan ---


def _setup_2_7(env: Env) -> dict[str, Any]:
    rng = env.rng
    hkickers = query_variables(env, "hkicker", "kick")
    bpms = query_variables(env, "monitor", "orbit.x")
    if not hkickers or not bpms:
        raise ValueError("Need horizontal correctors and BPMs for scan")
    corr = hkickers[int(rng.integers(len(hkickers)))]
    bpm = bpms[int(rng.integers(len(bpms)))]
    return {
        "corrector_var": corr["variable"],
        "bpm_var": bpm["variable"],
        "max_kick": 0.0005,
        "n": 5,
    }


def _verify_2_7(result: dict, env: Env, setup_data: dict) -> bool:
    corrector_var = setup_data["corrector_var"]
    bpm_var = setup_data["bpm_var"]

    # Corrector should be restored
    design = env.get_design(corrector_var)
    if abs(env.machine.get(corrector_var) - design) > 1e-8:
        return False

    data = result.get("data")
    if not isinstance(data, list) or len(data) < setup_data["n"] - 1:
        return False

    # Verify each reported point by replaying the kick
    for point in data:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            kick_val, orbit_val = float(point[0]), float(point[1])
        elif isinstance(point, dict):
            kick_val = float(point.get("kick", 0))
            orbit_val = float(point.get("orbit", 0))
        else:
            return False
        env.machine.set(corrector_var, kick_val)
        expected = env.machine.get(bpm_var)
        if not close(orbit_val, expected, rtol=0.1, atol=1e-8):
            env.machine.set(corrector_var, design)
            return False

    env.machine.set(corrector_var, design)
    return True


# --- 2.8 Find and Zero a Corrector ---


def _setup_2_8(env: Env) -> dict[str, Any]:
    rng = env.rng
    hkickers = query_variables(env, "hkicker", "kick")
    if not hkickers:
        raise ValueError("No horizontal correctors found in lattice")
    pick = hkickers[int(rng.integers(len(hkickers)))]
    sign = 1 if float(rng.random()) > 0.5 else -1
    kick = float(rng.uniform(1e-4, 5e-4)) * sign
    env.machine.set(pick["variable"], kick)
    return {"_corrector_var": pick["variable"]}


def _verify_2_8(result: dict, env: Env, setup_data: dict) -> bool:
    return abs(env.machine.get(setup_data["_corrector_var"])) < 1e-10


# --- Task list ---

TIER2_TASKS = [
    TaskDef(
        id="2.1",
        name="Maximum Beta Function",
        tier=2,
        budget=5,
        prompt_template=(
            "Read the horizontal beta function at the following elements: "
            "{variables}. Report the maximum value and which element it occurs at."
            "\n\n"
            'Return `{{"max_beta": <number>, "element": "<device_id>"}}`.'
        ),
        setup=_setup_2_1,
        verify=_verify_2_1,
    ),
    TaskDef(
        id="2.2",
        name="Orbit RMS",
        tier=2,
        budget=5,
        prompt_template=(
            "Read the horizontal orbit at the following BPMs: {bpm_variables}. "
            "Report the RMS value.\n\n"
            'Return `{{"rms": <number_in_meters>}}`.'
        ),
        setup=_setup_2_2,
        verify=_verify_2_2,
    ),
    TaskDef(
        id="2.3",
        name="Corrector Orbit Response",
        tier=2,
        budget=8,
        prompt_template=(
            "Apply a kick of {kick} rad to `{corrector_var}`, read the horizontal "
            "orbit at the following BPMs: {bpm_variables}, then restore the "
            "corrector to its original value. Report the orbit change at each BPM."
            "\n\n"
            'Return `{{"orbit_change": {{"<bpm_name>": <change_in_meters>, ...}}}}`.'
        ),
        setup=_setup_2_3,
        verify=_verify_2_3,
    ),
    TaskDef(
        id="2.4",
        name="Quad Tune Shift",
        tier=2,
        budget=7,
        prompt_template=(
            "Change `{quad_var}` by +1%, read the horizontal tune (`{tune_var}`), "
            "then restore the quadrupole. Report the tune change.\n\n"
            'Return `{{"tune_change": <number>}}`.'
        ),
        setup=_setup_2_4,
        verify=_verify_2_4,
    ),
    TaskDef(
        id="2.5",
        name="Increase Magnet Strength",
        tier=2,
        budget=5,
        prompt_template=(
            "Read the current value of `{variable}`, increase it by {pct}%, "
            "and set the new value.\n\n"
            'Return `{{"status": "done"}}`.'
        ),
        setup=_setup_2_5,
        verify=_verify_2_5,
    ),
    TaskDef(
        id="2.6",
        name="Zero Correctors and Read Orbit",
        tier=2,
        budget=8,
        prompt_template=(
            "Set the following horizontal correctors to zero: {corrector_variables}. "
            "Then read the horizontal orbit at these BPMs: {bpm_variables}.\n\n"
            'Return `{{"orbit": {{"<bpm_name>": <orbit_in_meters>, ...}}}}`.'
        ),
        setup=_setup_2_6,
        verify=_verify_2_6,
    ),
    TaskDef(
        id="2.7",
        name="Corrector Scan",
        tier=2,
        budget=12,
        prompt_template=(
            "Scan `{corrector_var}` through {n} evenly spaced values from 0 to "
            "{max_kick} rad. At each step, read the horizontal orbit at "
            "`{bpm_var}`. Restore the corrector when done.\n\n"
            'Return `{{"data": [[<kick>, <orbit>], ...]}}`.'
        ),
        setup=_setup_2_7,
        verify=_verify_2_7,
    ),
    TaskDef(
        id="2.8",
        name="Find and Zero a Corrector",
        tier=2,
        budget=5,
        prompt_template=(
            "One of the horizontal correctors has a nonzero kick. Find it and "
            "set it to zero.\n\n"
            'Return `{{"status": "done", "corrector": "<variable_name>"}}`.'
        ),
        setup=_setup_2_8,
        verify=_verify_2_8,
    ),
]
