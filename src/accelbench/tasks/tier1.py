"""Tier 1: Direct — single tool call tasks (budget: 3)."""

from __future__ import annotations

from typing import Any

from accelbench.prompts import TASK_PROMPTS
from accelbench.types import TaskDef, Env
from ._helpers import query_variables, close


# --- 1.1 Read a Variable ---


def _setup_1_1(env: Env) -> dict[str, Any]:
    rng = env.rng
    pool: list[dict[str, Any]] = []
    for dt, attr in [("quadrupole", "K1"), ("monitor", "orbit.x")]:
        pool.extend(query_variables(env, dt, attr))
    globs = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a "
        "JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.system = 'global'"
    )
    pool.extend(globs)
    if not pool:
        raise ValueError("No readable variables found")
    return {"variable": pool[int(rng.integers(len(pool)))]["variable"]}


def _verify_1_1(result: dict, env: Env, setup_data: dict) -> bool:
    val = result.get("value")
    if val is None:
        return False
    return close(float(val), env.machine.get(setup_data["variable"]))


# --- 1.2 Batch Read ---


def _setup_1_2(env: Env) -> dict[str, Any]:
    rng = env.rng
    # Build pools per category; pick one from each available category
    pools: list[list[dict[str, Any]]] = []
    pools.append(query_variables(env, "quadrupole", "K1"))
    pools.append(query_variables(env, "monitor", "orbit.x"))
    globs = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a "
        "JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.system = 'global'"
    )
    if globs:
        pools.append(globs)
    # Need at least 3 pools; pad from available if needed
    all_vars: list[dict[str, Any]] = [v for p in pools for v in p]
    if len(all_vars) < 3:
        raise ValueError("Need at least 3 readable variables for batch read")
    picks: list[str] = []
    for pool in pools[:3]:
        if pool:
            picks.append(pool[int(rng.integers(len(pool)))]["variable"])
    # Fill remaining slots from any pool
    while len(picks) < 3:
        v = all_vars[int(rng.integers(len(all_vars)))]["variable"]
        if v not in picks:
            picks.append(v)
    return {"var1": picks[0], "var2": picks[1], "var3": picks[2]}


def _verify_1_2(result: dict, env: Env, setup_data: dict) -> bool:
    values = result.get("values")
    if not isinstance(values, dict):
        return False
    for key in ("var1", "var2", "var3"):
        var = setup_data[key]
        val = values.get(var)
        if val is None:
            return False
        if not close(float(val), env.machine.get(var)):
            return False
    return True


# --- 1.3 Set a Variable ---


def _setup_1_3(env: Env) -> dict[str, Any]:
    rng = env.rng
    writable = env.machine.catalog.query(
        "SELECT a.variable, a.lower_limit, a.upper_limit "
        "FROM attributes a WHERE a.writable = 1"
    )
    pick = writable[int(rng.integers(len(writable)))]
    lo, hi = pick["lower_limit"], pick["upper_limit"]
    if lo is not None and hi is not None:
        target = round(float(rng.uniform(lo, hi)), 6)
    else:
        current = env.machine.get(pick["variable"])
        target = round(current * 1.1 + 0.01, 6)
    return {"variable": pick["variable"], "target": target}


def _verify_1_3(result: dict, env: Env, setup_data: dict) -> bool:
    return close(
        env.machine.get(setup_data["variable"]), setup_data["target"], rtol=1e-6
    )


# --- 1.4 Batch Set ---


def _setup_1_4(env: Env) -> dict[str, Any]:
    rng = env.rng
    writable = env.machine.catalog.query(
        "SELECT a.variable, a.lower_limit, a.upper_limit "
        "FROM attributes a WHERE a.writable = 1"
    )
    idxs = rng.choice(len(writable), size=min(2, len(writable)), replace=False)
    picks = [writable[int(i)] for i in idxs]
    vals = []
    for p in picks:
        lo, hi = p["lower_limit"], p["upper_limit"]
        if lo is not None and hi is not None:
            vals.append(round(float(rng.uniform(lo, hi)), 6))
        else:
            vals.append(round(env.machine.get(p["variable"]) * 1.1 + 0.01, 6))
    return {
        "var1": picks[0]["variable"],
        "val1": vals[0],
        "var2": picks[1]["variable"],
        "val2": vals[1],
    }


def _verify_1_4(result: dict, env: Env, setup_data: dict) -> bool:
    for vk, tk in [("var1", "val1"), ("var2", "val2")]:
        if not close(env.machine.get(setup_data[vk]), setup_data[tk], rtol=1e-6):
            return False
    return True


# --- 1.5 Reset Machine ---


def _setup_1_5(env: Env) -> dict[str, Any]:
    rng = env.rng
    writable = env.machine.catalog.query(
        "SELECT a.variable, a.lower_limit, a.upper_limit "
        "FROM attributes a WHERE a.writable = 1"
    )
    pick = writable[int(rng.integers(len(writable)))]
    current = env.machine.get(pick["variable"])
    lo, hi = pick["lower_limit"], pick["upper_limit"]
    if lo is not None and hi is not None:
        perturbed = float(rng.uniform(lo, hi))
        while abs(perturbed - current) < 1e-8:
            perturbed = float(rng.uniform(lo, hi))
    else:
        perturbed = current + 0.1
    env.machine.set(pick["variable"], perturbed)
    return {"_variable": pick["variable"]}


def _verify_1_5(result: dict, env: Env, setup_data: dict) -> bool:
    var = setup_data["_variable"]
    return close(env.machine.get(var), env.get_design(var), rtol=1e-6)


# --- 1.6 Count Aggregation ---


def _setup_1_6(env: Env) -> dict[str, Any]:
    rng = env.rng
    types = env.machine.catalog.query(
        "SELECT DISTINCT device_type FROM devices WHERE system != 'global'"
    )
    pick = types[int(rng.integers(len(types)))]
    return {"device_type": pick["device_type"]}


def _verify_1_6(result: dict, env: Env, setup_data: dict) -> bool:
    count = result.get("count")
    if count is None:
        return False
    expected = env.machine.catalog.query(
        "SELECT COUNT(*) as cnt FROM devices WHERE device_type = ?",
        (setup_data["device_type"],),
    )[0]["cnt"]
    return int(count) == expected


# --- 1.7 Attribute Lookup ---


def _setup_1_7(env: Env) -> dict[str, Any]:
    rng = env.rng
    devices = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices WHERE s_position IS NOT NULL"
    )
    pick = devices[int(rng.integers(len(devices)))]
    return {"device_id": pick["device_id"]}


def _verify_1_7(result: dict, env: Env, setup_data: dict) -> bool:
    val = result.get("value")
    if val is None:
        return False
    expected = env.machine.catalog.query(
        "SELECT s_position FROM devices WHERE device_id = ?",
        (setup_data["device_id"],),
    )[0]["s_position"]
    return close(float(val), expected)


# --- 1.8 Range Filter ---


def _setup_1_8(env: Env) -> dict[str, Any]:
    rng = env.rng
    devices = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices "
        "WHERE s_position IS NOT NULL ORDER BY s_position"
    )
    s_vals = [d["s_position"] for d in devices]
    s_min, s_max = min(s_vals), max(s_vals)
    span = s_max - s_min
    lo = round(s_min + float(rng.uniform(0.1, 0.3)) * span, 2)
    hi = round(s_min + float(rng.uniform(0.5, 0.8)) * span, 2)
    return {"lo": lo, "hi": hi}


def _verify_1_8(result: dict, env: Env, setup_data: dict) -> bool:
    devices = result.get("devices")
    if not isinstance(devices, list):
        return False
    lo, hi = setup_data["lo"], setup_data["hi"]
    expected = env.machine.catalog.query(
        "SELECT device_id FROM devices "
        "WHERE s_position IS NOT NULL AND s_position >= ? AND s_position <= ?",
        (lo, hi),
    )
    return sorted(devices) == sorted(r["device_id"] for r in expected)


# --- 1.9 Browse Root ---


def _setup_1_9(env: Env) -> dict[str, Any]:
    return {}


def _verify_1_9(result: dict, env: Env, setup_data: dict) -> bool:
    cats = result.get("categories")
    if not isinstance(cats, list):
        return False
    expected = env.machine.catalog.browse("/")["children"]
    return sorted(cats) == sorted(expected)


# --- 1.10 Browse Leaf ---


def _setup_1_10(env: Env) -> dict[str, Any]:
    rng = env.rng
    devices = env.machine.catalog.query(
        "SELECT device_id, system, device_type FROM devices "
        "WHERE system != 'global'"
    )
    pick = devices[int(rng.integers(len(devices)))]
    path = f"/{pick['system']}/{pick['device_type']}/{pick['device_id']}"
    return {"path": path}


def _verify_1_10(result: dict, env: Env, setup_data: dict) -> bool:
    attrs = result.get("attributes")
    if not isinstance(attrs, list):
        return False
    browse_result = env.machine.catalog.browse(setup_data["path"])
    expected = browse_result.get("children", [])
    return sorted(attrs) == sorted(expected)


# --- Task list ---

TIER1_TASKS = [
    TaskDef(
        id="1.1",
        name="Read a Variable",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.1"],
        setup=_setup_1_1,
        verify=_verify_1_1,
    ),
    TaskDef(
        id="1.2",
        name="Batch Read",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.2"],
        setup=_setup_1_2,
        verify=_verify_1_2,
    ),
    TaskDef(
        id="1.3",
        name="Set a Variable",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.3"],
        setup=_setup_1_3,
        verify=_verify_1_3,
    ),
    TaskDef(
        id="1.4",
        name="Batch Set",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.4"],
        setup=_setup_1_4,
        verify=_verify_1_4,
    ),
    TaskDef(
        id="1.5",
        name="Reset Machine",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.5"],
        setup=_setup_1_5,
        verify=_verify_1_5,
    ),
    TaskDef(
        id="1.6",
        name="Count Aggregation",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.6"],
        setup=_setup_1_6,
        verify=_verify_1_6,
    ),
    TaskDef(
        id="1.7",
        name="Attribute Lookup",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.7"],
        setup=_setup_1_7,
        verify=_verify_1_7,
    ),
    TaskDef(
        id="1.8",
        name="Range Filter",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.8"],
        setup=_setup_1_8,
        verify=_verify_1_8,
    ),
    TaskDef(
        id="1.9",
        name="Browse Root",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.9"],
        setup=_setup_1_9,
        verify=_verify_1_9,
    ),
    TaskDef(
        id="1.10",
        name="Browse Leaf",
        tier=1,
        budget=3,
        prompt_template=TASK_PROMPTS["1.10"],
        setup=_setup_1_10,
        verify=_verify_1_10,
    ),
]
