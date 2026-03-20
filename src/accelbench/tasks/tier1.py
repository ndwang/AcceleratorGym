"""Tier 1: Direct tasks (budget: 3). Basic interface competence."""

from __future__ import annotations

from typing import Any

from accelbench.tasks.helpers import random_element
from accelbench.types import Env, TaskDef


# --- 1.1: Read a Parameter ---

def _setup_1_1(env: Env) -> dict[str, Any]:
    var = random_element(env, "quadrupole", "K1")
    name = var.split(":")[0]
    return {"element": name, "variable": var}


def _verify_1_1(result: dict, env: Env, setup: dict) -> bool:
    truth = env.machine.get(setup["variable"])
    return abs(result["value"] - truth) < 1e-8


# --- 1.2: Set a Parameter ---

def _setup_1_2(env: Env) -> dict[str, Any]:
    var = random_element(env, "quadrupole", "K1")
    name = var.split(":")[0]
    # Pick a target value: small perturbation from current
    current = env.machine.get(var)
    target = round(current * 1.05, 6) if abs(current) > 0.01 else 0.5
    return {"element": name, "variable": var, "target": target}


def _verify_1_2(result: dict, env: Env, setup: dict) -> bool:
    return abs(env.machine.get(setup["variable"]) - setup["target"]) < 1e-8


# --- 1.3: Read a BPM ---

def _setup_1_3(env: Env) -> dict[str, Any]:
    var = random_element(env, "monitor", "orbit.y")
    name = var.split(":")[0]
    return {"element": name, "variable": var}


def _verify_1_3(result: dict, env: Env, setup: dict) -> bool:
    truth = env.machine.get(setup["variable"])
    return abs(result["value"] - truth) < 1e-9


# --- 1.4: Count Elements by Type ---

def _setup_1_4(env: Env) -> dict[str, Any]:
    return {}


def _verify_1_4(result: dict, env: Env, setup: dict) -> bool:
    rows = env.machine.catalog.query(
        "SELECT COUNT(*) as n FROM devices WHERE device_type='quadrupole'"
    )
    return result["count"] == rows[0]["n"]


# --- 1.5: Read a Global Parameter ---

def _setup_1_5(env: Env) -> dict[str, Any]:
    return {}


def _verify_1_5(result: dict, env: Env, setup: dict) -> bool:
    truth = env.machine.get("params:tune.a")
    return abs(result["value"] - truth) < 1e-4


# --- 1.6: Identify an Element ---

def _setup_1_6(env: Env) -> dict[str, Any]:
    # Pick a random device
    rows = env.machine.catalog.query("SELECT device_id, device_type FROM devices")
    idx = env.rng.integers(0, len(rows))
    return {"element": rows[idx]["device_id"], "expected_type": rows[idx]["device_type"]}


def _verify_1_6(result: dict, env: Env, setup: dict) -> bool:
    return result["type"] == setup["expected_type"]


# --- Aggregation ---

TIER1_TASKS: list[TaskDef] = [
    TaskDef(
        id="1.1",
        name="Read a Parameter",
        tier=1,
        prompt_template=(
            "What is K1 of {element}?\n"
            'Answer: {{"value": <number>}}'
        ),
        budget=3,
        abilities=["io"],
        setup=_setup_1_1,
        verify=_verify_1_1,
    ),
    TaskDef(
        id="1.2",
        name="Set a Parameter",
        tier=1,
        prompt_template=(
            "Set K1 of {element} to {target}.\n"
            'Answer: {{"status": "done"}}'
        ),
        budget=3,
        abilities=["io"],
        setup=_setup_1_2,
        verify=_verify_1_2,
    ),
    TaskDef(
        id="1.3",
        name="Read a BPM",
        tier=1,
        prompt_template=(
            "What is the vertical orbit at {element}?\n"
            'Answer: {{"value": <number>}}'
        ),
        budget=3,
        abilities=["io"],
        setup=_setup_1_3,
        verify=_verify_1_3,
    ),
    TaskDef(
        id="1.4",
        name="Count Elements by Type",
        tier=1,
        prompt_template=(
            "How many quadrupoles are in the ring?\n"
            'Answer: {{"count": <integer>}}'
        ),
        budget=3,
        abilities=["discovery"],
        setup=_setup_1_4,
        verify=_verify_1_4,
    ),
    TaskDef(
        id="1.5",
        name="Read a Global Parameter",
        tier=1,
        prompt_template=(
            "What is the horizontal tune?\n"
            'Answer: {{"value": <number>}}'
        ),
        budget=3,
        abilities=["analysis"],
        setup=_setup_1_5,
        verify=_verify_1_5,
    ),
    TaskDef(
        id="1.6",
        name="Identify an Element",
        tier=1,
        prompt_template=(
            "What type of element is {element}?\n"
            'Answer: {{"type": "<device_type>"}}'
        ),
        budget=3,
        abilities=["discovery"],
        setup=_setup_1_6,
        verify=_verify_1_6,
    ),
]
