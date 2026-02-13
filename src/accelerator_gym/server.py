"""MCP server exposing accelerator-gym Machine as tool calls."""

from __future__ import annotations

import argparse
import fnmatch
from typing import Any

from mcp.server.fastmcp import FastMCP

from accelerator_gym.core.machine import Machine

mcp = FastMCP("accelerator-gym")

_machine: Machine | None = None


def _get_machine() -> Machine:
    assert _machine is not None, "Machine not initialized"
    return _machine


@mcp.tool()
def list_variables(filter: str | None = None) -> dict[str, Any]:
    """List available variables with metadata (units, limits, read-only status), optionally filtered by glob pattern."""
    machine = _get_machine()
    items = []
    for name, var in machine.variables.items():
        if filter and not fnmatch.fnmatch(name, filter):
            continue
        info: dict[str, Any] = {"name": name}
        if var.description:
            info["description"] = var.description
        if var.units:
            info["units"] = var.units
        if var.read_only:
            info["read_only"] = True
        if var.limits:
            info["limits"] = list(var.limits)
        items.append(info)
    return {"variables": items}


@mcp.tool()
def get_variable(name: str) -> dict[str, Any]:
    """Read a single variable value."""
    machine = _get_machine()
    var = machine.variables.get(name)
    if var is None:
        return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
    value = machine.get(name)
    result: dict[str, Any] = {"name": name, "value": value}
    if var.units:
        result["units"] = var.units
    return result


@mcp.tool()
def get_variables(names: list[str]) -> dict[str, Any]:
    """Read multiple variables at once."""
    machine = _get_machine()
    values = {}
    for name in names:
        if name not in machine.variables:
            return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
        values[name] = machine.get(name)
    return {"values": values}


@mcp.tool()
def set_variable(name: str, value: float) -> dict[str, Any]:
    """Write a single variable value."""
    machine = _get_machine()
    var = machine.variables.get(name)
    if var is None:
        return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
    try:
        machine.set(name, value)
    except ValueError as e:
        result: dict[str, Any] = {
            "error": "validation_error",
            "variable": name,
            "value": value,
        }
        if var.limits:
            result["limits"] = list(var.limits)
        result["message"] = str(e)
        return result
    return {"success": True, "name": name, "value": value}


@mcp.tool()
def set_variables(values: dict[str, float]) -> dict[str, Any]:
    """Write multiple variables atomically. All-or-nothing: if any value violates limits, none are applied."""
    machine = _get_machine()
    for name in values:
        if name not in machine.variables:
            return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
    try:
        machine.set_many(values)
    except ValueError as e:
        return {"error": "validation_error", "message": str(e)}
    return {"success": True, "values": values}


@mcp.tool()
def get_state() -> dict[str, Any]:
    """Get a snapshot of all readable variable values."""
    machine = _get_machine()
    variables = machine.variables
    names = list(variables.keys())
    values = machine.get_many(names)
    return {"variables": values}


@mcp.tool()
def reset() -> dict[str, Any]:
    """Reset the machine to its initial state."""
    machine = _get_machine()
    machine.reset()
    return {"success": True}


CONFIG_FILENAME = "accelerator-gym.yaml"


def _find_config() -> str:
    """Look for accelerator-gym.yaml in the current directory."""
    from pathlib import Path

    path = Path.cwd() / CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{CONFIG_FILENAME} not found in {Path.cwd()}. "
            "Provide --config or create an accelerator-gym.yaml file."
        )
    return str(path)


def main():
    global _machine
    parser = argparse.ArgumentParser(description="Accelerator Gym MCP Server")
    parser.add_argument("--config", default=None, help="Path to machine config YAML (auto-detected if omitted)")
    args = parser.parse_args()
    config_path = args.config or _find_config()
    _machine = Machine.from_config(config_path)
    mcp.run()


if __name__ == "__main__":
    main()
