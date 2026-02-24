"""MCP server exposing accelerator-gym Machine as tool calls."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from accelerator_gym.core.machine import Machine

# Configure logging to stderr so MCP clients can see it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("accelerator-gym")

_machine: Machine | None = None


def _get_machine() -> Machine:
    if _machine is None:
        error_msg = "Machine not initialized. Server startup may have failed."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    return _machine


@mcp.tool()
def browse_devices(path: str = "/", depth: int = 1) -> dict[str, Any]:
    """Browse the device tree using filesystem-like paths.

    Path levels: "/" -> systems, "/system" -> device types,
    "/system/type" -> devices, "/system/type/device" -> attributes,
    "/system/type/device/attr" -> attribute metadata.

    Use depth > 1 to see multiple levels at once.
    """
    try:
        machine = _get_machine()
        return machine.catalog.browse(path, depth)
    except Exception:
        logger.exception("Error in browse_devices")
        raise


@mcp.tool()
def query_devices(sql: str) -> dict[str, Any]:
    """Run a read-only SQL query against the device metadata database.

    Tables: systems(name), device_types(name, system),
    devices(name, device_type, system, description),
    attributes(device_name, device_type, system, attr_name, variable,
    description, units, readable, writable, limit_low, limit_high).
    """
    try:
        machine = _get_machine()
        rows = machine.catalog.query(sql)
        return {"rows": rows, "count": len(rows)}
    except ValueError as e:
        return {"error": "query_error", "message": str(e)}
    except Exception as e:
        logger.exception("Error in query_devices")
        return {"error": "query_error", "message": str(e)}


@mcp.tool()
def get_variable(name: str) -> dict[str, Any]:
    """Read a single variable value."""
    try:
        machine = _get_machine()
        var = machine.variables.get(name)
        if var is None:
            return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
        value = machine.get(name)
        result: dict[str, Any] = {"name": name, "value": value}
        if var.units:
            result["units"] = var.units
        return result
    except ValueError as e:
        return {"error": "validation_error", "message": str(e)}
    except Exception:
        logger.exception(f"Error in get_variable({name})")
        raise


@mcp.tool()
def get_variables(names: list[str]) -> dict[str, Any]:
    """Read multiple variables at once."""
    try:
        machine = _get_machine()
        for name in names:
            if name not in machine.variables:
                return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
        values = machine.get_many(names)
        return {"values": values}
    except ValueError as e:
        return {"error": "validation_error", "message": str(e)}
    except Exception:
        logger.exception(f"Error in get_variables({names})")
        raise


@mcp.tool()
def set_variable(name: str, value: float) -> dict[str, Any]:
    """Write a single variable value."""
    try:
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
    except Exception:
        logger.exception(f"Error in set_variable({name}, {value})")
        raise


@mcp.tool()
def set_variables(values: dict[str, float]) -> dict[str, Any]:
    """Write multiple variables atomically. All-or-nothing: if any value violates limits, none are applied."""
    try:
        machine = _get_machine()
        for name in values:
            if name not in machine.variables:
                return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
        try:
            machine.set_many(values)
        except ValueError as e:
            return {"error": "validation_error", "message": str(e)}
        return {"success": True, "values": values}
    except Exception:
        logger.exception(f"Error in set_variables({values})")
        raise


@mcp.tool()
def get_state() -> dict[str, Any]:
    """Get a snapshot of all readable variable values."""
    try:
        machine = _get_machine()
        variables = machine.variables
        readable_names = [n for n, v in variables.items() if v.readable]
        values = machine.get_many(readable_names)
        return {"variables": values}
    except Exception:
        logger.exception("Error in get_state")
        raise


@mcp.tool()
def reset() -> dict[str, Any]:
    """Reset the machine to its initial state."""
    try:
        machine = _get_machine()
        machine.reset()
        return {"success": True}
    except Exception:
        logger.exception("Error in reset")
        raise


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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config_path = args.config or _find_config()
        logger.info(f"Loading machine configuration from: {config_path}")
        _machine = Machine.from_config(config_path)
        logger.info(f"Machine initialized successfully: {_machine._config.name}")
        logger.info(f"Available variables: {len(_machine.variables)}")
        mcp.run()
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        logger.exception("Failed to initialize MCP server")
        sys.exit(1)


if __name__ == "__main__":
    main()
