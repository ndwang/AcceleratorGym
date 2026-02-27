"""MCP server exposing accelerator-gym Machine as tool calls."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
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
    """Browse the device tree to discover devices using filesystem-like paths.
    Use this to discover what you can read/write. The catalog is tree-shaped.

    Path levels: "/" -> systems, "/system" -> device types,
    "/system/type" -> devices, "/system/type/device" -> attributes,
    "/system/type/device/attr" -> attribute metadata.

    When you browse to an attribute (or use depth so attributes are included), each
    attribute has a "variable" field: that is the exact string to pass to get_variable
    and set_variable. Example: browsing /magnets/quadrupole/QF may show K1 with
    variable "QF:K1" — use name "QF:K1" in get_variable("QF:K1") and set_variable("QF:K1", value).

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

    Tables: devices(device_id, system, device_type, s_position, tree_path),
    attributes(device_id, attribute_name, value, unit, readable, writable, lower_limit, upper_limit, variable).
    JOIN attributes with devices on device_id to filter by system/device_type.
    Example: SELECT a.variable, a.unit FROM attributes a JOIN devices d ON a.device_id = d.device_id WHERE d.system = 'magnets';
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


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write variable readings to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "value", "units"])
        writer.writeheader()
        writer.writerows(rows)


@mcp.tool()
def get_variables(names: list[str], output_file: str | None = None) -> str:
    """Read one or more variables. Each name must be a variable name (e.g. "QF:K1").

    Variable names are flat strings like "QF:K1", "BPM1:X". Get them from browse_devices
    (see the "variable" field when you browse to an attribute) or by querying the metadata database.

    If output_file is provided, results are written as CSV to that path instead of
    being returned inline. This saves tokens when reading many variables. The CSV has
    columns: name, value, units.
    """
    try:
        machine = _get_machine()
        values = machine.get_many(names)

        if output_file is not None:
            rows = [
                {
                    "name": name,
                    "value": values[name],
                    "units": machine.get_variable(name).units or "",
                }
                for name in names
            ]
            _write_csv(output_file, rows)
            return f"Wrote {len(rows)} variables to {output_file}"

        lines = []
        for name in names:
            var = machine.get_variable(name)
            if var.units:
                lines.append(f"{name} = {values[name]} ({var.units})")
            else:
                lines.append(f"{name} = {values[name]}")
        return "\n".join(lines)
    except (KeyError, ValueError) as e:
        return f"Error: {e}"
    except Exception:
        logger.exception(f"Error in get_variables({names})")
        raise


@mcp.tool()
def set_variables(values: dict[str, float]) -> str:
    """Write one or more variables atomically. Keys must be variable names (e.g. "QF:K1").
    All-or-nothing: if any value violates limits, none are applied.
    """
    try:
        machine = _get_machine()
        machine.set_many(values)
        lines = [f"{name} set to {v}" for name, v in values.items()]
        return "\n".join(lines)
    except (KeyError, ValueError, TypeError) as e:
        return f"Error: {e}"
    except Exception:
        logger.exception(f"Error in set_variables({values})")
        raise


@mcp.tool()
def reset() -> str:
    """Reset the machine to its initial state."""
    try:
        machine = _get_machine()
        machine.reset()
        return "Machine reset to initial state"
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
        try:
            mcp.run()
        finally:
            _machine.close()
            _machine = None
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
