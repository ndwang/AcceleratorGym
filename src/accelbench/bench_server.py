"""Benchmark MCP server — instrumented server for AccelBench.

Launched by Claude CLI via mcp-config. Creates a Machine, runs task setup,
wraps in InstrumentedMachine for budget enforcement and trace recording,
then serves the 5 accelerator-gym tools over stdio MCP.

On exit (when Claude CLI disconnects), dumps trace data to --trace-file
so the parent harness can replay agent mutations for verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("accelbench")

_inst: "InstrumentedMachine | None" = None


def _get() -> "InstrumentedMachine":
    assert _inst is not None, "InstrumentedMachine not initialized"
    return _inst


@mcp.tool()
def browse_devices(path: str = "/", depth: int = 1) -> str:
    """Browse the device tree to discover devices using filesystem-like paths.
    Use this to discover what you can read/write. The catalog is tree-shaped.

    Path levels: "/" -> systems, "/system" -> device types,
    "/system/type" -> devices, "/system/type/device" -> attributes,
    "/system/type/device/attr" -> attribute metadata.

    When you browse to an attribute (or use depth so attributes are included), each
    attribute has a "variable" field: that is the exact string to pass to get_variables
    and set_variables. Use depth > 1 to see multiple levels at once.
    """
    return _get().browse_devices(path, depth)


@mcp.tool()
def query_devices(sql: str) -> str:
    """Run a read-only SQL query against the device metadata database.

    Tables: devices(device_id, system, device_type, s_position, tree_path),
    attributes(device_id, attribute_name, value, unit, readable, writable,
    lower_limit, upper_limit, variable).
    JOIN attributes with devices on device_id to filter by system/device_type.
    Example: SELECT a.variable, a.unit FROM attributes a JOIN devices d
    ON a.device_id = d.device_id WHERE d.system = 'magnets';
    """
    return _get().query_devices(sql)


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
    (see the "variable" field when you browse to an attribute) or by querying the
    metadata database.

    If output_file is provided, results are written as CSV to that path instead of
    being returned inline. This saves tokens when reading many variables. The CSV has
    columns: name, value, units.
    """
    inst = _get()
    if output_file is not None:
        # Still count as one tool call via the instrumented get
        result = inst.get_variables(names)
        if result.startswith("Error"):
            return result
        # Write CSV from the machine's current values
        machine = inst.machine
        rows = [
            {
                "name": name,
                "value": machine.get(name),
                "units": machine.get_variable(name).units or "",
            }
            for name in names
        ]
        _write_csv(output_file, rows)
        return f"Wrote {len(rows)} variables to {output_file}"
    return inst.get_variables(names)


@mcp.tool()
def set_variables(values: dict[str, float]) -> str:
    """Write one or more variables atomically. Keys must be variable names (e.g. "QF:K1").
    All-or-nothing: if any value violates limits, none are applied.
    """
    return _get().set_variables(values)


@mcp.tool()
def reset() -> str:
    """Reset the machine to its initial state."""
    return _get().reset()


def _safe_serialize(obj: Any) -> Any:
    """Convert an object to JSON-safe types."""
    import numpy as np

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    global _inst

    parser = argparse.ArgumentParser(description="AccelBench MCP Server")
    parser.add_argument("--config", required=True, help="Path to accelerator-gym YAML config")
    parser.add_argument("--task-id", required=True, help="Task ID to set up")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--budget", type=int, required=True, help="Tool call budget")
    parser.add_argument("--trace-file", required=True, help="Path to write trace JSON on exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    from accelerator_gym.core.machine import Machine
    from accelbench.harness import task_seed
    from accelbench.instrument import InstrumentedMachine
    from accelbench.tasks import TASKS_BY_ID
    from accelbench.types import Env

    import numpy as np

    task = TASKS_BY_ID[args.task_id]
    machine = Machine.from_config(args.config)
    rng = np.random.default_rng(task_seed(args.seed, args.task_id))
    env = Env(machine=machine, rng=rng)

    logger.info(f"Setting up task {task.id}: {task.name}")
    setup_data = task.setup(env)

    _inst = InstrumentedMachine(machine, args.budget)
    logger.info(f"Serving with budget={args.budget}")

    try:
        mcp.run()
    finally:
        trace = {
            "call_count": _inst.call_count,
            "trace": [tc.to_dict() for tc in _inst.trace],
            "setup_data": _safe_serialize(setup_data),
        }
        try:
            with open(args.trace_file, "w") as f:
                json.dump(trace, f, default=str)
            logger.info(f"Trace written to {args.trace_file}")
        except Exception:
            logger.exception("Failed to write trace file")
        machine.close()


if __name__ == "__main__":
    main()
