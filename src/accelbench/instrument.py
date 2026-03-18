"""Instrumented machine wrapper for tool call counting and budget enforcement."""

from __future__ import annotations

from typing import Any

from accelerator_gym.core.machine import Machine


class InstrumentedMachine:
    """Wraps a Machine, counting each tool-level call and enforcing budgets."""

    def __init__(self, machine: Machine, budget: int) -> None:
        self._machine = machine
        self._budget = budget
        self._call_count = 0

    @property
    def machine(self) -> Machine:
        return self._machine

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def budget(self) -> int:
        return self._budget

    def _check_budget(self) -> str | None:
        """Return error string if budget exceeded, else None."""
        if self._call_count >= self._budget:
            return "Error: tool call budget exceeded"
        self._call_count += 1
        return None

    def browse_devices(self, path: str = "/", depth: int = 1) -> str:
        err = self._check_budget()
        if err:
            return err
        try:
            result = self._machine.catalog.browse(path, depth)
            return _format_json(result)
        except Exception as e:
            return f"Error: {e}"

    def query_devices(self, sql: str) -> str:
        err = self._check_budget()
        if err:
            return err
        try:
            rows = self._machine.catalog.query(sql)
            return _format_json({"rows": rows, "count": len(rows)})
        except (ValueError, Exception) as e:
            return _format_json({"error": "query_error", "message": str(e)})

    def get_variables(self, names: list[str]) -> str:
        err = self._check_budget()
        if err:
            return err
        try:
            values = self._machine.get_many(names)
            lines = []
            for name in names:
                var = self._machine.get_variable(name)
                if var.units:
                    lines.append(f"{name} = {values[name]} ({var.units})")
                else:
                    lines.append(f"{name} = {values[name]}")
            return "\n".join(lines)
        except (KeyError, ValueError) as e:
            return f"Error: {e}"

    def set_variables(self, values: dict[str, float]) -> str:
        err = self._check_budget()
        if err:
            return err
        try:
            self._machine.set_many(values)
            lines = [f"{name} set to {v}" for name, v in values.items()]
            return "\n".join(lines)
        except (KeyError, ValueError, TypeError) as e:
            return f"Error: {e}"

    def reset(self) -> str:
        err = self._check_budget()
        if err:
            return err
        try:
            self._machine.reset()
            return "Machine reset to initial state"
        except Exception as e:
            return f"Error: {e}"


def _format_json(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)


def make_call_tool(instrumented: InstrumentedMachine):
    """Build a call_tool(name, args) -> str closure for an adapter."""

    _dispatch = {
        "browse_devices": lambda args: instrumented.browse_devices(
            path=args.get("path", "/"), depth=args.get("depth", 1)
        ),
        "query_devices": lambda args: instrumented.query_devices(
            sql=args["sql"]
        ),
        "get_variables": lambda args: instrumented.get_variables(
            names=args["names"]
        ),
        "set_variables": lambda args: instrumented.set_variables(
            values=args["values"]
        ),
        "reset": lambda args: instrumented.reset(),
    }

    def call_tool(name: str, arguments: dict[str, Any]) -> str:
        fn = _dispatch.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        return fn(arguments)

    return call_tool


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "browse_devices",
        "description": (
            "Browse the device tree to discover devices using filesystem-like paths. "
            "Use this to discover what you can read/write. The catalog is tree-shaped.\n\n"
            "Path levels: \"/\" -> systems, \"/system\" -> device types, "
            "\"/system/type\" -> devices, \"/system/type/device\" -> attributes, "
            "\"/system/type/device/attr\" -> attribute metadata.\n\n"
            "When you browse to an attribute (or use depth so attributes are included), each "
            "attribute has a \"variable\" field: that is the exact string to pass to get_variables "
            "and set_variables. Use depth > 1 to see multiple levels at once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Tree path to browse (default: \"/\")",
                    "default": "/",
                },
                "depth": {
                    "type": "integer",
                    "description": "How many levels below path to include (default: 1)",
                    "default": 1,
                },
            },
        },
    },
    {
        "name": "query_devices",
        "description": (
            "Run a read-only SQL query against the device metadata database.\n\n"
            "Tables: devices(device_id, system, device_type, s_position, tree_path), "
            "attributes(device_id, attribute_name, description, value, unit, readable, writable, "
            "lower_limit, upper_limit, variable).\n"
            "JOIN attributes with devices on device_id to filter by system/device_type.\n"
            "Example: SELECT a.variable, a.unit FROM attributes a JOIN devices d "
            "ON a.device_id = d.device_id WHERE d.system = 'magnets';"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_variables",
        "description": (
            "Read one or more variables. Each name must be a variable name (e.g. \"QF:K1\").\n\n"
            "Variable names are flat strings like \"QF:K1\", \"BPM1:X\". Get them from browse_devices "
            "(see the \"variable\" field when you browse to an attribute) or by querying the metadata database."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of variable names to read",
                },
            },
            "required": ["names"],
        },
    },
    {
        "name": "set_variables",
        "description": (
            "Write one or more variables atomically. Keys must be variable names (e.g. \"QF:K1\"). "
            "All-or-nothing: if any value violates limits, none are applied."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Map of variable names to values",
                },
            },
            "required": ["values"],
        },
    },
    {
        "name": "reset",
        "description": "Reset the machine to its initial state.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]
