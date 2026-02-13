from __future__ import annotations

import fnmatch
from typing import Any

from accelerator_gym.core.machine import Machine


class LLMInterface:
    """Adapts a Machine for LLM tool calling.

    Generates OpenAI-compatible tool schemas and dispatches tool calls
    to Machine methods, returning structured results.
    """

    def __init__(self, machine: Machine) -> None:
        self._machine = machine

    def get_tools(self) -> list[dict]:
        """Return OpenAI-compatible tool/function schemas."""
        variables = self._machine.variables
        var_names = sorted(variables.keys())
        var_descriptions = []
        for name in var_names:
            v = variables[name]
            parts = [name]
            if v.description:
                parts.append(f"- {v.description}")
            if v.units:
                parts.append(f"({v.units})")
            if v.read_only:
                parts.append("[read-only]")
            if v.limits:
                parts.append(f"[{v.limits[0]}, {v.limits[1]}]")
            var_descriptions.append(" ".join(parts))

        var_list_text = "\n".join(var_descriptions) if var_descriptions else "No variables available."

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_variables",
                    "description": f"List available variables, optionally filtered by glob pattern.\n\nAvailable variables:\n{var_list_text}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter": {
                                "type": "string",
                                "description": "Glob pattern to filter variable names (e.g. 'Q*:K1')",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_variable",
                    "description": "Read a single variable value.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Variable name to read",
                                "enum": var_names,
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_variables",
                    "description": "Read multiple variables at once.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "names": {
                                "type": "array",
                                "items": {"type": "string", "enum": var_names},
                                "description": "List of variable names to read",
                            }
                        },
                        "required": ["names"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "set_variable",
                    "description": "Write a single variable value.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Variable name to write",
                                "enum": [n for n in var_names if not variables[n].read_only],
                            },
                            "value": {
                                "type": "number",
                                "description": "Value to set",
                            },
                        },
                        "required": ["name", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "set_variables",
                    "description": "Write multiple variables atomically. All-or-nothing: if any value violates limits, none are applied.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "values": {
                                "type": "object",
                                "description": "Mapping of variable names to values",
                            }
                        },
                        "required": ["values"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_state",
                    "description": "Get a snapshot of all readable variable values.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reset",
                    "description": "Reset the machine to its initial state.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]
        return tools

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call and return a structured result."""
        try:
            if tool_name == "list_variables":
                return self._list_variables(arguments.get("filter"))
            elif tool_name == "get_variable":
                return self._get_variable(arguments["name"])
            elif tool_name == "get_variables":
                return self._get_variables(arguments["names"])
            elif tool_name == "set_variable":
                return self._set_variable(arguments["name"], arguments["value"])
            elif tool_name == "set_variables":
                return self._set_variables(arguments["values"])
            elif tool_name == "get_state":
                return self._get_state()
            elif tool_name == "reset":
                return self._reset()
            else:
                return {"error": "unknown_tool", "message": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": type(e).__name__, "message": str(e)}

    def _list_variables(self, pattern: str | None) -> dict[str, Any]:
        variables = self._machine.variables
        items = []
        for name, var in sorted(variables.items()):
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue
            info: dict[str, Any] = {"name": name, "dtype": var.dtype}
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

    def _get_variable(self, name: str) -> dict[str, Any]:
        var = self._machine.variables.get(name)
        if var is None:
            return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
        value = self._machine.get(name)
        result: dict[str, Any] = {"name": name, "value": value}
        if var.units:
            result["units"] = var.units
        return result

    def _get_variables(self, names: list[str]) -> dict[str, Any]:
        values = {}
        for name in names:
            if name not in self._machine.variables:
                return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
            values[name] = self._machine.get(name)
        return {"values": values}

    def _set_variable(self, name: str, value: Any) -> dict[str, Any]:
        var = self._machine.variables.get(name)
        if var is None:
            return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
        try:
            self._machine.set(name, value)
        except ValueError as e:
            return self._format_set_error(name, value, var, e)
        return {"success": True, "name": name, "value": value}

    def _set_variables(self, values: dict[str, Any]) -> dict[str, Any]:
        for name in values:
            if name not in self._machine.variables:
                return {"error": "not_found", "message": f"Unknown variable: '{name}'"}
        try:
            self._machine.set_many(values)
        except ValueError as e:
            return {"error": "validation_error", "message": str(e)}
        return {"success": True, "values": values}

    def _get_state(self) -> dict[str, Any]:
        variables = self._machine.variables
        names = sorted(variables.keys())
        values = self._machine.get_many(names)
        return {"variables": values}

    def _reset(self) -> dict[str, Any]:
        self._machine.reset()
        return {"success": True}

    def _format_set_error(
        self, name: str, value: Any, var: Any, error: ValueError
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"error": "validation_error", "variable": name, "value": value}
        if var.limits:
            result["limits"] = list(var.limits)
        result["message"] = str(error)
        return result
