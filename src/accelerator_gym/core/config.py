from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from accelerator_gym.core.variable import Variable


@dataclass
class MachineConfig:
    """Parsed machine configuration."""

    name: str = ""
    description: str = ""
    backend_type: str = ""
    backend_settings: dict[str, Any] = field(default_factory=dict)
    discover_include: list[str] = field(default_factory=list)
    discover_exclude: list[str] = field(default_factory=list)
    definitions: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def discovery_enabled(self) -> bool:
        return len(self.discover_include) > 0


def load_config(path: str | Path) -> MachineConfig:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    machine_section = raw.get("machine", {})
    backend_section = raw.get("backend", {})
    variables_section = raw.get("variables", {})

    backend_type = backend_section.pop("type", "")
    backend_settings = backend_section  # remaining keys are backend-specific

    discover = variables_section.get("discover", {})
    discover_include = discover.get("include", []) if discover else []
    discover_exclude = discover.get("exclude", []) if discover else []

    definitions = variables_section.get("definitions", {}) or {}

    return MachineConfig(
        name=machine_section.get("name", ""),
        description=machine_section.get("description", ""),
        backend_type=backend_type,
        backend_settings=backend_settings,
        discover_include=discover_include,
        discover_exclude=discover_exclude,
        definitions=definitions,
    )


def filter_variables(
    variables: list[Variable],
    include: list[str],
    exclude: list[str],
) -> list[Variable]:
    """Filter variables using glob include/exclude patterns."""
    result = []
    for var in variables:
        if not any(fnmatch.fnmatch(var.name, pat) for pat in include):
            continue
        if any(fnmatch.fnmatch(var.name, pat) for pat in exclude):
            continue
        result.append(var)
    return result


def merge_definitions(
    discovered: dict[str, Variable],
    definitions: dict[str, dict[str, Any]],
) -> dict[str, Variable]:
    """Merge explicit definitions on top of discovered variables.

    For variables in both, definition fields override discovered fields.
    Definition-only variables are created as new Variables.
    """
    registry: dict[str, Variable] = dict(discovered)

    for name, defn in definitions.items():
        if name in registry:
            # Overlay fields from definition onto discovered variable
            var = registry[name]
            registry[name] = Variable(
                name=name,
                description=defn.get("description", var.description),
                dtype=defn.get("dtype", var.dtype),
                units=defn.get("units", var.units),
                read_only=defn.get("read_only", var.read_only),
                limits=tuple(defn["limits"]) if "limits" in defn else var.limits,
            )
        else:
            # New variable from definitions only
            limits = tuple(defn["limits"]) if "limits" in defn else None
            registry[name] = Variable(
                name=name,
                description=defn.get("description", ""),
                dtype=defn.get("dtype", "float"),
                units=defn.get("units"),
                read_only=defn.get("read_only", False),
                limits=limits,
            )

    return registry
