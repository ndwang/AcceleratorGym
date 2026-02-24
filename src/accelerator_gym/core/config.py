from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MachineConfig:
    """Parsed machine configuration."""

    name: str = ""
    description: str = ""
    backend_type: str = ""
    backend_settings: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_config(path: str | Path) -> MachineConfig:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    machine_section = raw.get("machine", {})
    backend_section = raw.get("backend", {})
    devices_section = raw.get("devices", {})

    backend_type = backend_section.pop("type", "")
    backend_settings = backend_section  # remaining keys are backend-specific

    return MachineConfig(
        name=machine_section.get("name", ""),
        description=machine_section.get("description", ""),
        backend_type=backend_type,
        backend_settings=backend_settings,
        devices=devices_section,
    )
