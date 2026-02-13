from __future__ import annotations

from typing import Any

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.config import (
    MachineConfig,
    filter_variables,
    load_config,
    merge_definitions,
)
from accelerator_gym.core.variable import Variable


class Machine:
    """Central orchestrator for accelerator control.

    Owns the variable registry, enforces safety limits, and routes
    reads/writes to the backend.
    """

    def __init__(self, backend: Backend, config: MachineConfig) -> None:
        self._backend = backend
        self._config = config
        self._variables = self._build_registry()

    @classmethod
    def from_config(cls, config_path: str) -> Machine:
        """Construct a Machine from a YAML config file."""
        from accelerator_gym.backends import get_backend_class

        config = load_config(config_path)
        backend_cls = get_backend_class(config.backend_type)
        backend = backend_cls(**config.backend_settings)
        backend.connect()
        return cls(backend, config)

    def _build_registry(self) -> dict[str, Variable]:
        """Build the variable registry from discovery + config definitions."""
        discovered: dict[str, Variable] = {}

        if self._config.discovery_enabled:
            raw = self._backend.discover_variables()
            filtered = filter_variables(
                raw,
                self._config.discover_include,
                self._config.discover_exclude,
            )
            discovered = {v.name: v for v in filtered}

        return merge_definitions(discovered, self._config.definitions)

    @property
    def variables(self) -> dict[str, Variable]:
        """All variables exposed to the agent."""
        return dict(self._variables)

    def get(self, name: str) -> Any:
        """Read a variable value."""
        if name not in self._variables:
            raise KeyError(f"Unknown variable: '{name}'")
        return self._backend.get(name)

    def set(self, name: str, value: Any) -> None:
        """Write a variable value. Validates limits before delegating."""
        if name not in self._variables:
            raise KeyError(f"Unknown variable: '{name}'")
        self._variables[name].validate_value(value)
        self._backend.set(name, value)

    def get_many(self, names: list[str]) -> dict[str, Any]:
        """Read multiple variables at once."""
        result = {}
        for name in names:
            result[name] = self.get(name)
        return result

    def set_many(self, values: dict[str, Any]) -> None:
        """Write multiple variables. All-or-nothing validation."""
        # Validate all first
        for name, value in values.items():
            if name not in self._variables:
                raise KeyError(f"Unknown variable: '{name}'")
            self._variables[name].validate_value(value)
        # Apply all
        for name, value in values.items():
            self._backend.set(name, value)

    def reset(self) -> None:
        """Reset the machine to its initial state."""
        self._backend.reset()
