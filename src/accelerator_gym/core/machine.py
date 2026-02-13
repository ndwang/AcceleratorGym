from __future__ import annotations

from typing import Any

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.config import MachineConfig, build_variables, load_config
from accelerator_gym.core.variable import Variable


class Machine:
    """Central orchestrator for accelerator control.

    Owns the variable registry, enforces safety limits, and routes
    reads/writes to the backend.
    """

    def __init__(self, backend: Backend, config: MachineConfig) -> None:
        self._backend = backend
        self._config = config
        self._variables = build_variables(config.variables)

    @classmethod
    def from_config(cls, config_path: str) -> Machine:
        """Construct a Machine from a YAML config file."""
        from accelerator_gym.backends import get_backend_class

        config = load_config(config_path)
        backend_cls = get_backend_class(config.backend_type)
        backend = backend_cls(**config.backend_settings)
        backend.connect()
        return cls(backend, config)

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
        """Read multiple variables at once. Validates all names first."""
        for name in names:
            if name not in self._variables:
                raise KeyError(f"Unknown variable: '{name}'")
        return {name: self._backend.get(name) for name in names}

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
