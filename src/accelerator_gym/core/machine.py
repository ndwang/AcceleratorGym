from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.config import MachineConfig, build_variables, load_config
from accelerator_gym.core.variable import Variable

logger = logging.getLogger(__name__)


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

        config_file = Path(config_path).resolve()
        config_dir = config_file.parent
        
        logger.debug(f"Loading config from: {config_path}")
        logger.debug(f"Config directory: {config_dir}")
        config = load_config(config_path)
        logger.debug(f"Config loaded: backend_type={config.backend_type}, backend_settings={config.backend_settings}")
        
        # Resolve relative paths in backend_settings relative to config file directory
        backend_settings = dict(config.backend_settings)
        if "init_file" in backend_settings:
            init_file = Path(backend_settings["init_file"])
            if not init_file.is_absolute():
                init_file = (config_dir / init_file).resolve()
                logger.debug(f"Resolved init_file path: {init_file}")
                backend_settings["init_file"] = str(init_file)
        
        logger.info(f"Initializing backend: {config.backend_type}")
        backend_cls = get_backend_class(config.backend_type)
        logger.debug(f"Backend class: {backend_cls}")
        
        logger.info(f"Creating backend instance with settings: {backend_settings}")
        backend = backend_cls(**backend_settings)
        logger.debug("Backend instance created successfully")
        
        logger.info("Connecting to backend...")
        try:
            backend.connect()
            logger.info("Backend connected successfully")
        except Exception as e:
            logger.exception("Backend connection failed")
            raise
        
        logger.debug("Building variable registry...")
        machine = cls(backend, config)
        logger.info(f"Machine created with {len(machine.variables)} variables")
        return machine

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
        # Apply all using backend's batch operation
        self._backend.set_many(values)

    def reset(self) -> None:
        """Reset the machine to its initial state."""
        self._backend.reset()
