from __future__ import annotations

from typing import Any

import pytest

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.config import MachineConfig
from accelerator_gym.core.machine import Machine


class MockBackend(Backend):
    """In-memory dict-based backend for testing."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._initial_state = dict(initial_state or {})
        self._state: dict[str, Any] = {}
        self._connected = False

    def connect(self) -> None:
        self._state = dict(self._initial_state)
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get(self, name: str) -> Any:
        if name not in self._state:
            raise KeyError(f"Backend has no value for '{name}'")
        return self._state[name]

    def set(self, name: str, value: Any) -> None:
        self._state[name] = value

    def reset(self) -> None:
        self._state = dict(self._initial_state)

    @property
    def connected(self) -> bool:
        return self._connected


@pytest.fixture
def mock_backend():
    """A MockBackend with some initial state."""
    backend = MockBackend(
        initial_state={
            "QF:K1": 0.5,
            "QD:K1": -0.5,
            "BPM1:X": 0.1,
            "BPM1:Y": -0.05,
        }
    )
    backend.connect()
    return backend


@pytest.fixture
def basic_config():
    """A basic MachineConfig."""
    return MachineConfig(
        name="Test Machine",
        description="A test machine",
        backend_type="mock",
        definitions={
            "QF:K1": {
                "description": "Focusing quad strength",
                "units": "1/m",
                "limits": [-5.0, 5.0],
            },
            "QD:K1": {
                "description": "Defocusing quad strength",
                "units": "1/m",
                "limits": [-5.0, 5.0],
            },
            "BPM1:X": {
                "description": "Horizontal position",
                "units": "mm",
                "read_only": True,
            },
            "BPM1:Y": {
                "description": "Vertical position",
                "units": "mm",
                "read_only": True,
            },
        },
    )


@pytest.fixture
def machine(mock_backend, basic_config):
    """A Machine with mock backend and basic config."""
    return Machine(mock_backend, basic_config)
