"""Mock backend for testing and development without a real accelerator."""

from __future__ import annotations

from typing import Any

from accelerator_gym.backends.base import Backend


class MockBackend(Backend):
    """In-memory dict-based backend that returns a default value for unknown variables.

    Unlike test mocks that require pre-populated state, this backend works
    automatically with ``Machine.from_config()`` — any variable read before
    being written simply returns ``default_value`` (0.0 by default).
    """

    def __init__(self, default_value: float = 0.0, **kwargs: Any) -> None:
        self._default_value = default_value
        self._state: dict[str, Any] = {}
        self._connected = False

    def connect(self) -> None:
        self._state = {}
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get(self, name: str) -> Any:
        return self._state.get(name, self._default_value)

    def set(self, name: str, value: Any) -> None:
        self._state[name] = value

    def reset(self) -> None:
        self._state = {}

    @property
    def connected(self) -> bool:
        return self._connected
