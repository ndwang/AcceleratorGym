from __future__ import annotations

from typing import Any

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.variable import Variable


class EPICSBackend(Backend):
    """EPICS Channel Access backend for real control systems.

    Requires the `pyepics` package. Install with: pip install accelerator-gym[epics]
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            import epics  # noqa: F401
        except ImportError:
            raise ImportError(
                "EPICSBackend requires pyepics. Install with: pip install accelerator-gym[epics]"
            )
        self._settings = kwargs
        raise NotImplementedError("EPICSBackend is not yet implemented")

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get(self, name: str) -> Any:
        raise NotImplementedError

    def set(self, name: str, value: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def discover_variables(self) -> list[Variable]:
        raise NotImplementedError
