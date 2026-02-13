from __future__ import annotations

from typing import Any

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.variable import Variable


class BmadBackend(Backend):
    """Bmad/Tao backend for lattice simulations.

    Requires the `pytao` package. Install with: pip install accelerator-gym[bmad]
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            import pytao  # noqa: F401
        except ImportError:
            raise ImportError(
                "BmadBackend requires pytao. Install with: pip install accelerator-gym[bmad]"
            )
        self._settings = kwargs
        raise NotImplementedError("BmadBackend is not yet implemented")

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
