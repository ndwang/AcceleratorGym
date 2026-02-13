from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class Backend(ABC):
    """Abstract base class for accelerator backends."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the accelerator/simulation."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources and close connections."""

    @abstractmethod
    def get(self, name: str) -> Any:
        """Read the current value of a variable."""

    @abstractmethod
    def set(self, name: str, value: Any) -> None:
        """Write a value to a variable."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to the initial state."""

    def __enter__(self) -> Backend:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()
