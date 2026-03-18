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

    def set_many(self, values: dict[str, Any]) -> None:
        """Write multiple values at once.

        Default implementation applies sets sequentially. Backends can override
        this to provide optimized batch operations.

        Args:
            values: Dictionary mapping variable names to their new values.
        """
        for name, value in values.items():
            self.set(name, value)

    def resolve_variable_name(
        self, system: str, device_type: str, device_name: str, attribute: str
    ) -> str:
        """Map tree coordinates to a backend-native variable name.

        Override in subclasses to implement backend-specific naming conventions.
        Default returns ``device_name:attribute``.
        """
        return f"{device_name}:{attribute}"

    def discover_devices(self) -> dict[str, dict[str, Any]]:
        """Auto-discover devices from the backend.

        Returns a devices dict in the same format as MachineConfig.devices.
        Override in backends that can introspect their lattice/hardware.
        Default returns empty dict (no auto-discovery).
        """
        return {}

    @abstractmethod
    def reset(self) -> None:
        """Reset to the initial state."""

    def __enter__(self) -> Backend:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()
