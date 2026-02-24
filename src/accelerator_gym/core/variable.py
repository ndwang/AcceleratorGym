from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Variable:
    """A single named parameter that can be read and/or written.

    All values are floats. Validation checks type, access permissions, and limits.
    """

    name: str
    description: str = ""
    units: str | None = None
    readable: bool = True
    writable: bool = True
    limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.limits is not None:
            lo, hi = self.limits
            if lo > hi:
                raise ValueError(
                    f"Variable '{self.name}': lower limit {lo} exceeds upper limit {hi}"
                )

    def validate_read(self) -> None:
        """Raise ValueError if this variable is not readable."""
        if not self.readable:
            raise ValueError(f"Variable '{self.name}' is not readable")

    def validate_value(self, value: float | int) -> None:
        """Raise TypeError/ValueError if *value* violates this variable's constraints."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"Variable '{self.name}': expected numeric value, got {type(value).__name__}"
            )
        if not self.writable:
            raise ValueError(f"Variable '{self.name}' is not writable")
        if self.limits is not None:
            lo, hi = self.limits
            if value < lo or value > hi:
                raise ValueError(
                    f"Variable '{self.name}': value {value} outside limits [{lo}, {hi}]"
                )
