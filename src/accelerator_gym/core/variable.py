from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Variable:
    """A single named parameter that can be read and optionally written."""

    name: str
    description: str = ""
    dtype: str = "float"
    units: str | None = None
    read_only: bool = False
    limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.limits is not None:
            lo, hi = self.limits
            if lo > hi:
                raise ValueError(
                    f"Variable '{self.name}': lower limit {lo} exceeds upper limit {hi}"
                )

    def validate_value(self, value: float | int) -> None:
        """Raise ValueError if value violates this variable's constraints."""
        if self.read_only:
            raise ValueError(f"Variable '{self.name}' is read-only")
        if self.limits is not None:
            lo, hi = self.limits
            if value < lo or value > hi:
                raise ValueError(
                    f"Variable '{self.name}': value {value} outside limits [{lo}, {hi}]"
                )
