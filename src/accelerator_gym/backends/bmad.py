from __future__ import annotations

from typing import Any

from accelerator_gym.backends.base import Backend


class BmadBackend(Backend):
    """Bmad/Tao backend for lattice simulations.

    Requires the `pytao` package. Install with: pip install accelerator-gym[bmad]

    Variable naming convention
    --------------------------
    Names use Tao's native syntax directly:

    - ``ele::{ele_name}[{attr}]`` — element attributes (e.g. ``ele::QF[K1]``)
    - ``lat::{attr}[{ele_name}]`` — lattice-level attributes (e.g. ``lat::orbit.x[BPM1]``)
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            import pytao  # noqa: F401
        except ImportError:
            raise ImportError(
                "BmadBackend requires pytao. Install with: pip install accelerator-gym[bmad]"
            )
        self._settings = kwargs
        self._tao: Any = None

    def connect(self) -> None:
        from pytao import Tao

        init_file = self._settings.get("init_file", "tao.init")
        self._tao = Tao(f"-init {init_file} -noplot")

    def disconnect(self) -> None:
        self._tao = None

    def get(self, name: str) -> float:
        result = self._tao.cmd(f"show value {name}")
        # `show value` returns lines like "   <expr>  =  1.23456"
        for line in result:
            line = line.strip()
            if "=" in line:
                return float(line.split("=")[-1].strip())
        raise ValueError(f"Could not parse Tao output for '{name}': {result}")

    def set(self, name: str, value: Any) -> None:
        if not name.startswith("ele::"):
            raise ValueError(f"Cannot set lattice-level variable: {name}")
        # ele::QF[K1] -> ele_name="QF", attr="K1"
        ele_name, attr = name[5:].rstrip("]").split("[", 1)
        self._tao.cmd(f"set element {ele_name} {attr} = {value}")

    def reset(self) -> None:
        self._tao.cmd("reinit tao")


