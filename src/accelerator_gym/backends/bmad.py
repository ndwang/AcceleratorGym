from __future__ import annotations

import logging
from typing import Any

from accelerator_gym.backends.base import Backend

logger = logging.getLogger(__name__)


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
        logger.debug(f"Initializing BmadBackend with settings: {kwargs}")
        logger.debug("Attempting to import pytao module...")
        try:
            import pytao  # noqa: F401
            logger.debug("pytao module imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import pytao: {e}")
            raise ImportError(
                "BmadBackend requires pytao. Install with: pip install accelerator-gym[bmad]"
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error importing pytao: {e}")
            raise
        logger.debug("Storing backend settings...")
        self._settings = kwargs
        self._tao: Any = None
        logger.debug("BmadBackend initialization complete")

    def connect(self) -> None:
        from pathlib import Path
        from pytao import Tao

        init_file = self._settings.get("init_file", "tao.init")
        init_path = Path(init_file)
        
        logger.info(f"Connecting to Tao with init_file: {init_file}")
        logger.debug(f"Init file path (absolute): {init_path.resolve()}")
        
        if not init_path.exists():
            error_msg = f"Tao init file not found: {init_path.resolve()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logger.debug(f"Creating Tao instance with args: -init {init_file} -noplot")
            # Change to init file's directory so relative paths in init file work correctly
            original_cwd = Path.cwd()
            init_dir = init_path.parent
            logger.debug(f"Changing working directory to: {init_dir}")
            import os
            os.chdir(init_dir)
            try:
                self._tao = Tao(f"-init {init_path.name} -noplot")
                logger.info("Tao instance created successfully")
            finally:
                os.chdir(original_cwd)
                logger.debug(f"Restored working directory to: {original_cwd}")
        except Exception as e:
            logger.exception(f"Failed to create Tao instance: {e}")
            raise RuntimeError(f"Failed to connect to Tao backend: {e}") from e

    def disconnect(self) -> None:
        self._tao = None

    def get(self, name: str) -> float:
        result = self._tao.cmd(f"show value {name}")
        # `show value` may return "   <expr>  =  1.23456" or just "  0.0E+00" (no equals)
        for line in result:
            line = line.strip()
            if not line:
                continue
            if "=" in line:
                value_str = line.split("=")[-1].strip()
            else:
                value_str = line
            try:
                return float(value_str)
            except ValueError:
                continue
        raise ValueError(f"Could not parse Tao output for '{name}': {result}")

    def set(self, name: str, value: Any) -> None:
        if not name.startswith("ele::"):
            raise ValueError(f"Cannot set lattice-level variable: {name}")
        # ele::QF[K1] -> ele_name="QF", attr="K1"
        ele_name, attr = name[5:].rstrip("]").split("[", 1)
        self._tao.cmd(f"set element {ele_name} {attr} = {value}")

    def set_many(self, values: dict[str, Any]) -> None:
        """Batch set multiple element attributes.

        More efficient than calling set() repeatedly as it validates all
        variables upfront and can potentially batch commands to Tao.
        """
        # Validate all names first
        for name in values:
            if not name.startswith("ele::"):
                raise ValueError(f"Cannot set lattice-level variable: {name}")

        self._tao.cmd("set global lattice_calc_on = F")
        # Execute all set commands
        for name, value in values.items():
            ele_name, attr = name[5:].rstrip("]").split("[", 1)
            self._tao.cmd(f"set element {ele_name} {attr} = {value}")
        self._tao.cmd("set global lattice_calc_on = T")

    def resolve_variable_name(
        self, system: str, device_type: str, device_name: str, attribute: str
    ) -> str:
        """Map tree coordinates to Tao variable names.

        Monitor devices use ``lat::{attribute}[{device_name}]`` syntax.
        All other devices use ``ele::{device_name}[{attribute}]`` syntax.
        """
        if device_type == "monitor":
            return f"lat::{attribute}[{device_name}]"
        return f"ele::{device_name}[{attribute}]"

    def reset(self) -> None:
        self._tao.cmd("reinit tao")


