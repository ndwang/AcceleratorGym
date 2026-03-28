from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from accelerator_gym.backends.base import Backend

logger = logging.getLogger(__name__)

# Attributes read via lat::{attr}[{element}] rather than ele::{element}[{attr}]
_LATTICE_ATTRIBUTES = frozenset({
    "orbit.x", "orbit.y",
    "beta.a", "beta.b", "alpha.a", "alpha.b",
    "phi.a", "phi.b", "eta.x",
})

# Global params that are element attributes (read from BEGINNING element)
# rather than Tao datums.
_GLOBAL_ELEMENT_ATTRIBUTES = frozenset({"e_tot"})

# Default control attributes per Bmad element type.
# Can be overridden via backend_settings["element_attributes"] in the YAML config.
_DEFAULT_ELEMENT_ATTRIBUTES: dict[str, list[dict[str, Any]]] = {
    "Quadrupole": [
        {"name": "K1", "units": "1/m^2", "write": True},
    ],
    "Sbend": [
        {"name": "ANGLE", "units": "rad", "write": True},
    ],
    "HKicker": [
        {"name": "kick", "units": "rad", "write": True},
    ],
    "VKicker": [
        {"name": "kick", "units": "rad", "write": True},
    ],
    "Sextupole": [
        {"name": "K2", "units": "1/m^3", "write": True},
    ],
}

# Twiss attributes added to every discovered element (read-only).
_TWISS_ATTRIBUTES: list[dict[str, Any]] = [
    {"name": "beta.a", "description": "Horizontal beta function", "units": "m"},
    {"name": "beta.b", "description": "Vertical beta function", "units": "m"},
    {"name": "alpha.a", "description": "Horizontal alpha function"},
    {"name": "alpha.b", "description": "Vertical alpha function"},
    {"name": "phi.a", "description": "Horizontal betatron phase", "units": "rad/2pi"},
    {"name": "phi.b", "description": "Vertical betatron phase", "units": "rad/2pi"},
    {"name": "eta.x", "description": "Horizontal dispersion", "units": "m"},
]

# Monitor-specific readback attributes (read-only).
_MONITOR_ATTRIBUTES: list[dict[str, Any]] = [
    {"name": "orbit.x", "description": "Horizontal orbit", "units": "m"},
    {"name": "orbit.y", "description": "Vertical orbit", "units": "m"},
]

# Bmad element types to skip during discovery.
_SKIP_ELEMENT_TYPES = frozenset({
    "Drift", "Marker", "Beginning_Ele", "Null_Ele", "Floor_Shift",
    "Fiducial", "Patch", "Taylor", "Match",
    "Ramper", "Girder", "Group",
})

# Bmad element types that map to the "magnets" system.
_MAGNET_TYPES = frozenset({
    "Quadrupole", "Sbend", "HKicker", "VKicker", "Sextupole",
})


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
        try:
            import pytao  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "BmadBackend requires pytao. Install with: pip install accelerator-gym[bmad]"
            ) from e
        self._settings = kwargs
        self._tao: Any = None

    def connect(self) -> None:
        from pytao import Tao

        init_file = self._settings.get("init_file", "tao.init")
        init_path = Path(init_file).resolve()

        logger.info(f"Connecting to Tao with init_file: {init_path}")

        if not init_path.exists():
            raise FileNotFoundError(
                f"Tao init file not found: {init_path}"
            )

        self._init_dir = init_path.parent

        import os
        original_cwd = Path.cwd()
        os.chdir(self._init_dir)
        try:
            tao_kwargs: dict[str, Any] = {}
            startup_file = self._settings.get("startup_file")
            if startup_file:
                tao_kwargs["startup_file"] = str(startup_file)
            self._tao = Tao(f"-init {init_path.name} -noplot", **tao_kwargs)
            logger.info("Tao instance created successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Tao backend: {e}") from e
        finally:
            os.chdir(original_cwd)

    def disconnect(self) -> None:
        self._tao = None

    def get(self, name: str) -> float:
        result = self._tao.cmd(f"show value {name}")
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
        ele_name, attr = name[5:].rstrip("]").split("[", 1)
        self._tao.cmd(f"set element {ele_name} {attr} = {value}")

    def set_many(self, values: dict[str, Any]) -> None:
        """Batch set multiple element attributes."""
        for name in values:
            if not name.startswith("ele::"):
                raise ValueError(f"Cannot set lattice-level variable: {name}")

        self._tao.cmd("set global lattice_calc_on = F")
        for name, value in values.items():
            ele_name, attr = name[5:].rstrip("]").split("[", 1)
            self._tao.cmd(f"set element {ele_name} {attr} = {value}")
        result = self._tao.cmd("set global lattice_calc_on = T", raises=False)
        for line in result:
            if "[ERROR" in line or "[FATAL" in line:
                logger.warning("Lattice recomputation warning: %s", line)

    def resolve_variable_name(
        self, system: str, device_type: str, device_name: str, attribute: str
    ) -> str:
        """Map tree coordinates to Tao variable names.

        Routing is attribute-based:
        - Lattice-computed attributes (orbit, Twiss) use ``lat::{attr}[{name}]``
        - Global lattice parameters use ``lat::{attr}[0]``
        - Element attributes use ``ele::{name}[{attr}]``
        """
        if attribute in _LATTICE_ATTRIBUTES:
            return f"lat::{attribute}[{device_name}]"
        if system == "global":
            if attribute in _GLOBAL_ELEMENT_ATTRIBUTES:
                return f"ele::beginning[{attribute}]"
            return f"lat::{attribute}"
        return f"ele::{device_name}[{attribute}]"

    def _discover_overlay_vars(self, name: str) -> list[str]:
        """Return the control variable names for an overlay element."""
        try:
            cv = self._tao.ele_control_var(name)
        except Exception:
            logger.debug("ele_control_var failed for %s, trying pipe fallback", name)
            return self._discover_overlay_vars_fallback(name)

        # pytao may return a dict keyed by variable name, or a list of dicts
        if isinstance(cv, dict):
            # Check for "name" column in a table-style dict
            if "name" in cv:
                names = cv["name"]
                if isinstance(names, str):
                    return [names]
                return list(names)
            return list(cv.keys())
        if isinstance(cv, (list, tuple)):
            out = []
            for item in cv:
                if isinstance(item, dict):
                    n = item.get("name") or item.get("var_name")
                    if n:
                        out.append(str(n))
                elif isinstance(item, str):
                    out.append(item)
            return out
        return []

    def _discover_overlay_vars_fallback(self, name: str) -> list[str]:
        """Fallback: parse raw pipe output for overlay control variables."""
        try:
            lines = self._tao.cmd(f"python ele:control_var {name}")
        except Exception:
            return []
        var_names = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            var_name = line.split(";")[0].strip()
            if var_name:
                var_names.append(var_name)
        return var_names

    def discover_devices(self) -> dict[str, dict[str, Any]]:
        """Auto-discover devices from the Tao lattice.

        Queries pytao for all elements, their types, and s-positions,
        then builds the device tree with control attributes, Twiss
        attributes, and global lattice parameters.
        """
        if self._tao is None:
            raise RuntimeError("Backend not connected. Call connect() first.")

        names = self._tao.lat_list("*", "ele.name")
        keys = self._tao.lat_list("*", "ele.key")
        s_positions = self._tao.lat_list("*", "ele.s")

        # Merge default element attributes with user overrides
        element_attributes = dict(_DEFAULT_ELEMENT_ATTRIBUTES)
        user_attrs = self._settings.get("element_attributes")
        if user_attrs:
            element_attributes.update(user_attrs)

        devices: dict[str, dict[str, dict[str, Any]]] = {}

        for name, key, s_pos in zip(names, keys, s_positions):
            if key in _SKIP_ELEMENT_TYPES:
                continue

            s_position = float(s_pos)

            if key == "Monitor":
                system = "diagnostics"
                device_type = "monitor"
            elif key in _MAGNET_TYPES:
                system = "magnets"
                device_type = key.lower()
            else:
                # Unknown element type — skip unless it has configured attributes
                if key not in element_attributes:
                    continue
                system = "magnets"
                device_type = key.lower()

            # Build attributes dict
            attrs: dict[str, dict[str, Any]] = {}

            # Control attributes (from mapping)
            if key in element_attributes:
                for attr_def in element_attributes[key]:
                    attrs[attr_def["name"]] = {
                        "description": attr_def.get("description", ""),
                        "units": attr_def.get("units"),
                        "read": True,
                        "write": attr_def.get("write", False),
                    }
                    if "limits" in attr_def:
                        attrs[attr_def["name"]]["limits"] = attr_def["limits"]

            # Monitor readbacks
            if key == "Monitor":
                for attr_def in _MONITOR_ATTRIBUTES:
                    attrs[attr_def["name"]] = {
                        "description": attr_def.get("description", ""),
                        "units": attr_def.get("units"),
                        "read": True,
                        "write": False,
                    }

            # Twiss attributes on every element
            for attr_def in _TWISS_ATTRIBUTES:
                attrs[attr_def["name"]] = {
                    "description": attr_def.get("description", ""),
                    "units": attr_def.get("units"),
                    "read": True,
                    "write": False,
                }

            # Insert into device tree
            if system not in devices:
                devices[system] = {}
            if device_type not in devices[system]:
                devices[system][device_type] = {}

            devices[system][device_type][name] = {
                "s_position": s_position,
                "attributes": attrs,
            }

        # Overlay elements (lord elements, queried separately)
        try:
            overlay_names = self._tao.lat_list("overlay::*", "ele.name")
        except Exception:
            overlay_names = []

        if overlay_names is not None and len(overlay_names) > 0:
            devices.setdefault("controls", {}).setdefault("overlay", {})
            for ol_name in overlay_names:
                ol_vars = self._discover_overlay_vars(ol_name)
                if not ol_vars:
                    continue
                attrs: dict[str, dict[str, Any]] = {}
                for var_name in ol_vars:
                    attrs[var_name] = {
                        "description": "Overlay control variable",
                        "read": True,
                        "write": True,
                    }
                devices["controls"]["overlay"][ol_name] = {
                    "s_position": 0.0,
                    "attributes": attrs,
                }

        # Global lattice parameters
        devices["global"] = {
            "lattice": {
                "params": {
                    "description": "Global lattice parameters",
                    "attributes": {
                        "tune.a": {
                            "description": "Horizontal tune",
                            "read": True,
                            "write": False,
                        },
                        "tune.b": {
                            "description": "Vertical tune",
                            "read": True,
                            "write": False,
                        },
                        "chrom.a": {
                            "description": "Horizontal chromaticity",
                            "read": True,
                            "write": False,
                        },
                        "chrom.b": {
                            "description": "Vertical chromaticity",
                            "read": True,
                            "write": False,
                        },
                        "e_tot": {
                            "description": "Total beam energy",
                            "units": "eV",
                            "read": True,
                            "write": False,
                        },
                    },
                }
            }
        }

        logger.info(
            f"Discovered {sum(len(devs) for types in devices.values() for devs in types.values())} devices"
        )
        return devices

    def get_design(self, name: str) -> float:
        """Read the design (unperturbed) value via Tao's ``|design`` suffix."""
        result = self._tao.cmd(f"show value {name}|design")
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
        raise ValueError(f"Could not parse Tao design output for '{name}': {result}")

    def reset(self) -> None:
        import os
        original_cwd = Path.cwd()
        os.chdir(self._init_dir)
        try:
            self._tao.cmd("reinit tao")
        finally:
            os.chdir(original_cwd)
