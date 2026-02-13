"""Backend registry with lazy imports for optional dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerator_gym.backends.base import Backend

_BACKEND_REGISTRY: dict[str, str] = {
    "bmad": "accelerator_gym.backends.bmad:BmadBackend",
    "epics": "accelerator_gym.backends.epics:EPICSBackend",
}


def get_backend_class(backend_type: str) -> type[Backend]:
    """Look up a backend class by type name, importing lazily."""
    if backend_type not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. Available: {available}"
        )

    module_path, class_name = _BACKEND_REGISTRY[backend_type].rsplit(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def register_backend(name: str, import_path: str) -> None:
    """Register a custom backend. import_path format: 'module.path:ClassName'."""
    _BACKEND_REGISTRY[name] = import_path
