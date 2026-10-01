"""
Registry system for dynamically discovering and instantiating systems and tasks.

Uses decorators to register classes and introspection to extract parameter metadata.
"""

import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Type, get_type_hints


# Global registries
SYSTEMS_REGISTRY: dict[str, Type] = {}
TASKS_REGISTRY: dict[str, Type] = {}

_SRC_ROOT = Path(__file__).resolve().parent
_TASKS_ROOT = _SRC_ROOT / "tasks"
_SYSTEMS_ROOT = _SRC_ROOT / "systems"


@lru_cache(maxsize=1)
def _discover_task_modules() -> dict[str, str]:
    """Discover task modules from `src/tasks/<name>/task.py`."""
    modules: dict[str, str] = {}
    for path in sorted(_TASKS_ROOT.glob("*/task.py")):
        task_name = path.parent.name
        modules[task_name] = f"src.tasks.{task_name}.task"
    return modules


@lru_cache(maxsize=1)
def _discover_system_modules() -> dict[str, str]:
    """Discover system modules from flat files or package directories."""
    modules: dict[str, str] = {}
    for path in sorted(_SYSTEMS_ROOT.glob("*.py")):
        if path.stem == "common" or path.stem.startswith("_"):
            continue
        modules[path.stem] = f"src.systems.{path.stem}"
    for path in sorted(_SYSTEMS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("_") or path.name in {"__pycache__", "utils"}:
            continue
        if not (path / "__init__.py").exists():
            continue
        modules[path.name] = f"src.systems.{path.name}"
    return modules


def register_system(name: str) -> Callable:
    """Decorator: register a system class under ``name`` for CLI discovery."""

    def decorator(cls: Type) -> Type:
        SYSTEMS_REGISTRY[name] = cls
        return cls

    return decorator


def register_task(name: str) -> Callable:
    """Decorator: register a task class under ``name`` for CLI discovery."""

    def decorator(cls: Type) -> Type:
        r_max = cls.__dict__.get("r_max")
        if not isinstance(r_max, (int, float)) or isinstance(r_max, bool):
            raise TypeError(
                f"Task '{name}' must define class attribute r_max as the mean "
                "per-instance maximum reward for the default evaluation schedule."
            )
        if r_max <= 0:
            raise ValueError(f"Task '{name}' r_max must be positive; got {r_max!r}.")
        TASKS_REGISTRY[name] = cls
        return cls

    return decorator


def get_system_class(name: str) -> Type:
    """Get a system class by name, lazy-importing its module on demand."""
    module_map = _discover_system_modules()
    if name not in SYSTEMS_REGISTRY:
        if name in module_map:
            importlib.import_module(module_map[name])
        if name not in SYSTEMS_REGISTRY:
            if name in module_map:
                raise KeyError(
                    f"System module '{module_map[name]}' did not register '{name}'. "
                    f'Expected @register_system("{name}").'
                )
            raise KeyError(
                f"System '{name}' not found. Available systems: {list_systems()}"
            )
    return SYSTEMS_REGISTRY[name]


def get_task_class(name: str) -> Type:
    """Get a task class by name, lazy-importing its module on demand."""
    module_map = _discover_task_modules()
    if name not in TASKS_REGISTRY:
        if name in module_map:
            importlib.import_module(module_map[name])
        if name not in TASKS_REGISTRY:
            if name in module_map:
                raise KeyError(
                    f"Task module '{module_map[name]}' did not register '{name}'. "
                    f'Expected @register_task("{name}").'
                )
            raise KeyError(f"Task '{name}' not found. Available tasks: {list_tasks()}")
    return TASKS_REGISTRY[name]


def list_systems() -> list[str]:
    return sorted(_discover_system_modules().keys())


def list_tasks() -> list[str]:
    return sorted(_discover_task_modules().keys())


def get_class_params(cls: Type) -> dict[str, Any]:
    """
    Extract parameter metadata from a class's __init__ signature.

    Args:
        cls: Class to inspect

    Returns:
        Dict mapping parameter names to metadata:
            {
                "param_name": {
                    "type": type annotation or None,
                    "default": default value or inspect.Parameter.empty,
                    "required": bool
                }
            }
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}

    try:
        resolved_hints = get_type_hints(cls.__init__)
    except Exception:
        resolved_hints = {}

    params = {}
    for param_name, param in sig.parameters.items():
        # Skip 'self'
        if param_name == "self":
            continue

        params[param_name] = {
            "type": resolved_hints.get(param_name)
            if param_name in resolved_hints
            else (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else None
            ),
            "default": param.default
            if param.default != inspect.Parameter.empty
            else inspect.Parameter.empty,
            "required": param.default == inspect.Parameter.empty,
        }

    return params


def clear_registry_cache() -> None:
    """Clear discovery caches for tests."""
    _discover_task_modules.cache_clear()
    _discover_system_modules.cache_clear()
