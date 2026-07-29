"""Ouroboros bridge system for ContinualLearningBench.

Plugs the Ouroboros self-modifying agent into CL-Bench as a `ContinualLearningSystem`.
Importing the class here runs the `@register_system("ouroboros")` decorator so the CLI
discovers it (see src/registry.py dir-based discovery).
"""

from .system import OuroborosSystem

__all__ = ["OuroborosSystem"]
