"""Minimal-governance Core 1,500 experiment path.

This package is deliberately separate from the historical Portfolio/Formal
launch, authorization, reservation, and receipt machinery.  It keeps only
the state required to run a fair experiment and resume it conservatively.
"""

from .engine import CoreFastEngine, FastPathError, ValidationSummary
from .models import CoreFastSpec, load_core_fast_spec

__all__ = [
    "CoreFastEngine",
    "CoreFastSpec",
    "FastPathError",
    "ValidationSummary",
    "load_core_fast_spec",
]
