"""
Backward-compatibility module forwarding to generalized_functions.py.
"""

from __future__ import annotations

from src.benchmarks.generalized_functions import (
    generalized_sphere as heterogeneous_sphere,
    generalized_rastrigin as heterogeneous_rastrigin,
    generalized_coupled as heterogeneous_coupled,
    GENERALIZED_BENCHMARKS as HETEROGENEOUS_BENCHMARKS,
)

__all__ = [
    "heterogeneous_sphere",
    "heterogeneous_rastrigin",
    "heterogeneous_coupled",
    "HETEROGENEOUS_BENCHMARKS",
]
