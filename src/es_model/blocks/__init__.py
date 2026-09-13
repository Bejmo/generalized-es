"""
blocks — Heterogeneous variable block definitions and registry.
"""

from __future__ import annotations

from .base import VariableBlock, resolve_bounds
from .real import RealBlock
from .count import CountBlock
from .binary import BinaryBlock
from .categorical import CategoricalBlock
from .fuzzy import FuzzyBlock
from .circular import CircularBlock

BLOCK_REGISTRY: dict[str, type[VariableBlock]] = {
    "real": RealBlock,
    "count": CountBlock,
    "binary": BinaryBlock,
    "categorical": CategoricalBlock,
    "fuzzy": FuzzyBlock,
    "circular": CircularBlock,
}

__all__ = [
    "VariableBlock",
    "resolve_bounds",
    "RealBlock",
    "CountBlock",
    "BinaryBlock",
    "CategoricalBlock",
    "FuzzyBlock",
    "CircularBlock",
    "BLOCK_REGISTRY",
]
