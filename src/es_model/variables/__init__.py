"""
variables: Decision variable representations and mutation operators.
"""

from __future__ import annotations

from .variable import Variable, resolve_bounds
from .real import RealVariable, mutate_real
from .count import CountVariable, mutate_count
from .binary import BinaryVariable, mutate_binary
from .categorical import CategoricalVariable, mutate_categorical, center_categorical_logits
from .fuzzy import FuzzyVariable, mutate_fuzzy, membership
from .circular import CircularVariable, mutate_circular

VARIABLE_REGISTRY: dict[str, type[Variable]] = {
    "real": RealVariable,
    "count": CountVariable,
    "binary": BinaryVariable,
    "categorical": CategoricalVariable,
    "fuzzy": FuzzyVariable,
    "circular": CircularVariable,
}

__all__ = [
    "Variable",
    "resolve_bounds",
    "RealVariable",
    "CountVariable",
    "BinaryVariable",
    "CategoricalVariable",
    "FuzzyVariable",
    "CircularVariable",
    "VARIABLE_REGISTRY",
    "mutate_real",
    "mutate_count",
    "mutate_binary",
    "mutate_categorical",
    "center_categorical_logits",
    "mutate_fuzzy",
    "membership",
    "mutate_circular",
]
