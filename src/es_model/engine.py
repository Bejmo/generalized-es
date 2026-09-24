"""
Backward-compatibility module forwarding to individual.py and generalized_es.py.
"""

from __future__ import annotations

from src.es_model.individual import (
    Individual,
    UnpackedGenome,
    unpack_genome,
    create_random_individual,
    clone_individual,
    recombine_individual,
    mutate_individual,
)
from src.es_model.generalized_es import (
    GeneralizedES,
    GeneralizedES as HeterogeneousES,
    select_survivors,
)

recombine_individuals = recombine_individual

__all__ = [
    "Individual",
    "UnpackedGenome",
    "unpack_genome",
    "create_random_individual",
    "clone_individual",
    "recombine_individual",
    "recombine_individuals",
    "GeneralizedES",
    "HeterogeneousES",
    "select_survivors",
]
