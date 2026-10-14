"""
test_benchmarks.py: Unit tests for Generalized mixed-variable synthetic benchmark landscapes.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.es_model.config import ESConfig
from src.es_model.individual import Individual
from src.benchmarks.generalized_functions import (
    generalized_sphere,
    generalized_rastrigin,
    generalized_coupled,
    get_default_targets,
    GENERALIZED_BENCHMARKS,
)


class TestGeneralizedBenchmarks(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.config = ESConfig(
            n_real=2,
            n_count=2,
            n_binary=2,
            n_categorical=2,
            categorical_k=np.array([3, 4], dtype=np.int64),
            n_fuzzy=1,
            n_circular=1,
            circular_k=np.array([7], dtype=np.int64),
        )

    def test_default_targets_optimum(self) -> None:
        targets = get_default_targets(self.config)
        ind = Individual.random(self.rng, self.config)

        # Force individual variables to exact target values
        ind.variables["real"].values = targets["real"].copy()
        ind.variables["count"].values = targets["count"].copy()
        ind.variables["binary"].values = targets["binary"].copy()
        ind.variables["categorical"].values = targets["categorical"].copy()
        ind.variables["fuzzy"].values = targets["fuzzy"].copy()
        ind.variables["circular"].values = targets["circular"].copy()

        # At the global optimum, sphere cost must be exactly 0.0
        sphere_val = generalized_sphere(ind, self.config, targets)
        self.assertAlmostEqual(sphere_val, 0.0, places=6)

        # At global optimum, rastrigin cost must be 0.0
        rastrigin_val = generalized_rastrigin(ind, self.config, targets)
        self.assertAlmostEqual(rastrigin_val, 0.0, places=6)

        # At global optimum, coupled cost must be 0.0
        coupled_val = generalized_coupled(ind, self.config, targets)
        self.assertAlmostEqual(coupled_val, 0.0, places=6)

    def test_benchmarks_registry(self) -> None:
        self.assertIn("Generalized_Sphere", GENERALIZED_BENCHMARKS)
        self.assertIn("Generalized_Rastrigin", GENERALIZED_BENCHMARKS)
        self.assertIn("Generalized_Coupled", GENERALIZED_BENCHMARKS)

        ind = Individual.random(self.rng, self.config)
        for name, spec in GENERALIZED_BENCHMARKS.items():
            fn = spec["fn"]
            val = fn(ind, self.config)
            self.assertIsInstance(val, float)
            self.assertGreaterEqual(val, 0.0)

    def test_decomposition_at_optimum(self) -> None:
        from experiments.utils.runners import compute_heterogeneous_decomposition
        targets = get_default_targets(self.config)
        ind = Individual.random(self.rng, self.config)

        ind.variables["real"].values = targets["real"].copy()
        ind.variables["count"].values = targets["count"].copy()
        ind.variables["binary"].values = targets["binary"].copy()
        ind.variables["categorical"].values = targets["categorical"].copy()
        ind.variables["fuzzy"].values = targets["fuzzy"].copy()
        ind.variables["circular"].values = targets["circular"].copy()

        for b_name in ["Generalized_Sphere", "Generalized_Rastrigin", "Generalized_Coupled"]:
            decomp = compute_heterogeneous_decomposition(ind, b_name, self.config)
            tot = sum(decomp.values())
            self.assertAlmostEqual(tot, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

