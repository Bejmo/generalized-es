"""
test_individual_and_es.py: Unit tests for Individual representation, operators, and GeneralizedES optimizer.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.es_model.config import ESConfig
from src.es_model.individual import (
    Individual,
    UnpackedGenome,
    unpack_genome,
    create_random_individual,
    clone_individual,
    recombine_individual,
    mutate_individual,
)
from src.es_model.generalized_es import GeneralizedES, select_survivors


class TestIndividualAndES(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.config = ESConfig(
            mu=10,
            lam=20,
            n_real=4,
            custom_real_bounds=(-5.0, 5.0),
            n_count=3,
            custom_count_bounds=(0, 10),
            n_binary=4,
            n_categorical=2,
            categorical_k=np.array([3, 4], dtype=np.int64),
            n_fuzzy=2,
            n_circular=2,
            circular_k=np.array([7, 12], dtype=np.int64),
        )

    # -------------------------------------------------------------------------
    # 1. Individual Lifecycle Tests
    # -------------------------------------------------------------------------
    def test_individual_creation_and_properties(self) -> None:
        ind = Individual.random(self.rng, self.config)
        self.assertIn("real", ind.variables)
        self.assertIn("count", ind.variables)
        self.assertIn("binary", ind.variables)
        self.assertIn("categorical", ind.variables)
        self.assertIn("fuzzy", ind.variables)
        self.assertIn("circular", ind.variables)

        # Backward compatibility property .blocks
        self.assertIs(ind.blocks, ind.variables)

        # Accessors
        self.assertEqual(ind.real.shape[0], 4)
        self.assertEqual(ind.count.shape[0], 3)
        self.assertEqual(ind.binary.shape[0], 4)
        self.assertEqual(ind.categorical.shape[0], 2)
        self.assertEqual(ind.fuzzy.shape[0], 4)  # 2 fuzzy vars * 2 dims (m, d)
        self.assertEqual(ind.circular.shape[0], 2)

        # Dimension interface
        expected_len = 4 + 3 + 4 + 2 + 4 + 2
        self.assertEqual(ind.genome.shape[0], expected_len)
        self.assertEqual(ind.total_dim, expected_len)
        self.assertEqual(ind.total_num_variables, 4 + 3 + 4 + 2 + 2 + 2)
        self.assertAlmostEqual(ind.distance_squared(ind), 0.0)

    def test_individual_distance_squared_properties(self) -> None:
        ind1 = Individual.random(self.rng, self.config)
        ind2 = Individual.random(self.rng, self.config)

        dist12 = ind1.distance_squared(ind2)
        dist21 = ind2.distance_squared(ind1)

        # Non-negativity and positive definiteness
        self.assertGreater(dist12, 0.0)
        # Symmetry
        self.assertAlmostEqual(dist12, dist21, places=6)

        # Additivity: sum of each variable's distance_squared equals individual's distance_squared
        manual_sum = sum(ind1.variables[k].distance_squared(ind2.variables[k]) for k in ind1.variables)
        self.assertAlmostEqual(dist12, manual_sum, places=6)

    def test_individual_clone(self) -> None:
        ind = Individual.random(self.rng, self.config)
        ind.fitness = 42.5
        cloned = ind.clone()

        self.assertIsNot(ind, cloned)
        self.assertEqual(ind.fitness, cloned.fitness)
        for k in ind.variables:
            self.assertIsNot(ind.variables[k], cloned.variables[k])
            self.assertIsNot(ind.variables[k].values, cloned.variables[k].values)
            np.testing.assert_array_equal(ind.variables[k].values, cloned.variables[k].values)

    def test_individual_recombination_and_mutation(self) -> None:
        parent_a = Individual.random(self.rng, self.config)
        parent_b = Individual.random(self.rng, self.config)
        parents = [parent_a, parent_b]

        # Recombine
        child = parent_a.recombine(parents, self.rng, self.config)
        self.assertEqual(child.genome.shape[0], parent_a.genome.shape[0])

        # Mutate
        init_strat = child.variables["real"].strategy.copy()
        child.mutate(self.rng, self.config)
        self.assertFalse(np.allclose(child.variables["real"].strategy, init_strat))

    def test_functional_wrappers(self) -> None:
        ind1 = create_random_individual(self.rng, self.config)
        ind2 = clone_individual(ind1)
        self.assertIsNot(ind1, ind2)
        child = recombine_individual(ind1, [ind1, ind2], self.rng, self.config)
        mutate_individual(child, self.rng, self.config)
        self.assertEqual(child.genome.shape[0], ind1.genome.shape[0])

    # -------------------------------------------------------------------------
    # 2. Unpack Genome Tests
    # -------------------------------------------------------------------------
    def test_unpack_genome(self) -> None:
        ind = Individual.random(self.rng, self.config)
        unpacked_from_ind = unpack_genome(ind, self.config)
        np.testing.assert_array_equal(unpacked_from_ind.real, ind.real)
        np.testing.assert_array_equal(unpacked_from_ind.count, ind.count)

        unpacked_from_vec = unpack_genome(ind.genome, self.config)
        np.testing.assert_allclose(unpacked_from_vec.real, ind.real)
        np.testing.assert_array_equal(unpacked_from_vec.count, ind.count)
        np.testing.assert_array_equal(unpacked_from_vec.binary, ind.binary)

    # -------------------------------------------------------------------------
    # 3. Survivor Selection Tests
    # -------------------------------------------------------------------------
    def test_select_survivors(self) -> None:
        parents = [Individual.random(self.rng, self.config) for _ in range(5)]
        children = [Individual.random(self.rng, self.config) for _ in range(10)]
        for i, ind in enumerate(parents):
            ind.fitness = float(i + 10)
        for i, ind in enumerate(children):
            ind.fitness = float(i)

        # Comma selection: picks top mu strictly from children
        survivors_comma = select_survivors(parents, children, mode="comma", mu=5)
        self.assertEqual(len(survivors_comma), 5)
        self.assertEqual(survivors_comma[0].fitness, 0.0)
        self.assertEqual(survivors_comma[4].fitness, 4.0)

        # Plus selection: picks top mu from combined pool
        survivors_plus = select_survivors(parents, children, mode="plus", mu=5)
        self.assertEqual(len(survivors_plus), 5)
        self.assertEqual(survivors_plus[0].fitness, 0.0)

    # -------------------------------------------------------------------------
    # 4. GeneralizedES Optimization Tests
    # -------------------------------------------------------------------------
    def test_generalized_es_optimization(self) -> None:
        cfg = ESConfig(
            mu=5,
            lam=15,
            n_real=3,
            custom_real_bounds=(-2.0, 2.0),
            n_count=0,
            n_binary=0,
            n_categorical=0,
            n_fuzzy=0,
            n_circular=0,
            seed=42,
        )
        es = GeneralizedES(config=cfg, seed=42)
        trace = es.optimize(lambda x: float(np.sum(x ** 2)), max_evals=50)

        self.assertEqual(len(trace), 50)
        self.assertLessEqual(trace[-1], trace[0])
        self.assertIsNotNone(es.best_individual)


if __name__ == "__main__":
    unittest.main()
