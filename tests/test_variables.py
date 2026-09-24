"""
test_variables.py: Comprehensive unit tests for decision variables and kernels.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.es_model.config import ESConfig
from src.es_model.variables import (
    Variable,
    resolve_bounds,
    RealVariable,
    CountVariable,
    BinaryVariable,
    CategoricalVariable,
    FuzzyVariable,
    CircularVariable,
    mutate_real,
    mutate_count,
    mutate_binary,
    mutate_categorical,
    center_categorical_logits,
    mutate_circular,
    mutate_fuzzy,
)
from src.es_model import kernels


class TestVariables(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.config = ESConfig(
            n_real=4,
            custom_real_bounds=(-5.0, 5.0),
            n_count=3,
            custom_count_bounds=(0, 10),
            n_binary=5,
            n_categorical=2,
            categorical_k=np.array([3, 4], dtype=np.int64),
            n_fuzzy=2,
            n_circular=2,
            circular_k=np.array([7, 12], dtype=np.int64),
        )

    # -------------------------------------------------------------------------
    # 1. Real Variable Tests
    # -------------------------------------------------------------------------
    def test_real_variable_lifecycle(self) -> None:
        var = RealVariable.random(self.config, self.rng)
        self.assertEqual(var.dim, 4)
        self.assertEqual(var.strategy_dim, 4)
        self.assertEqual(var.adaptation_type, "lognormal")
        self.assertEqual(var.strategy_recomb, "geometric")

        # Copy independence
        copied = var.copy()
        self.assertIsNot(var.values, copied.values)
        self.assertIsNot(var.strategy, copied.strategy)
        np.testing.assert_array_equal(var.values, copied.values)

        # Adapt strategy
        init_strat = var.strategy.copy()
        var.adapt_strategy(self.rng, tau_prime=0.1, tau=0.1)
        self.assertFalse(np.allclose(var.strategy, init_strat))
        self.assertTrue(np.all(var.strategy >= var.sigma_floor))
        if var.strategy_ceiling is not None:
            self.assertTrue(np.all(var.strategy <= var.strategy_ceiling + 1e-12))

        # Mutate within bounds
        var.mutate(self.rng)
        self.assertTrue(np.all(var.values >= var.bounds[:, 0]))
        self.assertTrue(np.all(var.values <= var.bounds[:, 1]))

    def test_real_standalone_mutation_and_empty(self) -> None:
        bounds = np.array([[-2.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        vals = np.array([1.9, 0.9], dtype=np.float64)
        sigmas = np.array([0.5, 0.5], dtype=np.float64)
        mutated = mutate_real(vals, sigmas, self.rng, bounds)
        self.assertTrue(np.all(mutated >= bounds[:, 0]))
        self.assertTrue(np.all(mutated <= bounds[:, 1]))

        # Empty variable
        empty_cfg = ESConfig(n_real=0)
        empty_var = RealVariable.random(empty_cfg, self.rng)
        self.assertEqual(empty_var.dim, 0)
        empty_var.adapt_strategy(self.rng, 0.1, 0.1)
        empty_var.mutate(self.rng)
        self.assertEqual(empty_var.values.shape[0], 0)

    # -------------------------------------------------------------------------
    # 2. Count Variable Tests
    # -------------------------------------------------------------------------
    def test_count_variable_lifecycle(self) -> None:
        var = CountVariable.random(self.config, self.rng)
        self.assertEqual(var.dim, 3)
        self.assertEqual(var.adaptation_type, "lognormal")

        # Copy independence
        copied = var.copy()
        self.assertIsNot(var.values, copied.values)
        np.testing.assert_array_equal(var.values, copied.values)

        # Adapt strategy
        init_strat = var.strategy.copy()
        var.adapt_strategy(self.rng, tau_prime=0.1, tau=0.1)
        self.assertFalse(np.allclose(var.strategy, init_strat))
        self.assertTrue(np.all(var.strategy >= var.sigma_floor))

        # Mutate
        var.mutate(self.rng)
        self.assertTrue(np.all(var.values >= var.bounds[:, 0]))
        self.assertTrue(np.all(var.values <= var.bounds[:, 1]))
        self.assertEqual(var.values.dtype, np.int64)

    def test_count_standalone_mutation_and_empty(self) -> None:
        bounds = np.array([[0, 5]], dtype=np.int64)
        vals = np.array([3], dtype=np.int64)
        lambdas = np.array([1.0], dtype=np.float64)
        mutated = mutate_count(vals, lambdas, self.rng, bounds)
        self.assertTrue(np.all(mutated >= 0) and np.all(mutated <= 5))

        empty_cfg = ESConfig(n_count=0)
        empty_var = CountVariable.random(empty_cfg, self.rng)
        self.assertEqual(empty_var.dim, 0)
        empty_var.adapt_strategy(self.rng, 0.1, 0.1)
        empty_var.mutate(self.rng)
        self.assertEqual(empty_var.values.shape[0], 0)

    # -------------------------------------------------------------------------
    # 3. Binary Variable Tests
    # -------------------------------------------------------------------------
    def test_binary_variable_lifecycle(self) -> None:
        var = BinaryVariable.random(self.config, self.rng)
        self.assertEqual(var.dim, 5)
        self.assertEqual(var.adaptation_type, "logit")
        self.assertEqual(var.strategy_recomb, "arithmetic")

        # Copy
        copied = var.copy()
        self.assertIsNot(var.values, copied.values)

        # Adapt strategy
        init_strat = var.strategy.copy()
        var.adapt_strategy(self.rng, tau_prime=0.1, tau=0.1)
        self.assertFalse(np.allclose(var.strategy, init_strat))
        if var.max_logit is not None:
            self.assertTrue(np.all(var.strategy <= var.max_logit + 1e-12))

        # Mutate
        var.mutate(self.rng)
        self.assertTrue(np.all(np.isin(var.values, [0, 1])))

    def test_binary_standalone_mutation_and_empty(self) -> None:
        vals = np.array([0, 1, 0], dtype=np.uint8)
        logits = np.array([-5.0, 5.0, 0.0], dtype=np.float64)
        mutated = mutate_binary(vals, logits, self.rng)
        self.assertTrue(np.all(np.isin(mutated, [0, 1])))

        empty_cfg = ESConfig(n_binary=0)
        empty_var = BinaryVariable.random(empty_cfg, self.rng)
        self.assertEqual(empty_var.dim, 0)
        empty_var.adapt_strategy(self.rng, 0.1, 0.1)
        empty_var.mutate(self.rng)
        self.assertEqual(empty_var.values.shape[0], 0)

    # -------------------------------------------------------------------------
    # 4. Categorical Variable Tests
    # -------------------------------------------------------------------------
    def test_categorical_variable_lifecycle(self) -> None:
        var = CategoricalVariable.random(self.config, self.rng)
        self.assertEqual(var.dim, 2)
        self.assertEqual(var.strategy_dim, 7)  # 3 + 4
        self.assertEqual(var.adaptation_type, "logit")

        # Copy
        copied = var.copy()
        self.assertIsNot(var.values, copied.values)

        # Adapt strategy with zero-mean centering hook
        var.adapt_strategy(self.rng, tau_prime=0.1, tau=0.1)
        slice1 = var.strategy[:3]
        slice2 = var.strategy[3:]
        self.assertAlmostEqual(float(np.mean(slice1)), 0.0, places=6)
        self.assertAlmostEqual(float(np.mean(slice2)), 0.0, places=6)

        # Mutate
        var.mutate(self.rng)
        self.assertTrue(0 <= var.values[0] < 3)
        self.assertTrue(0 <= var.values[1] < 4)

    def test_categorical_standalone_centering_and_empty(self) -> None:
        k = np.array([2, 3], dtype=np.int64)
        logits = np.array([1.0, 3.0, 2.0, 4.0, 6.0], dtype=np.float64)
        centered = center_categorical_logits(logits, k)
        self.assertAlmostEqual(float(np.mean(centered[:2])), 0.0, places=6)
        self.assertAlmostEqual(float(np.mean(centered[2:])), 0.0, places=6)

        empty_cfg = ESConfig(n_categorical=0, categorical_k=[])
        empty_var = CategoricalVariable.random(empty_cfg, self.rng)
        self.assertEqual(empty_var.dim, 0)
        empty_var.adapt_strategy(self.rng, 0.1, 0.1)
        empty_var.mutate(self.rng)
        self.assertEqual(empty_var.values.shape[0], 0)

    # -------------------------------------------------------------------------
    # 5. Circular Variable Tests
    # -------------------------------------------------------------------------
    def test_circular_variable_lifecycle(self) -> None:
        var = CircularVariable.random(self.config, self.rng)
        self.assertEqual(var.dim, 2)
        self.assertEqual(var.adaptation_type, "lognormal")

        # Copy
        copied = var.copy()
        self.assertIsNot(var.values, copied.values)

        # Adapt
        var.adapt_strategy(self.rng, tau_prime=0.1, tau=0.1)
        self.assertTrue(np.all(var.strategy >= var.sigma_floor))

        # Mutate
        var.mutate(self.rng)
        self.assertTrue(0 <= var.values[0] < 7)
        self.assertTrue(0 <= var.values[1] < 12)

    def test_circular_standalone_mutation_and_empty(self) -> None:
        k = np.array([5], dtype=np.int64)
        vals = np.array([4], dtype=np.int64)
        lambdas = np.array([2.0], dtype=np.float64)
        mutated = mutate_circular(vals, lambdas, self.rng, k)
        self.assertTrue(0 <= mutated[0] < 5)

        empty_cfg = ESConfig(n_circular=0, circular_k=[])
        empty_var = CircularVariable.random(empty_cfg, self.rng)
        self.assertEqual(empty_var.dim, 0)
        empty_var.adapt_strategy(self.rng, 0.1, 0.1)
        empty_var.mutate(self.rng)
        self.assertEqual(empty_var.values.shape[0], 0)

    # -------------------------------------------------------------------------
    # 6. Fuzzy Variable Tests
    # -------------------------------------------------------------------------
    def test_fuzzy_variable_lifecycle(self) -> None:
        var = FuzzyVariable.random(self.config, self.rng)
        self.assertEqual(var.dim, 6)  # 2 triplets
        self.assertEqual(var.adaptation_type, "lognormal")

        # Initial monotonicity check
        for i in range(var.n_fuzzy):
            a, b, c = var.values[i * 3 : i * 3 + 3]
            self.assertTrue(0.0 <= a <= b <= c <= 1.0)

        # Copy
        copied = var.copy()
        self.assertIsNot(var.values, copied.values)

        # Adapt
        var.adapt_strategy(self.rng, tau_prime=0.1, tau=0.1)
        self.assertTrue(np.all(var.strategy >= var.sigma_floor))

        # Mutate
        var.mutate(self.rng)
        for i in range(var.n_fuzzy):
            a, b, c = var.values[i * 3 : i * 3 + 3]
            self.assertTrue(0.0 <= a <= b <= c <= 1.0)
            self.assertGreaterEqual(b, a)
            self.assertGreaterEqual(c, b)

    def test_fuzzy_standalone_mutation_and_empty(self) -> None:
        pts = np.array([0.2, 0.5, 0.8], dtype=np.float64)
        sigmas = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        mutated = mutate_fuzzy(pts, sigmas, self.rng, n_fuzzy=1)
        self.assertTrue(0.0 <= mutated[0] <= mutated[1] <= mutated[2] <= 1.0)

        empty_cfg = ESConfig(n_fuzzy=0)
        empty_var = FuzzyVariable.random(empty_cfg, self.rng)
        self.assertEqual(empty_var.dim, 0)
        empty_var.adapt_strategy(self.rng, 0.1, 0.1)
        empty_var.mutate(self.rng)
        self.assertEqual(empty_var.values.shape[0], 0)

    # -------------------------------------------------------------------------
    # 7. Variable Base Class Recombination & Flatten Tests
    # -------------------------------------------------------------------------
    def test_recombinations(self) -> None:
        v1 = RealVariable.random(self.config, self.rng)
        v2 = RealVariable.random(self.config, self.rng)

        # Recombine values
        child_vals = v1.recombine_values(v2, self.rng)
        for i in range(v1.dim):
            self.assertTrue(child_vals[i] == v1.values[i] or child_vals[i] == v2.values[i])

        # Recombine strategy (geometric)
        child_strat = v1.recombine_strategy([v1, v2], self.rng)
        self.assertEqual(child_strat.shape[0], v1.strategy_dim)
        self.assertTrue(np.all(child_strat > 0))

        # Flatten
        flat = v1.flatten()
        self.assertEqual(flat.dtype, np.float64)
        self.assertEqual(flat.shape[0], v1.dim)

    # -------------------------------------------------------------------------
    # 8. Kernels Primitives Tests
    # -------------------------------------------------------------------------
    def test_kernels_primitives(self) -> None:
        logits = np.array([-1000.0, 0.0, 1000.0])
        probs = kernels.sigmoid(logits)
        self.assertAlmostEqual(probs[0], 0.0, places=5)
        self.assertAlmostEqual(probs[1], 0.5, places=5)
        self.assertAlmostEqual(probs[2], 1.0, places=5)

        vec = np.array([1.0, 2.0, 3.0])
        sm = kernels.softmax(vec)
        self.assertAlmostEqual(float(np.sum(sm)), 1.0, places=6)
        self.assertTrue(np.all(sm > 0))


if __name__ == "__main__":
    unittest.main()
