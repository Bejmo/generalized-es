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
    membership,
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

    def test_binary_logit_bounds_enforce_mutation_rates(self) -> None:
        """Verify that logit clamping strictly enforces mutation flip probability in [1%, 50%]."""
        cfg = ESConfig(n_binary=10, min_logit_binary=-4.6, max_logit_binary=0.0)
        var = BinaryVariable.random(cfg, self.rng)
        # Attempt to drive logits to extreme values
        var.strategy[:] = 100.0
        var.adapt_strategy(self.rng, tau_prime=0.0, tau=0.0)
        self.assertTrue(np.all(var.strategy <= 0.0 + 1e-12))
        probs_max = kernels.sigmoid(var.strategy)
        self.assertTrue(np.all(probs_max <= 0.5 + 1e-12))

        var.strategy[:] = -100.0
        var.adapt_strategy(self.rng, tau_prime=0.0, tau=0.0)
        self.assertTrue(np.all(var.strategy >= -4.6 - 1e-12))
        probs_min = kernels.sigmoid(var.strategy)
        self.assertTrue(np.all(probs_min >= kernels.sigmoid(-4.6) - 1e-6))

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
        self.assertEqual(empty_var.values.shape[0], 0)

    def test_categorical_gauge_centering_prevents_translation_drift(self) -> None:
        """Verify that zero-mean centering projects arbitrary translation drifts to zero-mean without altering Softmax."""
        k = np.array([4], dtype=np.int64)
        logits = np.array([1.5, 0.5, -0.5, -1.5], dtype=np.float64)
        probs_original = kernels.softmax(logits)

        # Inject huge constant drift (gauge freedom: +500 along null vector)
        drifted = logits + 500.0
        # Centering removes the +500 drift
        centered = center_categorical_logits(drifted, k)
        self.assertAlmostEqual(float(np.mean(centered)), 0.0, places=10)
        np.testing.assert_allclose(centered, logits, atol=1e-10)

        probs_centered = kernels.softmax(centered)
        np.testing.assert_allclose(probs_centered, probs_original, atol=1e-10)

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
        self.assertEqual(empty_var.values.shape[0], 0)

    def test_circular_modular_wrap_around_boundary(self) -> None:
        """Verify that circular modular mutation crosses the periodic boundary and evaluates geodesic distance."""
        # 1. Modular wrap-around: 11 + 1 mod 12 = 0
        k = np.array([12], dtype=np.int64)
        vals = np.array([11], dtype=np.int64)
        # Mock deterministic Skellam step of +1
        delta_plus_one = 1
        mutated_val = (vals + delta_plus_one) % k
        self.assertEqual(int(mutated_val[0]), 0)

        # 2. Geodesic distance formula: distance between 11 and 1 on Z_12 is 2 (squared distance = 4)
        var = CircularVariable(values=np.array([11], dtype=np.int64), strategy=np.array([1.0]), circular_k=k)
        target = np.array([1], dtype=np.int64)
        dist_sq = var.distance_squared(target)
        self.assertEqual(dist_sq, 4.0)

    # -------------------------------------------------------------------------
    # 6. Fuzzy Variable Tests
    # -------------------------------------------------------------------------
    def test_fuzzy_variable_lifecycle(self) -> None:
        var = FuzzyVariable.random(self.config, self.rng)
        self.assertEqual(var.dim, 4)  # 2 fuzzy vars * 2 dims (m, d)
        self.assertEqual(var.num_variables, 2)
        self.assertEqual(var.dims_per_var, 2)
        self.assertEqual(var.adaptation_type, "lognormal")

        # Initial bounds check
        m = var.values[0::2]
        d = var.values[1::2]
        self.assertTrue(np.all((m >= 0.0) & (m <= 1.0)))
        self.assertTrue(np.all((d >= 0.05) & (d <= 0.4)))

        # Triplet decode
        triplets = var.decode_triplets()
        self.assertEqual(triplets.shape[0], 6)
        for i in range(var.n_fuzzy):
            a, b, c = triplets[i * 3 : i * 3 + 3]
            self.assertTrue(0.0 <= a <= b <= c <= 1.0)

        # Copy
        copied = var.copy()
        self.assertIsNot(var.values, copied.values)

        # Adapt
        var.adapt_strategy(self.rng, tau_prime=0.1, tau=0.1)
        self.assertTrue(np.all(var.strategy >= var.sigma_floor))

        # Mutate
        var.mutate(self.rng)
        m_mut = var.values[0::2]
        d_mut = var.values[1::2]
        self.assertTrue(np.all((m_mut >= 0.0) & (m_mut <= 1.0)))
        self.assertTrue(np.all(d_mut >= var.fuzzy_epsilon))
        self.assertTrue(np.all(d_mut <= var.d_max))

    def test_fuzzy_standalone_mutation_and_empty(self) -> None:
        # 1 fuzzy var: [m=0.5, d=0.3]
        pts = np.array([0.5, 0.3], dtype=np.float64)
        sigmas = np.array([0.1, 0.1], dtype=np.float64)
        mutated = mutate_fuzzy(pts, sigmas, self.rng, n_fuzzy=1, fuzzy_epsilon=1e-4, d_max=1.0)
        self.assertTrue(0.0 <= mutated[0] <= 1.0)
        self.assertTrue(1e-4 <= mutated[1] <= 1.0)

        empty_cfg = ESConfig(n_fuzzy=0)
        empty_var = FuzzyVariable.random(empty_cfg, self.rng)
        self.assertEqual(empty_var.dim, 0)
        self.assertEqual(empty_var.num_variables, 0)
        empty_var.adapt_strategy(self.rng, 0.1, 0.1)
        empty_var.mutate(self.rng)
        self.assertEqual(empty_var.values.shape[0], 0)

    def test_dimensions_interface_across_variables(self) -> None:
        cfg = self.config
        real = RealVariable.random(cfg, self.rng)
        count = CountVariable.random(cfg, self.rng)
        binary = BinaryVariable.random(cfg, self.rng)
        cat = CategoricalVariable.random(cfg, self.rng)
        circ = CircularVariable.random(cfg, self.rng)
        fuzzy = FuzzyVariable.random(cfg, self.rng)

        # Check dims_per_var
        self.assertEqual(real.dims_per_var, 1)
        self.assertEqual(count.dims_per_var, 1)
        self.assertEqual(binary.dims_per_var, 1)
        self.assertEqual(cat.dims_per_var, 1)
        self.assertEqual(circ.dims_per_var, 1)
        self.assertEqual(fuzzy.dims_per_var, 2)

        # Check num_variables
        self.assertEqual(real.num_variables, cfg.n_real)
        self.assertEqual(count.num_variables, cfg.n_count)
        self.assertEqual(binary.num_variables, cfg.n_binary)
        self.assertEqual(cat.num_variables, cfg.n_categorical)
        self.assertEqual(circ.num_variables, cfg.n_circular)
        self.assertEqual(fuzzy.num_variables, cfg.n_fuzzy)

        # Check dim
        self.assertEqual(real.dim, cfg.n_real)
        self.assertEqual(fuzzy.dim, cfg.n_fuzzy * 2)

    def test_distance_squared_interface(self) -> None:
        cfg = self.config
        real = RealVariable.random(cfg, self.rng)
        self.assertAlmostEqual(real.distance_squared(real), 0.0)

        count = CountVariable.random(cfg, self.rng)
        self.assertAlmostEqual(count.distance_squared(count), 0.0)

        binary = BinaryVariable.random(cfg, self.rng)
        self.assertAlmostEqual(binary.distance_squared(binary), 0.0)

        cat = CategoricalVariable.random(cfg, self.rng)
        self.assertAlmostEqual(cat.distance_squared(cat), 0.0)

        circ = CircularVariable.random(cfg, self.rng)
        self.assertAlmostEqual(circ.distance_squared(circ), 0.0)

        fuzzy = FuzzyVariable.random(cfg, self.rng)
        self.assertAlmostEqual(fuzzy.distance_squared(fuzzy), 0.0)

        # Test fuzzy distance formula explicitly: 3*(m1-m2)^2 + 2*(d1-d2)^2
        v1 = FuzzyVariable(np.array([0.5, 0.3]), np.array([0.05, 0.05]), n_fuzzy=1)
        target = np.array([0.2, 0.1])
        self.assertAlmostEqual(v1.distance_squared(target), 0.35)

    def test_fuzzy_atomic_recombination(self) -> None:
        v1 = FuzzyVariable(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.05, 0.05, 0.05, 0.05]), n_fuzzy=2)
        v2 = FuzzyVariable(np.array([0.6, 0.7, 0.8, 0.9]), np.array([0.05, 0.05, 0.05, 0.05]), n_fuzzy=2)
        for _ in range(20):
            child = v1.recombine_values(v2, self.rng)
            pair0 = child[0:2]
            self.assertTrue(np.array_equal(pair0, v1.values[0:2]) or np.array_equal(pair0, v2.values[0:2]))
            pair1 = child[2:4]
            self.assertTrue(np.array_equal(pair1, v1.values[2:4]) or np.array_equal(pair1, v2.values[2:4]))

    def test_fuzzy_membership_properties(self) -> None:
        m, d = 0.5, 0.2
        # Peak
        self.assertAlmostEqual(float(membership(0.5, m, d)), 1.0)
        # Half-way
        self.assertAlmostEqual(float(membership(0.4, m, d)), 0.5)
        self.assertAlmostEqual(float(membership(0.6, m, d)), 0.5)
        # Zero at edges
        self.assertAlmostEqual(float(membership(0.3, m, d)), 0.0)
        self.assertAlmostEqual(float(membership(0.7, m, d)), 0.0)
        # Outside support
        self.assertAlmostEqual(float(membership(0.1, m, d)), 0.0)
        self.assertAlmostEqual(float(membership(0.9, m, d)), 0.0)

        # Vectorized evaluation
        grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        expected = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        np.testing.assert_allclose(membership(grid, m, d), expected, atol=1e-12)

    def test_fuzzy_mutation_stress_invariants(self) -> None:
        pts = np.array([0.5, 0.2, 0.1, 0.05], dtype=np.float64)
        sigmas = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)  # Very large perturbations
        eps = 1e-4
        d_max = 1.0

        for _ in range(2000):
            pts = mutate_fuzzy(pts, sigmas, self.rng, n_fuzzy=2, fuzzy_epsilon=eps, d_max=d_max)
            m = pts[0::2]
            d = pts[1::2]
            # Center invariant
            self.assertTrue(np.all((m >= 0.0) & (m <= 1.0)))
            # Dispersion non-collapse and ceiling invariant
            self.assertTrue(np.all(d >= eps))
            self.assertTrue(np.all(d <= d_max))

    def test_fuzzy_absolute_value_reflection(self) -> None:
        # Deterministic check for reflection when d + perturbation < 0
        # If d = 0.05 and perturbation = -0.20, raw = -0.15, reflected = |-0.15| = 0.15
        pts = np.array([0.5, 0.05], dtype=np.float64)
        mock_rng = np.random.default_rng(1234)
        # Mutate multiple times to ensure negative excursions reflect to positive
        for _ in range(50):
            mutated = mutate_fuzzy(pts, np.array([0.01, 0.5]), mock_rng, n_fuzzy=1, fuzzy_epsilon=1e-4, d_max=1.0)
            self.assertGreaterEqual(mutated[1], 1e-4)
            self.assertLessEqual(mutated[1], 1.0)

    def test_fuzzy_acute_compression_non_collapse(self) -> None:
        """
        Verify that under acute compression toward a crisp point (d_target = 0.0),
        the dispersion floor d >= epsilon prevents collapse into a crisp real scalar,
        strictly maintaining support width 2*d >= 2*epsilon.
        """
        from src.es_model.generalized_es import GeneralizedES
        from src.es_model.individual import unpack_genome

        eps = 1e-4
        cfg = ESConfig(
            mu=10, lam=50, max_generations=50,
            n_real=0, n_count=0, n_binary=0, n_categorical=0, categorical_k=[],
            n_fuzzy=2, n_circular=0, circular_k=[],
            fuzzy_epsilon=eps,
            seed=42
        )
        # Target with zero dispersion (acute compression)
        t_crisp = np.array([0.5, 0.0, 0.3, 0.0], dtype=np.float64)
        eval_fn = lambda ind: float(np.sum((unpack_genome(ind, cfg).fuzzy - t_crisp) ** 2))

        es = GeneralizedES(config=cfg, seed=42)
        es.init_population(eval_fn)

        for _ in range(50):
            es.step(eval_fn)
            best_p = es.parents[0]
            vals = unpack_genome(best_p, cfg).fuzzy
            dispersions = vals[1::2]
            # Verify dispersion never breaches epsilon floor
            self.assertTrue(np.all(dispersions >= eps), f"Dispersion breached floor: {dispersions} < {eps}")
            # Verify support width 2*d >= 2*eps
            support_widths = 2.0 * dispersions
            self.assertTrue(np.all(support_widths >= 2.0 * eps))


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
