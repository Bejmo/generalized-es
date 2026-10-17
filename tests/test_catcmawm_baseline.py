"""
test_catcmawm_baseline.py: Unit tests for CatCMAwM baseline adapter and invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
from cmaes import CatCMAwM

from src.es_model.config import ESConfig
from src.benchmarks.generalized_functions import generalized_sphere, get_default_targets
from experiments.experiment_3.runners.run_exp3_baselines import (
    CONFIG,
    build_catcmawm_spaces,
    decode_catcmawm_solution,
    run_catcmawm_seed,
)


def test_catcmawm_initialization():
    """Verify explicit initialization parameters and shapes for CatCMAwM."""
    spaces = build_catcmawm_spaces(CONFIG)
    x_space = spaces["x_space"]
    z_space = spaces["z_space"]
    c_space = spaces["c_space"]
    mean = spaces["mean"]
    sigma = spaces["sigma"]
    cat_param = spaces["cat_param"]

    # 5 real + 4 fuzzy = 9 continuous dimensions
    assert len(x_space) == 9
    assert x_space[0] == [-5.12, 5.12]
    # Fuzzy bounds: [0, 1] for center, [1e-4, 1.0] for dispersion
    assert x_space[5] == [0.0, 1.0]
    assert x_space[6] == [1e-4, 1.0]
    assert x_space[7] == [0.0, 1.0]
    assert x_space[8] == [1e-4, 1.0]

    # 3 count dimensions with domain [0, 20]
    assert len(z_space) == 3
    for row in z_space:
        assert len(row) == 21
        assert row[0] == 0 and row[-1] == 20

    # 4 binary + 2 categorical + 2 circular = 8 categorical dimensions
    assert c_space == [2, 2, 2, 2, 3, 4, 7, 12]

    # Mean vector length: Nmi = 9 continuous + 3 integer = 12
    assert len(mean) == 12
    assert np.allclose(mean[:5], 0.0)
    assert np.allclose(mean[5:9], [0.5, 0.50005, 0.5, 0.50005])
    assert np.allclose(mean[9:12], 10.0)

    # Step size
    assert sigma == 1.0

    # Categorical probability matrix: shape (8, 12)
    assert cat_param.shape == (8, 12)
    for i, k in enumerate(c_space):
        assert np.isclose(np.sum(cat_param[i, :k]), 1.0)
        assert np.all(cat_param[i, k:] == 0.0)


def test_phenotype_decoding_invariants():
    """Verify that decoded solutions strictly respect all domain invariants."""
    spaces = build_catcmawm_spaces(CONFIG)
    optimizer = CatCMAwM(
        x_space=spaces["x_space"],
        z_space=spaces["z_space"],
        c_space=spaces["c_space"],
        mean=spaces["mean"],
        sigma=spaces["sigma"],
        cat_param=spaces["cat_param"],
        seed=123,
    )

    targets = get_default_targets(CONFIG)

    for _ in range(50):
        sol = optimizer.ask()
        vec = decode_catcmawm_solution(sol, CONFIG)

        assert len(vec) == 20, f"Expected 20D phenotype, got {len(vec)}"

        # 1. Real: [-5.12, 5.12]
        assert np.all(vec[0:5] >= -5.12) and np.all(vec[0:5] <= 5.12)

        # 2. Count: integers in [0, 20]
        assert np.all(np.equal(np.mod(vec[5:8], 1), 0))
        assert np.all(vec[5:8] >= 0) and np.all(vec[5:8] <= 20)

        # 3. Binary: {0, 1}
        assert np.all(np.isin(vec[8:12], [0, 1]))

        # 4. Categorical: c0 in {0, 1, 2}, c1 in {0, 1, 2, 3}
        assert vec[12] in [0, 1, 2]
        assert vec[13] in [0, 1, 2, 3]

        # 5. Fuzzy: m in [0, 1], d in [1e-4, 1.0]
        assert 0.0 <= vec[14] <= 1.0  # m1
        assert 1e-4 <= vec[15] <= 1.0  # d1
        assert 0.0 <= vec[16] <= 1.0  # m2
        assert 1e-4 <= vec[17] <= 1.0  # d2

        # 6. Circular: c0 in {0..6}, c1 in {0..11}
        assert 0 <= vec[18] <= 6
        assert 0 <= vec[19] <= 11

        # Evaluation should execute cleanly without NaN or Inf
        cost = generalized_sphere(vec, config=CONFIG, targets=targets)
        assert np.isfinite(cost)
        assert cost >= 0.0


def test_evaluation_budget_invariance():
    """Verify that execution consumes exactly the allocated budget and trace is monotonic."""
    targets = get_default_targets(CONFIG)
    budget = 500
    seed = 42

    f_best, trace = run_catcmawm_seed(
        fn=generalized_sphere,
        seed=seed,
        max_evals=budget,
        targets=targets,
    )

    assert len(trace) == budget, f"Expected {budget} evaluations in trace, got {len(trace)}"
    assert trace[-1] == f_best, "Last trace entry must match returned best fitness"
    # Best-so-far trace must be monotonically non-increasing
    assert np.all(np.diff(trace) <= 1e-12), "Trace should be monotonically non-increasing"
