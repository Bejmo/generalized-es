"""
Benchmark functions.
"""

import numpy as np

def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    n = len(x)
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x):
    n = len(x)
    sum_sq = np.sum(x ** 2) / n
    sum_cos = np.sum(np.cos(2.0 * np.pi * x)) / n
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20.0 + np.e)

# @TODO Change the limit
limit = 10.0
BENCHMARKS = {
    'Sphere':     {'fn': sphere,     'bounds': (-limit, limit), 'optimum': np.zeros},
    'Rastrigin':  {'fn': rastrigin,  'bounds': (-limit, limit), 'optimum': np.zeros},
    'Rosenbrock': {'fn': rosenbrock, 'bounds': (-limit, limit),  'optimum': np.ones},
    'Ackley':     {'fn': ackley,     'bounds': (-5.0, 5.0),   'optimum': np.zeros}, # Limit must be (-5,5), otherwise the function is not defined at the optimum.
}
