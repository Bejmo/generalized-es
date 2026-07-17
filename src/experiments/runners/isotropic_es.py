"""
Runner wrapper for Isotropic ES (Simple Self-Adaptive).
Uses a single global sigma for the entire individual (n_sigma = 1, n_alpha = 0).
"""

import numpy as np

class IsotropicIndividual:
    def __init__(self, x, sigma, fitness=float('inf')):
        self.x = x.copy()
        self.sigma = sigma
        self.fitness = fitness

def run(fn, dim, bounds, seed, max_evals, *, mu=15, lam=100, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    # Initialize learning rate for global sigma
    # tau0 = 1 / sqrt(2 * dim) (standard global learning rate)
    tau0 = 1.0 / np.sqrt(2.0 * dim)
    
    # Initial sigma: 0.5 (matches Prototype 1 initialization)
    init_sigma = 0.5
    sigma_floor = 1e-6
    
    # Create initial population
    parents = []
    trace = []
    best_so_far = float('inf')
    
    for _ in range(mu):
        x = rng.uniform(bounds[0], bounds[1], size=dim)
        ind = IsotropicIndividual(x, init_sigma)
        ind.fitness = fn(ind.x)
        best_so_far = min(best_so_far, ind.fitness)
        trace.append(best_so_far)
        parents.append(ind)
        if len(trace) >= max_evals:
            break
            
    while len(trace) < max_evals:
        children = []
        for _ in range(lam):
            # Select parent uniformly
            parent = parents[rng.integers(0, mu)]
            
            # Adapt sigma (log-normal)
            # sigma_new = sigma * exp(tau0 * N(0,1))
            sigma_new = parent.sigma * np.exp(tau0 * rng.standard_normal())
            sigma_new = max(sigma_new, sigma_floor)
            
            # Mutate x
            x_new = parent.x + sigma_new * rng.standard_normal(dim)
            x_new = np.clip(x_new, bounds[0], bounds[1])
            
            child = IsotropicIndividual(x_new, sigma_new)
            child.fitness = fn(child.x)
            best_so_far = min(best_so_far, child.fitness)
            trace.append(best_so_far)
            children.append(child)
            
            if len(trace) >= max_evals:
                break
                
        # (mu, lambda)-ES selection: select best mu from children
        # Stable sort children by fitness
        children.sort(key=lambda ind: ind.fitness)
        parents = children[:mu]
        
    return np.array(trace[:max_evals])
