import numpy as np
from .gbm import generate_gbm_paths

def price_call_naive(S0, K, r, mu, sigma, T, steps, paths, seed=None):
    S = generate_gbm_paths(S0, mu, sigma, T, steps, paths, seed)
    ST = S[:, -1]

    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    return discounted.mean()
