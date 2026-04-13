import numpy as np
from .utils import validate_parameters, get_rng

def generate_gbm_paths(S0, mu, sigma, T, steps, paths, seed=None):
    """
    Generate Geometric Brownian Motion paths.
    
    Args:
        S0: Initial stock price
        mu: Drift rate
        sigma: Volatility
        T: Time to maturity
        steps: Number of time steps
        paths: Number of simulation paths
        seed: Random seed or Generator instance
    
    Returns:
        Array of shape (paths, steps+1) containing GBM paths
    """
    validate_parameters(S0, 1.0, 0.0, mu, sigma, T, steps, paths)
    
    rng = get_rng(seed)
    dt = T / steps
    drift = (mu - 0.5 * sigma**2) * dt
    shock_scale = sigma * np.sqrt(dt)

    Z = rng.standard_normal((paths, steps))
    increments = drift + shock_scale * Z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)

    S = np.concatenate([S0 * np.ones((paths, 1)), S], axis=1)
    return S
