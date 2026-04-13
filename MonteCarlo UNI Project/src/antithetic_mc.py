import numpy as np
from .utils import validate_parameters, get_rng

def price_call_antithetic(S0, K, r, mu, sigma, T, steps, paths, seed=None):
    """
    Price European call option using antithetic variates Monte Carlo simulation.
    
    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        mu: Drift rate
        sigma: Volatility
        T: Time to maturity
        steps: Number of time steps
        paths: Number of simulation paths (must be even)
        seed: Random seed or Generator instance
    
    Returns:
        tuple: (price, standard_error, payoffs)
    """
    validate_parameters(S0, K, r, mu, sigma, T, steps, paths)
    
    if paths % 2 != 0:
        paths += 1
    
    rng = get_rng(seed)
    dt = T / steps
    drift = (mu - 0.5 * sigma**2) * dt
    shock_scale = sigma * np.sqrt(dt)

    half = paths // 2
    Z = rng.standard_normal((half, steps))
    Z_neg = -Z

    def evolve(Zmat):
        increments = drift + shock_scale * Zmat
        log_paths = np.cumsum(increments, axis=1)
        S = S0 * np.exp(log_paths)
        return S[:, -1]

    ST_pos = evolve(Z)
    ST_neg = evolve(Z_neg)

    ST = 0.5 * (ST_pos + ST_neg)
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    
    price = discounted.mean()
    standard_error = np.exp(-r * T) * payoffs.std(ddof=1) / np.sqrt(paths)
    
    return price, standard_error, payoffs
