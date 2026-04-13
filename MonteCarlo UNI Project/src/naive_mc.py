import numpy as np
from .gbm import generate_gbm_paths
from .utils import validate_parameters

def price_call_naive(S0, K, r, mu, sigma, T, steps, paths, seed=None):
    """
    Price European call option using naive Monte Carlo simulation.
    
    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        mu: Drift rate
        sigma: Volatility
        T: Time to maturity
        steps: Number of time steps
        paths: Number of simulation paths
        seed: Random seed or Generator instance
    
    Returns:
        tuple: (price, standard_error, payoffs)
    """
    validate_parameters(S0, K, r, mu, sigma, T, steps, paths)
    
    S = generate_gbm_paths(S0, mu, sigma, T, steps, paths, seed)
    ST = S[:, -1]

    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    
    price = discounted.mean()
    standard_error = np.exp(-r * T) * payoffs.std(ddof=1) / np.sqrt(paths)
    
    return price, standard_error, payoffs
