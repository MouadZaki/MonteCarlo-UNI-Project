import numpy as np
from .gbm import generate_gbm_paths
from .utils import validate_parameters

def price_call_control_var(S0, K, r, mu, sigma, T, steps, paths, seed=None):
    """
    Price European call option using control variates Monte Carlo simulation.
    
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
        tuple: (price, standard_error, payoffs, adjusted_payoffs)
    """
    validate_parameters(S0, K, r, mu, sigma, T, steps, paths)
    
    S = generate_gbm_paths(S0, mu, sigma, T, steps, paths, seed)
    ST = S[:, -1]

    payoffs = np.maximum(ST - K, 0.0)
    df = np.exp(-r * T)

    # Use terminal stock price as control variate
    X = ST
    EX = S0 * np.exp(mu * T)

    cov = np.cov(payoffs, X, ddof=1)[0,1]
    var_X = np.var(X, ddof=1)
    
    # Handle case where variance is zero
    beta = cov / var_X if var_X > 0 else 0.0

    adjusted_payoffs = payoffs - beta * (X - EX)
    price = df * adjusted_payoffs.mean()
    standard_error = df * adjusted_payoffs.std(ddof=1) / np.sqrt(paths)
    
    return price, standard_error, payoffs, adjusted_payoffs
