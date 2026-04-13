import numpy as np
from typing import Optional, Union

def validate_parameters(S0: float, K: float, r: float, mu: float, sigma: float, 
                       T: float, steps: int, paths: int) -> None:
    """
    Validate input parameters for Monte Carlo simulation.
    
    Args:
        S0: Initial stock price (must be > 0)
        K: Strike price (must be > 0)
        r: Risk-free rate (can be any real number)
        mu: Drift rate (can be any real number)
        sigma: Volatility (must be >= 0)
        T: Time to maturity (must be > 0)
        steps: Number of time steps (must be > 0)
        paths: Number of simulation paths (must be > 0)
    
    Raises:
        ValueError: If any parameter is invalid
    """
    if S0 <= 0:
        raise ValueError("Initial stock price S0 must be positive")
    if K <= 0:
        raise ValueError("Strike price K must be positive")
    if sigma < 0:
        raise ValueError("Volatility sigma must be non-negative")
    if T <= 0:
        raise ValueError("Time to maturity T must be positive")
    if steps <= 0:
        raise ValueError("Number of steps must be positive")
    if paths <= 0:
        raise ValueError("Number of paths must be positive")
    if not isinstance(steps, int) or not isinstance(paths, int):
        raise ValueError("Steps and paths must be integers")

def get_rng(seed: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    """
    Get a random number generator with consistent seeding.
    
    Args:
        seed: Random seed or Generator instance. If None, uses default RNG.
    
    Returns:
        numpy.random.Generator instance
    """
    if isinstance(seed, np.random.Generator):
        return seed
    elif seed is not None:
        return np.random.default_rng(seed)
    else:
        return np.random.default_rng()

def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Black-Scholes price for European call option.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
    
    Returns:
        Black-Scholes call price
    """
    from scipy.stats import norm
    
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
