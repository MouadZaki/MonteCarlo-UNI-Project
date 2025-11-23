import numpy as np
from .gbm import generate_gbm_paths

def price_call_control_var(S0, K, r, mu, sigma, T, steps, paths, seed=None):
    S = generate_gbm_paths(S0, mu, sigma, T, steps, paths, seed)
    ST = S[:, -1]

    payoffs = np.maximum(ST - K, 0.0)
    df = np.exp(-r * T)

    X = ST
    EX = S0 * np.exp(mu * T)

    cov = np.cov(payoffs, X, ddof=1)[0,1]
    var = np.var(X, ddof=1)
    b = cov / var

    corrected = payoffs - b * (X - EX)
    price = df * corrected.mean()
    return price
