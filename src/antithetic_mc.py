import numpy as np

def price_call_antithetic(S0, K, r, mu, sigma, T, steps, paths, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    drift = (mu - 0.5 * sigma**2) * dt
    shock_scale = sigma * np.sqrt(dt)

    half = paths // 2
    Z = np.random.randn(half, steps)
    Z_neg = -Z

    def evolve(Zmat):
        increments = drift + shock_scale * Zmat
        log_paths = np.cumsum(increments, axis=1)
        S = S0 * np.exp(log_paths)
        return S[:, -1]

    ST_pos = evolve(Z)
    ST_neg = evolve(Z_neg)

    ST = 0.5 * (ST_pos + ST_neg)
    payoffs = np.maximum(ST - K, 0)
    discounted = np.exp(-r * T) * payoffs
    return discounted.mean()
