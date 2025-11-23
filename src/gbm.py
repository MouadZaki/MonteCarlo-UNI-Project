import numpy as np

def generate_gbm_paths(S0, mu, sigma, T, steps, paths, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    drift = (mu - 0.5 * sigma**2) * dt
    shock_scale = sigma * np.sqrt(dt)

    Z = np.random.randn(paths, steps)
    increments = drift + shock_scale * Z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)

    S = np.concatenate([S0 * np.ones((paths, 1)), S], axis=1)
    return S
