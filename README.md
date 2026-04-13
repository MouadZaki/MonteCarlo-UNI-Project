# Monte Carlo Option Pricing Project

This project implements Monte Carlo simulation methods for pricing European call options under a Geometric Brownian Motion (GBM) model.

## Project Structure

```
MonteCarlo UNI Project/
|
+-- src/                          # Source code modules
|   |-- __init__.py               # Package initialization
|   |-- gbm.py                    # GBM path generation
|   |-- naive_mc.py               # Naive Monte Carlo pricing
|   |-- antithetic_mc.py          # Antithetic variates pricing
|   |-- control_variate_mc.py     # Control variates pricing
|   |-- utils.py                  # Shared utilities and validation
|
+-- notebook/                     # Jupyter notebooks
|   |-- Monte_Carlo_Analysis.ipynb    # Complete analysis notebook
|
+-- tests/                        # Unit tests
|   |-- test_mc_pricing.py        # Basic tests for all modules
|
+-- plots/                        # Generated plots (empty initially)
|
+-- README.md                     # This file
+-- requirements.txt              # Dependencies
```

## Features

### Pricing Methods
- **Naive Monte Carlo**: Standard Monte Carlo simulation
- **Antithetic Variates**: Variance reduction using antithetic sampling
- **Control Variates**: Variance reduction using terminal stock price as control

### Implementation Features
- **Input Validation**: Comprehensive parameter checking
- **Standard Error Calculation**: All methods return standard error estimates
- **Consistent RNG**: Standardized random number generation with reproducible seeds
- **Error Handling**: Robust error handling and edge case management
- **Modular Design**: Clean separation of concerns across modules
- **Documentation**: Comprehensive docstrings and type hints

### Risk Analysis
- Probability of exceeding loss thresholds
- Position sizing recommendations
- Risk metrics and visualizations

## Quick Start

```python
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src import price_call_naive, price_call_antithetic, price_call_control_var, black_scholes_call

# Define parameters
S0, K, r, mu, sigma, T, steps, paths = 100.0, 100.0, 0.03, 0.05, 0.2, 1.0, 252, 10000

# Calculate prices
naive_price, naive_se, _ = price_call_naive(S0, K, r, mu, sigma, T, steps, paths, seed=42)
antithetic_price, antithetic_se, _ = price_call_antithetic(S0, K, r, mu, sigma, T, steps, paths, seed=42)
cv_price, cv_se, _, _ = price_call_control_var(S0, K, r, mu, sigma, T, steps, paths, seed=42)

# Compare with Black-Scholes
bs_price = black_scholes_call(S0, K, r, sigma, T)
```

## Usage Examples

### Basic Pricing
```python
from src import price_call_naive

price, se, payoffs = price_call_naive(
    S0=100, K=100, r=0.03, mu=0.05, sigma=0.2, 
    T=1.0, steps=252, paths=10000, seed=42
)
```

### Variance Reduction Comparison
```python
from src import price_call_naive, price_call_antithetic, price_call_control_var

# Compare methods
naive_price, naive_se, _ = price_call_naive(...)
antithetic_price, antithetic_se, _ = price_call_antithetic(...)
cv_price, cv_se, _, _ = price_call_control_var(...)

# Calculate variance reduction factors
vrf_anti = (naive_se**2) / (antithetic_se**2)
vrf_cv = (naive_se**2) / (cv_se**2)
```

### Risk Analysis
```python
from src.gbm import generate_gbm_paths
from src.utils import get_rng

def calculate_risk_metrics(q, loss_threshold, S0, mu, sigma, T, paths, steps):
    rng = get_rng(42)
    S_T = generate_gbm_paths(S0, mu, sigma, T, steps, paths, 42)[:, -1]
    pnl = q * (S_T - S0)
    return np.mean(pnl < -loss_threshold)
```

## Performance Results

Based on testing with 50,000 paths:
- **Control Variates**: ~6-7x variance reduction vs naive
- **Antithetic Variates**: ~1x variance reduction vs naive
- **All methods**: Converge to Black-Scholes price as paths increase

## Testing

Run the test suite:
```bash
python -m pytest tests/test_mc_pricing.py -v
```

## Requirements

See `requirements.txt` for dependencies. Key packages:
- numpy
- scipy
- pandas
- matplotlib
- pytest (for testing)

## Notes

- All pricing functions return standard error estimates for statistical accuracy assessment
- Input validation ensures robust parameter handling
- Random number generation is consistent and reproducible across methods
- The control variate method uses terminal stock price as the control variable
