# Monte Carlo Option Pricing Project

This project implements Monte Carlo simulation methods for pricing European call options under a Geometric Brownian Motion (GBM) model.

## Structure
- `src/gbm.py`: GBM path generation functions.
- `src/pricing.py`: Naive, antithetic, and control-variate pricing functions.
- `src/utils.py`: Shared utilities (e.g., seed, helpers).
- `Monte Carlo Project.ipynb`: Main notebook running experiments and saving plots.
- `plots/`: Folder containing saved visualization images.

## Methods Implemented
- Naive Monte Carlo
- Antithetic Variates
- Control Variates
- Error comparison vs Black–Scholes analytical price

## Requirements
See `requirements.txt` for dependencies.

