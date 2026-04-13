"""
Monte Carlo Option Pricing Package

This package implements Monte Carlo simulation methods for pricing European call options
under a Geometric Brownian Motion (GBM) model.

Modules:
- gbm: GBM path generation functions
- naive_mc: Naive Monte Carlo pricing
- antithetic_mc: Antithetic variates pricing
- control_variate_mc: Control variate pricing
- utils: Shared utilities and validation
"""

from .gbm import generate_gbm_paths
from .naive_mc import price_call_naive
from .antithetic_mc import price_call_antithetic
from .control_variate_mc import price_call_control_var
from .utils import validate_parameters, get_rng, black_scholes_call

__all__ = [
    'generate_gbm_paths',
    'price_call_naive', 
    'price_call_antithetic',
    'price_call_control_var',
    'validate_parameters',
    'get_rng',
    'black_scholes_call'
]
