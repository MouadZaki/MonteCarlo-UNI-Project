"""
Basic tests for Monte Carlo option pricing modules.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import price_call_naive, price_call_antithetic, price_call_control_var, black_scholes_call, validate_parameters


class TestValidation:
    """Test parameter validation."""
    
    def test_valid_parameters(self):
        """Should not raise for valid parameters."""
        validate_parameters(100, 100, 0.03, 0.05, 0.2, 1.0, 252, 10000)
    
    def test_invalid_S0(self):
        """Should raise for negative S0."""
        with pytest.raises(ValueError):
            validate_parameters(-100, 100, 0.03, 0.05, 0.2, 1.0, 252, 10000)
    
    def test_invalid_K(self):
        """Should raise for negative K."""
        with pytest.raises(ValueError):
            validate_parameters(100, -100, 0.03, 0.05, 0.2, 1.0, 252, 10000)
    
    def test_invalid_sigma(self):
        """Should raise for negative sigma."""
        with pytest.raises(ValueError):
            validate_parameters(100, 100, 0.03, 0.05, -0.2, 1.0, 252, 10000)
    
    def test_invalid_T(self):
        """Should raise for non-positive T."""
        with pytest.raises(ValueError):
            validate_parameters(100, 100, 0.03, 0.05, 0.2, 0, 252, 10000)
    
    def test_invalid_steps(self):
        """Should raise for non-positive steps."""
        with pytest.raises(ValueError):
            validate_parameters(100, 100, 0.03, 0.05, 0.2, 1.0, 0, 10000)
    
    def test_invalid_paths(self):
        """Should raise for non-positive paths."""
        with pytest.raises(ValueError):
            validate_parameters(100, 100, 0.03, 0.05, 0.2, 1.0, 252, 0)


class TestPricingMethods:
    """Test Monte Carlo pricing methods."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.S0, self.K, self.r, self.mu, self.sigma = 100.0, 100.0, 0.03, 0.05, 0.2
        self.T, self.steps, self.paths = 1.0, 252, 10000
        self.seed = 42
    
    def test_naive_mc_returns(self):
        """Test naive MC returns correct tuple."""
        result = price_call_naive(self.S0, self.K, self.r, self.mu, self.sigma, 
                                 self.T, self.steps, self.paths, self.seed)
        
        assert len(result) == 3
        price, se, payoffs = result
        
        assert isinstance(price, (int, float))
        assert isinstance(se, (int, float))
        assert isinstance(payoffs, np.ndarray)
        assert len(payoffs) == self.paths
        assert price >= 0  # Price should be non-negative
        assert se >= 0    # Standard error should be non-negative
    
    def test_antithetic_mc_returns(self):
        """Test antithetic MC returns correct tuple."""
        result = price_call_antithetic(self.S0, self.K, self.r, self.mu, self.sigma,
                                      self.T, self.steps, self.paths, self.seed)
        
        assert len(result) == 3
        price, se, payoffs = result
        
        assert isinstance(price, (int, float))
        assert isinstance(se, (int, float))
        assert isinstance(payoffs, np.ndarray)
        assert price >= 0
        assert se >= 0
    
    def test_control_variate_mc_returns(self):
        """Test control variate MC returns correct tuple."""
        result = price_call_control_var(self.S0, self.K, self.r, self.mu, self.sigma,
                                       self.T, self.steps, self.paths, self.seed)
        
        assert len(result) == 4
        price, se, payoffs, adjusted_payoffs = result
        
        assert isinstance(price, (int, float))
        assert isinstance(se, (int, float))
        assert isinstance(payoffs, np.ndarray)
        assert isinstance(adjusted_payoffs, np.ndarray)
        assert price >= 0
        assert se >= 0
    
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        price1, _, _ = price_call_naive(self.S0, self.K, self.r, self.mu, self.sigma,
                                       self.T, self.steps, self.paths, 42)
        price2, _, _ = price_call_naive(self.S0, self.K, self.r, self.mu, self.sigma,
                                       self.T, self.steps, self.paths, 42)
        
        assert abs(price1 - price2) < 1e-10
    
    def test_convergence(self):
        """Test that more paths gives lower standard error."""
        small_paths = 1000
        large_paths = 10000
        
        _, se_small, _ = price_call_naive(self.S0, self.K, self.r, self.mu, self.sigma,
                                          self.T, self.steps, small_paths, self.seed)
        _, se_large, _ = price_call_naive(self.S0, self.K, self.r, self.mu, self.sigma,
                                          self.T, self.steps, large_paths, self.seed)
        
        # Standard error should decrease with more paths
        assert se_large < se_small


class TestBlackScholes:
    """Test Black-Scholes formula."""
    
    def test_atm_option(self):
        """Test at-the-money option."""
        price = black_scholes_call(100, 100, 0.03, 0.2, 1.0)
        assert price > 0
    
    def test_deep_itm_option(self):
        """Test deep in-the-money option."""
        price = black_scholes_call(150, 100, 0.03, 0.2, 1.0)
        assert price > 50  # Should be close to intrinsic value
    
    def test_deep_otm_option(self):
        """Test deep out-of-the-money option."""
        price = black_scholes_call(50, 100, 0.03, 0.2, 1.0)
        assert price < 5  # Should be small
    
    def test_zero_time(self):
        """Test option at expiry."""
        price = black_scholes_call(120, 100, 0.03, 0.2, 0.0)
        assert abs(price - 20) < 1e-10  # Should equal intrinsic value
    
    def test_zero_volatility(self):
        """Test option with zero volatility."""
        price = black_scholes_call(120, 100, 0.03, 0.0, 1.0)
        discounted_intrinsic = (120 - 100) * np.exp(-0.03 * 1.0)
        assert abs(price - discounted_intrinsic) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
