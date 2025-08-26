import numpy as np
import pytest
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg


class TestMakeUtilityCfg:
    """Tests for make_utility_cfg function."""
    
    def test_log_utility_creation(self):
        """Test that log utility functions are created correctly."""
        w0 = 50.0
        cfg = make_utility_cfg("log", w0=w0)
        
        # Check that all required functions exist
        assert "u" in cfg
        assert "k" in cfg
        assert "link_function" in cfg
        
        # Test utility function
        x = np.array([10.0, 20.0])
        u_values = cfg["u"](x)
        expected = np.log(x + w0)
        np.testing.assert_array_almost_equal(u_values, expected)
        
        # Test inverse utility function
        k_values = cfg["k"](u_values)
        np.testing.assert_array_almost_equal(k_values, x)
        
        # Test link function
        z = np.array([60.0, 70.0])
        g_values = cfg["link_function"](z)
        expected_g = np.log(np.maximum(z, w0))
        np.testing.assert_array_almost_equal(g_values, expected_g)
    
    def test_crra_utility_creation(self):
        """Test that CRRA utility functions are created correctly."""
        w0 = 50.0
        gamma = 0.5
        cfg = make_utility_cfg("crra", w0=w0, gamma=gamma)
        
        assert "u" in cfg
        assert "k" in cfg
        assert "link_function" in cfg
        
        # Test utility function
        x = np.array([10.0, 20.0])
        u_values = cfg["u"](x)
        expected = np.power(x + w0, 1 - gamma) / (1 - gamma)
        np.testing.assert_array_almost_equal(u_values, expected)
    
    def test_cara_utility_creation(self):
        """Test that CARA utility functions are created correctly."""
        w0 = 50.0
        alpha = 0.1
        cfg = make_utility_cfg("cara", w0=w0, alpha=alpha)
        
        assert "u" in cfg
        assert "k" in cfg
        assert "link_function" in cfg
        
        # Test utility function
        x = np.array([10.0, 20.0])
        u_values = cfg["u"](x)
        expected = -np.exp(-alpha * (x + w0)) / alpha
        np.testing.assert_array_almost_equal(u_values, expected)
    
    def test_invalid_utility_type(self):
        """Test that invalid utility type raises ValueError."""
        with pytest.raises(ValueError, match="utility must be one of"):
            make_utility_cfg("invalid", w0=50.0)
    
    def test_crra_missing_gamma(self):
        """Test that CRRA without gamma defaults to log case."""
        # CRRA without gamma defaults to gamma=1, which becomes log case
        cfg = make_utility_cfg("crra", w0=50.0)
        
        # Should still return the expected functions
        assert "u" in cfg
        assert "k" in cfg
        assert "link_function" in cfg
        
        # Test that it behaves like log utility
        x = np.array([10.0, 20.0])
        u_values = cfg["u"](x)
        expected = np.log(x + 50.0)
        np.testing.assert_array_almost_equal(u_values, expected)
    
    def test_cara_missing_alpha(self):
        """Test that CARA without alpha raises ValueError."""
        with pytest.raises(ValueError, match="For CARA, provide alpha > 0"):
            make_utility_cfg("cara", w0=50.0)
    
    def test_cara_invalid_alpha(self):
        """Test that CARA with invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="For CARA, provide alpha > 0"):
            make_utility_cfg("cara", w0=50.0, alpha=0.0)


class TestMakeDistributionCfg:
    """Tests for make_distribution_cfg function."""
    
    def test_gaussian_distribution_creation(self):
        """Test that Gaussian distribution functions are created correctly."""
        sigma = 10.0
        cfg = make_distribution_cfg("gaussian", sigma=sigma)
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test PDF function
        y = np.array([0.0, 10.0])
        a = np.array([5.0, 15.0])
        f_values = cfg["f"](y, a)
        
        # Check that PDF values are positive
        assert np.all(f_values > 0)
        
        # Test score function
        score_values = cfg["score"](y, a)
        expected_score = (y - a) / (sigma * sigma)
        np.testing.assert_array_almost_equal(score_values, expected_score)
    
    def test_gaussian_default_sigma(self):
        """Test that Gaussian with no sigma uses default sigma=1.0."""
        cfg = make_distribution_cfg("gaussian")
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test that it uses default sigma=1.0
        y = np.array([0.0, 1.0])
        a = np.array([0.0, 0.0])
        score_values = cfg["score"](y, a)
        expected_score = (y - a) / (1.0 * 1.0)  # default sigma=1.0
        np.testing.assert_array_almost_equal(score_values, expected_score)
    
    def test_poisson_distribution_creation(self):
        """Test that Poisson distribution functions are created correctly."""
        cfg = make_distribution_cfg("poisson")
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test PMF function
        y = np.array([0, 1, 2])
        a = np.array([1.0, 2.0, 3.0])
        f_values = cfg["f"](y, a)
        
        # Check that PMF values are positive
        assert np.all(f_values > 0)
        
        # Test score function
        score_values = cfg["score"](y, a)
        expected_score = (y - a) / a
        np.testing.assert_array_almost_equal(score_values, expected_score)
    
    def test_bernoulli_distribution_creation(self):
        """Test that Bernoulli distribution functions are created correctly."""
        cfg = make_distribution_cfg("bernoulli")
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test PMF function
        y = np.array([0, 1])
        a = np.array([0.3, 0.7])
        f_values = cfg["f"](y, a)
        
        # Check that PMF values are positive
        assert np.all(f_values > 0)
        
        # Test score function
        score_values = cfg["score"](y, a)
        expected_score = (y - a) / (a - a * a)
        np.testing.assert_array_almost_equal(score_values, expected_score)
    
    def test_binomial_distribution_creation(self):
        """Test that Binomial distribution functions are created correctly."""
        n = 5
        cfg = make_distribution_cfg("binomial", n=n)
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test PMF function
        y = np.array([0, 1, 2])
        a = np.array([0.3, 0.5, 0.7])
        f_values = cfg["f"](y, a)
        
        # Check that PMF values are positive
        assert np.all(f_values > 0)
    
    def test_binomial_default_n(self):
        """Test that Binomial with no n uses default n=1.0."""
        cfg = make_distribution_cfg("binomial")
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test that it uses default n=1.0
        y = np.array([0, 1])
        a = np.array([0.3, 0.7])
        score_values = cfg["score"](y, a)
        expected_score = (y - 1.0 * a) / (a - a * a)  # default n=1.0
        np.testing.assert_array_almost_equal(score_values, expected_score)
    
    def test_gamma_distribution_creation(self):
        """Test that Gamma distribution functions are created correctly."""
        n = 2.0
        cfg = make_distribution_cfg("gamma", n=n)
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test PDF function
        y = np.array([1.0, 2.0])
        a = np.array([0.5, 1.0])
        f_values = cfg["f"](y, a)
        
        # Check that PDF values are positive
        assert np.all(f_values > 0)
    
    def test_gamma_default_n(self):
        """Test that Gamma with no n uses default n=1.0."""
        cfg = make_distribution_cfg("gamma")
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test that it uses default n=1.0
        y = np.array([1.0, 2.0])
        a = np.array([0.5, 1.0])
        score_values = cfg["score"](y, a)
        expected_score = (y - 1.0 * a) / (a * a)  # default n=1.0
        np.testing.assert_array_almost_equal(score_values, expected_score)
    
    def test_student_t_distribution_creation(self):
        """Test that Student's t distribution functions are created correctly."""
        nu = 5.0
        sigma = 2.0
        cfg = make_distribution_cfg("student_t", nu=nu, sigma=sigma)
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test PDF function
        y = np.array([0.0, 1.0])
        a = np.array([0.0, 0.0])
        f_values = cfg["f"](y, a)
        
        # Check that PDF values are positive
        assert np.all(f_values > 0)
    
    def test_student_t_default_params(self):
        """Test that Student's t with no params uses defaults nu=5.0, sigma=1.0."""
        cfg = make_distribution_cfg("student_t")
        
        assert "f" in cfg
        assert "score" in cfg
        
        # Test that it uses default nu=5.0, sigma=1.0
        y = np.array([0.0, 1.0])
        a = np.array([0.0, 0.0])
        score_values = cfg["score"](y, a)
        # Default: nu=5.0, sigma=1.0
        expected_score = ((5.0 + 1.0) * (y - a)) / (5.0 * (1.0 * 1.0) + (y - a) ** 2)
        np.testing.assert_array_almost_equal(score_values, expected_score)
    
    def test_invalid_distribution_type(self):
        """Test that invalid distribution type raises ValueError."""
        with pytest.raises(ValueError, match="dist must be one of"):
            make_distribution_cfg("invalid")
    
    def test_binomial_invalid_n(self):
        """Test that Binomial with invalid n raises ValueError."""
        with pytest.raises(ValueError, match="binomial requires integer n >= 1"):
            make_distribution_cfg("binomial", n=0.5)


class TestConfigMakerIntegration:
    """Integration tests for config maker functions."""
    
    def test_utility_and_distribution_combination(self):
        """Test that utility and distribution functions can be combined."""
        # Create utility functions
        util_cfg = make_utility_cfg("log", w0=50.0)
        
        # Create distribution functions
        dist_cfg = make_distribution_cfg("gaussian", sigma=10.0)
        
        # Test that they work together
        x = np.array([10.0, 20.0])
        a = np.array([5.0, 15.0])
        
        # Utility should work
        u_values = util_cfg["u"](x)
        assert len(u_values) == 2
        
        # Distribution should work
        f_values = dist_cfg["f"](x, a)
        assert len(f_values) == 2
        
        # Score should work
        score_values = dist_cfg["score"](x, a)
        assert len(score_values) == 2
        
        print("Integration test passed: utility and distribution functions work together")
    
    def test_broadcasting_behavior(self):
        """Test that functions handle broadcasting correctly."""
        util_cfg = make_utility_cfg("log", w0=10.0)
        dist_cfg = make_distribution_cfg("gaussian", sigma=5.0)
        
        # Test broadcasting with different shapes
        y = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        a = np.array([0.5, 1.5])  # 1x2
        
        # These should broadcast correctly
        f_values = dist_cfg["f"](y, a)
        score_values = dist_cfg["score"](y, a)
        
        assert f_values.shape == (2, 2)
        assert score_values.shape == (2, 2)
        
        print("Broadcasting test passed: functions handle different shapes correctly")

