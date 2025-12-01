"""
Unit tests for _maximize_1d_robust function in moralhazard.utils
"""
import numpy as np
import pytest
from moralhazard.utils import _maximize_1d_robust


class TestMaximize1DRobust:
    """Test suite for _maximize_1d_robust function."""

    def test_basic_functionality(self):
        """Test that the function returns valid results for a simple quadratic."""
        # Simple quadratic: -(x - 5)^2, maximum at x = 5
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 5.0) ** 2
            return -(x - 5.0) ** 2

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=10.0,
            n_grid_points=10,
        )

        # Check return types
        assert isinstance(best_x, float)
        assert isinstance(best_value, float)

        # Check that x is within bounds
        assert 0.0 <= best_x <= 10.0

        # Check that value is finite
        assert np.isfinite(best_value)

        # Should be close to the true maximum at x=5
        assert abs(best_x - 5.0) < 0.1
        assert abs(best_value - 0.0) < 0.1

    def test_vectorized_objective(self):
        """Test that vectorized objective functions work correctly."""
        # Test with a function that handles arrays
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 3.0) ** 2 + 2.0
            return -(x - 3.0) ** 2 + 2.0

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=6.0,
            n_grid_points=20,
        )

        assert 0.0 <= best_x <= 6.0
        assert np.isfinite(best_value)
        # Should find maximum near x=3
        assert abs(best_x - 3.0) < 0.2

    def test_returns_optimal_value(self):
        """Test that the function finds a better or equal value compared to grid search alone."""
        # Function with maximum at x = 7.5
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 7.5) ** 2
            return -(x - 7.5) ** 2

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=10.0,
            n_grid_points=10,
        )

        # Compare with grid search only
        x_grid = np.linspace(0.0, 10.0, 10)
        grid_values = objective(x_grid)
        grid_best_idx = np.argmax(grid_values)
        grid_best_value = grid_values[grid_best_idx]

        # The optimized result should be at least as good as grid search
        assert best_value >= grid_best_value - 1e-6  # Allow small numerical error

    def test_small_range(self):
        """Test with a small range."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return -x ** 2 + 10 * x
            return -x ** 2 + 10 * x

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=4.0,
            upper_bound=6.0,
            n_grid_points=5,
        )

        assert 4.0 <= best_x <= 6.0
        assert np.isfinite(best_value)
        # Maximum of -x^2 + 10x is at x = 5
        assert abs(best_x - 5.0) < 0.5

    def test_single_grid_point(self):
        """Test with only one grid point (edge case)."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return np.ones_like(x) * 42.0
            return 42.0

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=5.0,
            upper_bound=5.0,
            n_grid_points=1,
        )

        assert best_x == 5.0
        assert best_value == 42.0

    def test_boundary_conditions(self):
        """Test that boundary conditions are handled correctly."""
        # Maximum at the boundary
        def objective(x):
            if isinstance(x, np.ndarray):
                return x  # Increasing function, max at upper bound
            return x

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=10.0,
            n_grid_points=5,
        )

        # Action should be within bounds (allowing for small floating point errors)
        assert 0.0 - 1e-10 <= best_x <= 10.0 + 1e-10
        # Should be close to upper bound
        assert best_x >= 9.0

    def test_different_grid_sizes(self):
        """Test with different numbers of grid points."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 5.0) ** 2
            return -(x - 5.0) ** 2

        for n_points in [5, 10, 20, 50]:
            best_x, best_value = _maximize_1d_robust(
                objective=objective,
                lower_bound=0.0,
                upper_bound=10.0,
                n_grid_points=n_points,
            )

            assert 0.0 <= best_x <= 10.0
            assert np.isfinite(best_value)

    def test_value_consistency(self):
        """Test that the returned value matches the computed value at the best x."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 3.0) ** 2 + 5.0
            return -(x - 3.0) ** 2 + 5.0

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=6.0,
            n_grid_points=10,
        )

        # Compute value directly at the best x
        computed_value = objective(best_x)

        # They should match (within numerical precision)
        assert np.abs(best_value - computed_value) < 1e-6

    def test_negative_values(self):
        """Test with an objective that yields negative values."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 5.0) ** 2 - 100.0  # Always negative
            return -(x - 5.0) ** 2 - 100.0

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=10.0,
            n_grid_points=10,
        )

        assert 0.0 <= best_x <= 10.0
        assert np.isfinite(best_value)
        # Value should be negative but still the best available
        assert best_value <= -100.0

    def test_very_small_range(self):
        """Test with a very small range."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 5.0) ** 2
            return -(x - 5.0) ** 2

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=4.9,
            upper_bound=5.1,
            n_grid_points=5,
        )

        assert 4.9 <= best_x <= 5.1
        assert np.isfinite(best_value)

    def test_multimodal_function(self):
        """Test with a function that has multiple local maxima."""
        # Function with local maxima at x=2 and x=8, global at x=8
        def objective(x):
            if isinstance(x, np.ndarray):
                return -((x - 2.0) ** 2) * ((x - 8.0) ** 2) / 100.0 + (x - 8.0) ** 2 / 10.0
            return -((x - 2.0) ** 2) * ((x - 8.0) ** 2) / 100.0 + (x - 8.0) ** 2 / 10.0

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=10.0,
            n_grid_points=20,
        )

        assert 0.0 <= best_x <= 10.0
        assert np.isfinite(best_value)

    def test_custom_tolerance(self):
        """Test with custom tolerance parameter."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return -(x - 5.0) ** 2
            return -(x - 5.0) ** 2

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=10.0,
            n_grid_points=10,
            xatol=1e-10,  # Very tight tolerance
        )

        assert 0.0 <= best_x <= 10.0
        assert np.isfinite(best_value)

    def test_linear_function(self):
        """Test with a linear function (maximum at boundary)."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return 2.0 * x + 1.0
            return 2.0 * x + 1.0

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=10.0,
            n_grid_points=10,
        )

        assert 0.0 <= best_x <= 10.0
        # Linear function should maximize at upper bound
        assert best_x >= 9.0

    def test_sine_function(self):
        """Test with a periodic function."""
        def objective(x):
            if isinstance(x, np.ndarray):
                return np.sin(x)
            return np.sin(x)

        best_x, best_value = _maximize_1d_robust(
            objective=objective,
            lower_bound=0.0,
            upper_bound=np.pi,
            n_grid_points=20,
        )

        assert 0.0 <= best_x <= np.pi
        assert np.isfinite(best_value)
        # sin(x) has maximum at x = π/2 in [0, π]
        assert abs(best_x - np.pi / 2) < 0.2
