"""
Unit tests for _agent_best_action function in moralhazard.core
"""
import numpy as np
import pytest
from moralhazard.core import _agent_best_action, _compute_expected_utility
from moralhazard.problem import MoralHazardProblem


@pytest.fixture
def simple_problem():
    """Create a simple MoralHazardProblem for testing."""
    x0 = 50.0
    sigma = 10.0
    first_best_effort = 100.0
    theta = 1.0 / first_best_effort / (first_best_effort + x0)

    def u(c):
        return np.log(x0 + c)

    def k(utils):
        return np.exp(utils) - x0

    def g(z):
        return np.log(np.maximum(z, x0))

    def C(a):
        return theta * a ** 2 / 2

    def Cprime(a):
        return theta * a

    def f(y, a):
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))

    def score(y, a):
        return (y - a) / (sigma ** 2)

    cfg = {
        "problem_params": {
            "u": u,
            "k": k,
            "link_function": g,
            "C": C,
            "Cprime": Cprime,
            "f": f,
            "score": score,
        },
        "computational_params": {
            "distribution_type": "continuous",
            "y_min": 0.0 - 3 * sigma,
            "y_max": 100.0 + 3 * sigma,
            "n": 51,  # odd number for Simpson's rule
        },
    }

    return MoralHazardProblem(cfg)


@pytest.fixture
def simple_contract(simple_problem):
    """Create a simple contract (value function v) for testing."""
    # Create a contract that gives higher utility for actions around 50
    # This is a simple linear contract: v(y) = y
    return simple_problem.y_grid.copy()


class TestAgentBestAction:
    """Test suite for _agent_best_action function."""

    def test_basic_functionality(self, simple_problem, simple_contract):
        """Test that the function returns valid results."""
        v = simple_contract
        a_lb = 0.0
        a_ub = 100.0
        n_a_grid_points = 10

        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        # Check return types
        assert isinstance(best_action, float)
        assert isinstance(best_utility, float)

        # Check that action is within bounds
        assert a_lb <= best_action <= a_ub

        # Check that utility is finite
        assert np.isfinite(best_utility)

    def test_returns_optimal_action(self, simple_problem, simple_contract):
        """Test that the function finds a better or equal action compared to grid search alone."""
        v = simple_contract
        a_lb = 0.0
        a_ub = 100.0
        n_a_grid_points = 10

        # Get result from _agent_best_action
        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        # Compare with grid search only
        a_grid = np.linspace(a_lb, a_ub, n_a_grid_points)
        grid_utilities = _compute_expected_utility(v, a_grid, problem=simple_problem)
        grid_best_idx = np.argmax(grid_utilities)
        grid_best_action = a_grid[grid_best_idx]
        grid_best_utility = grid_utilities[grid_best_idx]

        # The optimized result should be at least as good as grid search
        assert best_utility >= grid_best_utility - 1e-6  # Allow small numerical error

    def test_small_action_range(self, simple_problem, simple_contract):
        """Test with a small action range."""
        v = simple_contract
        a_lb = 40.0
        a_ub = 60.0
        n_a_grid_points = 5

        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        assert a_lb <= best_action <= a_ub
        assert np.isfinite(best_utility)

    def test_single_grid_point(self, simple_problem, simple_contract):
        """Test with only one grid point (edge case)."""
        v = simple_contract
        a_lb = 50.0
        a_ub = 50.0
        n_a_grid_points = 1

        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        assert best_action == a_lb
        assert np.isfinite(best_utility)

    def test_boundary_conditions(self, simple_problem, simple_contract):
        """Test that boundary conditions are handled correctly."""
        v = simple_contract
        a_lb = 0.0
        a_ub = 100.0
        n_a_grid_points = 3  # Small grid to test boundaries

        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        # Action should be within bounds (allowing for small floating point errors)
        assert a_lb - 1e-10 <= best_action <= a_ub + 1e-10

    def test_different_grid_sizes(self, simple_problem, simple_contract):
        """Test with different numbers of grid points."""
        v = simple_contract
        a_lb = 0.0
        a_ub = 100.0

        for n_points in [5, 10, 20, 50]:
            best_action, best_utility = _agent_best_action(
                v=v,
                a_lb=a_lb,
                a_ub=a_ub,
                n_a_grid_points=n_points,
                problem=simple_problem,
            )

            assert a_lb <= best_action <= a_ub
            assert np.isfinite(best_utility)

    def test_utility_consistency(self, simple_problem, simple_contract):
        """Test that the returned utility matches the computed utility at the best action."""
        v = simple_contract
        a_lb = 0.0
        a_ub = 100.0
        n_a_grid_points = 10

        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        # Compute utility directly at the best action
        computed_utility = _compute_expected_utility(v, best_action, problem=simple_problem)

        # They should match (within numerical precision)
        assert np.abs(best_utility - computed_utility) < 1e-6

    def test_negative_utilities(self, simple_problem):
        """Test with a contract that yields negative utilities."""
        # Create a contract that gives negative utility
        v = -np.ones_like(simple_problem.y_grid) * 100

        a_lb = 0.0
        a_ub = 100.0
        n_a_grid_points = 10

        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        assert a_lb <= best_action <= a_ub
        assert np.isfinite(best_utility)
        # Utility should be negative but still the best available
        assert best_utility <= 0

    def test_very_small_range(self, simple_problem, simple_contract):
        """Test with a very small action range."""
        v = simple_contract
        a_lb = 49.0
        a_ub = 51.0
        n_a_grid_points = 5

        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=a_lb,
            a_ub=a_ub,
            n_a_grid_points=n_a_grid_points,
            problem=simple_problem,
        )

        assert a_lb <= best_action <= a_ub
        assert np.isfinite(best_utility)

    def test_contract_shape_validation(self, simple_problem):
        """Test that the function works with contracts of correct shape."""
        v = simple_problem.y_grid.copy()

        # Should work with correct shape
        best_action, best_utility = _agent_best_action(
            v=v,
            a_lb=0.0,
            a_ub=100.0,
            n_a_grid_points=10,
            problem=simple_problem,
        )

        assert np.isfinite(best_action)
        assert np.isfinite(best_utility)
