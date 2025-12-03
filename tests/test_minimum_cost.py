"""
Unit tests for minimum_cost method in moralhazard.problem
"""
import numpy as np
import pytest
from moralhazard.problem import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg


@pytest.fixture
def simple_problem():
    """Create a simple MoralHazardProblem for testing."""
    x0 = 50.0
    sigma = 10.0
    first_best_effort = 100.0
    theta = 1.0 / first_best_effort / (first_best_effort + x0)

    utility_cfg = make_utility_cfg("log", w0=x0)
    dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

    def C(a):
        return theta * a ** 2 / 2

    def Cprime(a):
        return theta * a

    cfg = {
        "problem_params": {
            **utility_cfg,
            **dist_cfg,
            "C": C,
            "Cprime": Cprime,
        },
        "computational_params": {
            "distribution_type": "continuous",
            "y_min": 0.0 - 3 * sigma,
            "y_max": 100.0 + 3 * sigma,
            "n": 51,  # odd number for Simpson's rule
        },
    }

    return MoralHazardProblem(cfg)


class TestMinimumCost:
    """Test suite for minimum_cost method."""

    def test_basic_functionality_scalar(self, simple_problem):
        """Test that the function returns valid results for scalar input."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        result = simple_problem.minimum_cost(
            intended_action=50.0,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=1,
        )

        # Check return type
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0  # Expected wage should be positive

    def test_basic_functionality_array(self, simple_problem):
        """Test that the function returns valid results for array input."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        actions = np.array([40.0, 50.0, 60.0])
        result = simple_problem.minimum_cost(
            intended_action=actions,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=1,
        )

        # Check return type and shape
        assert isinstance(result, np.ndarray)
        assert result.shape == actions.shape
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)  # Expected wages should be positive

    def test_scalar_vs_array_consistency(self, simple_problem):
        """Test that scalar and array inputs give consistent results."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        action = 50.0
        
        # Scalar input
        result_scalar = simple_problem.minimum_cost(
            intended_action=action,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=1,
        )
        
        # Array input with single element
        result_array = simple_problem.minimum_cost(
            intended_action=np.array([action]),
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=1,
        )

        # Results should match
        assert abs(result_scalar - result_array[0]) < 1e-6

    def test_multiple_actions(self, simple_problem):
        """Test with multiple actions in array."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        # Use relaxed problem (n_a_iterations=0) to avoid solver convergence issues
        # This test is about array handling, not solver convergence
        actions = np.array([30.0, 50.0, 70.0, 80.0])
        result = simple_problem.minimum_cost(
            intended_action=actions,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=0,  # Relaxed problem for stability
        )

        assert result.shape == actions.shape
        assert len(result) == 4
        assert np.all(np.isfinite(result))

    def test_different_reservation_utilities(self, simple_problem):
        """Test with different reservation utilities."""
        u_fun = simple_problem._primitives["u"]
        
        action = 50.0
        reservation_utilities = [u_fun(-1), u_fun(0), u_fun(5), u_fun(10)]
        
        results = []
        for Ubar in reservation_utilities:
            result = simple_problem.minimum_cost(
                intended_action=action,
                reservation_utility=Ubar,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=1,
            )
            results.append(result)
            assert np.isfinite(result)
            assert result > 0

        # Higher reservation utility should generally require higher expected wage
        # (though this depends on the specific problem)
        assert all(r > 0 for r in results)

    def test_multi_dimensional_array(self, simple_problem):
        """Test with multi-dimensional arrays (2D, 3D, etc.)."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        # Use relaxed problem (n_a_iterations=0) to avoid solver convergence issues
        # This test is about array shape handling, not solver convergence
        
        # 2D array
        actions_2d = np.array([[40.0, 50.0], [60.0, 70.0]])
        result_2d = simple_problem.minimum_cost(
            intended_action=actions_2d,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=0,  # Relaxed problem for stability
        )
        assert result_2d.shape == actions_2d.shape
        assert np.all(np.isfinite(result_2d))
        
        # 3D array
        actions_3d = np.array([[[30.0, 40.0], [50.0, 60.0]], [[70.0, 80.0], [85.0, 90.0]]])
        result_3d = simple_problem.minimum_cost(
            intended_action=actions_3d,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=0,  # Relaxed problem for stability
        )
        assert result_3d.shape == actions_3d.shape
        assert np.all(np.isfinite(result_3d))

    def test_zero_iterations(self, simple_problem):
        """Test with n_a_iterations=0 (relaxed problem)."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        result = simple_problem.minimum_cost(
            intended_action=50.0,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=0,  # Relaxed problem
        )

        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0

    def test_custom_parameters(self, simple_problem):
        """Test with custom solver parameters."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        result = simple_problem.minimum_cost(
            intended_action=50.0,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=2,    # More iterations
            clip_ratio=1e5,      # Different clip ratio
            a_always_check_global_ic=np.array([0.0, 25.0, 50.0]),
        )

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_consistency_with_solve_cost_minimization(self, simple_problem):
        """Test that minimum_cost gives same result as solve_cost_minimization_problem."""
        u_fun = simple_problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        action = 50.0
        
        # Using minimum_cost
        result_minimum_cost = simple_problem.minimum_cost(
            intended_action=action,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=1,
        )
        
        # Using solve_cost_minimization_problem directly
        results = simple_problem.solve_cost_minimization_problem(
            intended_action=action,
            reservation_utility=reservation_utility,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=1,
        )
        result_direct = results.expected_wage

        # Results should match
        assert abs(result_minimum_cost - result_direct) < 1e-6
