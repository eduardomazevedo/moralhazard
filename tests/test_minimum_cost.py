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


@pytest.fixture
def cvxpy_compatible_problem():
    """Create a MoralHazardProblem with k that accepts xp argument for CVXPY."""
    x0 = 50.0
    sigma = 10.0
    first_best_effort = 100.0
    theta = 1.0 / first_best_effort / (first_best_effort + x0)

    # CVXPY-compatible k function with xp argument
    def u(c):
        return np.log(x0 + c)
    
    def k(utils, xp=np):
        return xp.exp(utils) - x0
    
    def link_function(z):
        return np.log(np.maximum(z, x0))

    def f(y, a):
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
    
    def score(y, a):
        return (y - a) / (sigma ** 2)

    def C(a):
        return theta * a ** 2 / 2

    def Cprime(a):
        return theta * a

    cfg = {
        "problem_params": {
            "u": u,
            "k": k,
            "link_function": link_function,
            "f": f,
            "score": score,
            "C": C,
            "Cprime": Cprime,
        },
        "computational_params": {
            "distribution_type": "continuous",
            "y_min": 0.0 - 3 * sigma,
            "y_max": 100.0 + 3 * sigma,
            "n": 101,  # odd number for Simpson's rule
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


class TestCrossMethodConsistency:
    """Test consistency between dual-based and CVXPY-based solvers."""

    def test_dual_solve_vs_minimum_cost(self, cvxpy_compatible_problem):
        """Dual: solve_cost_minimization_problem should match minimum_cost exactly."""
        problem = cvxpy_compatible_problem
        u_fun = problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        # Use actions where dual solver works (skip 20, 30 which fail)
        actions = [40.0, 50.0, 70.0]
        
        for action in actions:
            # solve_cost_minimization_problem
            result_solve = problem.solve_cost_minimization_problem(
                intended_action=action,
                reservation_utility=reservation_utility,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=2,
            ).expected_wage
            
            # minimum_cost
            result_min = problem.minimum_cost(
                intended_action=action,
                reservation_utility=reservation_utility,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=2,
            )
            
            # Should be exactly the same (same underlying implementation)
            assert abs(result_solve - result_min) < 1e-10, \
                f"Dual solve vs minimum_cost mismatch at a={action}: {result_solve} vs {result_min}"

    def test_cvxpy_solve_vs_minimum_cost(self, cvxpy_compatible_problem):
        """CVXPY: solve_cost_minimization_problem_cvxpy should match minimum_cost_cvxpy exactly."""
        problem = cvxpy_compatible_problem
        u_fun = problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        a_hat = np.linspace(0.0, 100.0, 101)
        actions = np.array([30.0, 50.0, 70.0])
        
        # minimum_cost_cvxpy (batch)
        results_batch = problem.minimum_cost_cvxpy(
            intended_actions=actions,
            reservation_utility=reservation_utility,
            a_hat=a_hat,
        )
        
        for i, action in enumerate(actions):
            # solve_cost_minimization_problem_cvxpy (single)
            result_solve = problem.solve_cost_minimization_problem_cvxpy(
                intended_action=action,
                reservation_utility=reservation_utility,
                a_hat=a_hat,
            )['expected_wage']
            
            # Should be very close (same underlying implementation, minor numerical differences)
            assert abs(result_solve - results_batch[i]) < 1e-6, \
                f"CVXPY solve vs minimum_cost mismatch at a={action}: {result_solve} vs {results_batch[i]}"

    def test_dual_vs_cvxpy_single_action(self, cvxpy_compatible_problem):
        """Cross-algorithm: dual and CVXPY single-action solvers should agree within tolerance."""
        problem = cvxpy_compatible_problem
        u_fun = problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        a_hat = np.linspace(0.0, 100.0, 101)
        # Include ALL actions, including problematic ones (20.0, 30.0)
        actions = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        
        failed_actions = []
        
        for action in actions:
            # CVXPY solver
            result_cvxpy = problem.solve_cost_minimization_problem_cvxpy(
                intended_action=action,
                reservation_utility=reservation_utility,
                a_hat=a_hat,
            )['expected_wage']
            
            # Dual solver with more iterations (may warn but returns result)
            result_dual = problem.solve_cost_minimization_problem(
                intended_action=action,
                reservation_utility=reservation_utility,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=10,
            ).expected_wage
            
            diff = abs(result_dual - result_cvxpy)
            if diff >= 1e-2:
                failed_actions.append({
                    'action': action,
                    'dual': result_dual,
                    'cvxpy': result_cvxpy,
                    'diff': diff,
                })
        
        if failed_actions:
            msg = "Dual vs CVXPY mismatch at the following actions:\n"
            for f in failed_actions:
                msg += f"  a={f['action']}: dual={f['dual']:.6f}, cvxpy={f['cvxpy']:.6f}, diff={f['diff']:.6f}\n"
            pytest.fail(msg)

    def test_dual_vs_cvxpy_batch(self, cvxpy_compatible_problem):
        """Cross-algorithm: dual and CVXPY batch solvers should agree within tolerance."""
        problem = cvxpy_compatible_problem
        u_fun = problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        a_hat = np.linspace(0.0, 100.0, 101)
        # Include ALL actions, including problematic ones (20.0, 30.0)
        actions = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        
        # CVXPY batch
        results_cvxpy = problem.minimum_cost_cvxpy(
            intended_actions=actions,
            reservation_utility=reservation_utility,
            a_hat=a_hat,
        )
        
        failed_actions = []
        
        # Test each action individually (may warn but returns result)
        for i, action in enumerate(actions):
            result_dual = problem.minimum_cost(
                intended_action=float(action),
                reservation_utility=reservation_utility,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=10,
            )
            
            diff = abs(result_dual - results_cvxpy[i])
            if diff >= 1e-2:
                failed_actions.append({
                    'action': action,
                    'dual': result_dual,
                    'cvxpy': results_cvxpy[i],
                    'diff': diff,
                })
        
        if failed_actions:
            msg = "Dual vs CVXPY batch mismatch at the following actions:\n"
            for f in failed_actions:
                msg += f"  a={f['action']}: dual={f['dual']:.6f}, cvxpy={f['cvxpy']:.6f}, diff={f['diff']:.6f}\n"
            pytest.fail(msg)

    def test_all_four_methods_consistency(self, cvxpy_compatible_problem):
        """All four methods should produce consistent results."""
        problem = cvxpy_compatible_problem
        u_fun = problem._primitives["u"]
        reservation_utility = u_fun(0) - 5.0
        
        a_hat = np.linspace(0.0, 100.0, 101)
        # Include ALL actions, including problematic ones (20.0, 30.0)
        actions = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        
        # CVXPY methods
        results_cvxpy_solve = np.array([
            problem.solve_cost_minimization_problem_cvxpy(
                intended_action=a,
                reservation_utility=reservation_utility,
                a_hat=a_hat,
            )['expected_wage']
            for a in actions
        ])
        
        results_cvxpy_min = problem.minimum_cost_cvxpy(
            intended_actions=actions,
            reservation_utility=reservation_utility,
            a_hat=a_hat,
        )
        
        # CVXPY methods should match exactly
        np.testing.assert_allclose(
            results_cvxpy_solve, results_cvxpy_min, rtol=1e-5,
            err_msg="CVXPY solve vs minimum_cost mismatch"
        )
        
        # Dual methods - collect results (may warn but returns result)
        results_dual_solve = np.array([
            problem.solve_cost_minimization_problem(
                intended_action=float(a),
                reservation_utility=reservation_utility,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=10,
            ).expected_wage
            for a in actions
        ])
        
        results_dual_min = np.array([
            problem.minimum_cost(
                intended_action=float(a),
                reservation_utility=reservation_utility,
                a_ic_lb=0.0,
                a_ic_ub=100.0,
                n_a_iterations=10,
            )
            for a in actions
        ])
        
        # Dual methods should match each other
        np.testing.assert_allclose(
            results_dual_solve, results_dual_min, rtol=1e-10,
            err_msg="Dual solve vs minimum_cost mismatch"
        )
        
        # Cross-algorithm comparison
        failed_actions = []
        for i, action in enumerate(actions):
            diff = abs(results_dual_solve[i] - results_cvxpy_solve[i])
            if diff >= 1e-2:
                failed_actions.append({
                    'action': action,
                    'dual': results_dual_solve[i],
                    'cvxpy': results_cvxpy_solve[i],
                    'diff': diff,
                })
        
        if failed_actions:
            msg = "Dual vs CVXPY mismatch at the following actions:\n"
            for f in failed_actions:
                msg += f"  a={f['action']}: dual={f['dual']:.6f}, cvxpy={f['cvxpy']:.6f}, diff={f['diff']:.6f}\n"
            pytest.fail(msg)
