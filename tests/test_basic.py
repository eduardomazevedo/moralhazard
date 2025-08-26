import numpy as np
import pytest
from moralhazard import MoralHazardProblem, SolveResults
import dataclasses


class TestMoralHazardProblem:
    """Basic tests for MoralHazardProblem class."""
    
    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        x0 = 50
        sigma = 10.0
        first_best_effort = 100
        theta = 1.0 / first_best_effort / (first_best_effort + x0)
        
        def u(c): return np.log(x0 + c)
        def k(utils): return np.exp(utils) - x0
        def g(z): return np.log(np.maximum(z, x0))
        def C(a): return theta * a ** 2 / 2
        def Cprime(a): return theta * a
        def f(y, a):
            return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
        def score(y, a):
            return (y - a) / (sigma ** 2)
        
        return {
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
                "y_min": 0.0 - 3 * sigma,
                "y_max": 120.0 + 3 * sigma,
                "n": 201,  # must be odd
            },
        }
    
    def test_valid_config_creation(self, valid_config):
        """Test that MoralHazardProblem can be created with valid config."""
        mhp = MoralHazardProblem(valid_config)
        assert isinstance(mhp, MoralHazardProblem)
        assert mhp.y_grid.shape == (201,)
        assert len(mhp.y_grid) == 201
    
    def test_invalid_config_missing_keys(self):
        """Test that invalid config raises appropriate errors."""
        with pytest.raises(TypeError):
            MoralHazardProblem({})
        
        with pytest.raises(TypeError):
            MoralHazardProblem({"problem_params": {}})
        
        with pytest.raises(TypeError):
            MoralHazardProblem({"computational_params": {}})
    
    def test_invalid_config_missing_callables(self, valid_config):
        """Test that missing callables raise KeyError."""
        config = valid_config.copy()
        del config["problem_params"]["u"]
        
        with pytest.raises(KeyError, match=r"problem_params\['u'\] is required"):
            MoralHazardProblem(config)
    
    def test_invalid_config_missing_computational_params(self, valid_config):
        """Test that missing computational params raise KeyError."""
        config = valid_config.copy()
        del config["computational_params"]["y_min"]
        
        with pytest.raises(KeyError, match=r"computational_params\['y_min'\] is required"):
            MoralHazardProblem(config)


class TestSolveCostMinimization:
    """Tests for solve_cost_minimization_problem method."""
    
    @pytest.fixture
    def mhp(self, valid_config):
        """MoralHazardProblem instance for testing."""
        return MoralHazardProblem(valid_config)
    
    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        x0 = 50
        sigma = 10.0
        first_best_effort = 100
        theta = 1.0 / first_best_effort / (first_best_effort + x0)
        
        def u(c): return np.log(x0 + c)
        def k(utils): return np.exp(utils) - x0
        def g(z): return np.log(np.maximum(z, x0))
        def C(a): return theta * a ** 2 / 2
        def Cprime(a): return theta * a
        def f(y, a):
            return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((y - a) ** 2) / (2 * sigma ** 2))
        def score(y, a):
            return (y - a) / (sigma ** 2)
        
        return {
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
                "y_min": 0.0 - 3 * sigma,
                "y_max": 120.0 + 3 * sigma,
                "n": 201,  # must be odd
            },
        }
    
    def test_solve_with_a_hat_single_element(self, mhp):
        """Test solving with a_hat containing only one element."""
        a_hat = np.array([0.0])  # Single element case
        
        results = mhp.solve_cost_minimization_problem(
            intended_action=80.0,
            reservation_utility=np.log(100),  # u(50)
            solver="a_hat",
            a_hat=a_hat,
        )
        
        # Verify results structure
        assert isinstance(results, SolveResults)
        assert results.a0 == 80.0
        assert results.Ubar == np.log(100)
        assert results.a_hat.shape == (1,)
        assert results.a_hat[0] == 0.0
        assert results.optimal_contract.shape == (201,)  # matches y_grid length
        assert isinstance(results.expected_wage, float)
        assert isinstance(results.multipliers, dict)
        assert isinstance(results.constraints, dict)
        assert isinstance(results.solver_state, dict)
        
        # Check that multipliers exist
        assert "lam" in results.multipliers
        assert "mu" in results.multipliers
        assert "mu_hat" in results.multipliers
        assert results.multipliers["mu_hat"].shape == (1,)  # matches a_hat length
    
    def test_solve_with_a_hat_two_elements(self, mhp):
        """Test solving with a_hat containing two elements."""
        a_hat = np.array([0.0, 0.0])  # Two element case
        
        results = mhp.solve_cost_minimization_problem(
            intended_action=80.0,
            reservation_utility=np.log(100),  # u(50)
            solver="a_hat",
            a_hat=a_hat,
        )
        
        # Verify results structure
        assert isinstance(results, SolveResults)
        assert results.a0 == 80.0
        assert results.Ubar == np.log(100)
        assert results.a_hat.shape == (2,)
        assert results.a_hat[0] == 0.0
        assert results.a_hat[1] == 0.0
        assert results.optimal_contract.shape == (201,)  # matches y_grid length
        assert isinstance(results.expected_wage, float)
        assert isinstance(results.multipliers, dict)
        assert isinstance(results.constraints, dict)
        assert isinstance(results.solver_state, dict)
        
        # Check that multipliers exist and have correct shapes
        assert "lam" in results.multipliers
        assert "mu" in results.multipliers
        assert "mu_hat" in results.multipliers
        assert results.multipliers["mu_hat"].shape == (2,)  # matches a_hat length
    
    def test_solve_with_iterative_solver(self, mhp):
        """Test solving with iterative solver."""
        results = mhp.solve_cost_minimization_problem(
            intended_action=80.0,
            reservation_utility=np.log(100),  # u(50)
            solver="iterative",
            a_min=0.0,
            a_max=120.0,
            n_a_iterations=1,
        )
        
        # Verify results structure
        assert isinstance(results, SolveResults)
        assert results.a0 == 80.0
        assert results.Ubar == np.log(100)
        assert results.a_hat.shape == (2,)  # iterative solver always uses 2 elements
        assert results.optimal_contract.shape == (201,)  # matches y_grid length
        assert isinstance(results.expected_wage, float)
        assert isinstance(results.multipliers, dict)
        assert isinstance(results.constraints, dict)
        assert isinstance(results.solver_state, dict)
    
    def test_invalid_solver_type(self, mhp):
        """Test that invalid solver type raises ValueError."""
        with pytest.raises(ValueError, match="solver must be 'a_hat' or 'iterative'"):
            mhp.solve_cost_minimization_problem(
                intended_action=80.0,
                reservation_utility=np.log(100),
                solver="invalid",
                a_hat=np.array([0.0]),
            )
    
    def test_a_hat_required_for_a_hat_solver(self, mhp):
        """Test that a_hat is required when using a_hat solver."""
        with pytest.raises(ValueError, match="a_hat is required when solver='a_hat'"):
            mhp.solve_cost_minimization_problem(
                intended_action=80.0,
                reservation_utility=np.log(100),
                solver="a_hat",
                # a_hat not provided
            )
    
    def test_a_max_required_for_iterative_solver(self, mhp):
        """Test that a_max is required when using iterative solver."""
        with pytest.raises(ValueError, match="a_max is required when solver='iterative'"):
            mhp.solve_cost_minimization_problem(
                intended_action=80.0,
                reservation_utility=np.log(100),
                solver="iterative",
                a_min=0.0,
                # a_max not provided
            )
    
    def test_a_hat_must_be_1d_array(self, mhp):
        """Test that a_hat must be a 1D array."""
        with pytest.raises(ValueError, match="a_hat must be a 1D array"):
            mhp.solve_cost_minimization_problem(
                intended_action=80.0,
                reservation_utility=np.log(100),
                solver="a_hat",
                a_hat=np.array([[0.0, 0.0], [0.0, 0.0]]),  # 2D array
            )


class TestSolveResults:
    """Tests for SolveResults dataclass."""
    
    def test_solve_results_immutable(self):
        """Test that SolveResults is immutable (frozen dataclass)."""
        # Create a minimal results object for testing
        results = SolveResults(
            a0=80.0,
            Ubar=np.log(100),
            a_hat=np.array([0.0]),
            optimal_contract=np.zeros(201),
            expected_wage=100.0,
            multipliers={"lam": 1.0, "mu": 0.5, "mu_hat": np.array([0.1])},
            constraints={"U0": 0.0, "IR": 0.0, "FOC": 0.0, "Uhat": np.array([0.0]), "IC": np.array([0.0]), "Ewage": 0.0},
            solver_state={"method": "test", "status": "success", "iterations": 1, "timing": 0.1, "grad_norm": 1e-6}
        )
        
        # Verify it's immutable
        with pytest.raises(dataclasses.FrozenInstanceError):
            results.a0 = 90.0
        
        with pytest.raises(dataclasses.FrozenInstanceError):
            results.expected_wage = 200.0
