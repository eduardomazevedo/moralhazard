"""Numerical regressions for the Gaussian fallback cases."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from moralhazard.core import _make_cache
from moralhazard.problem import MoralHazardProblem
from moralhazard.solver import _dual_value_and_grad


@pytest.fixture(scope="module")
def gaussian_problem():
    x0, sigma = 50.0, 10.0
    theta = 1.0 / 100.0 / 150.0

    def u(c): return np.log(x0 + c)
    def k(v, xp=np): return xp.exp(v) - x0
    def g(z): return np.log(np.maximum(z, x0))
    def f(y, a):
        return np.exp(-((y - a) ** 2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    def score(y, a): return (y - a) / sigma**2
    def C(a): return theta * a**2 / 2
    def Cprime(a): return theta * a

    problem = MoralHazardProblem({
        "problem_params": {
            "u": u, "k": k, "link_function": g, "f": f,
            "score": score, "C": C, "Cprime": Cprime,
        },
        "computational_params": {
            "distribution_type": "continuous", "y_min": -30.0,
            "y_max": 130.0, "n": 101,
        },
    })
    return problem, float(u(0) - 5.0)


def test_fixed_master_danskin_gradient(gaussian_problem):
    problem, ubar = gaussian_problem
    cache = _make_cache(20.0, np.array([0.0, 100.0]), problem=problem)
    theta = np.array([55.0, 8.0, 0.1, 0.1])
    _, analytic = _dual_value_and_grad(theta, cache, problem, ubar)
    numerical = np.empty_like(theta)
    for i in range(theta.size):
        h = 1e-5 * max(1.0, abs(theta[i]))
        plus, minus = theta.copy(), theta.copy()
        plus[i] += h
        minus[i] -= h
        numerical[i] = (
            _dual_value_and_grad(plus, cache, problem, ubar)[0]
            - _dual_value_and_grad(minus, cache, problem, ubar)[0]
        ) / (2 * h)
    np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize(
    ("action", "baseline_wage"),
    [(20.0, 0.6577627565190345), (30.0, 1.4757198950773227)],
)
def test_fallback_cases_are_certified_without_fallback(
    gaussian_problem, action, baseline_wage
):
    problem, ubar = gaussian_problem
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = problem.solve_cost_minimization_problem(
            intended_action=action,
            reservation_utility=ubar,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=10,
        )

    assert not any("Triggering CVXPY fallback" in str(w.message) for w in caught)
    assert result.solver_state["success"]
    assert result.solver_state["outer_converged"]
    assert result.solver_state["projected_grad_inf"] <= 1e-4
    assert result.solver_state["primal_residual"] <= 1e-5

    dense_actions = np.linspace(0.0, 100.0, 2001)
    dense_utilities = problem.U(result.optimal_contract, dense_actions)
    intended_utility = float(np.asarray(problem.U(result.optimal_contract, action)).item())
    assert float(np.max(dense_utilities) - intended_utility) <= 1e-5

    # Preserve the pre-change, CVXPY-backed answer to the examples.
    assert result.expected_wage == pytest.approx(baseline_wage, abs=1e-3)
