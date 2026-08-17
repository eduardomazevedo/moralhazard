"""Tests for the primitive factory functions."""

import cvxpy as cp
import numpy as np
import pytest

from moralhazard.config_maker import make_utility_cfg


@pytest.mark.parametrize(
    ("utility", "kwargs", "lower_bound", "upper_bound"),
    [
        ("log", {}, 0.0, 2.0),
        ("crra", {"gamma": 0.5}, 0.0, 2.0),
        ("crra", {"gamma": 1.5}, -2.0, -0.1),
        ("cara", {"alpha": 1.2}, -2.0, -0.1),
    ],
)
def test_inverse_utility_is_cvxpy_dcp(
    utility, kwargs, lower_bound, upper_bound
):
    """Built-in inverse utilities produce valid convex objectives."""
    k = make_utility_cfg(utility, w0=1.0, **kwargs)["k"]
    utility_values = cp.Variable(3)
    problem = cp.Problem(
        cp.Minimize(cp.sum(k(utility_values, xp=cp))),
        [utility_values >= lower_bound, utility_values <= upper_bound],
    )

    assert problem.is_dcp()


@pytest.mark.parametrize(
    ("utility", "kwargs", "utility_values", "expected_wages"),
    [
        ("crra", {"gamma": 0.5}, [-1.0, 0.0, 2.0], [np.inf, -1.0, 0.0]),
        ("crra", {"gamma": 1.5}, [-2.0, 0.0, 1.0], [0.0, np.inf, np.inf]),
        ("cara", {"alpha": 2.0}, [-0.5, 0.0, 0.5], [-1.0, np.inf, np.inf]),
    ],
)
def test_numpy_inverse_utility_returns_infinity_outside_domain(
    utility, kwargs, utility_values, expected_wages
):
    """Invalid inverse utility inputs have infinite rather than clamped cost."""
    k = make_utility_cfg(utility, w0=1.0, **kwargs)["k"]

    with np.errstate(divide="raise", invalid="raise"):
        wages = k(utility_values)

    np.testing.assert_allclose(wages, expected_wages)
