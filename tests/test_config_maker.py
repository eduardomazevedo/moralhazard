"""Tests for the primitive factory functions."""

import cvxpy as cp
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
