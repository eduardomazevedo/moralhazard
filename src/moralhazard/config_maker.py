"""Factory functions for utility and distribution primitives.

Provides make_utility_cfg() and make_distribution_cfg() to generate the
callable primitives (u, k, link_function, f, score) needed to initialize a MoralHazardProblem.
"""
from __future__ import annotations
import math
from typing import Callable, Dict, Optional
import numpy as np

ArrayLike = np.ndarray | float

def _asarray(x: ArrayLike) -> np.ndarray:
    """Convert input to float64 numpy array.

    Args:
        x: Scalar or array-like input.

    Returns:
        Numpy array with dtype float64.
    """
    return np.asarray(x, dtype=float)


def _safe_log(x: ArrayLike) -> ArrayLike:
    """Compute natural logarithm with domain protection.

    Returns -inf for x <= 0 (standard NumPy behavior), which is appropriate
    since callers typically exponentiate the result later.

    Args:
        x: Scalar or array-like input.

    Returns:
        Natural logarithm of x.
    """
    x = _asarray(x)
    return np.log(x)


def _lgamma(x: ArrayLike) -> ArrayLike:
    """Compute vectorized log-gamma function.

    Args:
        x: Scalar or array-like input.

    Returns:
        Log of absolute value of gamma function at x.
    """
    vlgamma = np.vectorize(math.lgamma, otypes=[float])
    return vlgamma(_asarray(x))


def _is_integer_array(x: np.ndarray) -> np.ndarray:
    """Check which array elements are finite integers.

    Args:
        x: Numpy array to check.

    Returns:
        Boolean array, True where x is finite and equals floor(x).
    """
    return np.isfinite(x) & (x == np.floor(x))


def make_utility_cfg(
    utility: str,
    *,
    w0: float,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None
) -> Dict[str, Callable]:
    """Create utility function primitives for moral hazard problems.

    Generates broadcastable callables (u, k, link_function) for common
    utility specifications used in contract theory.

    Args:
        utility: Utility type, one of {"log", "crra", "cara"} (case-insensitive).
        w0: Baseline wealth. Agent consumes x + w0 when receiving transfer x.
        gamma: CRRA coefficient. Required if utility == "crra" and gamma != 1.
            When gamma = 1, equivalent to log utility.
        alpha: CARA coefficient. Required if utility == "cara". Must be > 0.

    Returns:
        Dictionary with callables:
            - u(x): Utility from transfer x (agent consumes x + w0).
            - k(u): Inverse utility mapping utils -> transfer (wage).
            - link_function(z): Link function mapping z = λ + μ S(y|a0)
              to utility units for the dual formulation.

    Raises:
        ValueError: If utility type is unknown or required parameters missing.

    Example:
        >>> cfg = make_utility_cfg("log", w0=1.0)
        >>> cfg["u"](0.0)  # u(0) = log(1) = 0
        0.0
    """
    kind = utility.strip().lower()

    if kind == "log" or (kind == "crra" and (gamma is None or np.isclose(gamma, 1.0))):
        # u(x) = log(x + w0)
        def u(x: ArrayLike) -> ArrayLike:
            x = _asarray(x)
            return np.log(x + w0)

        # k(u) = exp(u) - w0
        def k(uval: ArrayLike, xp=np) -> ArrayLike:
            if xp is np:
                uval = _asarray(uval)
            return xp.exp(uval) - w0

        # link_function(z) = log(max(w0, z))
        def link_function(z: ArrayLike) -> ArrayLike:
            z = _asarray(z)
            return np.log(np.maximum(z, w0))

        return {"u": u, "k": k, "link_function": link_function}

    if kind == "crra":
        if gamma is None or np.isclose(gamma, 1.0):
            raise ValueError("For CRRA, provide gamma != 1 (gamma=1 is log case).")

        one_minus_g = 1.0 - gamma
        inv_power = 1.0 / one_minus_g
        inv_gamma = 1.0 / gamma

        # u(x) = (x + w0)^{1-γ}/(1-γ)
        def u(x: ArrayLike) -> ArrayLike:
            x = _asarray(x)
            return np.power(np.maximum(x + w0, 0.0), one_minus_g) / one_minus_g

        # k(u) = (( (1-γ) u )^{1/(1-γ)}) - w0
        def k(uval: ArrayLike, xp=np) -> ArrayLike:
            if xp is np:
                uval = _asarray(uval)
                base = np.maximum((one_minus_g * uval), 0.0)
            else:
                # CVXPY: use cp.maximum
                base = xp.maximum((one_minus_g * uval), 0.0)
            return xp.power(base, inv_power) - w0

        # link_function(z) = max(w0^γ, z)^{(1-γ)/γ}/(1-γ)
        def link_function(z: ArrayLike) -> ArrayLike:
            z = _asarray(z)
            z_clamped = np.maximum(z, np.power(w0, gamma))
            return np.power(z_clamped, one_minus_g * inv_gamma) / one_minus_g

        return {"u": u, "k": k, "link_function": link_function}

    if kind == "cara":
        if alpha is None or alpha <= 0:
            raise ValueError("For CARA, provide alpha > 0.")

        # u(x) = -exp(-α (x + w0)) / α
        def u(x: ArrayLike) -> ArrayLike:
            x = _asarray(x)
            return -np.exp(-alpha * (x + w0)) / alpha

        # k(u) = - (1/α) log(-α u) - w0
        def k(uval: ArrayLike, xp=np) -> ArrayLike:
            if xp is np:
                uval = _asarray(uval)
                # -α u must be > 0; clamp tiny to avoid NaNs
                t = np.maximum(-alpha * uval, 1e-300)
            else:
                # CVXPY: use cp.maximum
                t = xp.maximum(-alpha * uval, 1e-300)
            return -(1.0 / alpha) * xp.log(t) - w0

        # link_function(z) = - 1 / (α * max(exp(α w0), z))
        def link_function(z: ArrayLike) -> ArrayLike:
            z = _asarray(z)
            z_clamped = np.maximum(z, np.exp(alpha * w0))
            return -1.0 / (alpha * z_clamped)

        return {"u": u, "k": k, "link_function": link_function}

    raise ValueError("utility must be one of {'log', 'crra', 'cara'}.")


def make_distribution_cfg(
    dist: str,
    **params
) -> Dict[str, Callable]:
    """Create distribution primitives for moral hazard problems.

    Generates broadcastable callables (f, score) for common outcome
    distributions.

    Args:
        dist: Distribution type, one of {'gaussian', 'poisson', 'exponential',
            'bernoulli', 'geometric', 'binomial', 'gamma', 'student_t'}.
        **params: Distribution-specific parameters:
            - gaussian: sigma (standard deviation, default 1.0).
            - poisson: (no extra params, mean = a > 0).
            - exponential: (no extra params, mean = a > 0).
            - bernoulli: (no extra params, a ∈ (0,1) is success prob).
            - geometric: (no extra params, mean = a > 1, support {1,2,...}).
            - binomial: n (number of trials, integer >= 1).
            - gamma: n (shape > 0, a is scale, mean = n*a).
            - student_t: nu (degrees of freedom > 0), sigma (scale > 0).

    Returns:
        Dictionary with callables:
            - f(y, a): PDF/PMF as function of outcome y and action a.
            - score(y, a): Score function ∂/∂a log f(y|a).

    Raises:
        ValueError: If distribution type is unknown or parameters invalid.

    Example:
        >>> cfg = make_distribution_cfg("gaussian", sigma=2.0)
        >>> cfg["f"](0.0, 0.0)  # f(0|0) for N(0, 2²)
        0.19947114...
    """
    kind = dist.strip().lower()

    if kind == "gaussian":
        sigma = params.get("sigma", None) or 1.0
        if sigma <= 0:
            raise ValueError("gaussian requires sigma > 0.")

        inv_s2 = 1.0 / (sigma * sigma)
        norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            return norm_const * np.exp(-0.5 * inv_s2 * (y - a) ** 2)

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            return (y - a) * inv_s2

        return {"f": f, "score": score}

    if kind == "poisson":
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            out = np.zeros_like(y, dtype=float)
            mask = (y >= 0) & _is_integer_array(y)  # Only non-negative integers
            out[mask] = np.exp(y[mask] * _safe_log(a[mask]) - a[mask] - _lgamma(y[mask] + 1.0))
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            out = np.zeros_like(y, dtype=float)
            mask = (y >= 0) & _is_integer_array(y)
            out[mask] = (y[mask] - a[mask]) / a[mask]
            return out

        return {"f": f, "score": score}

    if kind == "exponential":
        # mean = a > 0, support y >= 0
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            out = np.zeros_like(y, dtype=float)
            mask = y >= 0
            out[mask] = (1.0 / a[mask]) * np.exp(-y[mask] / a[mask])
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)  # Avoid divide by zero
            return (y - a) / (a * a)

        return {"f": f, "score": score}

    if kind == "bernoulli":
        # support y in {0,1}
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            out = np.zeros_like(y, dtype=float)
            mask = _is_integer_array(y) & ((y == 0) | (y == 1))
            out[mask] = np.power(a[mask], y[mask]) * np.power(1.0 - a[mask], 1.0 - y[mask])
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            out = np.zeros_like(y, dtype=float)
            mask = _is_integer_array(y) & ((y == 0) | (y == 1))
            out[mask] = (y[mask] - a[mask]) / (a[mask] - a[mask] * a[mask])
            return out

        return {"f": f, "score": score}

    if kind == "geometric":
        # mean = a > 1, support y in {1,2,...}
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1.0 + 1e-12)
            out = np.zeros_like(y, dtype=float)
            mask = _is_integer_array(y) & (y >= 1)
            q = 1.0 - 1.0 / a
            out[mask] = np.power(q[mask], y[mask] - 1.0) * (1.0 / a[mask])
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            out = np.zeros_like(y, dtype=float)
            mask = _is_integer_array(y) & (y >= 1)
            out[mask] = (y[mask] - a[mask]) / (a[mask] * a[mask] - a[mask])
            return out

        return {"f": f, "score": score}

    if kind == "binomial":
        n = params.get("n", None) or 1.0
        if n <= 0 or abs(n - round(n)) > 1e-9:
            raise ValueError("binomial requires integer n >= 1.")
        n = int(n)

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            out = np.zeros_like(y, dtype=float)
            mask = _is_integer_array(y) & (y >= 0) & (y <= n)
            yy = y[mask]
            coeff_log = _lgamma(n + 1.0) - _lgamma(yy + 1.0) - _lgamma(n - yy + 1.0)
            out[mask] = np.exp(coeff_log + yy * _safe_log(a[mask]) + (n - yy) * _safe_log(1.0 - a[mask]))
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            out = np.zeros_like(y, dtype=float)
            mask = _is_integer_array(y) & (y >= 0) & (y <= n)
            out[mask] = (y[mask] - n * a[mask]) / (a[mask] - a[mask] * a[mask])
            return out

        return {"f": f, "score": score}

    if kind == "gamma":
        # shape n > 0, scale a (mean = n * a)
        n = params.get("n", None) or 1.0
        if n <= 0:
            raise ValueError("gamma requires n > 0 (shape).")

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            out = np.zeros_like(y, dtype=float)
            mask = y > 0
            yy = y[mask]; aa = a[mask]
            log_pdf = ((n - 1.0) * _safe_log(yy) - yy / aa - _lgamma(n) - n * _safe_log(aa))
            out[mask] = np.exp(log_pdf)
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            return (y - n * a) / (a * a)

        return {"f": f, "score": score}

    if kind == "student_t":
        nu = params.get("nu", None) or 5.0
        sigma = params.get("sigma", None) or 1.0
        if nu <= 0 or sigma <= 0:
            raise ValueError("student_t requires nu > 0 and sigma > 0.")

        c = math.gamma((nu + 1.0) / 2.0) / (math.gamma(nu / 2.0) * math.sqrt(np.pi * nu) * sigma)

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            t = (y - a) / sigma
            return c * np.power(1.0 + (t * t) / nu, -(nu + 1.0) / 2.0)

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            return ((nu + 1.0) * (y - a)) / (nu * (sigma * sigma) + (y - a) ** 2)

        return {"f": f, "score": score}

    raise ValueError(
        "dist must be one of {'gaussian','poisson','exponential',"
        "'bernoulli','geometric','binomial','gamma','student_t'}."
    )
