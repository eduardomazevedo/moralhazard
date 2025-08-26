from __future__ import annotations
import math
from typing import Callable, Dict, Optional
import numpy as np

ArrayLike = np.ndarray | float

# ---------- helpers ----------
def _pos(x: ArrayLike) -> ArrayLike:
    return np.maximum(x, 0.0)

def _asarray(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)

def _safe_log(x: ArrayLike) -> ArrayLike:
    # log with domain protection (returns -inf for x<=0 which then gets clamped where needed)
    x = _asarray(x)
    return np.log(x)

def _lgamma(x: ArrayLike) -> ArrayLike:
    # vectorized math.lgamma
    vlgamma = np.vectorize(math.lgamma, otypes=[float])
    return vlgamma(_asarray(x))

# ---------- utility factories (u, k, g) ----------
def make_utility_cfg(
    utility: str,
    *,
    w0: float,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None
) -> Dict[str, Callable]:
    """
    Create broadcastable callables (u, k, g) consistent with the LaTeX tables.

    Args:
        utility: one of {"log", "crra", "cara"} (case-insensitive).
        w0: baseline wealth (float).
        gamma: CRRA coefficient (required if utility == "crra" and gamma != 1).
        alpha: CARA coefficient (required if utility == "cara").

    Returns:
        dict with:
          - u(x): utility from transfer x (agent consumes x + w0)
          - k(u): inverse utility -> transfer (wage) that delivers utility u
          - g(z): link from z = λ + μ S(y|a0) into utility units
    """
    kind = utility.strip().lower()

    if kind == "log" or (kind == "crra" and (gamma is None or np.isclose(gamma, 1.0))):
        # u(x) = log(x + w0)
        def u(x: ArrayLike) -> ArrayLike:
            x = _asarray(x)
            return np.log(x + w0)

        # k(u) = exp(u) - w0
        def k(uval: ArrayLike) -> ArrayLike:
            uval = _asarray(uval)
            return np.exp(uval) - w0

        # g(z) = log(max(w0, z))
        def g(z: ArrayLike) -> ArrayLike:
            z = _asarray(z)
            return np.log(np.maximum(z, w0))

        return {"u": u, "k": k, "link_function": g}

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
        def k(uval: ArrayLike) -> ArrayLike:
            uval = _asarray(uval)
            base = np.maximum((one_minus_g * uval), 0.0)
            return np.power(base, inv_power) - w0

        # g(z) = max(w0^γ, z)^{(1-γ)/γ}/(1-γ)
        def g(z: ArrayLike) -> ArrayLike:
            z = _asarray(z)
            z_clamped = np.maximum(z, np.power(w0, gamma))
            return np.power(z_clamped, one_minus_g * inv_gamma) / one_minus_g

        return {"u": u, "k": k, "link_function": g}

    if kind == "cara":
        if alpha is None or alpha <= 0:
            raise ValueError("For CARA, provide alpha > 0.")

        # u(x) = -exp(-α (x + w0)) / α
        def u(x: ArrayLike) -> ArrayLike:
            x = _asarray(x)
            return -np.exp(-alpha * (x + w0)) / alpha

        # k(u) = - (1/α) log(-α u) - w0
        def k(uval: ArrayLike) -> ArrayLike:
            uval = _asarray(uval)
            # -α u must be > 0; clamp tiny to avoid NaNs
            t = np.maximum(-alpha * uval, 1e-300)
            return -(1.0 / alpha) * np.log(t) - w0

        # g(z) = - 1 / (α * max(exp(α w0), z))
        def g(z: ArrayLike) -> ArrayLike:
            z = _asarray(z)
            z_clamped = np.maximum(z, np.exp(alpha * w0))
            return -1.0 / (alpha * z_clamped)

        return {"u": u, "k": k, "link_function": g}

    raise ValueError("utility must be one of {'log', 'crra', 'cara'}.")


# ---------- distribution factories (f, score) ----------
def make_distribution_cfg(
    dist: str,
    **params
) -> Dict[str, Callable]:
    """
    Create broadcastable callables (f, score) for the distributions in your table.

    Args:
        dist: one of {
          'gaussian', 'lognormal', 'poisson', 'exponential',
          'bernoulli', 'geometric', 'binomial', 'gamma', 'student_t'
        }
        params:
          gaussian: sigma
          lognormal: sigma
          poisson: (no extra params)
          exponential: (no extra params)
          bernoulli: (no extra params)
          geometric: (no extra params)  # mean = a > 1
          binomial: n (trials)
          gamma: n (shape > 0)
          student_t: nu (df > 0), sigma (>0)

    Returns:
        dict with:
          - f(y, a): PDF/PMF as function of outcome y and action a
          - score(y, a): ∂/∂a log f(y|a)
    """
    kind = dist.strip().lower()

    if kind == "gaussian":
        sigma = float(params.get("sigma", None) or 1.0)
        if sigma <= 0:
            raise ValueError("gaussian requires sigma > 0.")

        inv_s2 = 1.0 / (sigma * sigma)
        norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            return norm_const * np.exp(-0.5 * inv_s2 * (y - a) ** 2)

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            return (y - a) * inv_s2

        return {"f": f, "score": score}

    if kind == "lognormal":
        sigma = float(params.get("sigma", None) or 1.0)
        if sigma <= 0:
            raise ValueError("lognormal requires sigma > 0.")
        inv_s2 = 1.0 / (sigma * sigma)
        root = np.sqrt(2.0 * np.pi) * sigma

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            out = np.zeros(np.broadcast(y, a).shape)
            mask = y > 0
            yy = y[mask]
            aa = a[mask]
            out[mask] = (1.0 / (yy * root)) * np.exp(-0.5 * inv_s2 * (np.log(yy) - aa) ** 2)
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            out = np.zeros(np.broadcast(y, a).shape)
            mask = y > 0
            out[mask] = (np.log(y[mask]) - a[mask]) * inv_s2
            return out

        return {"f": f, "score": score}

    if kind == "poisson":
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            # PMF: a^y e^{-a} / y!
            # Use exp(log form) for stability; treat non-integer y as 0 by formula anyway
            # Handle negative y values by setting them to 0
            out = np.zeros_like(y, dtype=float)
            mask = (y >= 0) & (y == np.floor(y))  # Only non-negative integers
            out[mask] = np.exp(y[mask] * _safe_log(a[mask]) - a[mask] - _lgamma(y[mask] + 1.0))
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            # Score is only defined for non-negative integer y
            out = np.zeros_like(y, dtype=float)
            mask = (y >= 0) & (y == np.floor(y))
            out[mask] = (y[mask] - a[mask]) / a[mask]
            return out

        return {"f": f, "score": score}

    if kind == "exponential":
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            # mean = a, support y >= 0
            y, a = _asarray(y), _asarray(a)
            # Ensure proper broadcasting
            y, a = np.broadcast_arrays(y, a)
            out = np.zeros_like(y, dtype=float)
            a = np.maximum(a, 1e-300)
            mask = y >= 0
            # Handle scalar case properly
            if np.isscalar(a) or a.size == 1:
                a_val = float(a)
                out[mask] = (1.0 / a_val) * np.exp(-y[mask] / a_val)
            else:
                out[mask] = (1.0 / a[mask]) * np.exp(-y[mask] / a[mask])
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)  # Avoid divide by zero
            return (y - a) / (a * a)

        return {"f": f, "score": score}

    if kind == "bernoulli":
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            return np.power(a, y) * np.power(1.0 - a, 1.0 - y)

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            return (y - a) / (a - a * a)

        return {"f": f, "score": score}

    if kind == "geometric":
        # mean = a > 1, support y in {1,2,...}
        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1.0 + 1e-12)
            out = np.zeros_like(y, dtype=float)
            mask = y >= 1
            # Handle scalar case properly
            if np.isscalar(a) or a.size == 1:
                a_val = float(a)
                q = 1.0 - 1.0 / a_val
                out[mask] = np.power(q, y[mask] - 1.0) * (1.0 / a_val)
            else:
                q = 1.0 - 1.0 / a[mask]
                out[mask] = np.power(q, y[mask] - 1.0) * (1.0 / a[mask])
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            return (y - a) / (a * a - a)

        return {"f": f, "score": score}

    if kind == "binomial":
        n = float(params.get("n", None) or 1.0)
        if n <= 0 or abs(n - round(n)) > 1e-9:
            raise ValueError("binomial requires integer n >= 1.")
        n = int(n)

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            # pmf = C(n,y) a^y (1-a)^{n-y}
            coeff_log = _lgamma(n + 1.0) - _lgamma(y + 1.0) - _lgamma(n - y + 1.0)
            return np.exp(coeff_log + y * _safe_log(a) + (n - y) * _safe_log(1.0 - a))

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            a = np.clip(a, 1e-12, 1 - 1e-12)
            return (y - n * a) / (a - a * a)

        return {"f": f, "score": score}

    if kind == "gamma":
        # shape n > 0, scale a (mean = n * a)
        n = float(params.get("n", None) or 1.0)
        if n <= 0:
            raise ValueError("gamma requires n > 0 (shape).")

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            out = np.zeros_like(y, dtype=float)
            a = np.maximum(a, 1e-300)
            mask = y > 0
            # Handle scalar case properly
            if np.isscalar(a) or a.size == 1:
                a_val = float(a)
                yy = y[mask]
                log_pdf = ( (n - 1.0) * _safe_log(yy)
                            - yy / a_val
                            - _lgamma(n)
                            - n * _safe_log(a_val) )
                out[mask] = np.exp(log_pdf)
            else:
                yy = y[mask]; aa = a[mask]
                log_pdf = ( (n - 1.0) * _safe_log(yy)
                            - yy / aa
                            - _lgamma(n)
                            - n * _safe_log(aa) )
                out[mask] = np.exp(log_pdf)
            return out

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            y, a = np.broadcast_arrays(y, a)
            a = np.maximum(a, 1e-300)
            return (y - n * a) / (a * a)

        return {"f": f, "score": score}

    if kind == "student_t":
        nu = float(params.get("nu", None) or 5.0)
        sigma = float(params.get("sigma", None) or 1.0)
        if nu <= 0 or sigma <= 0:
            raise ValueError("student_t requires nu > 0 and sigma > 0.")

        c = math.gamma((nu + 1.0) / 2.0) / (math.gamma(nu / 2.0) * math.sqrt(np.pi * nu) * sigma)

        def f(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            t = (y - a) / sigma
            return c * np.power(1.0 + (t * t) / nu, -(nu + 1.0) / 2.0)

        def score(y: ArrayLike, a: ArrayLike) -> ArrayLike:
            y, a = _asarray(y), _asarray(a)
            return ((nu + 1.0) * (y - a)) / (nu * (sigma * sigma) + (y - a) ** 2)

        return {"f": f, "score": score}

    raise ValueError(
        "dist must be one of {'gaussian','lognormal','poisson','exponential',"
        "'bernoulli','geometric','binomial','gamma','student_t'}."
    )
