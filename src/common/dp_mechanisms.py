"""DP mechanisms: Analytic Gaussian (Balle & Wang, ICML'18) and Exponential."""
import numpy as np
from math import sqrt
from scipy.special import erf


def _calibrate_analytic_gaussian(epsilon, delta, GS, tol=1e-12):
    """Calibrate Gaussian noise for (epsilon, delta)-DP.

    Uses the high-precision mpmath version for large epsilon to avoid overflow.
    Adapted from: https://github.com/BorjaBalle/analytic-gaussian-mechanism
    """
    # for large epsilon, exp(epsilon) overflows float64 — use mpmath
    use_mpmath = epsilon > 500

    if use_mpmath:
        import mpmath
        from mpmath import mp

        if epsilon <= 1000:
            mp.dps = 500
        elif epsilon <= 2500:
            mp.dps = 1100
        else:
            mp.dps = 2200

        _exp = mpmath.exp
        _sqrt = mpmath.sqrt

        def Phi(t):
            return 0.5 * (1.0 + mpmath.erf(t / _sqrt(2.0)))
    else:
        from math import exp as _exp

        _sqrt = sqrt

        def Phi(t):
            return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def caseA(eps, s):
        return Phi(_sqrt(eps * s)) - _exp(eps) * Phi(-_sqrt(eps * (s + 2.0)))

    def caseB(eps, s):
        return Phi(-_sqrt(eps * s)) - _exp(eps) * Phi(-_sqrt(eps * (s + 2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if delta == delta_thr:
        alpha = 1.0
    else:
        if delta > delta_thr:
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: _sqrt(1.0 + s / 2.0) - _sqrt(s / 2.0)
        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: _sqrt(1.0 + s / 2.0) + _sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha * GS / _sqrt(2.0 * epsilon)
    return float(sigma)


class AnalyticGaussianMechanism:
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = _calibrate_analytic_gaussian(epsilon, delta, sensitivity)

    def privatize(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.sigma, size=x.shape).astype(x.dtype)
        return x + noise


class ExponentialMechanism:
    """Exponential mechanism for selecting a cluster based on utility scores."""

    def __init__(self, epsilon: float, sensitivity: float):
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def select(self, utilities: np.ndarray) -> int:
        scores = (self.epsilon * utilities) / (2 * self.sensitivity)
        scores = scores - scores.max()  # numerical stability
        probs = np.exp(scores)
        probs = probs / probs.sum()
        return np.random.choice(len(probs), p=probs)
