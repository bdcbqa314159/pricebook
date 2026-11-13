"""The bivariate normal CDF (L0) - `Phi2(h,k;rho) = P(X <= h, Y <= k)` for correlated standard normals.

Needed by wrong-way risk (tranche 36b), and new: the tree had no bivariate normal, and `src/` does
not use scipy (see `quadrature.py` for why a pinned reproducibility contract excludes an adaptive
integrator whose node set can shift with a dependency's minor release).

**Derived, not transcribed** - §4 rates a derivation above a transcription, and one is available
here. Differentiating the bivariate density in the *correlation* (Plackett's identity) gives

    d Phi2 / d rho = exp(-(h^2 - 2 rho h k + k^2) / (2(1 - rho^2))) / (2 pi sqrt(1 - rho^2))

so

    Phi2(h,k;rho) = Phi(h) Phi(k) + integral_0^rho (that density) dt

**That form is chosen for a property the oracles rely on, not for elegance.** At `rho = 0` the
integral is empty, so independence `Phi2 = Phi(h)Phi(k)` is **exact by construction** rather than a
numerical limit - which is what lets tranche 36b's `rho -> 0` reduction to the independent-default
CVA be bit-identical instead of merely close. An implementation built on the double integral, or on
a series in `rho`, would make that reduction a tolerance question.

**Fixed Gauss-Legendre, for `quadrature.py`'s reason.** The integrand is smooth on `[0, rho]` for
`|rho| < 1`, so a fixed rule is deterministic, identical on both CI operating systems, and has no
tolerance knob to turn until a test passes. Its accuracy is a property the oracles measure rather
than a parameter the author tunes.

**The validity regime, measured - and a consumer must respect both halves.**

* **In the value.** `Phi2 = Phi(h)Phi(k) + integral`, and in the deep joint tail with `rho < 0` the
  integral very nearly cancels the product: at `h = k = -3.5, rho = -0.5` the answer is 8e-14 while
  the intermediate is 5e-8, so six digits cancel. Relative accuracy is ~1e-7 or better while
  `Phi2 >~ 1e-8`, and degrades below; absolute accuracy (~1e-16) holds throughout. Below ~1e-14 the
  question is moot, because **scipy itself returns exactly 0.0 there** - at `h=k=-4, rho=-0.5` this
  gives 3.5e-17 and scipy gives 0. The high-precision reference fails before we do, which is the
  tranche-20a lesson in its sharpest form.
* **In the correlation.** The integrand is peaked as `|rho| -> 1` and a fixed rule cannot resolve
  it: relative error against scipy at `h=k=0.2` is 4e-16 at `rho=0.95`, 8e-16 at 0.99, **4.3e-08 at
  0.999** and **9.9e-05 at 0.9999**. So `|rho| <~ 0.99`; `|rho| = 1` itself is exact (below), and
  the gap between is where a consumer must not live.

**`|rho| = 1` is handled in closed form, not by quadrature**, because the integrand is singular
there: the comonotone limit is `Phi(min(h,k))` and the countermonotone one is `max(Phi(h)+Phi(k)-1, 0)`.
Approaching a singularity with a fixed rule is the kind of silent inaccuracy this module exists to
avoid.

Provenance:
  stock:  none - the tree had no bivariate normal
  source: none external - Plackett's correlation-derivative identity is standard and is DERIVED in
          the docstring above rather than transcribed, which §4 rates higher. The cross-check is
          `scipy.stats.multivariate_normal`, used by the ORACLE only and never on the production
          path (the tranche-20a arrangement)
  oracle: tests/L0/test_bivariate_normal.py (PRE-REGISTERED) - independence EXACT at `rho = 0`; the
          marginals, the impossible corner, the comonotone limit, the reflection identity
          `Phi2(h,k;rho) + Phi2(-h,k;-rho) = Phi(k)`, symmetry and monotonicity; agreement with
          scipy in the body at 1e-12 absolute AND **in the joint tail at RELATIVE tolerance**,
          because an absolute check is blind where the value is ~1e-9 - the regime wrong-way risk
          actually uses
  crosscheck: scipy.stats.multivariate_normal - in the oracle, deliberately not in `src/`
  slice:  tranche 36b - wrong-way risk: the bivariate primitive it needs
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.foundation.distributions import norm_cdf

#: 96 nodes. Measured: the reflection identity and the scipy body agreement hold to ~1e-13 from 48
#: nodes upward and stop improving past ~96, so this sits just past where the rule converges rather
#: than at a round number someone liked. The joint tail is checked separately, at relative
#: tolerance, because that is the regime the consumer lives in.
_NODES = 96
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _legendre_rule(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on `[-1, 1]`, computed once per node count."""
    return np.polynomial.legendre.leggauss(nodes)


_RULE = _legendre_rule(_NODES)


def bivariate_normal_cdf(h: float, k: float, correlation: float) -> float:
    """`P(X <= h, Y <= k)` for standard normals with correlation `correlation`."""
    if not math.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
        raise ValueError(f"the correlation must be finite and in [-1, 1], got {correlation}")
    if correlation == 1.0:
        return norm_cdf(min(h, k))
    if correlation == -1.0:
        return max(norm_cdf(h) + norm_cdf(k) - 1.0, 0.0)
    if correlation == 0.0:
        # Exact, and deliberately not routed through the (empty) quadrature: the consumer's
        # reduction oracle asserts bit-identity, which a zero-width rule would only approximate.
        return norm_cdf(h) * norm_cdf(k)

    abscissae, weights = _RULE
    # Map [-1,1] onto [0, correlation].
    half = 0.5 * correlation
    points = half * (abscissae + 1.0)
    one_minus = 1.0 - points * points
    density = np.exp(
        -(h * h - 2.0 * points * h * k + k * k) / (2.0 * one_minus)
    ) / (2.0 * math.pi * np.sqrt(one_minus))
    return norm_cdf(h) * norm_cdf(k) + float(half * np.dot(weights, density))
