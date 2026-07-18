"""Standard-normal distribution — finance-free numerics (L0).

Provenance:
  quarry: python/pricebook/core/ (distributions)
  source: standard normal; `erfc` (stdlib) for the CDF
  oracle: norm_cdf(0)=0.5 and published values; norm_ppf inverts norm_cdf
  slice:  numerics-config (Topic 0 S6)
"""

from __future__ import annotations

import math

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    """Standard-normal probability density."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: float) -> float:
    """Standard-normal cumulative distribution (via `erfc`, machine-accurate)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile). Bisection on `norm_cdf` — accurate and
    dependency-free; the vectorised approximation arrives with its first MC consumer."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)
