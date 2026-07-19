"""Standard-normal distribution — thin scipy adapter (L0, S17).

`scipy.stats.norm` behind our own API + provenance, so there is **one** place to pin
behaviour and **one** swap point for the C++ port. Never call scipy from engines/models —
call these.

Provenance:
  quarry: python/pricebook/core/ (distributions)
  source: scipy.stats.norm (pdf/cdf/ppf)
  oracle: norm_cdf(0)=0.5 and published values; norm_ppf inverts norm_cdf; p∉(0,1) raises
  slice:  l0-numerics (Topic 0 gate S17)
"""

from __future__ import annotations

from scipy.stats import norm


def norm_pdf(x: float) -> float:
    """Standard-normal probability density."""
    return float(norm.pdf(x))


def norm_cdf(x: float) -> float:
    """Standard-normal cumulative distribution."""
    return float(norm.cdf(x))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile). `p` must be strictly in (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    return float(norm.ppf(p))
