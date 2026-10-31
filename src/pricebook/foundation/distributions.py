"""The normal distribution, behind our own API (L0).

**A thin adapter over scipy, deliberately.** The library's value is in the *finance* — the
convexity, the measure change, the forward construction — not in re-deriving a cumulative normal
that scipy already computes to full double precision. Wrapping it rather than calling scipy
directly from pricers buys two things: one place to pin the behaviour, and one swap point if a
port ever needs its own implementation.

**The oracles must test the finance, never scipy's numerics.** `Phi(0) = 1/2`, the symmetry
`Phi(x) + Phi(-x) = 1` and the tail limits are asserted because *this adapter's* correctness is
what is in question — not scipy's. The put-call parity oracle in the cap/floor tranche leans on
that symmetry holding exactly, which is why it is tested here rather than assumed.

Provenance:
  stock:  _stock/src/pricebook/foundation/distributions.py (purified: the CDF only. `norm_pdf`,
          `norm_ppf` and the vectorised sibling arrive with their consumers — a vega, an implied
          solve, a copula)
  source: standard normal CDF; scipy.stats.norm as the numerical provider (pinned in pyproject,
          since a scipy change can move a last digit and the version is part of the contract)
  oracle: Phi(0)=1/2 exactly; Phi(x)+Phi(-x)=1 to 1e-15 across the range; monotone; tails -> 0/1
  slice:  tranche 5 -- the inflation cap/floor (its first consumer)
"""

from __future__ import annotations

from scipy.stats import norm


def norm_cdf(x: float) -> float:
    """The standard-normal cumulative distribution `Phi(x)`."""
    return float(norm.cdf(x))
