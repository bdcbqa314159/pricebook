"""The normal distribution, behind our own API (L0).

**`Phi(x) = (1 + erf(x/sqrt2))/2`, from `math.erf` — and it was `scipy.stats.norm.cdf` until
tranche 11.** The identity is exact and `math.erf` is the C library's, accurate to about one ULP; the
reason for the change was measured, not aesthetic. `scipy.stats.norm.cdf` costs **10.70 us per
call** against **0.04 us** for the erf form — 270x — because `scipy.stats` carries a large
per-call distribution-object overhead that a scalar pricer pays on every caplet. Two `norm_cdf`
calls per caplet made it roughly 40% of the whole suite's runtime, and it is what pushed tranche
11's calibration file over the `slow` gate's ceiling on CI (10.6s against a 10.0s limit) while
passing locally. The swap took the full suite from 4.5s to 0.78s.

**The swap also made the cap/floor oracles MORE independent, which is the better reason to keep
it.** Every test that checks a Black price transcribes the formula using `scipy.stats.norm` — so
with the code on `math.erf` there are now genuinely two implementations, one on each side of the
oracle, where before the test and the code shared scipy's. Charter section 4 asks for exactly that.

The adapter remains the one place the behaviour is pinned, and the one swap point if a port needs
its own implementation — which is what made this a four-line change rather than a sweep.

**The oracles must test the finance, never scipy's numerics.** `Phi(0) = 1/2`, the symmetry
`Phi(x) + Phi(-x) = 1` and the tail limits are asserted because *this adapter's* correctness is
what is in question — not scipy's. The put-call parity oracle in the cap/floor tranche leans on
that symmetry holding exactly, which is why it is tested here rather than assumed.

Provenance:
  stock:  _stock/src/pricebook/foundation/distributions.py (purified: the CDF only. `norm_pdf`,
          `norm_ppf` and the vectorised sibling arrive with their consumers — a vega, an implied
          solve, a copula)
  source: standard normal CDF via the exact identity `Phi(x) = (1 + erf(x/sqrt2))/2`; `math.erf`
          from the C library is the numerical provider. (It was `scipy.stats.norm.cdf` through
          tranche 10; scipy is still pinned in pyproject and still used by the TESTS, which is now
          a feature — the oracle and the implementation no longer share a provider.)
  oracle: Phi(0)=1/2 exactly; Phi(x)+Phi(-x)=1 to 1e-15 across the range; monotone; tails -> 0/1
  slice:  tranche 5 -- the inflation cap/floor (its first consumer); re-based on `math.erf` in
          tranche 11 to clear the `slow` gate on CI (see above)
"""

from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """The standard-normal cumulative distribution `Phi(x) = (1 + erf(x/sqrt2))/2`."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))
