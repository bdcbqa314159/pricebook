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
          and the vectorised sibling arrive with their consumers — a vega, a copula. `norm_ppf`
          arrived at tranche 20 as `inverse_norm_cdf`, pulled by Sobol's need for a monotone
          transform)
  source: standard normal CDF via the exact identity `Phi(x) = (1 + erf(x/sqrt2))/2`; `math.erf`
          from the C library is the numerical provider. (It was `scipy.stats.norm.cdf` through
          tranche 10; scipy is still pinned in pyproject and still used by the TESTS, which is now
          a feature — the oracle and the implementation no longer share a provider.)
          The INVERSE: ACKLAM-2000 (docs/references.md) — the rational approximation's 21
          coefficients, its `p_low = 0.02425` break-points and its Halley refinement, transcribed
          from the held capture of the author's own page. Recommended by JAECKEL-2002 §9.3, which
          prints neither it nor Moro's; tranche 15 deferred the inversion for exactly that reason.
  oracle: Phi(0)=1/2 exactly; Phi(x)+Phi(-x)=1 to 1e-15 across the range; monotone; tails -> 0/1.
          The inverse: agreement with an INDEPENDENT root-find of `norm_cdf` (sharing no coefficient,
          branch or break-point) swept across both break-points; the round-trip to 1e-14; the
          standard quantiles to thirteen digits; oddness about 1/2; strict monotonicity (what Sobol
          requires); Acklam's published 1.15e-9 MEASURED on the unrefined approximation — which is
          what proves the coefficients — with the refinement then reaching machine precision; the
          Halley step shown to beat the Newton step it degrades to; the open-interval domain refused
  crosscheck: none -- the standard normal CDF and its inverse, no finance convention
  slice:  tranche 5 -- the inflation cap/floor (its first consumer); re-based on `math.erf` in
          tranche 11 to clear the `slow` gate on CI (see above); the inverse added at tranche 20
"""

from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)
_PI = math.pi


def norm_cdf(x: float) -> float:
    """The standard-normal cumulative distribution `Phi(x) = (1 + erf(x/sqrt2))/2`."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# ---------------------------------------------------------------------------------------------
# The inverse normal CDF — ACKLAM-2000, refined on our own exact `norm_cdf`.
#
# **Why a transcribed approximation at all, when we own `Phi` exactly?** Because the inverse has no
# closed form, and the alternative — a root-find per call — is both slower and, more importantly,
# *not what a variate generator can afford* once Sobol multiplies the call count. Acklam gives a
# rational approximation good to 1.15e-9, and one Halley step against an exact `Phi` lifts that to
# machine precision. The two halves come from different places and that is the point: the
# coefficients are transcribed, the refinement is anchored on arithmetic we already own.
#
# **The coefficients are 21 magic numbers, which is a transcription risk of a particular kind.** A
# single mistyped digit yields a function that is still monotone, still odd, still roughly right —
# every structural oracle passes. Nothing but a comparison against an *independently computed*
# inverse can catch it, so the oracle's anchor is a bracketed root-find of `norm_cdf` that shares no
# coefficient, no branch and no break-point with the code below, and Acklam's own published accuracy
# bound is measured rather than believed.
# ---------------------------------------------------------------------------------------------

#: Rational-approximation coefficients, transcribed from ACKLAM-2000's pseudo-code. Named `_A`/`_B`
#: for the central region and `_C`/`_D` for the tails, matching the source's `a`/`b`/`c`/`d`; the
#: source indexes from 1, these from 0.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)

#: The break-point between the tail and central approximations. `p_high` is `1 - p_low` by the
#: source's own definition, so it is derived rather than stored — two constants that must agree is
#: one constant too many.
_P_LOW = 0.02425


def _acklam_rational(p: float) -> float:
    """Acklam's rational approximation of `Phi^-1(p)` — accurate to a relative 1.15e-9, unrefined.

    Three regions, exactly as the source writes them: a tail form in `q = sqrt(-2 ln p)` below
    `p_low`, the same form negated above `p_high`, and a central form in `q = p - 1/2`, `r = q^2`.

    Separate from `inverse_norm_cdf` because the oracle must be able to measure *this* — the
    published 1.15e-9 is a claim about the approximation before refinement, and measuring it is what
    proves the 21 coefficients were transcribed correctly. Folded into the refinement it would be
    unreachable, and the coefficients would rest on nothing but a careful read."""
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        return (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        ) / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    )


def _lower_tail_cdf(x: float) -> float:
    """`Phi(x)` as `erfc(-x/sqrt2)/2` — the SAME quantity as `norm_cdf`, computed to survive the tail.

    **Acklam offers this as an explicit alternative and the reason is accuracy, measured:** at
    `x = -7.03`, `(1 + erf(x/sqrt2))/2` carries a *relative* error of **6.1e-06** against an
    independent high-precision CDF, while `erfc(-x/sqrt2)/2` carries **7.4e-15** — nine orders of
    magnitude, because `1 + erf` cancels catastrophically when `erf -> -1`. His pseudo-code prints
    both lines side by side (*"use this line ... OR this; not both"*), and he notes he used Cody's
    `erfc` for exactly this role.

    It is only ever called at `x <= 0` (the refinement is reduced to the lower half by symmetry), so
    this is the branch where the two forms differ and `erfc` is unambiguously the right one.

    **Not a duplicate definition of `norm_cdf` but a tail-stable evaluation of it** — and the gap it
    exposes in `norm_cdf` itself is raised rather than silently patched: that function has shipped
    since tranche 5 and nineteen tranches of oracles read it, so changing its form is not this
    tranche's call. Its own oracle says only *"tails -> 0/1"*, which is an ABSOLUTE statement and so
    passes however bad the relative error gets — §4's domain rule, found again."""
    return 0.5 * math.erfc(-x / _SQRT2)


def inverse_norm_cdf(p: float) -> float:
    """`Phi^-1(p)` to machine precision — Acklam's approximation plus one Halley step.

    *"The relative error of the approximation has absolute value less than 1.15e-9. One iteration of
    Halley's rational method (third order) gives full machine precision."* The step is the source's:

        e <- Phi(x) - p ;  u <- e*sqrt(2*pi)*exp(x^2/2) ;  x <- x - u/(1 + x*u/2)

    `u` is `e` divided by the normal density at `x`, so the update is Newton's with a third-order
    correction in the denominator. Acklam used Cody's FORTRAN `erfc` for the `Phi` it needs; we use
    our own exact `norm_cdf`, which is the same role played by arithmetic we already verify.

    **The domain is the open interval, and the endpoints are refused.** `Phi^-1(0)` is `-inf`, not a
    large number, and a generator that silently returned one would put an infinity into whatever consumes the draw.
    This is also where tranche 15's removed redraw loop relocated to: `UniformSource` no longer
    guards against drawing an exact `0.0` (that branch was unreachable from any test), because the
    check belongs here, where a test can call it directly."""
    if not math.isfinite(p):
        raise ValueError(f"a probability is not finite ({p}); cannot invert it")
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"a probability must lie strictly between 0 and 1 to be inverted, got {p}; "
            f"the endpoints map to -inf and +inf, which are not values a draw may take"
        )

    # REDUCED TO THE LOWER HALF, and this is a correctness matter rather than a tidiness one.
    # The refinement forms `Phi(x) - p`; for `p` near 1 both terms are near 1 and the difference
    # cancels away every significant digit, so the "refinement" makes the answer worse than the raw
    # approximation (measured: it broke the root-find agreement at `p = 1 - 1e-12` by 5e-8). Mapping
    # `p > 1/2` to `1 - p` and negating keeps the CDF evaluation where it is accurate. The oddness
    # of `Phi^-1` is exact, so nothing is approximated by doing this.
    upper = p > 0.5
    q = 1.0 - p if upper else p

    x = _acklam_rational(q)
    error = _lower_tail_cdf(x) - q
    u = error * math.sqrt(2.0 * _PI) * math.exp(x * x / 2.0)
    refined = x - u / (1.0 + x * u / 2.0)
    return -refined if upper else refined
