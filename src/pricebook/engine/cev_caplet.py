"""The CEV caplet pricer (L4). BROCKHAUS-2016 §6.1 (Cox 1996); MCINERNEY-ZASTAWNIAK §8.3.

**The formula is transcribed from BROCKHAUS-2016, not from the arc's own text, and that is a
deliberate re-sourcing.** MCINERNEY-ZASTAWNIAK §8.3 gives the CEV SDE, the absorbing-boundary
discussion and two *simulated histograms* (Figures 8.1/8.2), then pivots to the displaced-diffusion
approximation as the practical answer. It prints no closed form and no worked example, and §4 forbids
fitting a number to a figure. BROCKHAUS-2016 §6.1 p.79-80 prints Cox's (1996) formula and is already
`worked-from`, so the capability is sourced rather than inferred.

**The printed formula is WRONG, and this module implements the corrected one.** As printed:

    C(K,T) = S0 Q(2w; 2 + 1/(1-rho), 2z) - K [1 - Q(2w; 2 - 1/(1-rho), 2z)]

Both chi-squared terms take `2w` as the argument and `2z` as the noncentrality, and the first term
is missing its complement. `-raw` and `-layout` print it identically, so transcription independence
is powerless here — this is the source's own misprint, which §4 says only a requirement oracle can
catch. Three requirements catch it, and all three are in the oracle beside the corrected form:
`K -> 0` must give the forward (as printed: 0), `K -> inf` must give zero (as printed: the forward),
and at `beta = 0.5` — the other source's stated market practice — the printed second term asks for
**zero degrees of freedom** and is simply undefined. The corrected form was then **ruled by Monte
Carlo**, not by recall: a 600k-path Euler scheme with absorption agrees to under 1 SE at three
strikes.

**The horizon is the TOTAL VARIANCE, not the expiry, and that is structural rather than a
convention.** Brockhaus reduces the general process to the driftless unit-volatility
`dS = S^rho dW` (eq 6.9) by a space-time change, under which the clock *becomes* the accumulated
variance. A caplet needs no more: the forward is already a martingale under its own forward measure,
so the drift terms of eq (6.10) do not arise and the time change is exactly
`V = int sigma^2 ds` to the FIXING date (convention C16).

Provenance:
  stock:  none — the migration tree carried no CEV pricer
  source: BROCKHAUS-2016 (docs/references.md) §6.1 p.79-80 — Cox's (1996) call formula, the `z`/`w`
          definitions and the space-time change of eq (6.9)/(6.10); MCINERNEY-ZASTAWNIAK §8.3 for
          the forward-rate SDE, the absorbing boundary and the absorbed-fraction claims the
          absorption probability reproduces
  oracle: tests/L4/test_cev_caplet.py — the pointwise transcription against a scipy-backed reference
          independent of our own chi-squared primitive; the three requirement oracles that condemned
          the printed form, each asserted to FAIL on it; an independent Monte-Carlo simulation with
          absorption, banded as 4*SE + a measured 0.5% discretisation ceiling; the source's two
          prose claims about the absorbed fraction; the asymmetric Black and Bachelier reductions;
          put-call parity, the intrinsic limit, monotonicity in vol and the measured skew
  crosscheck: quantlib has no CEV cap engine — the chi-squared primitive is cross-checked against
          scipy.stats.ncx2 instead (foundation/noncentral_chisquare.py), deferred to the optional
          QuantLib tier
  slice:  tranche 72 — the CEV smile model
"""

from __future__ import annotations

import math

from pricebook.engine.registry import register
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money
from pricebook.foundation.noncentral_chisquare import (
    noncentral_chisquare_cdf,
    noncentral_chisquare_sf,
)
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.cev import CevModel
from pricebook.models.hull_white import CapletPeriod
from pricebook.products.nominal_cap import NominalCap

#: Noncentrality above which the chi-squared mixture is not affordable, and the price is REFUSED.
#:
#: The mixture needs O(sqrt(lambda)) terms around the Poisson mode, so its cost grows without bound:
#: measured on this primitive, 0.12s at 1e6, 0.38s at 1e7 and 1.18s at 1e8. 1e7 is the ceiling that
#: keeps a single call under half a second.
#:
#: **Beyond it this refuses rather than returning the intrinsic, and the first draft of this module
#: got that wrong in the dangerous direction.** The tempting guard is "a large `z` means the variance
#: is negligible, so the value IS the intrinsic" — and that premise is FALSE. `z` is
#: `F^(2(1-beta)) / (2 (1-beta)^2 V)`, so it also diverges as `beta -> 1` with the variance *fixed*:
#: at `beta = 1 - 1e-5` and a 20% vol the guard returned 0.0 for an at-the-money caplet worth
#: 0.00398. A silent plausible wrong number is exactly what §3's failure-is-a-value rule exists to
#: prevent, and it was caught only because the convergence-rate oracle was written first.
_MAX_NONCENTRALITY = 1.0e7

#: Moneyness, in units of the forward's own standard deviation over the period, beyond which the
#: option is deterministic and its value is EXACTLY the intrinsic. `Phi(-40)` is about 1e-350 — below
#: anything a double represents — so no path reaches the strike. This is the honest form of the
#: guard the constant above describes getting wrong: the deterministic case is a statement about
#: MONEYNESS relative to the dispersion, never about `z` alone.
_DETERMINISTIC_SIGMAS = 40.0


def cev_absorption_probability(forward: float, variance: float, beta: float) -> float:
    """`P(F_T = 0)` — the mass the absorbing barrier at zero collects by the fixing date.

    The central chi-squared survival `Qbar(2z; 1/(1-beta))`, which falls out of the same `z` the call
    is built on. It is public because it is what separates this model from Bachelier at `beta = 0`
    (they agree only where this is negligible) and because it is the quantity MCINERNEY-ZASTAWNIAK
    states in prose for its two figures — the structural way to reproduce a plot (§4)."""
    if variance <= 0.0:
        return 0.0
    return noncentral_chisquare_sf(2.0 * _z(forward, variance, beta), _dof(beta), 0.0)


def _dof(beta: float) -> float:
    """`1/(1-beta)` — the degrees of freedom both chi-squareds are built from."""
    return 1.0 / (1.0 - beta)


def _z(level: float, variance: float, beta: float) -> float:
    """`level^(2(1-beta)) / (2 (1-beta)^2 V)` — Brockhaus's `z` for the forward, `w` for the strike."""
    return level ** (2.0 * (1.0 - beta)) / (2.0 * (1.0 - beta) ** 2 * variance)


def cev_call(forward: float, strike: float, variance: float, beta: float) -> float:
    """Cox's CEV call, undiscounted and per unit notional — the CORRECTED form.

        C = F * Qbar(2w; 2 + 1/(1-beta), 2z)  -  K * Q(2z; 1/(1-beta), 2w)

    Note which function each term takes: the first is the **survival** and the second the **CDF**,
    and the argument/noncentrality pairs are **swapped** between them. Both matter — the source
    prints `1 - Q` for the first, and forming it that way destroys the deep out-of-the-money tail
    (`foundation/noncentral_chisquare` measures it: exactly 0.0 where the truth is 8.7e-32), which is
    why the primitive supplies a native survival.
    """
    dispersion = forward**beta * math.sqrt(max(variance, 0.0))
    if dispersion == 0.0 or abs(forward - strike) > _DETERMINISTIC_SIGMAS * dispersion:
        return max(forward - strike, 0.0)
    z, w = _z(forward, variance, beta), _z(strike, variance, beta)
    if max(z, w) * 2.0 > _MAX_NONCENTRALITY:
        raise ArithmeticError(
            f"the CEV chi-squared mixture is not affordable at a noncentrality of "
            f"{max(z, w) * 2.0:.3g} (forward {forward}, strike {strike}, variance {variance}, "
            f"beta {beta}); this is reached as beta approaches 1, where the degrees of freedom "
            f"1/(1-beta) diverge — measured, the closed form is sound to 1-beta = 1e-3 and not "
            f"beyond"
        )
    dof = _dof(beta)
    return forward * noncentral_chisquare_sf(2.0 * w, 2.0 + dof, 2.0 * z) - strike * (
        noncentral_chisquare_cdf(2.0 * z, dof, 2.0 * w)
    )


def cev_caplet(
    unit: CapletPeriod, strike: float, variance: float, beta: float, kind: CapFloorKind
) -> float:
    """One caplet or floorlet of the strip, per unit notional.

    The floorlet comes from **put-call parity** rather than a second closed form: the forward is a
    martingale under its own forward measure — absorption at zero is consistent with that, the
    barrier redistributes mass without moving the mean — so `floor = call - (F - K)` exactly. One
    formula, and parity then holds by construction rather than by two transcriptions agreeing.
    """
    forward = unit.forward_rate()
    call = cev_call(forward, strike, variance, beta)
    value = call if kind is CapFloorKind.CAP else call - (forward - strike)
    return unit.accrual * unit.df_pay * value


@register(NominalCap, CevModel)
def price_cev_cap(cap: NominalCap, model: CevModel) -> PricingResult | PricingFailure:
    """Sum the CEV caplet over the strip's remaining periods."""
    market = model.market
    currency = cap.notional.currency

    # Invariant 6: an already-paid period is the shell's business, checked before any market lookup.
    periods = [p for p in cap.schedule.periods() if not is_historical(p[1], market.valuation_date)]
    if not periods:
        return PricingResult(pv=Money(0.0, currency))

    try:
        curve = market.curves.discount(currency)
        measure = curve.time_measure
        total = 0.0
        for start, pay, accrual in periods:
            unit = CapletPeriod(curve.df(start), curve.df(pay), accrual)
            variance = model.vol * model.vol * measure.year_fraction(start)
            total += cap.notional.amount * cev_caplet(
                unit, cap.strike, variance, model.beta, cap.kind
            )
    except (KeyError, ArithmeticError) as exc:
        return PricingFailure(f"cannot price the cap: {exc}")

    return PricingResult(pv=Money(total, currency))
