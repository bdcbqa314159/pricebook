"""The year-on-year inflation swap under Jarrow-Yildirim (L4). BERHAN-2012 eq 2.5.13.

    YYIIS(0,T,Psi,N) = N*psi_1*[Pr(0,T1) - Pn(0,T1)]
                     + N*sum_{i=2..M} psi_i*[ Pn(0,Ti-1)*Pr(0,Ti)/Pr(0,Ti-1)*e^C - Pn(0,Ti) ]

**The first cash flow is priced by the ZERO-COUPON inflation leg formula** (1.2.13), because its
previous fixing is already known -- the thesis is explicit about this. So the first term composes
tranche 2's verified `inflation_leg_value` rather than restating it, and the cross-product
reconciliation oracle is a statement of the formula's own structure.

**Every later period carries `e^C`, the convexity** (2.5.11), which the model supplies. `C` is
where the model enters at all: with `sigma_r = 0` it is exactly zero and the whole swap collapses
to the deterministic-real-rate limit -- the tranche's anchor oracle, exact by construction rather
than numerically.

**At inception this prices off curves alone** -- eq 2.5.13 contains no index level, the first term
being `Pr - Pn` and the rest forward ratios. Requiring a fixing would be a plausible-looking error,
so a test asserts a YoY swap prices without one.

**t = 0 only.** The seasoned form (eq 2.5.12, `iota(t) > 1`) needs the *historical* fixing
`I(T_{iota-1})`, i.e. an index **series** in the snapshot -- a new market-data shape, deferred with
the indexation-lag resolution it shares a consumer with.

  crosscheck: quantlib YoYInflationSwap -- deferred to the optional QuantLib tier (0013/0014).

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py (the ZC/YoY structure, mined for
          shape). Its `inflation_advanced.yoy_convexity_adjustment` is deliberately NOT used -- a
          first-order approximation with rho as a factor and an unused real_vol argument.
  source: BERHAN-2012 (docs/references.md) eqs 1.2.17-1.2.18, 2.5.9-2.5.13
  oracle: eq 2.5.13 against an independent transcription; sigma_r -> 0 gives the deterministic
          limit exactly (the anchor), with a companion test that the convexity MOVES the leg so
          the anchor is not vacuous; the first period equals tranche 2's ZCIIS leg; par-rate
          reprice; matured -> PV 0; prices with no index fixing
  slice:  tranche 4 -- the year-on-year swap and Jarrow-Yildirim
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.engine.registry import register
from pricebook.foundation import Money, PricingFailure, PricingResult
from pricebook.market.fixing_series import reference_level
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.jarrow_yildirim import JarrowYildirim
from pricebook.products.inflation import YearOnYearInflationSwap


def _seasoning_ratio(
    swap: YearOnYearInflationSwap, market: MarketSnapshot, reset_date: date
) -> float:
    """`I(t) / I(T_{iota(t)-1})` — the realised index growth since the current period's reset.

    Exactly 1.0 at inception, where `t` and `reset_date` are the same day and so resolve to the
    same print. Both levels come from the one shared resolution atom, so the numerator and
    denominator cannot be read under different conventions."""
    if market.valuation_date <= reset_date:
        # **Not an assumption — a definition.** The period has not begun accruing, so no time has
        # passed since its reset and the ratio is exactly 1. This is the one case where eq 2.5.13
        # is the whole truth, and it is why a year-on-year swap at inception needs no fixing at
        # all: the first term really is `Pr − Pn`. Tranche 4 asserted that
        # (`test_the_yoy_swap_needs_NO_index_fixing_at_inception`) and the assertion is correct, so
        # requiring a series here would be a regression dressed as rigour.
        #
        # The guard is on the DATES, never on "is a series present" — a missing series must still
        # fail for a seasoned trade rather than fall back to 1.0, which was the original defect.
        return 1.0
    try:
        series = market.fixings[swap.index]
    except KeyError:
        raise KeyError(
            f"no fixing series for {swap.index}; refusing to assume the index has not moved "
            f"since the period's reset on {reset_date}"
        ) from None
    current = reference_level(series, market.valuation_date, swap.index)
    reset = reference_level(series, reset_date, swap.index)
    return current / reset


def yoy_inflation_leg_value(swap: YearOnYearInflationSwap, model: JarrowYildirim) -> float:
    """The inflation leg of eq **2.5.12** — the general `t`, seasoned form. Raises on missing market
    data; the pricer turns that into a `PricingFailure`.

    **This was eq 2.5.13 until tranche 7, and that was a silent wrong number at any seasoned
    valuation date.** 2.5.13 is 2.5.12 specialised to `t = 0`, where `I(t)/I(T_0) = I0/I0 = 1`; the
    tranche 4 implementation applied that unit ratio *unconditionally*, so valuing a seasoned trade
    it returned the answer for "the index has not moved since the last reset" — never reading the
    published reset, and never refusing either. Measured on a 3y annual swap 18 months in, with
    inflation realised at 2%/yr since the reset: **98.7 bp of notional, 34.9% of this leg** (196.5 bp
    at 4%). No oracle caught it because every oracle valued at inception, where the two forms
    coincide exactly — the oracle's domain excluded the failure (charter §4)."""
    market = model.market
    currency = swap.notional.currency
    nominal = market.curves.discount(currency)
    real = market.curves.real(currency)
    measure = nominal.time_measure
    schedule = list(swap.leg.schedule.periods())
    periods = [p for p in schedule if p[1] > market.valuation_date]
    if not periods:
        return 0.0

    # The FIRST remaining period is `iota(t)`: in progress (or, at inception, just starting). It
    # pays on the ratio to an ALREADY-PUBLISHED reset, so it is the zero-coupon form (1.2.13) with
    # `I(T_{iota-1})` as its base -- a print, never a curve-implied forward. At inception the
    # valuation date IS that period's start, so both resolve to the same print and the ratio is
    # exactly 1.0, recovering 2.5.13 by construction rather than by a numerical limit.
    first_start, first_pay, first_psi = periods[0]
    ratio = _seasoning_ratio(swap, market, first_start)
    total = first_psi * (ratio * real.df(first_pay) - nominal.df(first_pay))

    # Later periods carry the forward ratio and the convexity.
    for start, pay, psi in periods[1:]:
        c = model.convexity(0.0, measure.year_fraction(start), measure.year_fraction(pay))
        forward_real = real.df(pay) / real.df(start)
        total += psi * (nominal.df(start) * forward_real * math.exp(c) - nominal.df(pay))

    return swap.notional.amount * total


def yoy_fixed_leg_value(swap: YearOnYearInflationSwap, model: JarrowYildirim) -> float:
    """`N·Σ ψᵢ·C_fix·Pn(0,Tᵢ)` — eq 1.2.18 discounted. Historical periods excluded (invariant 6)."""
    market = model.market
    nominal = market.curves.discount(swap.notional.currency)
    return swap.notional.amount * sum(
        swap.leg.fixed_rate * psi * nominal.df(pay)
        for _, pay, psi in swap.leg.schedule.periods()
        if pay > market.valuation_date
    )


def yoy_par_rate(swap: YearOnYearInflationSwap, model: JarrowYildirim) -> float:
    """The fixed rate that makes the swap worth zero: the leg over the annuity.

    Solved in closed form rather than by iteration — the fixed leg is linear in the rate — and it
    composes the **same** `yoy_inflation_leg_value` and annuity the pricer composes, so a par
    reprice cannot disagree with the price. (That also means the par oracle telescopes, which is
    why it is kept as a consistency guard only and the anchor is the `σr → 0` reduction.)"""
    market = model.market
    nominal = market.curves.discount(swap.notional.currency)
    annuity = swap.notional.amount * sum(
        psi * nominal.df(pay)
        for _, pay, psi in swap.leg.schedule.periods()
        if pay > market.valuation_date
    )
    if annuity == 0.0:
        raise ValueError("a matured swap has no par rate")
    return yoy_inflation_leg_value(swap, model) / annuity


@register(YearOnYearInflationSwap)
def price_yoy_inflation_swap(
    swap: YearOnYearInflationSwap, model: JarrowYildirim
) -> PricingResult | PricingFailure:
    """Mark a YoY inflation swap from the inflation **receiver's** side: leg minus fixed leg."""
    market = model.market
    currency = swap.notional.currency

    # Invariant 6: all periods historical ⇒ the trade is settled. PV 0 before any market lookup,
    # per tranche 2's precedent, so a matured swap needs no curves.
    if all(pay <= market.valuation_date for _, pay, _ in swap.leg.schedule.periods()):
        return PricingResult(pv=Money(0.0, currency))

    try:
        pv = yoy_inflation_leg_value(swap, model) - yoy_fixed_leg_value(swap, model)
    except KeyError as exc:
        return PricingFailure(f"cannot price the year-on-year swap: {exc}")
    return PricingResult(pv=Money(pv, currency))
