"""The nominal cap/floor pricer (L4). MCINERNEY-ZASTAWNIAK §3.5 eq 3.21.

Sums the Hull-White caplets of a strip. The analytics live at L3 (`models/hull_white.py`); this
module's job is the schedule walk, the curve lookups and the valuation-date partition.

**The model's `σ` is a NOMINAL rate volatility here, and that is the point of the tranche.**
BERHAN-2012 §5.3.1 sources `σn` by fitting `(κn, σn)` to ATM nominal cap volatilities — so this is
the forward model that calibration inverts, and `σn` stops being the hand-supplied input it has been
since tranche 6.

Provenance:
  stock:  — built fresh
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §3.5 eq 3.21 (the Hull-White caplet as
          `(1+τK)` puts on a zero-coupon bond); BERHAN-2012 §5.3.1 for the calibration this serves
  oracle: tests/L4/test_nominal_cap.py — the strip equals the sum of its caplets priced
          independently from eq 3.21; `σ → 0` gives the intrinsic strip; cap − floor equals the
          forward-rate swap value (a cross-product identity, different code each side); invariant 6
          excludes settled periods at PV 0; a missing curve is a `PricingFailure`
  crosscheck: quantlib AnalyticCapFloorEngine / HullWhite — deferred to the optional QuantLib tier
  slice:  tranche 10(a) — the Hull-White nominal cap
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.models.hull_white import (
    CapletPeriod,
    HullWhiteModel,
    bond_option_variance,
    hw_caplet,
)
from pricebook.products.nominal_cap import NominalCap


def nominal_caplet_value(
    cap: NominalCap, model: HullWhiteModel, period: tuple
) -> float:
    """One caplet of the strip, for a `(start, pay, accrual)` schedule period.

    `τ` comes from the schedule's own day count (it is what `periods()` yields), never a hardcoded
    1.0 — the carried-field lesson from tranche 3, and the same reason tranche 8's annuity reads it.
    Note the accrual is used **twice and for two different things**: inside `CapletPeriod` as the
    `(1 + τK)` scaling of eq 3.21, and as the payment's own accrual factor. They are the same `τ`,
    which is why it is read once."""
    start, pay, accrual = period
    curve = model.market.curves.discount(cap.notional.currency)
    measure = curve.time_measure
    variance = bond_option_variance(
        model.params, measure.year_fraction(start), measure.year_fraction(pay)
    )
    unit = CapletPeriod(df_start=curve.df(start), df_pay=curve.df(pay), accrual=accrual)
    return cap.notional.amount * hw_caplet(unit, cap.strike, variance, cap.kind)


@register(NominalCap)
def price_nominal_cap(
    cap: NominalCap, model: HullWhiteModel
) -> PricingResult | PricingFailure:
    """Sum eq 3.21 over the strip's remaining periods."""
    market = model.market
    currency = cap.notional.currency

    # Invariant 6: a period that has already paid is the shell's business, excluded from the mark.
    # Checked before any market lookup, so a fully-settled cap needs no curve — tranche 2's
    # precedent, and the reason a matured trade is PV 0 rather than a PricingFailure.
    periods = [p for p in cap.schedule.periods() if p[1] > market.valuation_date]
    if not periods:
        return PricingResult(pv=Money(0.0, currency))

    try:
        total = sum(nominal_caplet_value(cap, model, p) for p in periods)
    except KeyError as exc:
        return PricingFailure(f"cannot price the nominal cap: {exc}")

    return PricingResult(pv=Money(total, currency))
