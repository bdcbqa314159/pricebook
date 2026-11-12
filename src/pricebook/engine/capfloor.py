"""The inflation cap/floor under Jarrow-Yildirim (L4). BERHAN-2012 eq 2.6.5.

    ILCFLT(t,Ti-1,Ti,psi,K,N,w) = w*N*psi*Pn(t,Ti)*[ m*Phi(w*d1) - K*Phi(w*d2) ]

    m  = (Pn(t,Ti-1)/Pn(t,Ti)) * (Pr(t,Ti)/Pr(t,Ti-1)) * e^{C}        (2.6.2)
    d1 = [ ln(m/K) + V^2/2 ] / V ,   d2 = d1 - V ,   K = 1 + kappa     (2.6.5)

This is 2.6.1 -- the lognormal call/put formula -- applied with `m` the `Ti`-forward expectation of
the index ratio (2.6.2) and `V` its log-standard-deviation (2.6.3/2.6.4). **Both come from the
model, composed**: `m` carries tranche 4's convexity `C` (2.5.11) and `V^2` is 2.6.4. Neither is
restated here, so a cap and a YoY swap cannot disagree about the forward ratio -- which is exactly
what the `K = 1` parity oracle checks, through different code on each side.

**Why `sigma_n = sigma_r = 0` is the reduction and not the correctness anchor.** It gives `C = 0`
and `V^2 = sigma_I^2*(Ti - Ti-1)`, collapsing each caplet to plain Black on a deterministic forward
ratio -- clean, exact, and it kills six of `V^2`'s eight terms. So the signs and factors of those
six are checked by the independently-transcribed sweep in `tests/L3/test_jy_variance.py`, not by
the reduction (thread 0031).

  crosscheck: quantlib YoYInflationCapFloor -- deferred to the optional QuantLib tier (0013/0014).

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py (its `jy_yoy_caplet`, structure only;
          its values are regression material, never an anchor -- the stock-is-not-authority rule
          tranche 4 earned the hard way)
  source: BERHAN-2012 (docs/references.md) eqs 1.2.22-1.2.23, 2.6.1-2.6.5
  oracle: eq 2.6.5 against an independent transcription over strikes and both kinds (primary);
          sigma_n=sigma_r=0 reduces to plain Black exactly, with a companion that the rate vols
          move the price; put-call parity, and at K=1 it IS tranche 4's YoY leg; deep-ITM tends to
          the forward annuity, deep-OTM to zero; no-vol gives intrinsic; matured -> PV 0
  slice:  tranche 5 -- the inflation cap/floor
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation import Money, PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.black import black_price
from pricebook.models.index_ratio import IndexRatioModel
from pricebook.products.inflation import InflationCapFloor


def caplet_value(
    product: InflationCapFloor, model: IndexRatioModel, period: tuple
) -> float:
    """One caplet or floorlet, eq 2.6.5, for a `(start, pay, accrual)` schedule period.

    Per unit notional times `N` at the end — the discounting is `Pn(t,Ti)` and the payoff is on the
    index *ratio*, so there is no separate real-curve discounting here: `Pr` enters only through
    `m`.

    **Takes the model as a PROTOCOL since tranche 12, not as a concrete Jarrow–Yildirim.** `m` and
    `V²` are the only things it asks of a model, so a second model is served by implementing
    `IndexRatioModel` rather than by a branch in here (§1's no-ladder law). It also now takes the
    schedule period as it comes from `periods()` rather than a pair of year fractions, which removes
    a date → year-fraction → date round trip that had no reason to exist."""
    nominal = model.market.curves.discount(product.notional.currency)
    start, pay, accrual = period
    return (
        product.notional.amount
        * accrual
        * nominal.df(pay)
        * black_price(
            model.expected_index_ratio(start, pay),
            1.0 + product.strike,
            model.index_ratio_variance(start, pay),
            product.kind,
        )
    )


@register(InflationCapFloor, IndexRatioModel)
def price_inflation_capfloor(
    product: InflationCapFloor, model: IndexRatioModel
) -> PricingResult | PricingFailure:
    """Sum eq 2.6.5 over the strip's remaining periods."""
    market = model.market
    currency = product.notional.currency

    # Invariant 6: a strip whose every payment has passed is settled. PV 0 before any market
    # lookup, per tranche 2's precedent.
    periods = [p for p in product.schedule.periods() if not is_historical(p[1], market.valuation_date)]
    if not periods:
        return PricingResult(pv=Money(0.0, currency))

    try:
        total = sum(caplet_value(product, model, period) for period in periods)
    except KeyError as exc:
        return PricingFailure(f"cannot price the inflation cap/floor: {exc}")

    return PricingResult(pv=Money(total, currency))
