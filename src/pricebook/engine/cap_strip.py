"""The caplet-strip walk the smile-family engines share (L4). Convention C16's structural guard.

Three registered pricers — Bachelier, displaced-diffusion and CEV — walk a `NominalCap`'s strip
identically: partition by the valuation date, read the discount curve, build each period's
`CapletPeriod`, take the **fixing-date** variance horizon, value, and accumulate. Only the
per-period valuation differs. That was duplicated *knowledge* (§6), and re-audit #8's duplication
pass measured it: an 8-engine skeleton, 5 byte-identical period filters, 3 `CapletPeriod` builds and
3 forward-variance expressions.

**What this extraction is FOR, and it is not tidiness.** Convention C16 — *a caplet's volatility
accumulates to the fixing date `T_{i-1}`, never the payment date* — was recorded **per-module**
precisely because no shared value existed to compare: each engine computed its own horizon, so
nothing could assert they agreed. Computing it once here **creates** that value, so a divergence
between the three is no longer expressible, and C16 becomes **guarded** rather than per-module (the
C10/C13 structural form). The error it guards against is silent: with a flat vol the payment horizon
over-states every caplet by `sqrt(T_i/T_{i-1})` while put-call parity, monotonicity and the
vanishing-variance limit all stay green.

**Who is NOT here, measured rather than assumed.** `engine/nominal_cap.py` (Hull-White) shares the
*walk* and not the *variance*: its quantity is a **bond-option** variance over `(expiry, maturity)`
— two dates — where these three take the **forward's** variance over one. There is no shared value
to extract, so folding it in would mean inventing one. `engine/capfloor.py` and `engine/yoy.py` take
an `InflationCapFloor` and a year-on-year swap: different payoffs, so a common walker would need a
protocol over products that share the word "leg" and not its content — §6's named line. Three
consumers, not eight, and the three are the ones that share the shape.

**The exception set is a parameter, not a default.** `cev_caplet` raises `ArithmeticError` when the
chi-squared mixture is unaffordable and the other two cannot; a shared walker with a default
`(KeyError,)` would silently drop that on the next edit, turning a `PricingFailure` into a traceback
out of a registered pricer (§3's failure-is-a-value). So each caller states its own and there is no
default to inherit by accident.

Provenance:
  stock:  none — the extraction is of this tree's own duplication
  source: n/a — no formula here; the mathematics lives in the three engines it serves.
          Convention C16 (`docs/conventions_register.md`) is what it enforces
  oracle: tests/L4/test_cap_strip.py — the three engines agree on the fixing-date horizon by
          construction (one expression), the historical partition, the empty-strip PV-0, and each
          caller's exception set reaching its own `PricingFailure`; plus the three engines' own
          suites, which price bit-identically across the refactor
  crosscheck: none — a control-flow walk, not a numerical convention a library would implement
  slice:  tranche 73 — re-audit #8 remediation, convention C16's guard
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.hull_white import CapletPeriod
from pricebook.products.nominal_cap import NominalCap

#: Values one period, per unit notional, given what the curve says about it and the **fixing-date**
#: variance horizon in years. The horizon arrives computed, which is the point: a caller cannot read
#: the payment date instead, because it is never handed one.
PeriodValue = Callable[[CapletPeriod, float], float]


def price_cap_strip(
    cap: NominalCap,
    market: MarketSnapshot,
    value_period: PeriodValue,
    failures: Sequence[type[BaseException]],
) -> PricingResult | PricingFailure:
    """Sum `value_period` over the strip's remaining periods.

    `failures` is the caller's own exception set, turned into a `PricingFailure` (§3). Anything else
    propagates, because a walker cannot know which exceptions a valuation considers expected.
    """
    currency = cap.notional.currency

    # Invariant 6: a period that has already paid is the shell's business and is excluded from the
    # mark, checked before any market lookup so a settled strip needs no curve.
    periods = [p for p in cap.schedule.periods() if not is_historical(p[1], market.valuation_date)]
    if not periods:
        return PricingResult(pv=Money(0.0, currency))

    try:
        curve = market.curves.discount(currency)
        measure = curve.time_measure
        total = 0.0
        for start, pay, accrual in periods:
            unit = CapletPeriod(curve.df(start), curve.df(pay), accrual)
            total += cap.notional.amount * value_period(unit, measure.year_fraction(start))
    except tuple(failures) as exc:
        return PricingFailure(f"cannot price the cap: {exc}")

    return PricingResult(pv=Money(total, currency))
