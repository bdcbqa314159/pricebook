"""Seasoned-trade guard — the honest failure for a current in-progress period (L4).

Invariant 6 excludes PAST periods from the mark (`future_periods`), but the CURRENT period — one
that started before the valuation date yet pays after it — cannot be priced by the curves alone: its
float coupon is partially realized and needs the historical fixings (the real fix is the accrued /
fixings-on-engine work, Batch F / #3b). Until then, every pricer fails with a NAMED message that says
what happened, not a raw L0 date-ordering complaint (#3a). Failure is a value (invariant 4).

Provenance:
  quarry: (none — new L4 guard; audit #3a)
  source: CLAUDE.md §2 (invariant 4 failure-as-value; invariant 6); AUDIT_FINDINGS.md #3
  oracle: a seasoned mid-period swap returns the NAMED failure, not a raw date-ordering error (repro_P)
  slice:  audit-batch-A (#3a)
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from pricebook_ng.foundation import (
    Accrual,
    AccrualMethod,
    FixingSource,
    PricingFailure,
    RateIndex,
    Schedule,
    SchedulePeriod,
    accrued_rate,
)
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.curve import CurveHandle


def current_period_failure(start: date, valuation_date: date) -> PricingFailure | None:
    """A `PricingFailure` naming the current-period situation when `start < valuation_date`
    (the period is in progress), else `None`."""
    if start < valuation_date:
        return PricingFailure(
            f"current period starts {start.isoformat()}, before valuation "
            f"{valuation_date.isoformat()} — fixings required (seasoned mid-period mark)"
        )
    return None


def split_current_period(
    schedule: Schedule, valuation_date: date
) -> tuple[SchedulePeriod | None, Schedule]:
    """From a FUTURE schedule (payment > vd), separate the CURRENT in-progress period (its accrual
    started on/before `valuation_date`) from the strictly-future ones — the strictly-future price
    through the unchanged `float_leg_pv` atom, the current one is spliced (#3b)."""
    periods = schedule.periods
    # strict: a period starting exactly ON the valuation date is fresh (prices via the forward atom)
    if periods and periods[0].accrual_start < valuation_date:
        future = replace(
            schedule,
            unadjusted=schedule.unadjusted[1:],
            adjusted=schedule.adjusted[1:],
            periods=periods[1:],
        )
        return periods[0], future
    return None, schedule


def current_period_float_rate(
    index: RateIndex,
    accrual: Accrual,
    projection: CurveHandle,
    fixings: FixingSource,
    valuation_date: date,
) -> float:
    """The float rate of the current in-progress period (its full `accrual`), from the past fixing
    (#3b). IBOR (FLAT): the single reset fixing over the whole accrual. RFR (COMPOUNDED): the
    realized-compounded rate over `[start, vd]` SPLICED with the projected forward over `[vd, end]`.
    Raises on a missing fixing — the caller turns that into a `PricingFailure` (invariant 4)."""
    if index.fixing.compounding is AccrualMethod.FLAT:  # IBOR — the whole-period reset fixing, no stub
        return accrued_rate(index, accrual, fixings)
    dc = accrual.day_count
    realized_acc = Accrual(accrual.start, valuation_date, dc)  # RFR — realized so far
    realized = accrued_rate(index, realized_acc, fixings)
    stub = Accrual(valuation_date, accrual.end, dc)  # projected remainder
    fwd = forward(projection, stub)
    compound = (1.0 + realized * realized_acc.year_fraction()) * (1.0 + fwd * stub.year_fraction())
    return (compound - 1.0) / accrual.year_fraction()
