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

from datetime import date

from pricebook_ng.foundation import PricingFailure


def current_period_failure(start: date, valuation_date: date) -> PricingFailure | None:
    """A `PricingFailure` naming the current-period situation when `start < valuation_date`
    (the period is in progress), else `None`."""
    if start < valuation_date:
        return PricingFailure(
            f"current period starts {start.isoformat()}, before valuation "
            f"{valuation_date.isoformat()} — fixings required (seasoned mid-period mark)"
        )
    return None
