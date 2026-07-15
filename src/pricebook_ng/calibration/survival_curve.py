"""Survival-curve bootstrap — the credit solver of the calibration front (L3).

`market -> calibrate -> model`: turns CDS par-spread quotes (L1) into a
`SurvivalCurve` (L1). Each quote adds a pillar whose survival probability makes
that CDS reprice to zero at its par spread — solved with the L1 `cds_pv` leg math
(no L4 engine), the credit sibling of the rates bootstrap and the HW vol fit.

Provenance:
  quarry: python/pricebook/core/survival_curve.py (bootstrap; re-homed L1 -> L3 front)
  source: standard single-name CDS; ISDA hazard-rate bootstrap
  oracle: each input CDS reprices to zero par spread (self-consistency) < 1e-10
  slice:  credit-hazard-bootstrap; calibration-front (bootstraps migrate under the L3 front)
"""

from __future__ import annotations

from datetime import date

from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.solvers import bisect_root
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote, SurvivalCurve, cds_pv


def bootstrap_survival_curve(
    market: MarketSnapshot, quotes: list[CDSQuote], recovery: float
) -> SurvivalCurve:
    """Bootstrap the survival curve from CDS par spreads, short end first: each
    quote adds a pillar whose Q makes that CDS reprice to zero."""
    if not quotes:
        raise ValueError("bootstrap requires at least one CDS quote")

    valuation = market.valuation_date
    discount = market.discount_curve
    pillars: list[tuple[date, float]] = [(valuation, 1.0)]

    for q in sorted(quotes, key=lambda x: x.maturity):
        schedule = generate_schedule(valuation, q.maturity, Frequency.ANNUAL)
        prev_q = pillars[-1][1]

        def repriced(q_mat: float, q=q, schedule=schedule) -> float:
            trial = SurvivalCurve(valuation, (*pillars, (q.maturity, q_mat)))
            return cds_pv(discount, trial, schedule, q.par_spread, recovery)

        # value is monotonic in Q(maturity); bracket (0, prev_Q]
        q_mat = bisect_root(repriced, 1e-12, prev_q)
        pillars.append((q.maturity, q_mat))

    return SurvivalCurve(valuation, tuple(pillars))
