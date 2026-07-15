"""Discount-curve bootstrap — the rates solver of the calibration front (L3).

`market -> calibrate -> model`: turns deposit + par-swap quotes (L1 market
observables) into a `DiscountCurve` (L1). The solver reprices each quote to par
using the curve's own `df` — no L4 engine — so it sits inside calibration's
L0/L1 dependency budget, the sibling of `calibrate_hull_white` (vol) and
`bootstrap_survival_curve` (credit).

Single-curve: discount = projection, so a par swap's float leg telescopes to
`1 - DF(maturity)` and each pillar solves in closed form (deposits) or one linear
step (swaps), short end first.

Provenance:
  quarry: python/pricebook/core/discount_curve.py (bootstrap; re-homed L1 -> L3 front)
  source: Hull, Options Futures & Other Derivatives ch.4; single-curve bootstrap
  oracle: inputs reprice to par (self-consistency) + closed-form deposit DFs < 1e-12
  slice:  S03; calibration-front (bootstraps migrate under the L3 front)
"""

from __future__ import annotations

from datetime import date

from pricebook_ng.foundation.schedule import generate_schedule
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.market.discount_curve import DepositQuote, DiscountCurve, ParSwapQuote


def bootstrap_discount_curve(
    valuation_date: date,
    deposits: list[DepositQuote],
    swaps: list[ParSwapQuote],
) -> DiscountCurve:
    """Bootstrap a discount curve from deposits then par swaps, short end first."""
    if not deposits and not swaps:
        raise ValueError("bootstrap requires at least one deposit or swap quote")

    pillars: list[tuple[date, float]] = [(valuation_date, 1.0)]

    for dep in sorted(deposits, key=lambda q: q.maturity):
        tau = year_fraction(valuation_date, dep.maturity, dep.day_count)
        pillars.append((dep.maturity, 1.0 / (1.0 + dep.rate * tau)))

    for sw in sorted(swaps, key=lambda q: q.maturity):
        curve = DiscountCurve(valuation_date, tuple(pillars))
        sched = generate_schedule(valuation_date, sw.maturity, sw.fixed_frequency)
        # par: rate * sum(tau_i * DF(t_i)) = 1 - DF(t_n).  DF(t_i<n) are known
        # pillars; solve the linear equation for the final DF(t_n).
        annuity_known = 0.0
        tau_last = 0.0
        for i in range(1, len(sched)):
            tau_i = year_fraction(sched[i - 1], sched[i], sw.day_count)
            if i < len(sched) - 1:
                annuity_known += tau_i * curve.df(sched[i])
            else:
                tau_last = tau_i
        df_n = (1.0 - sw.rate * annuity_known) / (1.0 + sw.rate * tau_last)
        pillars.append((sw.maturity, df_n))

    return DiscountCurve(valuation_date, tuple(pillars))
