"""Measure-consistency binding oracle (L5 risk & capital) — Amendment A6.1.

The exposure stack runs two simulators: the per-date **forward-measure** engine
(`exposure_profiles`/`pfe_profile`, one Gaussian per date) and the **risk-neutral
joint-path** engine (`_simulate_rate_paths`, used for MPOR). A6.1 rules these are one
model under a change of numeraire — the same discounted exposure
`E^Q[D(0,t)·max(V,0)] = P(0,t)·E^{T_t}[max(V,0)]` — and mandates a binding oracle so the
two can never silently diverge (the project's core failure mode).

Concretely the risk-neutral marginal `r(t) ~ N(alpha(t), v(t))` and the forward marginal
`r(t) ~ N(alpha(t)+m(t), v(t))` differ only by the analytic forward-measure drift `m(t)`.
So shifting the joint-path marginal by `m(t)` must reproduce the forward-measure EE and PFE
per date (two independent MC estimates of the same truth — deterministic under fixed seeds).
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.swaption import coupon_bond_cashflows
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.exposure import (
    _quantile,
    _simulate_rate_paths,
    exposure_profiles,
    pfe_profile,
)

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
MATURITY = date(2031, 1, 15)
QUANTILE = 0.95
NUM = NumericalConfig(mc_paths=50_000, mc_seed=101)
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _swap():
    return vanilla_swap(Money(100_000_000.0, Currency.USD), 0.03, D0, MATURITY, TERMS)


def _model():
    market = MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(0.03, D0, ACT365))
    return HullWhite(a=0.05, sigma=0.012, market=market)


def _forward_measure_drift(model, t):
    """m(t): the T_t-forward-measure drift shift of r(t) vs risk-neutral (change of numeraire)."""
    a, s = model.a, model.sigma
    return -(s**2 / a**2) * ((1.0 - math.exp(-a * t)) - 0.5 * (1.0 - math.exp(-2.0 * a * t)))


def test_joint_paths_reproduce_forward_measure_ee_and_pfe():
    swap, model = _swap(), _model()
    fwd_epe = exposure_profiles(swap, model, NUM).epe
    fwd_pfe = pfe_profile(swap, model, NUM, QUANTILE)

    dates, amounts, notional = coupon_bond_cashflows(swap)
    exposure_dates = fwd_epe.grid[1:]                       # skip valuation (t=0, deterministic)
    times = [year_fraction(D0, d, ACT365) for d in exposure_dates]
    paths = _simulate_rate_paths(model, times, NUM)         # risk-neutral joint paths

    for k, (d_j, t_j) in enumerate(zip(exposure_dates, times)):
        drift = _forward_measure_drift(model, t_j)
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > d_j]
        # shift the risk-neutral marginal to the forward measure, then reprice
        positives = sorted(
            max(notional - sum(amt * model.zero_bond(d_j, d, p[k] + drift) for d, amt in remaining), 0.0)
            for p in paths
        )
        ee = sum(positives) / len(positives)
        pfe = _quantile(positives, QUANTILE)
        # two independent MC estimates of the same discounted exposure (A6.1) -> agree
        assert ee == pytest.approx(fwd_epe.ee[k + 1], rel=0.04)
        assert pfe == pytest.approx(fwd_pfe.ee[k + 1], rel=0.04)
