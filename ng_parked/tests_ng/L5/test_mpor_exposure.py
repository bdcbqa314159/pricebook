"""Margin-period-of-risk (MPOR) path-simulated exposure oracle (L5 risk & capital).

Under full collateralisation exposure is NOT zero: at a counterparty default the last
collateral reflects the value one MPOR ago, while close-out happens at today's value, so
the residual exposure is the value move over the close-out gap —

    E_mpor(t) = mean_paths max( V(t) - V(t - MPOR), 0 )

which needs the JOINT distribution of V(t) and V(t-MPOR), i.e. correlated short-rate
*paths* (not the per-date marginals the EE profiles use). This slice adds a risk-neutral
joint-path simulator (exact Ornstein-Uhlenbeck steps on the HW state).

Oracles: (1) the path simulator reproduces the analytic HW/OU moments — marginal mean
alpha(t), variance sigma^2(1-e^{-2at})/2a, and cross-date covariance e^{-a(t-s)} var(s);
(2) a zero gap gives exactly zero exposure; (3) exposure grows with the gap; (4) it feeds
a positive CVA.
"""

import math
import statistics
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.exposure import _simulate_rate_paths, mpor_exposure
from pricebook_ng.risk.xva import cva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
MATURITY = date(2031, 1, 15)
NOTIONAL = 100_000_000.0
KEY = MarketKey(AssetClass.CREDIT, "CPTY")
NUM = NumericalConfig(mc_paths=20_000, mc_seed=7)
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _swap():
    return vanilla_swap(Money(NOTIONAL, Currency.USD), 0.03, D0, MATURITY, TERMS)


def _model(sigma=0.012):
    market = MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(0.03, D0, ACT365))
    return HullWhite(a=0.05, sigma=sigma, market=market)


def test_rate_paths_reproduce_ou_moments():
    model = _model()
    a, sigma, r0 = model.a, model.sigma, model.market.discount_curve.rate
    path_dates = [date(2027, 1, 15), date(2029, 1, 15), date(2031, 1, 15)]  # ~1y, 3y, 5y
    times = [year_fraction(D0, d, DC.ACT_365_FIXED) for d in path_dates]
    paths = _simulate_rate_paths(model, path_dates, NumericalConfig(mc_paths=60_000, mc_seed=5))

    def alpha(t):
        return r0 + (sigma**2 / (2 * a**2)) * (1 - math.exp(-a * t)) ** 2

    def var(t):
        return sigma**2 * (1 - math.exp(-2 * a * t)) / (2 * a)

    for k, t in enumerate(times):
        col = [p[k] for p in paths]
        assert statistics.mean(col) == pytest.approx(alpha(t), abs=3e-4)
        assert statistics.pvariance(col) == pytest.approx(var(t), rel=0.05)

    # cross-date OU covariance: cov(r_s, r_t) = e^{-a(t-s)} var(s)
    cs, ct = [p[0] for p in paths], [p[1] for p in paths]
    ms, mt = statistics.mean(cs), statistics.mean(ct)
    cov = sum((x - ms) * (y - mt) for x, y in zip(cs, ct)) / len(cs)
    assert cov == pytest.approx(math.exp(-a * (times[1] - times[0])) * var(times[0]), rel=0.05)


def test_zero_gap_gives_zero_exposure():
    prof = mpor_exposure(_swap(), _model(), NUM, mpor_days=0)
    assert all(e == 0.0 for e in prof.ee)


def test_exposure_grows_with_gap():
    swap, model = _swap(), _model()
    small = mpor_exposure(swap, model, NUM, mpor_days=5)
    large = mpor_exposure(swap, model, NUM, mpor_days=20)
    assert sum(large.ee) > sum(small.ee) > 0.0  # longer close-out gap -> more residual exposure


def test_mpor_exposure_feeds_cva():
    swap, model = _swap(), _model()
    disc = model.market.discount_curve
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.02)], 0.4)
    market = MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival})
    prof = mpor_exposure(swap, model, NUM, mpor_days=10)
    assert cva(prof, market, KEY, 0.4) > 0.0
