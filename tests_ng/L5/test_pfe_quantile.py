"""PFE-quantile / dynamic-IM engine oracle (L5 risk & capital).

Potential future exposure at confidence `q`: `PFE_q(t_j) = quantile_q(max(V(t_j), 0))`
from the simulated exposure distribution. Because the remaining swap value is a
*monotonic* function of the Gaussian short rate, the q-quantile of V is V evaluated at
the q-quantile of r — an exact analytic target:

    PFE_q(t_j) = max( V(t_j; r_q), 0 ),   r_q = forward_short_rate(t_j, Phi^{-1}(q))

Oracles: (1) MC PFE matches that analytic monotone-transform value; (2) sigma = 0 collapses
it to the deterministic exposure for any q; (3) PFE rises with q, and a high-quantile PFE
serves as a dynamic-IM proxy that feeds MVA to a positive charge.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.engine.swaption import coupon_bond_cashflows
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.exposure import pfe_profile
from pricebook_ng.risk.xva import mva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
MATURITY = date(2031, 1, 15)
NOTIONAL = 100_000_000.0
Z95 = 1.6448536269514722  # Phi^{-1}(0.95)
KEY = MarketKey(AssetClass.CREDIT, "SELF")
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _swap():
    # in-the-money payer (fixed below market) -> V > 0 across quantiles
    return vanilla_swap(Money(NOTIONAL, Currency.USD), 0.02, D0, MATURITY, TERMS)


def _model(sigma=0.012, rate=0.03):
    market = MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(rate, D0, ACT365))
    return HullWhite(a=0.05, sigma=sigma, market=market)


def test_pfe_matches_analytic_monotone_transform():
    swap, model = _swap(), _model()
    pfe = pfe_profile(swap, model, NumericalConfig(mc_paths=120_000, mc_seed=11), 0.95)
    dates, amounts, notional = coupon_bond_cashflows(swap)
    for t_j, p in zip(pfe.grid, pfe.ee):
        r_q = model.forward_short_rate(year_fraction(D0, t_j, ACT365), Z95)
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > t_j]
        v = notional - sum(amt * model.zero_bond(t_j, d, r_q) for d, amt in remaining)
        assert p == pytest.approx(max(v, 0.0), rel=0.02)


def test_sigma_zero_pfe_is_deterministic_exposure():
    swap = _swap()
    model = _model(sigma=0.0)
    curve = model.market.discount_curve
    pfe = pfe_profile(swap, model, NumericalConfig(mc_paths=16, mc_seed=1), 0.99)
    dates, amounts, notional = coupon_bond_cashflows(swap)
    for t_j, p in zip(pfe.grid, pfe.ee):
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > t_j]
        v = notional - sum(amt * curve.df(d) / curve.df(t_j) for d, amt in remaining)
        assert p == pytest.approx(max(v, 0.0), abs=1e-6)


def test_pfe_rises_with_quantile_and_feeds_mva():
    swap, model = _swap(), _model()
    num = NumericalConfig(mc_paths=40_000, mc_seed=3)
    pfe95 = pfe_profile(swap, model, num, 0.95)
    pfe99 = pfe_profile(swap, model, num, 0.99)
    assert all(p99 >= p95 for p99, p95 in zip(pfe99.ee, pfe95.ee))  # higher confidence -> higher PFE

    disc = model.market.discount_curve
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.015)], 0.4)
    market = MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival})
    assert mva(pfe99, market, KEY, 0.008) > 0.0  # dynamic-IM proxy funds a positive MVA
