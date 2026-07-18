"""Margined (collateralised) exposure oracle (L5 risk & capital).

Under a two-way CSA with variation margin and an uncollateralised threshold `H`,
collateral posts the mark-to-market beyond `H`, capping exposure:

    E_coll(t) = min( max(+-V(t), 0), H )

so `H = 0` is fully collateralised (exposure 0, ignoring the margin-period-of-risk
gap) and `H = inf` recovers the uncollateralised exposure. Collateral reduces
counterparty risk, so the collateralised CVA is below the uncollateralised one.

Oracles: (1) a huge threshold reproduces `exposure_profiles` exactly (same draws);
(2) zero threshold gives zero exposure; (3) sigma=0 caps the deterministic exposure
exactly; (4) collateralised CVA < uncollateralised CVA.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.engine.swaption import coupon_bond_cashflows
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.exposure import collateralized_exposure, exposure_profiles
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
    return vanilla_swap(Money(NOTIONAL, Currency.USD), 0.02, D0, MATURITY, TERMS)  # ITM payer


def _model(sigma=0.012):
    market = MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(0.03, D0, ACT365))
    return HullWhite(a=0.05, sigma=sigma, market=market)


def test_huge_threshold_recovers_uncollateralized():
    swap, model = _swap(), _model()
    coll = collateralized_exposure(swap, model, NUM, 1e15)
    unc = exposure_profiles(swap, model, NUM)
    assert coll.epe.ee == pytest.approx(unc.epe.ee, abs=1e-6)
    assert coll.ene.ee == pytest.approx(unc.ene.ee, abs=1e-6)


def test_zero_threshold_gives_zero_exposure():
    coll = collateralized_exposure(_swap(), _model(), NUM, 0.0)
    assert all(e == 0.0 for e in coll.epe.ee)
    assert all(e == 0.0 for e in coll.ene.ee)


def test_sigma_zero_caps_deterministic_exposure():
    swap, model = _swap(), _model(sigma=0.0)
    curve = model.market.discount_curve
    threshold = 3_000_000.0
    coll = collateralized_exposure(swap, model, NumericalConfig(mc_paths=8, mc_seed=1), threshold)
    dates, amounts, notional = coupon_bond_cashflows(swap)
    capped = False
    for t_j, e in zip(coll.epe.grid, coll.epe.ee):
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > t_j]
        v = notional - sum(amt * curve.df(d) / curve.df(t_j) for d, amt in remaining)
        assert e == pytest.approx(min(max(v, 0.0), threshold), abs=1e-6)
        capped = capped or max(v, 0.0) > threshold
    assert capped  # the threshold actually bites somewhere


def test_collateral_reduces_cva():
    swap, model = _swap(), _model()
    disc = model.market.discount_curve
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.02)], 0.4)
    market = MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival})

    unc = exposure_profiles(swap, model, NUM)
    coll = collateralized_exposure(swap, model, NUM, 2_000_000.0)
    assert cva(coll.epe, market, KEY, 0.4) < cva(unc.epe, market, KEY, 0.4)
    assert cva(coll.epe, market, KEY, 0.4) > 0.0
