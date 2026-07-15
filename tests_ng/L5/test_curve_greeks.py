"""Curve greeks oracle — generic keyed `curve01` on the Priceable protocol (L5).

The curve analogue of A5's `spot_delta`/`vol_vega`: one `curve01`/`bump_curve`
keyed by `MarketKey`, parallel-shifting the curve at that key and repricing —
whatever the curve type. A flat discount curve shifts its rate; a survival curve
shifts its hazard (so `credit01` is `curve01` on a CREDIT key). This gives rate
risk on the FX foreign curve and the inflation real (breakeven) curve for free.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.cds import CDSEngine
from pricebook_ng.engine.fx_forward import FXForwardEngine
from pricebook_ng.engine.inflation import ZCISEngine
from pricebook_ng.products.cds import cds
from pricebook_ng.products.fx_forward import fx_forward
from pricebook_ng.products.inflation import ZeroCouponInflationSwap
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.risk.greeks import bump_curve, credit01, curve01
from pricebook_ng.risk.priceable import credit_priceable, discounting_priceable

EUR, USD = Currency.EUR, Currency.USD
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
T = 365.0 / 365.0
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()
_ONE_BP = 1e-4


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=ACT365)


# ---- FX foreign-curve rate risk -----------------------------------------------
def test_curve01_on_fx_foreign_curve_matches_analytic():
    fx_key = MarketKey(AssetClass.FX, "EUR")
    notional, spot, base_rate = 1_000_000.0, 1.10, 0.01
    market = MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(0.03),
        curves={fx_key: _curve(base_rate)}, spots={fx_key: spot},
    )
    fwd = fx_forward(Money(notional, EUR), USD, 1.15, MATURITY)
    priceable = discounting_priceable(fwd, FXForwardEngine(), NUM)
    df_base = market.curves[fx_key].df(MATURITY)
    # d(PV_buy)/d(r_base) = base_notional * spot * (-T * DF_base); x 1bp
    analytic = -notional * spot * T * df_base * _ONE_BP
    assert curve01(priceable, market, fx_key, NUM) == pytest.approx(analytic, abs=1e-6)


# ---- inflation real-curve (breakeven01) ---------------------------------------
def test_curve01_on_inflation_real_curve_matches_analytic():
    index = "EUHICP"
    infl_key = MarketKey(AssetClass.INFLATION, index)
    notional, real_rate = 1_000_000.0, 0.01
    market = MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(0.03), curves={infl_key: _curve(real_rate)},
    )
    zcis = ZeroCouponInflationSwap(index, notional, 0.015, MATURITY, EUR, receive_inflation=True)
    priceable = discounting_priceable(zcis, ZCISEngine(), NUM)
    df_real = market.curves[infl_key].df(MATURITY)
    # PV_recv = notional*(DF_real - DF_r*(1+K)^T); d/d(r_real) = notional*(-T*DF_real); x 1bp
    analytic = notional * (-T * df_real) * _ONE_BP
    assert curve01(priceable, market, infl_key, NUM) == pytest.approx(analytic, abs=1e-6)


# ---- credit01 is curve01 on a survival curve ----------------------------------
def _credit_market():
    bare = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.02))
    survival = bootstrap_survival_curve(bare, [CDSQuote(date(2031, 1, 5), 0.02)], 0.4)
    key = MarketKey(AssetClass.CREDIT, "ACME_CO")
    return MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.02), curves={key: survival}), key


def test_credit01_equals_curve01_on_survival_key():
    market, key = _credit_market()
    c = cds(Money(10_000_000.0, USD), "ACME_CO", 0.025, D0, date(2031, 1, 5))
    priceable = credit_priceable(c, 0.4, CDSEngine(), NUM)
    assert credit01(priceable, market, key, NUM) == pytest.approx(
        curve01(priceable, market, key, NUM), abs=1e-12
    )


def test_bump_curve_shifts_only_the_keyed_curve():
    fx_key = MarketKey(AssetClass.FX, "EUR")
    market = MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(0.03), curves={fx_key: _curve(0.01)},
    )
    bumped = bump_curve(market, fx_key, 0.001)
    assert bumped.curves[fx_key].df(MATURITY) < market.curves[fx_key].df(MATURITY)  # rate up -> DF down
    assert bumped.discount_curve is market.discount_curve                          # home untouched
