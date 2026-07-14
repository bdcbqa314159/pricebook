"""Inflation oracle — zero-coupon inflation swap (ZCIS) (L4).

The building block of inflation: at maturity the inflation leg pays
`notional*(I(T) - 1)`, the fixed leg `notional*((1+K)^T - 1)`, both discounted by
the nominal curve. The inflation forward index ratio is `I(T) = DF_real(T)/DF_r(T)`
(Fisher), with the real curve keyed by index under `AssetClass.INFLATION`.
Receiver (of inflation) PV per notional = `DF_r(T) * (I(T) - (1+K)^T)`.
"""

from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.inflation import ZCISEngine
from pricebook_ng.products.inflation import ZeroCouponInflationSwap
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel

EUR = Currency.EUR
INDEX = "EUHICP"
KEY = MarketKey(AssetClass.INFLATION, INDEX)
D0 = date(2026, 1, 5)
MATURITY = date(2031, 1, 5)     # 5Y
NOTIONAL = 1_000_000.0
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=ACT365)


def _market(nominal=0.03, real=0.01):
    return MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(nominal),   # nominal home curve
        curves={KEY: _curve(real)},                          # real curve under INFLATION key
    )


def _index_ratio(market, maturity=MATURITY):
    return market.curves[KEY].df(maturity) / market.discount_curve.df(maturity)


def _par_rate(market, maturity=MATURITY):
    t = year_fraction(D0, maturity, ACT365)
    return _index_ratio(market, maturity) ** (1.0 / t) - 1.0


def _zcis(fixed_rate, receive_inflation=True):
    return ZeroCouponInflationSwap(
        index=INDEX, notional=NOTIONAL, fixed_rate=fixed_rate,
        maturity=MATURITY, currency=EUR, receive_inflation=receive_inflation,
    )


def _price(zcis, market):
    return ZCISEngine().price(zcis, DiscountingModel(market), NUM)


def test_par_zcis_reprices_to_zero():
    market = _market()
    result = _price(_zcis(_par_rate(market)), market)
    assert isinstance(result, PricingResult)
    assert result.pv.currency is EUR
    assert result.pv.amount == pytest.approx(0.0, abs=1e-4)


def test_pv_matches_formula():
    market = _market()
    k = 0.015
    t = year_fraction(D0, MATURITY, ACT365)
    df_r = market.discount_curve.df(MATURITY)
    expected = NOTIONAL * df_r * (_index_ratio(market) - (1.0 + k) ** t)
    assert _price(_zcis(k), market).pv.amount == pytest.approx(expected, abs=1e-6)


def test_receiver_is_negative_payer():
    market = _market()
    recv = _price(_zcis(0.015, receive_inflation=True), market).pv.amount
    payer = _price(replace(_zcis(0.015), receive_inflation=False), market).pv.amount
    assert payer == pytest.approx(-recv, abs=1e-9)


def test_below_par_fixed_is_valuable_to_inflation_receiver():
    market = _market()
    cheap = _price(_zcis(_par_rate(market) - 0.005), market).pv.amount
    assert cheap > 0.0  # receiving inflation while paying below breakeven -> positive


def test_missing_inflation_curve_is_a_failure():
    bare = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.03))
    assert isinstance(_price(_zcis(0.02), bare), PricingFailure)


def test_is_pure_data_no_pricing():
    assert not hasattr(_zcis(0.02), "pv")
