"""Commodity oracle — European commodity option (Black-Scholes on carry) (L4).

Structurally identical to the equity option: Black-76 on the commodity forward
`F = spot * DF_carry / DF_r` (carry = convenience-yield net of storage), keyed by
ticker in the snapshot under `AssetClass.CMDTY`. The greeks are FREE — the generic
`spot_delta`/`vol_vega` (A5) work with no new code.
"""

import math
from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.distributions import norm_cdf
from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.commodity_option import CommodityOptionEngine
from pricebook_ng.products.commodity_option import CommodityOption
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.risk.greeks import spot_delta, vol_vega
from pricebook_ng.risk.priceable import discounting_priceable

USD = Currency.USD
TICKER = "WTI"
KEY = MarketKey(AssetClass.CMDTY, TICKER)
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
SPOT, VOL, QTY = 80.0, 0.30, 1000.0
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=ACT365)


def _market(vol=VOL):
    return MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(0.04),
        curves={KEY: _curve(0.01)}, spots={KEY: SPOT}, vols={KEY: vol},   # carry curve at KEY
    )


def _fwd(market):
    return SPOT * market.curves[KEY].df(MATURITY) / market.discount_curve.df(MATURITY)


def _opt(strike, is_call=True):
    return CommodityOption(TICKER, QTY, strike, MATURITY, USD, is_call)


def _price(opt, market):
    return CommodityOptionEngine().price(opt, DiscountingModel(market), NUM)


def test_priced_in_the_option_currency():
    r = _price(_opt(85.0), _market())
    assert isinstance(r, PricingResult)
    assert r.pv.currency is USD


def test_put_call_parity_ties_to_forward_value():
    market = _market()
    k = 85.0
    call = _price(_opt(k, is_call=True), market).pv.amount
    put = _price(_opt(k, is_call=False), market).pv.amount
    df_r = market.discount_curve.df(MATURITY)
    assert (call - put) == pytest.approx(QTY * df_r * (_fwd(market) - k), abs=1e-4)


def test_matches_independent_black():
    market = _market()
    k = 85.0
    f = _fwd(market)
    t = year_fraction(D0, MATURITY, ACT365)
    df_r = market.discount_curve.df(MATURITY)
    std = VOL * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * std * std) / std
    d2 = d1 - std
    expected = QTY * df_r * (f * norm_cdf(d1) - k * norm_cdf(d2))
    assert _price(_opt(k, is_call=True), market).pv.amount == pytest.approx(expected, abs=1e-4)


def test_zero_vol_is_discounted_intrinsic():
    market = _market(vol=0.0)
    f = _fwd(market)
    df_r = market.discount_curve.df(MATURITY)
    call = _price(_opt(70.0, is_call=True), market).pv.amount  # ITM (K < F)
    assert call == pytest.approx(QTY * df_r * max(f - 70.0, 0.0), abs=1e-4)


def test_missing_commodity_market_is_a_failure():
    bare = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.04))
    assert isinstance(_price(_opt(85.0), bare), PricingFailure)


def test_greeks_are_free_via_generic_spot_delta_and_vol_vega():
    # A5 payoff: no commodity-specific greek code — the generic ones just work.
    market = _market()
    priceable = discounting_priceable(_opt(85.0), CommodityOptionEngine(), NUM)
    df_carry = market.curves[KEY].df(MATURITY)
    t = year_fraction(D0, MATURITY, ACT365)
    d1 = (math.log(_fwd(market) / 85.0) + 0.5 * (VOL * math.sqrt(t)) ** 2) / (VOL * math.sqrt(t))
    assert spot_delta(priceable, market, KEY, NUM) == pytest.approx(
        QTY * df_carry * norm_cdf(d1), abs=1e-3
    )
    assert vol_vega(priceable, market, KEY, NUM) > 0.0
