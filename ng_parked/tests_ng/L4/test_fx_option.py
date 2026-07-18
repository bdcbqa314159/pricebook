"""FX oracle — European FX option (Garman-Kohlhagen) (L4).

GK is Black-76 on the FX forward `F = spot*DF_base/DF_quote`, discounted by the
quote curve; the flat FX vol lives in the snapshot (`fx_vols`, market data §5.1).
Oracles: put-call parity ties to the FX forward PV; an independent Black recompute;
sigma -> 0 collapses to intrinsic; ATM-forward call == put.
"""

import math
from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.distributions import norm_cdf
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.fx_forward import FXForwardEngine
from pricebook_ng.engine.fx_option import FXOptionEngine
from pricebook_ng.products.fx_forward import fx_forward
from pricebook_ng.products.fx_option import FXOption, fx_option
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.models.discounting_model import DiscountingModel

EUR, USD = Currency.EUR, Currency.USD
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
NOTIONAL = 1_000_000.0
SPOT, VOL = 1.10, 0.10
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=ACT365)


def _market(vol=VOL):
    return MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(0.03),
        curves={MarketKey(AssetClass.FX, EUR.value): _curve(0.01)}, spots={MarketKey(AssetClass.FX, EUR.value): SPOT}, vols={MarketKey(AssetClass.FX, EUR.value): vol},
    )


def _fwd_rate(market, maturity=MATURITY):
    return market.spots[MarketKey(AssetClass.FX, EUR.value)] * market.curves[MarketKey(AssetClass.FX, EUR.value)].df(maturity) / market.discount_curve.df(maturity)


def _opt(strike, is_call=True):
    return fx_option(Money(NOTIONAL, EUR), quote_ccy=USD, strike=strike, maturity=MATURITY, is_call=is_call)


def _price(opt, market):
    return FXOptionEngine().price(opt, DiscountingModel(market), NUM)


def test_priced_in_quote_currency():
    r = _price(_opt(1.15), _market())
    assert isinstance(r, PricingResult)
    assert r.pv.currency is USD


def test_put_call_parity_ties_to_the_fx_forward():
    market = _market()
    strike = 1.15
    call = _price(_opt(strike, is_call=True), market).pv.amount
    put = _price(_opt(strike, is_call=False), market).pv.amount
    fwd = FXForwardEngine().price(
        fx_forward(Money(NOTIONAL, EUR), USD, strike, MATURITY), DiscountingModel(market), NUM
    ).pv.amount
    assert (call - put) == pytest.approx(fwd, abs=1e-4)


def test_matches_independent_black():
    market = _market()
    strike = 1.15
    f = _fwd_rate(market)
    t = year_fraction(D0, MATURITY, ACT365)
    df_quote = market.discount_curve.df(MATURITY)
    sig = VOL * math.sqrt(t)
    d1 = (math.log(f / strike) + 0.5 * sig * sig) / sig
    d2 = d1 - sig
    expected = NOTIONAL * df_quote * (f * norm_cdf(d1) - strike * norm_cdf(d2))
    assert _price(_opt(strike, is_call=True), market).pv.amount == pytest.approx(expected, abs=1e-4)


def test_zero_vol_is_discounted_intrinsic():
    market = _market(vol=0.0)
    f = _fwd_rate(market)
    df_quote = market.discount_curve.df(MATURITY)
    call = _price(_opt(1.05, is_call=True), market).pv.amount  # ITM (K < F)
    assert call == pytest.approx(NOTIONAL * df_quote * max(f - 1.05, 0.0), abs=1e-4)


def test_atm_forward_call_equals_put():
    market = _market()
    k = _fwd_rate(market)
    call = _price(_opt(k, is_call=True), market).pv.amount
    put = _price(_opt(k, is_call=False), market).pv.amount
    assert call == pytest.approx(put, abs=1e-4)


def test_is_pure_data_no_pricing():
    opt = _opt(1.10)
    assert isinstance(opt, FXOption)
    assert not hasattr(opt, "pv")
