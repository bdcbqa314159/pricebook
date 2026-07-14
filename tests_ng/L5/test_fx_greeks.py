"""FX greeks oracle — delta + vega on the Priceable protocol (L5).

FX market data (curves, spot, vol) lives in the snapshot (§5.1), so FX risk flows
through the same `Priceable` as `dv01`: `fx_delta` bumps the spot, `fx_vega` bumps
the vol. A plain `discounting_priceable` binds any FX product (linear or GK) to a
`DiscountingModel` over the snapshot.
"""

import math
from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.fx_forward import FXForwardEngine
from pricebook_ng.engine.fx_option import FXOptionEngine
from pricebook_ng.products.fx_forward import fx_forward
from pricebook_ng.products.fx_option import fx_option
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.risk.greeks import bump_fx_spot, bump_fx_vol, dv01, fx_delta, fx_vega
from pricebook_ng.risk.priceable import discounting_priceable

EUR, USD = Currency.EUR, Currency.USD
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
NOTIONAL = 1_000_000.0
VOL = 0.10
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=ACT365)


MARKET = MarketSnapshot(
    valuation_date=D0, discount_curve=_curve(0.03),
    fx_curves={EUR: _curve(0.01)}, fx_spots={EUR: 1.10}, fx_vols={EUR: VOL},
)


def _fwd_rate():
    return MARKET.fx_spots[EUR] * MARKET.fx_curves[EUR].df(MATURITY) / MARKET.discount_curve.df(MATURITY)


# ---- delta (on the FX forward) ------------------------------------------------
def _fwd_priceable(strike=1.15, buy=True):
    f = fx_forward(Money(NOTIONAL, EUR), USD, strike, MATURITY)
    return discounting_priceable(f if buy else replace(f, buy_base=False), FXForwardEngine(), NUM)


def test_fx_delta_matches_analytic():
    d = fx_delta(_fwd_priceable(), MARKET, EUR, NUM)
    assert d == pytest.approx(NOTIONAL * MARKET.fx_curves[EUR].df(MATURITY), abs=1e-3)


def test_same_priceable_gives_quote_rate_dv01():
    assert dv01(_fwd_priceable(), MARKET, NUM) != 0.0


# ---- vega (on the FX option) --------------------------------------------------
def _opt_priceable(strike=1.15):
    return discounting_priceable(
        fx_option(Money(NOTIONAL, EUR), USD, strike, MATURITY, is_call=True), FXOptionEngine(), NUM
    )


def test_bump_fx_vol_moves_only_that_vol():
    bumped = bump_fx_vol(MARKET, EUR, 0.01)
    assert bumped.fx_vols[EUR] == pytest.approx(0.11, abs=1e-12)
    assert bumped.fx_spots[EUR] == 1.10                    # spot untouched
    assert MARKET.fx_vols[EUR] == VOL                      # original unchanged


def test_fx_vega_matches_analytic_black():
    strike = 1.15
    f = _fwd_rate()
    t = year_fraction(D0, MATURITY, ACT365)
    df_quote = MARKET.discount_curve.df(MATURITY)
    std = VOL * math.sqrt(t)
    d1 = (math.log(f / strike) + 0.5 * std * std) / std
    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    analytic = NOTIONAL * df_quote * f * pdf_d1 * math.sqrt(t)   # dPV/dsigma (per unit vol)
    assert fx_vega(_opt_priceable(strike), MARKET, EUR, NUM) == pytest.approx(analytic, abs=1e-2)


def test_fx_vega_is_positive():
    assert fx_vega(_opt_priceable(), MARKET, EUR, NUM) > 0.0   # long option is long vol
