"""MC expected-exposure engine oracle (L5 risk & capital).

Generates the EE profile `EE(t_j) = E[(V(t_j))^+]` that feeds CVA, by simulating
the Hull-White short rate to each grid date under its own t_j-forward measure
(one exact Gaussian draw, as the MC swaption does) and repricing the remaining
swap analytically. Two oracles pin it:

  1. sigma = 0 -> no randomness: EE(t_j) is the deterministic forward swap value's
     positive part, exact to machine precision (validates the repricing mechanics).
  2. sigma > 0: the forward-measure identity `P(0,t_j) * EE(t_j) = ` the analytic
     co-terminal swaption expiring at t_j (Jamshidian, already oracle-checked) —
     the discounted expected exposure IS a swaption strip. MC matches within error.

And end-to-end: the profile feeds `cva` to a positive number.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.engine.swaption import SwaptionEngine, coupon_bond_cashflows
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.products.swaption import Swaption
from pricebook_ng.risk.exposure import expected_exposure
from pricebook_ng.risk.xva import cva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
USD = Currency.USD
MATURITY = date(2031, 1, 15)
FACE = Money(1_000_000.0, USD)
FIXED_RATE = 0.03
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _market(rate=0.03):
    return MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(rate, D0, ACT365))


def _swap(start=D0, maturity=MATURITY):
    return vanilla_swap(FACE, FIXED_RATE, start, maturity, TERMS)


def test_sigma_zero_exposure_is_deterministic_forward_value():
    market = _market()
    swap = _swap()
    model = HullWhite(a=0.05, sigma=0.0, market=market)
    profile = expected_exposure(swap, model, NumericalConfig(mc_paths=8, mc_seed=1))

    dates, amounts, _ = coupon_bond_cashflows(swap)
    curve = market.discount_curve
    for tj, ee in zip(profile.grid, profile.ee):
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > tj]
        v = FACE.amount - sum(amt * curve.df(d) / curve.df(tj) for d, amt in remaining)  # payer
        assert ee == pytest.approx(max(v, 0.0), abs=1e-8)


def test_discounted_exposure_matches_coterminal_swaptions():
    market = _market()
    swap = _swap()
    model = HullWhite(a=0.05, sigma=0.012, market=market)
    num = NumericalConfig(mc_paths=120_000, mc_seed=7)
    profile = expected_exposure(swap, model, num)

    engine = SwaptionEngine()
    for tj, ee in zip(profile.grid, profile.ee):
        if tj == D0:
            continue  # covered exactly by the sigma=0 / t0 case
        coterminal = Swaption(expiry=tj, swap=_swap(start=tj))
        analytic = engine.price(coterminal, model, num).pv.amount
        # P(0,t_j) * EE(t_j) == swaption(t_j); compare on the discounted scale
        assert ee * market.discount_curve.df(tj) == pytest.approx(analytic, rel=0.02)


def test_profile_feeds_cva_positive():
    market = _market()
    model = HullWhite(a=0.05, sigma=0.012, market=market)
    profile = expected_exposure(_swap(), model, NumericalConfig(mc_paths=20_000, mc_seed=3))

    key = MarketKey(AssetClass.CREDIT, "CPTY")
    survival = bootstrap_survival_curve(market, [CDSQuote(MATURITY, 0.02)], 0.4)
    credit_market = MarketSnapshot(
        valuation_date=D0, discount_curve=market.discount_curve, curves={key: survival}
    )
    assert cva(profile, credit_market, key, 0.4) > 0.0
    assert not math.isnan(cva(profile, credit_market, key, 0.4))
