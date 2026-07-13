"""Risk → L5 oracle — greeks on the Priceable protocol (bump-and-reprice).

The structural fix (spine): risk lives above the engine and depends only on a
`Priceable` (`snapshot -> PV`), never on concrete product/model classes. One
generic `dv01` computes rate delta for a single cashflow, a bond, and an HW
swaption — the swaption case rebuilds the model under the bumped snapshot (A1).
No `isinstance` ladders.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.engine.swaption import SwaptionEngine
from pricebook_ng.products.fixed_cashflow import FixedCashflow
from pricebook_ng.products.fixed_rate_bond import fixed_rate_bond
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.products.swaption import Swaption
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.risk.greeks import bump_rate, dv01
from pricebook_ng.risk.priceable import Priceable, discounting_priceable, hull_white_priceable

_ONE_BP = 1e-4
D0 = date(2026, 1, 15)
RATE = 0.03
CCY = Currency.USD
CURVE_DC = DC.ACT_365_FIXED
NUM = NumericalConfig()


def _market(rate=RATE, valuation=D0):
    return MarketSnapshot(valuation_date=valuation,
                          discount_curve=FlatDiscountCurve(rate, valuation, CURVE_DC))


def test_dv01_single_cashflow_matches_analytic():
    notional, mat = 1_000_000.0, date(2028, 1, 1)
    trade = FixedCashflow(Cashflow(date=mat, amount=Money(notional, CCY)))
    priceable = discounting_priceable(trade, DiscountingEngine(), NUM)
    t = year_fraction(D0, mat, CURVE_DC)
    analytic = -notional * t * math.exp(-RATE * t) * _ONE_BP
    assert dv01(priceable, _market(), NUM) == pytest.approx(analytic, abs=1e-6)


def test_dv01_bond_matches_analytic_sum():
    bond = fixed_rate_bond(
        face=Money(1_000_000.0, CCY), coupon_rate=0.04, start=D0, maturity=date(2029, 1, 15),
        terms=ScheduleTerms(Frequency.SEMI_ANNUAL, DC.THIRTY_360),
    )
    priceable = discounting_priceable(bond, DiscountingEngine(), NUM)
    analytic = -_ONE_BP * sum(
        cf.amount.amount * year_fraction(D0, cf.date, CURVE_DC)
        * math.exp(-RATE * year_fraction(D0, cf.date, CURVE_DC))
        for cf in bond.cashflows
    )
    assert dv01(priceable, _market(), NUM) == pytest.approx(analytic, abs=1e-4)


def test_dv01_is_generic_over_any_priceable():
    # risk depends only on the Priceable interface — feed it a raw closure
    notional, mat = 500_000.0, date(2030, 1, 15)
    raw = Priceable(lambda snap: notional * snap.discount_curve.df(mat))
    t = year_fraction(D0, mat, CURVE_DC)
    analytic = -notional * t * math.exp(-RATE * t) * _ONE_BP
    assert dv01(raw, _market(), NUM) == pytest.approx(analytic, abs=1e-6)


def test_dv01_swaption_rebuilds_the_model():
    expiry, maturity = date(2027, 1, 15), date(2032, 1, 15)
    swap = vanilla_swap(
        face=Money(1_000_000.0, CCY), fixed_rate=0.035, start=expiry, maturity=maturity,
        terms=SwapTerms(ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
                        ScheduleTerms(Frequency.ANNUAL, DC.ACT_360)),
    )
    swaption = Swaption(expiry=expiry, swap=swap)
    a, sigma = 0.05, 0.012
    priceable = hull_white_priceable(swaption, a, sigma, SwaptionEngine(), NUM)
    generic = dv01(priceable, _market(), NUM)

    # manual bump: rebuild HW on the bumped snapshot and reprice (what risk does)
    def manual(dr):
        m = _market(RATE + dr)
        return SwaptionEngine().price(swaption, HullWhite(a, sigma, m), NUM).pv.amount
    h = NUM.fd_bump
    manual_dv01 = (manual(h) - manual(-h)) / (2 * h) * _ONE_BP

    assert generic == pytest.approx(manual_dv01, abs=1e-9)
    assert abs(generic) > 1e-3  # a real, non-trivial sensitivity


def test_bump_rate_shifts_the_flat_curve():
    m = _market(0.03)
    up = bump_rate(m, 0.001)
    assert up.discount_curve.rate == pytest.approx(0.031, abs=1e-15)
    assert m.discount_curve.rate == pytest.approx(0.03, abs=1e-15)  # original untouched
