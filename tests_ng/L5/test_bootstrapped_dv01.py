"""Bootstrapped-curve dv01 oracle (L5).

`dv01`/`curve01` parallel-shift a curve via its own `bumped`. For the flat curve
that shifts one rate; for a **bootstrapped** `DiscountCurve` it shifts every
pillar's zero by the same amount (DF -> DF*exp(-shift*t)), so log-linear
interpolation keeps the shift uniform between pillars. This closes the gap noted
when generic curve greeks landed: risk now works on a real bootstrapped curve,
not only a flat one.

Oracle: for a parallel zero shift, d(DF(t))/d(shift) = -t*DF(t), so
    dv01 = -1bp * sum_i  cf_i * t_i * DF_boot(t_i)
with DF_boot the bootstrapped curve's own (interpolated) discount factors —
analytic vs central-difference on the actual bumped curve.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.market.discount_curve import DepositQuote, bootstrap_discount_curve
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.products.fixed_rate_bond import fixed_rate_bond
from pricebook_ng.risk.greeks import dv01
from pricebook_ng.risk.priceable import discounting_priceable

CCY = Currency.USD
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
NUM = NumericalConfig()
D0 = date(2026, 1, 15)
_ONE_BP = 1e-4


def _boot_curve():
    # rising deposit curve so the bootstrapped DFs are genuinely non-flat
    deposits = [
        DepositQuote(date(2027, 1, 15), 0.030, ACT360),
        DepositQuote(date(2028, 1, 15), 0.035, ACT360),
        DepositQuote(date(2029, 1, 15), 0.040, ACT360),
    ]
    return bootstrap_discount_curve(D0, deposits, [])


def test_dv01_on_bootstrapped_home_curve_matches_analytic():
    curve = _boot_curve()
    market = MarketSnapshot(valuation_date=D0, discount_curve=curve)
    bond = fixed_rate_bond(
        face=Money(1_000_000.0, CCY), coupon_rate=0.04, start=D0, maturity=date(2029, 1, 15),
        terms=ScheduleTerms(Frequency.SEMI_ANNUAL, DC.THIRTY_360),
    )
    priceable = discounting_priceable(bond, DiscountingEngine(), NUM)
    # d(DF)/d(shift) = -t*DF(t) under a parallel zero shift; DF from the boot curve itself
    analytic = -_ONE_BP * sum(
        cf.amount.amount * year_fraction(D0, cf.date, ACT365) * curve.df(cf.date)
        for cf in bond.cashflows
    )
    assert dv01(priceable, market, NUM) == pytest.approx(analytic, abs=1e-4)


def test_bumped_is_a_uniform_parallel_zero_shift():
    curve = _boot_curve()
    shift = 0.0010
    bumped = curve.bumped(shift)
    # every pillar's continuously-compounded zero rose by exactly `shift`
    for d, _ in curve.pillars[1:]:                       # skip the anchor (t=0)
        t = year_fraction(D0, d, ACT365)
        z0 = -math.log(curve.df(d)) / t
        z1 = -math.log(bumped.df(d)) / t
        assert z1 - z0 == pytest.approx(shift, abs=1e-12)
    assert bumped.pillars[0] == curve.pillars[0]         # anchor DF stays 1.0
