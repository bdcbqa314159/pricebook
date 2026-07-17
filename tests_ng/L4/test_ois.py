"""OIS oracle (L4) — CP-2c #4, fixed_income spine (+ RateCurve / forward_rate).

An overnight index swap: fixed vs the compounded overnight rate. Single-curve, the
compounded rate over a period equals the curve's simply-compounded forward
`L(a,b) = (P(0,a)/P(0,b)-1)/τ` (new curve building block `forward_rate`), so the float
leg telescopes to `N·(P(0,start)-P(0,maturity))` exactly like a vanilla IRS. So the
load-bearing oracle is `OIS == vanilla IRS` for the same terms (single-curve); the
multi-curve OIS/IBOR basis and daily-fixing compounding are the deferred parity gap.

Oracles: `curve.forward_rate` equals the DF-ratio forward (flat + bootstrapped); a par
OIS reprices to zero; and the OIS NPV equals the equivalent vanilla-swap NPV.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.discount_curve import bootstrap_discount_curve
from pricebook_ng.engine.ois import OISEngine
from pricebook_ng.engine.swap import SwapEngine
from pricebook_ng.market.discount_curve import DepositQuote
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot, RateCurve
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.ois import overnight_index_swap
from pricebook_ng.products.swap import SwapTerms, vanilla_swap

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
USD = Currency.USD
FACE = Money(100_000_000.0, USD)
MATURITY = date(2029, 1, 15)
NUM = NumericalConfig()
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _model(curve):
    return DiscountingModel(MarketSnapshot(valuation_date=D0, discount_curve=curve))


def _par_rate(curve):
    dates = generate_schedule(D0, MATURITY, Frequency.ANNUAL)
    annuity = sum(
        year_fraction(a, b, DC.THIRTY_360) * curve.df(b) for a, b in zip(dates[:-1], dates[1:])
    )
    return (curve.df(D0) - curve.df(MATURITY)) / annuity


def test_curve_forward_rate_matches_df_ratio():
    for curve in (
        FlatDiscountCurve(0.03, D0, ACT365),
        bootstrap_discount_curve(D0, [DepositQuote(date(2027, 1, 15), 0.03, ACT360)], []),
    ):
        assert isinstance(curve, RateCurve)  # the rate-curve capability protocol
        d1, d2 = date(2027, 1, 15), date(2027, 7, 15)
        expected = (curve.df(d1) / curve.df(d2) - 1.0) / year_fraction(d1, d2, ACT360)
        assert curve.forward_rate(d1, d2, ACT360) == pytest.approx(expected, abs=1e-14)


def test_par_ois_prices_to_zero():
    curve = FlatDiscountCurve(0.03, D0, ACT365)
    ois = overnight_index_swap(FACE, _par_rate(curve), D0, MATURITY, TERMS)
    assert OISEngine().price(ois, _model(curve), NUM).pv.amount == pytest.approx(0.0, abs=1e-4)


def test_ois_equals_vanilla_irs_single_curve():
    for curve in (
        FlatDiscountCurve(0.03, D0, ACT365),
        bootstrap_discount_curve(
            D0, [DepositQuote(date(2028, 1, 15), 0.035, ACT360), DepositQuote(date(2030, 1, 15), 0.04, ACT360)], []
        ),
    ):
        model = _model(curve)
        ois = overnight_index_swap(FACE, 0.03, D0, MATURITY, TERMS)
        swap = vanilla_swap(FACE, 0.03, D0, MATURITY, TERMS)
        assert OISEngine().price(ois, model, NUM).pv.amount == pytest.approx(
            SwapEngine().price(swap, model, NUM).pv.amount, abs=1e-6
        )
