"""General-curve Hull-White oracle (L3) — CP-2b #2.

HW now reads `df` + `instantaneous_forward` from ANY curve (flat or bootstrapped),
replacing the flat `r0` with the market forward `f(0,t)`. Oracles:
  1. Curve refit: `zero_bond(0, S, f(0,0)) == P^M(0,S)` on a bootstrapped curve (the
     model reprices the initial curve exactly).
  2. ZCB-option put-call parity on the bootstrapped curve.
  3. Flat equivalence: a `DiscountCurve` built from `exp(-r t)` pillars gives byte-identical
     HW outputs to a `FlatDiscountCurve(r)` (the flat curve is the degenerate general curve).
  4. Integration: on the bootstrapped curve the analytic swaption == the MC swaption (exercises
     the general `forward_short_rate` + `zero_bond` together).
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.discount_curve import bootstrap_discount_curve
from pricebook_ng.engine.swaption import SwaptionEngine
from pricebook_ng.engine.swaption_mc import SwaptionMCEngine
from pricebook_ng.market.discount_curve import DepositQuote, DiscountCurve
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.products.swaption import Swaption
import math

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
A, SIGMA = 0.05, 0.010


def _rising_curve():
    deposits = [
        DepositQuote(date(2027, 1, 15), 0.030, ACT360),
        DepositQuote(date(2028, 1, 15), 0.035, ACT360),
        DepositQuote(date(2029, 1, 15), 0.040, ACT360),
        DepositQuote(date(2030, 1, 15), 0.043, ACT360),
        DepositQuote(date(2031, 1, 15), 0.045, ACT360),
    ]
    return bootstrap_discount_curve(D0, deposits, [])


def _model(curve):
    return HullWhite(A, SIGMA, MarketSnapshot(valuation_date=D0, discount_curve=curve))


def test_model_refits_the_initial_curve():
    curve = _rising_curve()
    model = _model(curve)
    f0 = curve.instantaneous_forward(D0)  # short rate r(0) = f(0,0)
    for s in (date(2028, 1, 15), date(2029, 7, 1), date(2031, 1, 15)):
        assert model.zero_bond(D0, s, f0) == pytest.approx(curve.df(s), abs=1e-10)


def test_zcb_option_put_call_parity_on_bootstrapped_curve():
    curve = _rising_curve()
    model = _model(curve)
    expiry, bond_mat, strike = date(2028, 1, 15), date(2030, 1, 15), 0.90
    call = model.zero_bond_option(expiry, bond_mat, strike, is_call=True)
    put = model.zero_bond_option(expiry, bond_mat, strike, is_call=False)
    assert call - put == pytest.approx(curve.df(bond_mat) - strike * curve.df(expiry), abs=1e-12)


def test_flat_pillars_curve_matches_flat_curve():
    r = 0.03
    dates = [D0, date(2027, 1, 15), date(2029, 1, 15), date(2031, 1, 15)]
    pillar_curve = DiscountCurve(D0, tuple((d, math.exp(-r * year_fraction(D0, d, ACT365))) for d in dates))
    flat = FlatDiscountCurve(r, D0, ACT365)
    expiry, bond_mat, strike = date(2028, 1, 15), date(2030, 1, 15), 0.92
    for is_call in (True, False):
        assert _model(pillar_curve).zero_bond_option(expiry, bond_mat, strike, is_call) == pytest.approx(
            _model(flat).zero_bond_option(expiry, bond_mat, strike, is_call), abs=1e-12
        )


def test_analytic_swaption_equals_mc_on_bootstrapped_curve():
    curve = _rising_curve()
    model = _model(curve)
    terms = SwapTerms(
        fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
        float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
        pay_fixed=True,
    )
    swap = vanilla_swap(Money(1_000_000.0, Currency.USD), 0.04, date(2028, 1, 15), date(2031, 1, 15), terms)
    swaption = Swaption(expiry=date(2028, 1, 15), swap=swap)
    num = NumericalConfig(mc_paths=200_000, mc_seed=1)
    analytic = SwaptionEngine().price(swaption, model, num).pv.amount
    mc = SwaptionMCEngine().price(swaption, model, num).pv.amount
    assert mc == pytest.approx(analytic, rel=0.02)
