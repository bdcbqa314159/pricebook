"""S07 oracle — Hull-White 1F analytic core (L3).

Closed-form self-consistency (the strongest oracle tier, no MC):
  1. the model refits the initial curve: P^HW(0,T) == curve.df(T), exact;
  2. B(t,T) -> (T-t) as a -> 0;
  3. European ZCB option put-call parity: call - put == P(0,S) - K*P(0,T), exact;
  4. sigma -> 0 collapses the option to its discounted intrinsic value;
  5. the option value matches an independent recompute of the HW ZBC/ZBP formula.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.distributions import norm_cdf
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite

ABS = 1e-12
D0 = date(2026, 1, 5)
R0 = 0.03
EXPIRY = date(2027, 1, 5)     # ~1Y
BOND_MAT = date(2031, 1, 5)   # ~5Y


def _curve(rate=R0):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=DC.ACT_365_FIXED)


def _market(rate=R0):
    return MarketSnapshot(valuation_date=D0, discount_curve=_curve(rate))


def _hw(a=0.05, sigma=0.01):
    return HullWhite(a=a, sigma=sigma, market=_market())


def test_refits_initial_curve():
    hw, curve = _hw(), _curve()
    for d in [date(2026, 7, 5), EXPIRY, BOND_MAT, date(2036, 1, 5)]:
        assert hw.discount_factor(d) == pytest.approx(curve.df(d), abs=ABS)


def test_b_tends_to_time_gap_as_a_small():
    hw = _hw(a=1e-9)
    t, T = 1.0, 5.0
    assert hw.b(t, T) == pytest.approx(T - t, abs=1e-4)


def test_zcb_option_put_call_parity():
    hw, curve = _hw(), _curve()
    K = 0.87
    call = hw.zero_bond_option(EXPIRY, BOND_MAT, K, is_call=True)
    put = hw.zero_bond_option(EXPIRY, BOND_MAT, K, is_call=False)
    assert (call - put) == pytest.approx(curve.df(BOND_MAT) - K * curve.df(EXPIRY), abs=ABS)


def test_zero_vol_collapses_to_intrinsic():
    hw, curve = _hw(sigma=0.0), _curve()
    K = 0.87
    call = hw.zero_bond_option(EXPIRY, BOND_MAT, K, is_call=True)
    put = hw.zero_bond_option(EXPIRY, BOND_MAT, K, is_call=False)
    intrinsic_call = max(curve.df(BOND_MAT) - K * curve.df(EXPIRY), 0.0)
    assert call == pytest.approx(intrinsic_call, abs=ABS)
    assert put == pytest.approx(max(K * curve.df(EXPIRY) - curve.df(BOND_MAT), 0.0), abs=ABS)


def test_matches_independent_zbc_formula():
    a, sigma, K = 0.05, 0.01, 0.87
    hw, curve = HullWhite(a=a, sigma=sigma, market=_market()), _curve()
    T = year_fraction(D0, EXPIRY, DC.ACT_365_FIXED)
    tau = year_fraction(EXPIRY, BOND_MAT, DC.ACT_365_FIXED)
    p_t, p_s = curve.df(EXPIRY), curve.df(BOND_MAT)
    b_ts = (1.0 - math.exp(-a * tau)) / a
    sigma_p = sigma * math.sqrt((1.0 - math.exp(-2.0 * a * T)) / (2.0 * a)) * b_ts
    h = math.log(p_s / (p_t * K)) / sigma_p + sigma_p / 2.0
    expected_call = p_s * norm_cdf(h) - K * p_t * norm_cdf(h - sigma_p)
    assert hw.zero_bond_option(EXPIRY, BOND_MAT, K, is_call=True) == pytest.approx(
        expected_call, abs=ABS
    )
