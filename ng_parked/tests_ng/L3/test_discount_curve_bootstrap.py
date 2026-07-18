"""S03 oracle — bootstrapped single-curve discount curve (L1).

Deposits give the short-end DFs in closed form; par swaps extend the curve by a
sequential closed-form solve (annual fixed coupons landing on prior pillars).
The oracle is self-consistency: every input reprices to par, exact.

Single-curve identity used for the swap par check: with notional exchange and
discount = projection, the float leg PV per unit notional telescopes to
`1 - DF(maturity)`, and par means `rate * sum(tau_i * DF(t_i)) == 1 - DF(maturity)`.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.market.discount_curve import DepositQuote, DiscountCurve, ParSwapQuote
from pricebook_ng.calibration.discount_curve import bootstrap_discount_curve
from pricebook_ng.market.snapshot import CurveHandle

ABS = 1e-12
D0 = date(2026, 1, 5)

DEPOSITS = [
    DepositQuote(maturity=date(2026, 4, 5), rate=0.030, day_count=DC.ACT_360),
    DepositQuote(maturity=date(2026, 7, 5), rate=0.031, day_count=DC.ACT_360),
    DepositQuote(maturity=date(2027, 1, 5), rate=0.032, day_count=DC.ACT_360),  # 1Y pillar
]
SWAPS = [
    ParSwapQuote(maturity=date(2028, 1, 5), rate=0.034,
                 fixed_frequency=Frequency.ANNUAL, day_count=DC.ACT_360),
    ParSwapQuote(maturity=date(2029, 1, 5), rate=0.036,
                 fixed_frequency=Frequency.ANNUAL, day_count=DC.ACT_360),
]


@pytest.fixture
def curve() -> DiscountCurve:
    return bootstrap_discount_curve(D0, DEPOSITS, SWAPS)


def test_is_a_curve_handle(curve):
    assert isinstance(curve, CurveHandle)


def test_df_at_valuation_is_one(curve):
    assert curve.df(D0) == pytest.approx(1.0, abs=ABS)


def test_deposits_reprice_to_closed_form_df(curve):
    for dep in DEPOSITS:
        tau = year_fraction(D0, dep.maturity, dep.day_count)
        assert curve.df(dep.maturity) == pytest.approx(1.0 / (1.0 + dep.rate * tau), abs=ABS)


def test_swaps_reprice_to_par(curve):
    for sw in SWAPS:
        sched = generate_schedule(D0, sw.maturity, sw.fixed_frequency)
        fixed_pv = sum(
            year_fraction(sched[i - 1], sched[i], sw.day_count) * curve.df(sched[i])
            for i in range(1, len(sched))
        ) * sw.rate
        float_pv = 1.0 - curve.df(sw.maturity)  # single-curve telescoping identity
        assert fixed_pv == pytest.approx(float_pv, abs=ABS)


def test_discount_factors_strictly_decreasing(curve):
    dfs = [curve.df(q.maturity) for q in DEPOSITS + SWAPS]
    assert all(0.0 < b < a <= 1.0 for a, b in zip([1.0, *dfs], dfs))


def test_loglinear_interpolation_between_pillars(curve):
    # midway (in ACT/365F time) between two pillars, ln(df) is the linear blend
    import math
    a, b = date(2027, 1, 5), date(2028, 1, 5)  # adjacent pillars (1Y, 2Y)
    mid = date(2027, 7, 6)
    ta = year_fraction(D0, a, DC.ACT_365_FIXED)
    tb = year_fraction(D0, b, DC.ACT_365_FIXED)
    tm = year_fraction(D0, mid, DC.ACT_365_FIXED)
    w = (tm - ta) / (tb - ta)
    expected_ln = (1 - w) * math.log(curve.df(a)) + w * math.log(curve.df(b))
    assert math.log(curve.df(mid)) == pytest.approx(expected_ln, abs=1e-12)


def test_empty_inputs_raises():
    with pytest.raises(ValueError):
        bootstrap_discount_curve(D0, [], [])
