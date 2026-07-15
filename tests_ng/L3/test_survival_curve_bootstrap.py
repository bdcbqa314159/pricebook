"""Credit oracle — hazard/survival curve bootstrapped from CDS par spreads (L1).

Mirrors the S03 discount-curve bootstrap: sequential solve of a piecewise-hazard
`SurvivalCurve` so that each input CDS reprices to zero at its par spread
(self-consistency). CDS legs use annual premiums, ACT/360 accrual, a discrete
protection integral on the premium grid, recovery R.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.survival_curve import (
    CDSQuote,
    SurvivalCurve,
    cds_par_spread,
    cds_pv,
)

ABS = 1e-10
D0 = date(2026, 1, 5)
RECOVERY = 0.4

MARKET = MarketSnapshot(
    valuation_date=D0,
    discount_curve=FlatDiscountCurve(rate=0.02, anchor=D0, day_count=DC.ACT_365_FIXED),
)
QUOTES = [
    CDSQuote(maturity=date(2027, 1, 5), par_spread=0.0100),   # 1Y, 100bp
    CDSQuote(maturity=date(2029, 1, 5), par_spread=0.0150),   # 3Y, 150bp
    CDSQuote(maturity=date(2031, 1, 5), par_spread=0.0200),   # 5Y, 200bp
]


@pytest.fixture
def curve() -> SurvivalCurve:
    return bootstrap_survival_curve(MARKET, QUOTES, RECOVERY)


def _sched(maturity):
    return generate_schedule(D0, maturity, Frequency.ANNUAL)


def test_survival_at_valuation_is_one(curve):
    assert curve.survival(D0) == pytest.approx(1.0, abs=ABS)


def test_each_input_cds_reprices_to_zero(curve):
    for q in QUOTES:
        pv = cds_pv(MARKET.discount_curve, curve, _sched(q.maturity), q.par_spread, RECOVERY)
        assert pv == pytest.approx(0.0, abs=ABS)


def test_curve_implied_par_spread_matches_quotes(curve):
    for q in QUOTES:
        implied = cds_par_spread(MARKET.discount_curve, curve, _sched(q.maturity), RECOVERY)
        assert implied == pytest.approx(q.par_spread, abs=1e-10)


def test_survival_strictly_decreasing_in_unit_interval(curve):
    qs = [curve.survival(q.maturity) for q in QUOTES]
    assert all(0.0 < b < a <= 1.0 for a, b in zip([1.0, *qs], qs))


def test_loglinear_survival_between_pillars(curve):
    import math
    a, b = date(2027, 1, 5), date(2029, 1, 5)  # 1Y, 3Y pillars
    mid = date(2028, 1, 5)
    ta = (a - D0).days / 365.0
    tb = (b - D0).days / 365.0
    tm = (mid - D0).days / 365.0
    w = (tm - ta) / (tb - ta)
    expected_ln = (1 - w) * math.log(curve.survival(a)) + w * math.log(curve.survival(b))
    assert math.log(curve.survival(mid)) == pytest.approx(expected_ln, abs=1e-12)


def test_empty_quotes_raises():
    with pytest.raises(ValueError):
        bootstrap_survival_curve(MARKET, [], RECOVERY)
