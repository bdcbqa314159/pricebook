"""KVA oracle — capital valuation adjustment (L5 risk & capital).

KVA is the cost of holding regulatory capital over the life of the trade: the
cost-of-capital hurdle `gamma_K` charged on the capital profile `K(t)`, discounted
and survival-weighted —

    KVA = gamma_K * sum_i  K(t_i) * DF(t_i) * S(t_i) * tau_i

the *same* survival-weighted funding annuity as FVA (the CDS RPV01 structure), with
the capital profile in place of net exposure and the cost of capital in place of the
funding spread. Load-bearing oracle: KVA on unit capital equals `gamma_K * RPV01`.

Capital *generation* (a regulatory SA-CCR / IRB model turning exposure into required
capital) is upstream and out of scope — `K(t)` is an input here, exactly as `EE(t)`
is an input to CVA. The test shows the intended wiring with a simple `K = k * EPE`.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.risk.xva import ExposureProfile, kva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ANNUITY_DC = DC.ACT_360
MATURITY = date(2031, 1, 15)
KEY = MarketKey(AssetClass.CREDIT, "SELF")
GAMMA = 0.10  # cost of capital (hurdle)


def _market():
    disc = FlatDiscountCurve(0.02, D0, ACT365)
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.015)], 0.4)
    return MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival}), survival


def _grid():
    return generate_schedule(D0, MATURITY, Frequency.ANNUAL)


def _annuity(market, survival, grid):
    return sum(
        year_fraction(grid[i - 1], grid[i], ANNUITY_DC)
        * market.discount_curve.df(grid[i])
        * survival.df(grid[i])
        for i in range(1, len(grid))
    )


def test_kva_unit_capital_is_cost_of_capital_times_annuity():
    market, survival = _market()
    grid = _grid()
    capital = ExposureProfile(tuple(grid), tuple(1.0 for _ in grid))
    assert kva(capital, market, KEY, GAMMA) == pytest.approx(
        GAMMA * _annuity(market, survival, grid), abs=1e-14
    )


def test_kva_from_capital_proportional_to_exposure():
    market, survival = _market()
    grid = _grid()
    epe = [max(1_000_000.0 * (1.0 - i / len(grid)), 0.0) for i in range(len(grid))]  # runoff profile
    k = 0.08  # capital factor (stand-in for a regulatory EAD model)
    capital = ExposureProfile(tuple(grid), tuple(k * e for e in epe))
    expected = GAMMA * sum(
        k * epe[i]
        * year_fraction(grid[i - 1], grid[i], ANNUITY_DC)
        * market.discount_curve.df(grid[i])
        * survival.df(grid[i])
        for i in range(1, len(grid))
    )
    assert kva(capital, market, KEY, GAMMA) == pytest.approx(expected, abs=1e-9)


def test_kva_is_linear_in_cost_of_capital():
    market, _ = _market()
    grid = _grid()
    capital = ExposureProfile(tuple(grid), tuple(500_000.0 for _ in grid))
    assert kva(capital, market, KEY, 2.0 * GAMMA) == pytest.approx(
        2.0 * kva(capital, market, KEY, GAMMA), abs=1e-14
    )
