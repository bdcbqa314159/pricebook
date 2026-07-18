"""FVA oracle — funding valuation adjustment (L5 risk & capital).

FVA is the funding cost/benefit of an uncollateralised position, reusing the same
EPE/ENE profiles as CVA/DVA but with a *funding-annuity* weight rather than a
protection-leg one:

    FVA = FCA - FBA = s_F * sum_i (EPE_i - ENE_i) * DF(t_i) * S(t_i) * tau_i

funding spread `s_F` carried over each interval `tau_i`, discounted, and weighted
by survival `S` (funding stops on default). Where CVA weights exposure by default
increments `(Q_{i-1}-Q_i)` and `(1-R)`, FVA weights it by the survival annuity `S*tau`
— exactly the CDS RPV01 structure. So the load-bearing oracle: FVA on unit positive
exposure equals `s_F * RPV01`. Plus linearity and a symmetric-exposure zero.
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
from pricebook_ng.risk.xva import ExposurePair, ExposureProfile, fva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
FUNDING_DC = DC.ACT_360
MATURITY = date(2031, 1, 15)
KEY = MarketKey(AssetClass.CREDIT, "SELF")
SPREAD = 0.01


def _market():
    disc = FlatDiscountCurve(0.02, D0, ACT365)
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.015)], 0.4)
    return MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival}), survival


def _grid():
    return generate_schedule(D0, MATURITY, Frequency.ANNUAL)


def _pair(epe_val, ene_val, grid):
    return ExposurePair(
        ExposureProfile(tuple(grid), tuple(epe_val for _ in grid)),
        ExposureProfile(tuple(grid), tuple(ene_val for _ in grid)),
    )


def test_fva_unit_positive_exposure_is_spread_times_funding_annuity():
    market, survival = _market()
    grid = _grid()
    pair = _pair(1.0, 0.0, grid)  # EPE=1, ENE=0 -> pure funding cost
    annuity = sum(
        year_fraction(grid[i - 1], grid[i], FUNDING_DC)
        * market.discount_curve.df(grid[i])
        * survival.df(grid[i])
        for i in range(1, len(grid))
    )
    assert fva(pair, market, KEY, SPREAD) == pytest.approx(SPREAD * annuity, abs=1e-14)


def test_symmetric_exposure_gives_zero_fva():
    market, _ = _market()
    grid = _grid()
    pair = _pair(3_000_000.0, 3_000_000.0, grid)  # EPE == ENE -> cost cancels benefit
    assert fva(pair, market, KEY, SPREAD) == pytest.approx(0.0, abs=1e-9)


def test_fva_is_linear_in_spread_and_exposure():
    market, _ = _market()
    grid = _grid()
    base = _pair(1_000_000.0, 200_000.0, grid)
    scaled = _pair(2_000_000.0, 400_000.0, grid)
    assert fva(scaled, market, KEY, SPREAD) == pytest.approx(
        2.0 * fva(base, market, KEY, SPREAD), abs=1e-9
    )
    assert fva(base, market, KEY, 2.0 * SPREAD) == pytest.approx(
        2.0 * fva(base, market, KEY, SPREAD), abs=1e-14
    )
