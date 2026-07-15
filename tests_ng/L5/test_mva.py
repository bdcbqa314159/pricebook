"""MVA oracle — margin valuation adjustment (L5 risk & capital).

MVA is the funding cost of posting initial margin over the trade's life — the same
survival-weighted funding annuity as FVA/KVA, with the IM profile in place of net
exposure / capital:

    MVA = s_F * sum_i  IM(t_i) * DF(t_i) * S(t_i) * tau_i

Oracles: MVA on unit IM equals `s_F * RPV01` (the funding annuity, CDS RPV01 structure);
linearity in spread and IM. Plus a vertical: an IM profile taken as the SA-CCR PFE
(AddOn) runoff feeds MVA to a positive charge that matches its annuity.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.saccr import forward_ead_profile
from pricebook_ng.risk.xva import ExposureProfile, mva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ANNUITY_DC = DC.ACT_360
MATURITY = date(2036, 1, 15)
KEY = MarketKey(AssetClass.CREDIT, "SELF")
SPREAD = 0.008  # IM funding spread


def _market():
    disc = FlatDiscountCurve(0.02, D0, ACT365)
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.015)], 0.4)
    return MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival}), survival


def _grid():
    return generate_schedule(D0, MATURITY, Frequency.ANNUAL)


def _annuity(market, survival, grid, values):
    return sum(
        values[i]
        * year_fraction(grid[i - 1], grid[i], ANNUITY_DC)
        * market.discount_curve.df(grid[i])
        * survival.df(grid[i])
        for i in range(1, len(grid))
    )


def test_mva_unit_im_is_spread_times_funding_annuity():
    market, survival = _market()
    grid = _grid()
    im = ExposureProfile(tuple(grid), tuple(1.0 for _ in grid))
    annuity = _annuity(market, survival, grid, im.ee)
    assert mva(im, market, KEY, SPREAD) == pytest.approx(SPREAD * annuity, abs=1e-14)


def test_mva_is_linear_in_spread_and_im():
    market, _ = _market()
    grid = _grid()
    base = ExposureProfile(tuple(grid), tuple(2_000_000.0 for _ in grid))
    scaled = ExposureProfile(tuple(grid), tuple(4_000_000.0 for _ in grid))
    assert mva(scaled, market, KEY, SPREAD) == pytest.approx(2.0 * mva(base, market, KEY, SPREAD), abs=1e-9)
    assert mva(base, market, KEY, 2.0 * SPREAD) == pytest.approx(2.0 * mva(base, market, KEY, SPREAD), abs=1e-14)


def test_mva_on_saccr_pfe_runoff_im():
    market, survival = _market()
    swap = vanilla_swap(
        Money(100_000_000.0, Currency.USD), 0.03, D0, MATURITY,
        SwapTerms(
            fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
            float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
            pay_fixed=True,
        ),
    )
    ead = forward_ead_profile(swap, D0)                                   # = alpha * AddOn
    im = ExposureProfile(ead.grid, tuple(e / 1.4 for e in ead.ee))       # IM proxy = SA-CCR PFE (AddOn)
    expected = SPREAD * _annuity(market, survival, im.grid, im.ee)
    assert mva(im, market, KEY, SPREAD) == pytest.approx(expected, abs=1e-6)
    assert mva(im, market, KEY, SPREAD) > 0.0
