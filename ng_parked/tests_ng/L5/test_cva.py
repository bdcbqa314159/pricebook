"""CVA oracle — unilateral credit valuation adjustment (L5 risk & capital).

CVA is the expected loss from counterparty default over a trade's exposure profile:
    CVA = (1-R) * sum_i EE(t_i) * DF(t_i) * (Q(t_{i-1}) - Q(t_i))
the *same* discounted-default "protection leg" as a CDS, weighted by expected
exposure `EE(t)` instead of a unit notional (independence of exposure and default
assumed — the standard unilateral form). So the load-bearing oracle is exact: on a
flat unit-exposure profile over a CDS grid, CVA equals that CDS's protection leg
(`cds_pv` at zero spread), which is already oracle-checked. Plus linearity in
exposure and zero CVA against a default-free (Q≡1) counterparty.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote, SurvivalCurve, cds_pv
from pricebook_ng.risk.xva import ExposureProfile, cva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
MATURITY = date(2031, 1, 15)
RECOVERY = 0.4
ISSUER = "ACME_CO"
KEY = MarketKey(AssetClass.CREDIT, ISSUER)
USD = Currency.USD


def _curve(rate):
    return FlatDiscountCurve(rate, D0, ACT365)


def _market():
    bare = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.02))
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.025)], RECOVERY)
    market = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.02), curves={KEY: survival})
    return market, survival


def _grid():
    return generate_schedule(D0, MATURITY, Frequency.ANNUAL)


def test_unit_exposure_cva_equals_cds_protection_leg():
    market, survival = _market()
    grid = _grid()
    unit = ExposureProfile(tuple(grid), tuple(1.0 for _ in grid))
    # cds_pv at zero spread is the protection leg alone: (1-R) sum DF*(Q_{i-1}-Q_i)
    protection = cds_pv(market.discount_curve, survival, grid, 0.0, RECOVERY)
    assert cva(unit, market, KEY, RECOVERY) == pytest.approx(protection, abs=1e-14)


def test_cva_is_linear_in_exposure():
    market, _ = _market()
    grid = _grid()
    single = ExposureProfile(tuple(grid), tuple(1_000_000.0 for _ in grid))
    double = ExposureProfile(tuple(grid), tuple(2_000_000.0 for _ in grid))
    assert cva(double, market, KEY, RECOVERY) == pytest.approx(
        2.0 * cva(single, market, KEY, RECOVERY), abs=1e-9
    )


def test_no_default_risk_gives_zero_cva():
    # Q ≡ 1 everywhere -> no default probability mass -> CVA = 0
    riskless = SurvivalCurve(D0, ((D0, 1.0), (MATURITY, 1.0)))
    market = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.02), curves={KEY: riskless})
    grid = _grid()
    profile = ExposureProfile(tuple(grid), tuple(5_000_000.0 for _ in grid))
    assert cva(profile, market, KEY, RECOVERY) == pytest.approx(0.0, abs=1e-14)
