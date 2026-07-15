"""Forward SA-CCR EAD profile -> KVA oracle (L5 risk & capital).

Closes the capital loop: SA-CCR EAD computed at each future coupon date on the
*remaining* trade gives a runoff EAD profile; scaling by `8% * risk_weight` turns it
into the capital profile KVA integrates. Under the ATM assumption (expected mark = 0)
the profile is deterministic — RC = 0, multiplier = 1 — so EAD(t_j) is the closed-form
supervisory PFE that shrinks as remaining maturity runs off.

Oracles: (1) each EAD(t_j) equals the closed-form `alpha * SF * notional * SD * MF`
on the remaining maturity, and the first point equals the single-date `saccr_ead`;
(2) the profile runs off monotonically; (3) `capital = 8% * EAD * RW`; (4) KVA on the
capital profile equals the cost-of-capital annuity.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.saccr import capital_profile, forward_ead_profile, saccr_ead
from pricebook_ng.risk.xva import kva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ANNUITY_DC = DC.ACT_360
MATURITY = date(2036, 1, 15)
NOTIONAL = 100_000_000.0
RISK_WEIGHT = 1.0
GAMMA = 0.10
KEY = MarketKey(AssetClass.CREDIT, "SELF")
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _swap():
    return vanilla_swap(Money(NOTIONAL, Currency.USD), 0.03, D0, MATURITY, TERMS)


def _market():
    disc = FlatDiscountCurve(0.02, D0, ACT365)
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.015)], 0.4)
    return MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival}), survival


def _closed_form_ead(remaining_years):
    sd = (1.0 - math.exp(-0.05 * remaining_years)) / 0.05  # S=0
    return 1.4 * 0.005 * NOTIONAL * sd * math.sqrt(min(remaining_years, 1.0))


def test_forward_ead_profile_matches_closed_form_runoff():
    swap = _swap()
    profile = forward_ead_profile(swap, D0)
    for t_j, ead in zip(profile.grid, profile.ee):
        remaining = year_fraction(t_j, MATURITY, ACT365)
        assert ead == pytest.approx(_closed_form_ead(remaining), abs=1e-6)
    # first grid point is the as-of-today single-date SA-CCR EAD
    assert profile.ee[0] == pytest.approx(saccr_ead(swap, 0.0, D0), abs=1e-9)


def test_ead_profile_runs_off_monotonically():
    profile = forward_ead_profile(_swap(), D0)
    assert all(a > b for a, b in zip(profile.ee, profile.ee[1:]))  # strictly decreasing
    assert profile.ee[-1] > 0.0


def test_capital_profile_is_eight_percent_of_rwa():
    ead = forward_ead_profile(_swap(), D0)
    cap = capital_profile(ead, RISK_WEIGHT)
    assert cap.grid == ead.grid
    for e, c in zip(ead.ee, cap.ee):
        assert c == pytest.approx(0.08 * e * RISK_WEIGHT, abs=1e-9)


def test_kva_on_forward_capital_profile():
    market, survival = _market()
    cap = capital_profile(forward_ead_profile(_swap(), D0), RISK_WEIGHT)
    expected = GAMMA * sum(
        cap.ee[i]
        * year_fraction(cap.grid[i - 1], cap.grid[i], ANNUITY_DC)
        * market.discount_curve.df(cap.grid[i])
        * survival.df(cap.grid[i])
        for i in range(1, len(cap.grid))
    )
    assert kva(cap, market, KEY, GAMMA) == pytest.approx(expected, abs=1e-6)
    assert kva(cap, market, KEY, GAMMA) > 0.0
