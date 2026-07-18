"""Stochastic-mark SA-CCR EAD profile oracle (L5 risk & capital).

Unifies the two halves of the capital stack: the MC exposure engine's expected
positive exposure becomes SA-CCR's replacement cost, added to the supervisory PFE
runoff —

    EAD(t_j) = alpha * ( EPE(t_j) + AddOn_remaining(t_j) )

(EPE >= 0 pins the multiplier at 1). So it decomposes exactly into the deterministic
PFE-only profile from the previous slice plus `alpha * EPE`:

    stochastic_ead(t_j) = forward_ead(t_j) + alpha * EPE(t_j)

both pieces already oracle-checked. Plus: it dominates the deterministic profile, and
its capital profile feeds KVA to a larger charge than the ATM one.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.exposure import exposure_profiles
from pricebook_ng.risk.saccr import (
    capital_profile,
    forward_ead_profile,
    stochastic_ead_profile,
)
from pricebook_ng.risk.xva import kva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ANNUITY_DC = DC.ACT_360
MATURITY = date(2036, 1, 15)
NOTIONAL = 100_000_000.0
ALPHA = 1.4
RISK_WEIGHT = 1.0
GAMMA = 0.10
KEY = MarketKey(AssetClass.CREDIT, "SELF")
NUM = NumericalConfig(mc_paths=10_000, mc_seed=7)
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _swap():
    return vanilla_swap(Money(NOTIONAL, Currency.USD), 0.04, D0, MATURITY, TERMS)


def _model():
    market = MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(0.03, D0, ACT365))
    return HullWhite(a=0.05, sigma=0.012, market=market)


def _credit_market():
    disc = FlatDiscountCurve(0.03, D0, ACT365)
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    survival = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.015)], 0.4)
    return MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={KEY: survival}), survival


def test_stochastic_ead_decomposes_into_pfe_plus_alpha_epe():
    swap, model = _swap(), _model()
    stochastic = stochastic_ead_profile(swap, model, NUM)
    epe = exposure_profiles(swap, model, NUM).epe
    pfe = forward_ead_profile(swap, D0)  # deterministic RC=0 profile (= alpha * AddOn)
    assert stochastic.grid == epe.grid == pfe.grid
    for s, e, p in zip(stochastic.ee, epe.ee, pfe.ee):
        assert s == pytest.approx(p + ALPHA * e, abs=1e-3)


def test_stochastic_ead_dominates_deterministic():
    swap, model = _swap(), _model()
    stochastic = stochastic_ead_profile(swap, model, NUM)
    pfe = forward_ead_profile(swap, D0)
    assert all(s >= p for s, p in zip(stochastic.ee, pfe.ee))
    assert any(s > p for s, p in zip(stochastic.ee, pfe.ee))  # positive expected exposure somewhere


def test_kva_on_stochastic_capital_exceeds_atm():
    market, survival = _credit_market()
    model = _model()
    swap = _swap()
    stochastic_cap = capital_profile(stochastic_ead_profile(swap, model, NUM), RISK_WEIGHT)
    atm_cap = capital_profile(forward_ead_profile(swap, D0), RISK_WEIGHT)

    expected = GAMMA * sum(
        stochastic_cap.ee[i]
        * year_fraction(stochastic_cap.grid[i - 1], stochastic_cap.grid[i], ANNUITY_DC)
        * market.discount_curve.df(stochastic_cap.grid[i])
        * survival.df(stochastic_cap.grid[i])
        for i in range(1, len(stochastic_cap.grid))
    )
    assert kva(stochastic_cap, market, KEY, GAMMA) == pytest.approx(expected, abs=1e-3)
    assert kva(stochastic_cap, market, KEY, GAMMA) > kva(atm_cap, market, KEY, GAMMA)
