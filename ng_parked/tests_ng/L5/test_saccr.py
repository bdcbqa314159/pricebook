"""SA-CCR oracle — Basel standardised counterparty EAD & RWA (L5 risk & capital).

`EAD = alpha * (RC + PFE)` for a single-trade interest-rate netting set (unmargined,
uncollateralised). Oracles mix a regulatory anchor with structural limits:

  1. Anchor: a 10y at-the-money $100mm IRS has EAD ~ 5.5% of notional — the widely
     published SA-CCR result (alpha=1.4, IR supervisory factor 0.5%, supervisory
     duration with 5% decay, maturity factor 1).
  2. Replacement cost: EAD is linear in a positive mark, `d(EAD)/d(mark)=alpha` when
     in the money (RC=max(V,0), multiplier pinned at 1).
  3. PFE multiplier floor: deep out of the money the multiplier hits its 0.05 floor,
     so EAD collapses to `0.05 * EAD_atm`.
  4. RWA = EAD * risk weight; capital = 8% * RWA.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.saccr import risk_weighted_assets, saccr_capital, saccr_ead

D0 = date(2026, 1, 15)
NOTIONAL = 100_000_000.0
TERMS = SwapTerms(
    fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
    pay_fixed=True,
)


def _swap(maturity=date(2036, 1, 15)):
    return vanilla_swap(Money(NOTIONAL, Currency.USD), 0.03, D0, maturity, TERMS)


def test_atm_10y_irs_ead_is_about_5pct_of_notional():
    ead = saccr_ead(_swap(), mark=0.0, valuation_date=D0)
    assert ead / NOTIONAL == pytest.approx(0.0551, rel=1e-2)  # ~5.5%, the published anchor


def test_replacement_cost_adds_alpha_times_positive_mark():
    swap = _swap()
    atm = saccr_ead(swap, mark=0.0, valuation_date=D0)
    itm = saccr_ead(swap, mark=10_000_000.0, valuation_date=D0)
    # in the money: multiplier pinned at 1, so RC just adds alpha * mark
    assert itm - atm == pytest.approx(1.4 * 10_000_000.0, rel=1e-9)


def test_deep_out_of_money_hits_multiplier_floor():
    swap = _swap()
    atm = saccr_ead(swap, mark=0.0, valuation_date=D0)          # PFE = AddOn (mult 1), RC 0
    deep = saccr_ead(swap, mark=-1e12, valuation_date=D0)       # multiplier -> 0.05 floor, RC 0
    assert deep == pytest.approx(0.05 * atm, rel=1e-9)


def test_rwa_and_capital():
    ead = saccr_ead(_swap(), mark=0.0, valuation_date=D0)
    assert risk_weighted_assets(ead, 1.0) == pytest.approx(ead)          # 100% corporate RW
    assert risk_weighted_assets(ead, 0.2) == pytest.approx(0.2 * ead)    # 20% bank RW
    assert saccr_capital(risk_weighted_assets(ead, 1.0)) == pytest.approx(0.08 * ead)
