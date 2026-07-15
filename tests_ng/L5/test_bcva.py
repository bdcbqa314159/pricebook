"""DVA / bilateral BCVA oracle (L5 risk & capital).

CVA and DVA are the *same* protection-leg integral seen from the two sides:
  - CVA = expected loss when the COUNTERPARTY defaults while we are in the money
    (positive exposure, EPE), against their survival curve.
  - DVA = expected gain when WE default while we are out of the money (negative
    exposure, ENE = E[(-V)^+]), against our own survival curve.
So `dva` is just `cva` on the negative-exposure profile with our own credit, and
  BCVA = CVA - DVA   (net credit charge; value adjustment = -BCVA).

Oracles: (1) ENE of a payer swap equals EPE of the mirror receiver swap, exactly
(same simulated rates) — ties ENE to the CVA-oracle'd exposure machinery; (2) BCVA
decomposes into CVA - DVA; (3) a default-free self (Q_self ≡ 1) zeroes DVA, so BCVA
collapses to the unilateral CVA.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.calibration.survival_curve import bootstrap_survival_curve
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote, SurvivalCurve
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.exposure import exposure_profiles
from pricebook_ng.risk.xva import CreditParty, bcva, cva, dva

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
USD = Currency.USD
MATURITY = date(2031, 1, 15)
FACE = Money(1_000_000.0, USD)
NUM = NumericalConfig(mc_paths=20_000, mc_seed=5)

CPTY = MarketKey(AssetClass.CREDIT, "CPTY")
SELF = MarketKey(AssetClass.CREDIT, "SELF")


def _terms(pay_fixed):
    leg = ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360)
    return SwapTerms(fixed_schedule=leg, float_schedule=leg, pay_fixed=pay_fixed)


def _swap(pay_fixed=True):
    return vanilla_swap(FACE, 0.03, D0, MATURITY, _terms(pay_fixed))


def _model():
    market = MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(0.03, D0, ACT365))
    return HullWhite(a=0.05, sigma=0.012, market=market)


def _market_with_credit(self_riskless=False):
    disc = FlatDiscountCurve(0.03, D0, ACT365)
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    cpty = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.03)], 0.4)
    own = (
        SurvivalCurve(D0, ((D0, 1.0), (MATURITY, 1.0)))
        if self_riskless
        else bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.01)], 0.4)
    )
    return MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={CPTY: cpty, SELF: own})


def test_ene_of_payer_equals_epe_of_receiver():
    model = _model()
    payer = exposure_profiles(_swap(pay_fixed=True), model, NUM)
    receiver = exposure_profiles(_swap(pay_fixed=False), model, NUM)
    # same simulated rates -> V_receiver = -V_payer path by path -> ENE_payer == EPE_receiver
    assert payer.ene.ee == pytest.approx(receiver.epe.ee, abs=1e-9)
    assert payer.epe.ee == pytest.approx(receiver.ene.ee, abs=1e-9)


def test_bcva_decomposes_into_cva_minus_dva():
    model = _model()
    exposure = exposure_profiles(_swap(), model, NUM)
    market = _market_with_credit()
    counterparty = CreditParty(CPTY, 0.4)
    myself = CreditParty(SELF, 0.4)

    cva_term = cva(exposure.epe, market, CPTY, 0.4)
    dva_term = dva(exposure.ene, market, SELF, 0.4)
    assert dva_term > 0.0
    assert bcva(exposure, market, counterparty, myself) == pytest.approx(
        cva_term - dva_term, abs=1e-12
    )


def test_riskless_self_collapses_bcva_to_cva():
    model = _model()
    exposure = exposure_profiles(_swap(), model, NUM)
    market = _market_with_credit(self_riskless=True)
    counterparty = CreditParty(CPTY, 0.4)
    myself = CreditParty(SELF, 0.4)
    assert dva(exposure.ene, market, SELF, 0.4) == pytest.approx(0.0, abs=1e-14)
    assert bcva(exposure, market, counterparty, myself) == pytest.approx(
        cva(exposure.epe, market, CPTY, 0.4), abs=1e-12
    )
