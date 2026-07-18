"""A6.2 oracle — the consolidated counterparty XVA report (L6 shell).

`xva_report` is the L6 object per A6.2: a per-counterparty netting set (a book of
swaps) simulated ONCE, returning CVA/DVA/BCVA/FVA/KVA/MVA + the EE/PFE/EAD profiles —
the six separate L5 calls consolidated. Oracles: for a single-trade netting set each
adjustment equals its standalone L5 value (one pass, not six); for a payer + mirror
receiver the portfolio exposure nets to zero, so the netted CVA collapses (netting is
real, not a sum of standalones).
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
from pricebook_ng.market.survival_curve import CDSQuote
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.exposure import exposure_profiles, pfe_profile
from pricebook_ng.risk.saccr import capital_profile, forward_ead_profile
from pricebook_ng.risk.xva import CreditParty, cva, dva, fva, kva, mva
from pricebook_ng.shell.xva_report import XvaReportConfig, xva_report

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
MATURITY = date(2031, 1, 15)
NOTIONAL = 100_000_000.0
CP = MarketKey(AssetClass.CREDIT, "CPTY")
SELF = MarketKey(AssetClass.CREDIT, "SELF")
NUM = NumericalConfig(mc_paths=20_000, mc_seed=7)
FUNDING, COC, RW, Q = 0.008, 0.10, 1.0, 0.99


def _terms(pay_fixed):
    leg = ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360)
    return SwapTerms(fixed_schedule=leg, float_schedule=leg, pay_fixed=pay_fixed)


def _swap(pay_fixed=True):
    return vanilla_swap(Money(NOTIONAL, Currency.USD), 0.03, D0, MATURITY, _terms(pay_fixed))


def _model():
    disc = FlatDiscountCurve(0.03, D0, ACT365)
    bare = MarketSnapshot(valuation_date=D0, discount_curve=disc)
    cp = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.02)], 0.4)
    me = bootstrap_survival_curve(bare, [CDSQuote(MATURITY, 0.01)], 0.4)
    market = MarketSnapshot(valuation_date=D0, discount_curve=disc, curves={CP: cp, SELF: me})
    return HullWhite(a=0.05, sigma=0.012, market=market)


def _config():
    return XvaReportConfig(
        counterparty=CreditParty(CP, 0.4), self_party=CreditParty(SELF, 0.4),
        funding_spread=FUNDING, cost_of_capital=COC, risk_weight=RW, pfe_quantile=Q,
    )


def test_single_trade_report_reproduces_standalone_adjustments():
    swap, model = _swap(), _model()
    market = model.market
    pair = exposure_profiles(swap, model, NUM)
    pfe = pfe_profile(swap, model, NUM, Q)
    cva_s = cva(pair.epe, market, CP, 0.4)
    dva_s = dva(pair.ene, market, SELF, 0.4)
    fva_s = fva(pair, market, SELF, FUNDING)
    mva_s = mva(pfe, market, SELF, FUNDING)
    kva_s = kva(capital_profile(forward_ead_profile(swap, D0), RW), market, SELF, COC)

    r = xva_report([swap], model, NUM, _config())
    assert r.cva == pytest.approx(cva_s, rel=1e-9)
    assert r.dva == pytest.approx(dva_s, rel=1e-9)
    assert r.bcva == pytest.approx(cva_s - dva_s, rel=1e-9)
    assert r.fva == pytest.approx(fva_s, rel=1e-9)
    assert r.mva == pytest.approx(mva_s, rel=1e-9)
    assert r.kva == pytest.approx(kva_s, rel=1e-9)
    assert r.epe.ee == pytest.approx(pair.epe.ee, abs=1e-9)  # one pass reproduces the profile


def test_mirror_hedge_nets_the_exposure_to_zero():
    model = _model()
    single = xva_report([_swap(pay_fixed=True)], model, NUM, _config())
    hedged = xva_report([_swap(pay_fixed=True), _swap(pay_fixed=False)], model, NUM, _config())
    assert hedged.cva == pytest.approx(0.0, abs=1e-6)   # portfolio value nets to 0 per path
    assert hedged.cva < single.cva
    assert all(e == pytest.approx(0.0, abs=1e-6) for e in hedged.epe.ee)
