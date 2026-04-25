"""XI10: End-to-End Multi-Currency Portfolio (capstone).

Multi-ccy curves → portfolio (swap + bond + CDS + FX fwd) → PV finite →
DV01 finite → stress test → VaR > 0. Portfolio PV = sum of trade PVs.

Bug hotspots:
- Most instruments lack pv_ctx — Trade wrapper only works for Swaption
- Multi-currency curve routing
- Stress test bumps all curves vs one

Finding: Trade.pv() requires pv_ctx but InterestRateSwap, FixedRateBond,
CDS, FXForward only have pv(curve). Tests use direct instrument calls.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap
from pricebook.bond import FixedRateBond
from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.currency import CurrencyPair
from pricebook.discount_curve import DiscountCurve
from pricebook.fx_forward import FXForward
from pricebook.schedule import Frequency
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.var import historical_var


# ---- Helpers ----

REF = date(2026, 4, 25)


def _usd_curve(ref: date) -> DiscountCurve:
    deposits = [(ref + timedelta(days=91), 0.045), (ref + timedelta(days=182), 0.044)]
    swaps = [
        (ref + timedelta(days=365), 0.043),
        (ref + timedelta(days=730), 0.042),
        (ref + timedelta(days=1825), 0.040),
        (ref + timedelta(days=3650), 0.038),
    ]
    return bootstrap(ref, deposits, swaps)


def _eur_curve(ref: date) -> DiscountCurve:
    deposits = [(ref + timedelta(days=91), 0.030), (ref + timedelta(days=182), 0.029)]
    swaps = [
        (ref + timedelta(days=365), 0.028),
        (ref + timedelta(days=730), 0.027),
        (ref + timedelta(days=1825), 0.025),
    ]
    return bootstrap(ref, deposits, swaps)


def _instruments(ref: date):
    """Build a multi-asset instrument set."""
    start = ref + timedelta(days=2)
    usd = _usd_curve(ref)
    eur = _eur_curve(ref)

    swap = InterestRateSwap(start, start + timedelta(days=3650),
                            fixed_rate=0.04, direction=SwapDirection.PAYER)
    bond = FixedRateBond(ref, ref + timedelta(days=1825), coupon_rate=0.035)
    surv = bootstrap_credit_curve(ref,
        [(ref + timedelta(days=1825), 0.0100)], usd, recovery=0.4)
    cds_inst = CDS(ref, ref + timedelta(days=1825), spread=0.0100)
    pair = CurrencyPair("EUR", "USD")
    fwd_rate = FXForward.forward_rate(1.085, ref + timedelta(days=365), eur, usd)
    fx_fwd = FXForward(pair, ref + timedelta(days=365), strike=fwd_rate)

    return swap, bond, cds_inst, fx_fwd, usd, eur, surv


def _portfolio_pv(ref: date, usd=None, eur=None, surv=None):
    """Compute total portfolio PV from all instruments."""
    swap, bond, cds_inst, fx_fwd, usd_d, eur_d, surv_d = _instruments(ref)
    usd = usd or usd_d
    eur = eur or eur_d
    surv = surv or surv_d

    pv_swap = swap.pv(usd)
    pv_bond = bond.dirty_price(usd) * 10000  # 10k face
    pv_cds = cds_inst.pv(usd, surv)
    pv_fx = fx_fwd.pv(1.085, eur, usd)

    return pv_swap + pv_bond + pv_cds + pv_fx, (pv_swap, pv_bond, pv_cds, pv_fx)


# ---- R1: Portfolio PV ----

class TestXI10R1PortfolioPV:
    """All instruments price to finite values, portfolio sums correctly."""

    def test_all_pvs_finite(self):
        total, (pv_swap, pv_bond, pv_cds, pv_fx) = _portfolio_pv(REF)
        assert math.isfinite(pv_swap), f"Swap PV non-finite: {pv_swap}"
        assert math.isfinite(pv_bond), f"Bond PV non-finite: {pv_bond}"
        assert math.isfinite(pv_cds), f"CDS PV non-finite: {pv_cds}"
        assert math.isfinite(pv_fx), f"FX fwd PV non-finite: {pv_fx}"

    def test_portfolio_pv_is_sum(self):
        total, components = _portfolio_pv(REF)
        assert total == pytest.approx(sum(components), rel=1e-10)

    def test_fx_forward_at_market_near_zero(self):
        """FX forward struck at market forward should have PV ≈ 0."""
        _, _, _, fx_fwd, usd, eur, _ = _instruments(REF)
        pv = fx_fwd.pv(1.085, eur, usd)
        assert abs(pv) < 100  # struck at market forward


# ---- R2: DV01 ----

class TestXI10R2DV01:
    """Portfolio DV01 via bump-and-reprice."""

    def test_dv01_finite(self):
        pv_base, _ = _portfolio_pv(REF)
        usd_up = _usd_curve(REF).bumped(0.0001)
        eur_up = _eur_curve(REF).bumped(0.0001)
        surv = bootstrap_credit_curve(REF,
            [(REF + timedelta(days=1825), 0.0100)], usd_up, recovery=0.4)
        pv_up, _ = _portfolio_pv(REF, usd=usd_up, eur=eur_up, surv=surv)
        dv01 = pv_up - pv_base
        assert math.isfinite(dv01)

    def test_dv01_nonzero(self):
        pv_base, _ = _portfolio_pv(REF)
        usd_up = _usd_curve(REF).bumped(0.0001)
        eur_up = _eur_curve(REF).bumped(0.0001)
        surv = bootstrap_credit_curve(REF,
            [(REF + timedelta(days=1825), 0.0100)], usd_up, recovery=0.4)
        pv_up, _ = _portfolio_pv(REF, usd=usd_up, eur=eur_up, surv=surv)
        assert abs(pv_up - pv_base) > 0


# ---- R3: Stress test ----

class TestXI10R3Stress:
    """Stress scenarios on the portfolio."""

    def test_parallel_up_50bp(self):
        pv_base, _ = _portfolio_pv(REF)
        usd_up = _usd_curve(REF).bumped(0.0050)
        eur_up = _eur_curve(REF).bumped(0.0050)
        surv = bootstrap_credit_curve(REF,
            [(REF + timedelta(days=1825), 0.0100)], usd_up, recovery=0.4)
        pv_stressed, _ = _portfolio_pv(REF, usd=usd_up, eur=eur_up, surv=surv)
        pnl = pv_stressed - pv_base
        assert math.isfinite(pnl)
        assert pnl != 0

    def test_up_down_opposite_sign(self):
        pv_base, _ = _portfolio_pv(REF)

        usd_up = _usd_curve(REF).bumped(0.0050)
        eur_up = _eur_curve(REF).bumped(0.0050)
        surv_up = bootstrap_credit_curve(REF,
            [(REF + timedelta(days=1825), 0.0100)], usd_up, recovery=0.4)
        pnl_up = _portfolio_pv(REF, usd=usd_up, eur=eur_up, surv=surv_up)[0] - pv_base

        usd_dn = _usd_curve(REF).bumped(-0.0050)
        eur_dn = _eur_curve(REF).bumped(-0.0050)
        surv_dn = bootstrap_credit_curve(REF,
            [(REF + timedelta(days=1825), 0.0100)], usd_dn, recovery=0.4)
        pnl_dn = _portfolio_pv(REF, usd=usd_dn, eur=eur_dn, surv=surv_dn)[0] - pv_base

        assert pnl_up * pnl_dn < 0 or (abs(pnl_up) < 1 and abs(pnl_dn) < 1)


# ---- R4: VaR ----

class TestXI10R4VaR:
    """Historical VaR from simulated P&L."""

    def test_var_positive(self):
        usd = _usd_curve(REF)
        eur = _eur_curve(REF)
        surv = bootstrap_credit_curve(REF,
            [(REF + timedelta(days=1825), 0.0100)], usd, recovery=0.4)
        pv_base, _ = _portfolio_pv(REF, usd=usd, eur=eur, surv=surv)

        rng = np.random.default_rng(42)
        daily_pnl = []
        for _ in range(250):
            shift = rng.normal(0, 0.0005)
            usd_b = usd.bumped(shift)
            eur_b = eur.bumped(shift)
            surv_b = bootstrap_credit_curve(REF,
                [(REF + timedelta(days=1825), 0.0100)], usd_b, recovery=0.4)
            pv_day, _ = _portfolio_pv(REF, usd=usd_b, eur=eur_b, surv=surv_b)
            daily_pnl.append(pv_day - pv_base)

        var_95 = historical_var(daily_pnl, confidence=0.95)
        assert var_95 > 0

    def test_var_99_geq_95(self):
        rng = np.random.default_rng(42)
        pnl = list(rng.normal(0, 10000, 500))
        var_95 = historical_var(pnl, confidence=0.95)
        var_99 = historical_var(pnl, confidence=0.99)
        assert var_99 >= var_95
