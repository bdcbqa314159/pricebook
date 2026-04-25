"""XI10: End-to-End Multi-Currency Portfolio (capstone).

Multi-ccy curves → portfolio (swap + bond + CDS + FX fwd) → PV finite →
DV01 finite → stress test → VaR > 0. Portfolio PV = sum of trade PVs.

Now uses Trade/Portfolio wrapper with pv_ctx on all instruments.
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
from pricebook.pricing_context import PricingContext
from pricebook.schedule import Frequency
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.trade import Trade, Portfolio
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


def _ctx(ref: date, usd=None, eur=None) -> PricingContext:
    usd = usd or _usd_curve(ref)
    eur = eur or _eur_curve(ref)
    surv = bootstrap_credit_curve(ref,
        [(ref + timedelta(days=1825), 0.0100)], usd, recovery=0.4)
    return PricingContext(
        valuation_date=ref,
        discount_curve=usd,
        discount_curves={"USD": usd, "EUR": eur},
        credit_curves={"default": surv},
        fx_spots={("EUR", "USD"): 1.0850},
    )


def _portfolio(ref: date) -> Portfolio:
    start = ref + timedelta(days=2)
    usd = _usd_curve(ref)
    eur = _eur_curve(ref)

    swap = InterestRateSwap(start, start + timedelta(days=3650),
                            fixed_rate=0.04, direction=SwapDirection.PAYER)
    bond = FixedRateBond(ref, ref + timedelta(days=1825), coupon_rate=0.035)
    cds_inst = CDS(ref, ref + timedelta(days=1825), spread=0.0100)
    pair = CurrencyPair("EUR", "USD")
    fwd_rate = FXForward.forward_rate(1.085, ref + timedelta(days=365), eur, usd)
    fx_fwd = FXForward(pair, ref + timedelta(days=365), strike=fwd_rate)

    port = Portfolio(name="XI10_test")
    port.add(Trade(swap, trade_id="USD_IRS_10Y"))
    port.add(Trade(bond, trade_id="USD_BOND_5Y"))
    port.add(Trade(cds_inst, trade_id="CDS_5Y"))
    port.add(Trade(fx_fwd, trade_id="EURUSD_FWD_1Y"))
    return port


# ---- R1: Portfolio PV via Trade wrapper ----

class TestXI10R1PortfolioPV:
    """Trade/Portfolio framework with pv_ctx on all instruments."""

    def test_portfolio_pv_finite(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        pv = port.pv(ctx)
        assert math.isfinite(pv)

    def test_portfolio_pv_equals_sum(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        total_pv = port.pv(ctx)
        by_trade = port.pv_by_trade(ctx)
        sum_pv = sum(pv for _, pv in by_trade)
        assert total_pv == pytest.approx(sum_pv, rel=1e-10)

    def test_all_trades_have_finite_pv(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        by_trade = port.pv_by_trade(ctx)
        for trade_id, pv in by_trade:
            assert math.isfinite(pv), f"Trade {trade_id} has non-finite PV: {pv}"

    def test_portfolio_length(self):
        port = _portfolio(REF)
        assert len(port) == 4


# ---- R2: DV01 ----

class TestXI10R2DV01:
    """Portfolio DV01 via bump-and-reprice through PricingContext."""

    def test_portfolio_dv01_finite(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        pv_base = port.pv(ctx)

        usd_up = ctx.discount_curves["USD"].bumped(0.0001)
        eur_up = ctx.discount_curves["EUR"].bumped(0.0001)
        ctx_up = _ctx(REF, usd=usd_up, eur=eur_up)
        pv_up = port.pv(ctx_up)
        dv01 = pv_up - pv_base
        assert math.isfinite(dv01)

    def test_portfolio_dv01_nonzero(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        pv_base = port.pv(ctx)

        usd_up = ctx.discount_curves["USD"].bumped(0.0001)
        eur_up = ctx.discount_curves["EUR"].bumped(0.0001)
        ctx_up = _ctx(REF, usd=usd_up, eur=eur_up)
        pv_up = port.pv(ctx_up)
        assert abs(pv_up - pv_base) > 0


# ---- R3: Stress test ----

class TestXI10R3Stress:
    """Stress scenarios on the portfolio."""

    def test_parallel_up_50bp(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        pv_base = port.pv(ctx)

        usd_up = ctx.discount_curves["USD"].bumped(0.0050)
        eur_up = ctx.discount_curves["EUR"].bumped(0.0050)
        ctx_stressed = _ctx(REF, usd=usd_up, eur=eur_up)
        pnl = port.pv(ctx_stressed) - pv_base
        assert math.isfinite(pnl)
        assert pnl != 0

    def test_up_down_opposite_sign(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        pv_base = port.pv(ctx)

        ctx_up = _ctx(REF,
            usd=ctx.discount_curves["USD"].bumped(0.0050),
            eur=ctx.discount_curves["EUR"].bumped(0.0050))
        ctx_dn = _ctx(REF,
            usd=ctx.discount_curves["USD"].bumped(-0.0050),
            eur=ctx.discount_curves["EUR"].bumped(-0.0050))

        pnl_up = port.pv(ctx_up) - pv_base
        pnl_dn = port.pv(ctx_dn) - pv_base
        assert pnl_up * pnl_dn < 0 or (abs(pnl_up) < 1 and abs(pnl_dn) < 1)


# ---- R4: VaR ----

class TestXI10R4VaR:
    """Historical VaR from simulated P&L."""

    def test_var_positive(self):
        ctx = _ctx(REF)
        port = _portfolio(REF)
        pv_base = port.pv(ctx)

        rng = np.random.default_rng(42)
        daily_pnl = []
        for _ in range(250):
            shift = rng.normal(0, 0.0005)
            ctx_day = _ctx(REF,
                usd=ctx.discount_curves["USD"].bumped(shift),
                eur=ctx.discount_curves["EUR"].bumped(shift))
            daily_pnl.append(port.pv(ctx_day) - pv_base)

        var_95 = historical_var(daily_pnl, confidence=0.95)
        assert var_95 > 0

    def test_var_99_geq_95(self):
        rng = np.random.default_rng(42)
        pnl = list(rng.normal(0, 10000, 500))
        var_95 = historical_var(pnl, confidence=0.95)
        var_99 = historical_var(pnl, confidence=0.99)
        assert var_99 >= var_95
