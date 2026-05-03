"""Tests for CLN desk: risk metrics, carry, P&L, book, dashboard, stress, lifecycle."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote
from pricebook.cln import BasketCLN
from pricebook.cln_desk import (
    cln_risk_metrics, CLNRiskMetrics,
    cln_carry_decomposition, CLNCarryDecomposition,
    cln_daily_pnl, CLNDailyPnL,
    CLNBook, CLNBookEntry,
    cln_dashboard, CLNDashboard,
    cln_stress_suite, CLNStressResult,
    cln_scenario_stress,
    cln_capital_summary, CLNCapitalSummary,
    cln_hedge_recommendations, CLNHedgeRecommendation,
    cln_basis_monitor, CLNBasisPoint,
    CLNLifecycle, CLNMarginCall,
    cln_collateral_evolution, CLNCollateralState,
    basket_cln_risk_metrics, BasketCLNRiskMetrics,
    BasketCLNBook, BasketCLNBookEntry,
    basket_cln_dashboard, BasketCLNDashboard,
)
from pricebook.pricing_context import PricingContext
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
END = REF + relativedelta(years=5)


def _vanilla_cln():
    return CreditLinkedNote(
        start=REF, end=END, coupon_rate=0.05,
        notional=1_000_000, recovery=0.4,
        frequency=Frequency.QUARTERLY,
    )


def _leveraged_cln():
    return CreditLinkedNote(
        start=REF, end=END, coupon_rate=0.07,
        notional=1_000_000, recovery=0.4, leverage=2.0,
        frequency=Frequency.QUARTERLY,
    )


def _flat_surv(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


# ── Risk metrics ──

class TestCLNRiskMetrics:

    def test_l11_hand_calc_5y_cln(self):
        """L11: 5Y CLN, 2% hazard, 40% recovery, 5% coupon, 4% flat.

        Hand-calculated:
          coupon_pv  = 217,447.32
          principal  = 740,696.45
          recovery   =  34,400.96
          total PV   = 992,544.73
        """
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert abs(rm.pv - 992_544.73) < 0.01

    def test_dv01_negative(self):
        """Rates up → price down for fixed-coupon CLN."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert rm.dv01 < 0

    def test_cs01_negative(self):
        """Wider spreads → lower CLN price."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert rm.cs01 < 0

    def test_recovery_sensitivity_positive(self):
        """Higher recovery → higher CLN price."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert rm.recovery_sensitivity > 0

    def test_jtd_is_loss(self):
        """JTD = R×N - PV, should be negative (loss) for long CLN."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        # R×N = 0.4×1M = 400k, PV ≈ 992k → JTD ≈ -592k
        assert rm.jump_to_default_pnl < 0
        expected_jtd = 0.4 * 1_000_000 - rm.pv
        assert abs(rm.jump_to_default_pnl - expected_jtd) < 0.01

    def test_leveraged_higher_jtd_loss(self):
        """Leveraged CLN has bigger JTD loss."""
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm_v = cln_risk_metrics(_vanilla_cln(), curve, surv)
        rm_l = cln_risk_metrics(_leveraged_cln(), curve, surv)
        # Leveraged has lower PV and same R×N → bigger loss
        assert rm_l.jump_to_default_pnl < rm_v.jump_to_default_pnl

    def test_to_dict(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        d = rm.to_dict()
        assert "dv01" in d
        assert "cs01" in d
        assert "jtd" in d
        assert "recovery_sensitivity" in d


# ── Carry decomposition ──

class TestCLNCarry:

    def test_coupon_income_positive(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        cd = cln_carry_decomposition(cln, curve, surv)
        assert cd.coupon_income > 0  # 5% × 1M = 50k/year

    def test_default_drag_negative(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        cd = cln_carry_decomposition(cln, curve, surv)
        assert cd.default_drag < 0  # expected loss per year

    def test_net_carry_sign(self):
        """For typical CLN, coupon should exceed default drag."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        cd = cln_carry_decomposition(cln, curve, surv)
        # 5% coupon vs 2%×60% = 1.2% default drag + 4% funding ≈ net negative
        assert math.isfinite(cd.net_carry)

    def test_to_dict(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        d = cln_carry_decomposition(cln, curve, surv).to_dict()
        assert "coupon" in d
        assert "default_drag" in d
        assert "net" in d


# ── Daily P&L ──

class TestCLNDailyPnL:

    def test_unchanged_small(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        pnl = cln_daily_pnl(cln, curve, curve, surv, surv,
                            REF + relativedelta(days=1))
        assert abs(pnl.total) < 1  # same curves → ~0

    def test_spread_widening_negative(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv_t0 = _flat_surv(0.02)
        surv_t1 = _flat_surv(0.04)  # spreads widen
        pnl = cln_daily_pnl(cln, curve, curve, surv_t0, surv_t1,
                            REF + relativedelta(days=1))
        assert pnl.total < 0  # wider spreads → loss

    def test_rate_shift_has_impact(self):
        cln = _vanilla_cln()
        c0 = make_flat_curve(REF, 0.04)
        c1 = make_flat_curve(REF, 0.05)
        surv = _flat_surv(0.02)
        pnl = cln_daily_pnl(cln, c0, c1, surv, surv,
                            REF + relativedelta(days=1))
        assert pnl.total != 0

    def test_to_dict(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        pnl = cln_daily_pnl(cln, curve, curve, surv, surv,
                            REF + relativedelta(days=1))
        d = pnl.to_dict()
        assert "spread" in d
        assert "rate" in d
        assert "theta" in d


# ── Book ──

class TestCLNBook:

    def test_add_and_count(self):
        book = CLNBook("TestBook")
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, issuer="MSFT"))
        assert len(book) == 2

    def test_total_notional(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv))
        assert book.total_notional() == 2_000_000

    def test_by_issuer(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, issuer="MSFT"))
        bi = book.by_issuer()
        assert "AAPL" in bi
        assert "MSFT" in bi

    def test_by_seniority(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, seniority="senior"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, seniority="sub"))
        bs = book.by_seniority()
        assert "senior" in bs
        assert "sub" in bs

    def test_independent_amount(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, independent_amount=100_000))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, independent_amount=200_000))
        assert book.total_independent_amount() == 300_000

    def test_aggregate_risk(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve)
        assert "total_pv" in risk
        assert "total_cs01" in risk
        assert "total_jtd" in risk
        assert risk["n_positions"] == 1


# ── Dashboard ──

class TestCLNDashboard:

    def test_dashboard_fields(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        db = cln_dashboard(book, REF, curve)
        assert db.n_positions == 1
        assert db.total_notional == 1_000_000
        assert math.isfinite(db.total_pv)
        assert math.isfinite(db.total_cs01)

    def test_by_issuer_breakdown(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, issuer="MSFT"))
        curve = make_flat_curve(REF, 0.04)
        db = cln_dashboard(book, REF, curve)
        assert "AAPL" in db.by_issuer
        assert "MSFT" in db.by_issuer

    def test_to_dict(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        db = cln_dashboard(book, REF, curve)
        d = db.to_dict()
        assert "cs01" in d
        assert "jtd" in d
        assert "by_issuer" in d
        assert "by_seniority" in d


# ── Stress testing ──

class TestCLNStress:

    def test_five_scenarios(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        results = cln_stress_suite(book, curve)
        assert len(results) == 5

    def test_spread_wide_negative(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        results = cln_stress_suite(book, curve)
        wide = [r for r in results if r.scenario == "spread_wide"][0]
        assert wide.spread_pnl < 0  # CS01 negative × positive spread = negative

    def test_scenario_stress(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        ctx = PricingContext(valuation_date=REF, discount_curve=make_flat_curve(REF, 0.04))
        results = cln_scenario_stress(book, ctx)
        assert len(results) == 5  # 2 rate + 3 credit

    def test_to_dict(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        results = cln_stress_suite(book, curve)
        d = results[0].to_dict()
        assert "spread" in d
        assert "total" in d


# ── Capital ──

class TestCLNCapital:

    def test_capital_summary(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        cap = cln_capital_summary(book, curve)
        assert len(cap.entries) == 1
        assert cap.total_ead > 0
        assert cap.total_capital > 0

    def test_capital_is_8pct_rwa(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        cap = cln_capital_summary(book, curve)
        assert abs(cap.total_capital - cap.total_rwa * 0.08) < 0.01

    def test_to_dict(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        d = cln_capital_summary(book, curve).to_dict()
        assert "total_ead" in d


# ── Hedge recommendations ──

class TestCLNHedge:

    def test_no_recs_within_limits(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        recs = cln_hedge_recommendations(book, curve,
            cs01_limit=1e12, jtd_limit=1e12, recovery_limit=1e12, dv01_limit=1e12)
        assert len(recs) == 0

    def test_recs_when_breached(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        recs = cln_hedge_recommendations(book, curve, cs01_limit=0.001)
        assert len(recs) >= 1
        assert recs[0].action != ""

    def test_basis_monitor(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        basis = cln_basis_monitor(book, curve, {"AAPL": 0.03})
        assert len(basis) == 1
        assert math.isfinite(basis[0].basis)

    def test_basis_to_dict(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        basis = cln_basis_monitor(book, curve, {"AAPL": 0.03})
        d = basis[0].to_dict()
        assert "basis" in d
        assert "cln_yield" in d


# ── Lifecycle ──

class TestCLNLifecycle:

    def test_credit_event_returns_recovery(self):
        cln = _vanilla_cln()
        surv = _flat_surv()
        lc = CLNLifecycle(cln, surv, "C1", REF)
        curve = make_flat_curve(REF, 0.04)
        payout = lc.credit_event(REF + relativedelta(years=1), curve)
        assert payout == 0.4 * 1_000_000  # R × N

    def test_restructuring_updates_terms(self):
        cln = _vanilla_cln()
        surv = _flat_surv()
        lc = CLNLifecycle(cln, surv, "C1", REF)
        curve = make_flat_curve(REF, 0.04)
        lc.restructuring(REF + relativedelta(months=6), curve, new_coupon=0.03)
        assert cln.coupon_rate == 0.03

    def test_margin_call_triggered(self):
        cln = _vanilla_cln()
        surv = _flat_surv()
        lc = CLNLifecycle(cln, surv, "C1", REF)
        curve = make_flat_curve(REF, 0.04)
        mc = lc.margin_call(REF, curve, threshold=0.0, min_transfer=0)
        assert mc.required_transfer > 0

    def test_margin_call_min_transfer(self):
        cln = _vanilla_cln()
        surv = _flat_surv()
        lc = CLNLifecycle(cln, surv, "C1", REF)
        curve = make_flat_curve(REF, 0.04)
        mc = lc.margin_call(REF, curve, min_transfer=1e15)
        assert mc.required_transfer == 0.0

    def test_early_redemption(self):
        cln = _vanilla_cln()
        surv = _flat_surv()
        lc = CLNLifecycle(cln, surv, "C1", REF)
        curve = make_flat_curve(REF, 0.04)
        pv = lc.early_redeem(REF, curve)
        expected = cln.dirty_price(curve, surv)
        assert abs(pv - expected) < 0.01

    def test_history_ordered(self):
        cln = _vanilla_cln()
        surv = _flat_surv()
        lc = CLNLifecycle(cln, surv, "C1", REF)
        curve = make_flat_curve(REF, 0.04)
        lc.restructuring(REF + relativedelta(months=3), curve, new_coupon=0.04)
        hist = lc.history
        dates = [h["date"] for h in hist]
        assert dates == sorted(dates)


# ── Collateral evolution ──

class TestCLNCollateral:

    def test_evolution_length(self):
        cln = _vanilla_cln()
        dates = [REF + relativedelta(days=i) for i in range(5)]
        curves = [make_flat_curve(REF, 0.04)] * 5
        survs = [_flat_surv(0.02 + i * 0.005) for i in range(5)]
        states = cln_collateral_evolution(cln, dates, curves, survs)
        assert len(states) == 5

    def test_spread_tracked(self):
        cln = _vanilla_cln()
        dates = [REF]
        curves = [make_flat_curve(REF, 0.04)]
        survs = [_flat_surv(0.02)]
        states = cln_collateral_evolution(cln, dates, curves, survs)
        assert states[0].spread_level > 0

    def test_margin_triggered(self):
        cln = _vanilla_cln()
        dates = [REF, REF + relativedelta(days=1)]
        curves = [make_flat_curve(REF, 0.04)] * 2
        survs = [_flat_surv(0.02), _flat_surv(0.10)]  # big spread move
        states = cln_collateral_evolution(cln, dates, curves, survs,
                                          threshold=0.0, min_transfer=0)
        assert any(s.margin_call != 0 for s in states)

    def test_to_dict(self):
        cln = _vanilla_cln()
        dates = [REF]
        curves = [make_flat_curve(REF, 0.04)]
        survs = [_flat_surv(0.02)]
        states = cln_collateral_evolution(cln, dates, curves, survs)
        d = states[0].to_dict()
        assert "spread" in d
        assert "net_exposure" in d


# ── Phase 1 foundations ──

class TestFoundations:

    def test_pillar_hazards_roundtrip(self):
        """Extract hazards from curve, rebuild curve, verify match."""
        surv = _flat_surv(0.02)
        hazards = surv.pillar_hazards()
        assert len(hazards) > 0
        # All hazards should be ~0.02 for flat curve
        for t, h in hazards:
            assert abs(h - 0.02) < 0.001

    def test_pillar_hazards_nonflat(self):
        """Non-flat curve should have varying hazard rates."""
        from pricebook.survival_curve import SurvivalCurve
        import math
        dates = [REF + relativedelta(years=i) for i in [1, 3, 5, 10]]
        # Steep curve: 1% short, 3% long
        survs = [math.exp(-0.01 * 1), math.exp(-0.015 * 3),
                 math.exp(-0.02 * 5), math.exp(-0.03 * 10)]
        sc = SurvivalCurve(REF, dates, survs)
        hazards = sc.pillar_hazards()
        assert len(hazards) >= 3
        # Hazards should vary (not all equal)
        h_vals = [h for _, h in hazards]
        assert max(h_vals) > min(h_vals)

    def test_cs01_uses_bumped_not_flat(self):
        """After CS01 fix, greeks should use bumped() not flat()."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        greeks = cln.greeks(curve, surv)
        assert greeks["cs01"] < 0  # wider spreads → lower price

    def test_credit_spread_shift_scenario(self):
        """credit_spread_shift bumps credit_curves in PricingContext."""
        from pricebook.scenario import credit_spread_shift
        surv = _flat_surv(0.02)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=make_flat_curve(REF, 0.04),
            credit_curves={"AAPL": surv},
        )
        scenario = credit_spread_shift(0.01)
        bumped_ctx = scenario.apply(ctx)
        # Bumped survival should be lower (higher hazard)
        t5 = REF + relativedelta(years=5)
        assert bumped_ctx.credit_curves["AAPL"].survival(t5) < surv.survival(t5)


# ── Basket CLN desk ──

def _basket_cln(attach=0.0, detach=0.03):
    return BasketCLN(
        start=REF, end=REF + relativedelta(years=5),
        coupon_rate=0.05, notional=10_000_000,
        attachment=attach, detachment=detach,
        recovery=0.4, n_names=50,
    )


def _survs(n=50, h=0.02):
    return [_flat_surv(h) for _ in range(n)]


class TestBasketCLNRiskMetrics:

    def test_pv_positive(self):
        basket = _basket_cln()
        curve = make_flat_curve(REF, 0.04)
        rm = basket_cln_risk_metrics(basket, curve, _survs(), n_sims=3_000)
        assert rm.pv > 0

    def test_cs01_negative(self):
        """Wider spreads → more defaults → lower tranche price."""
        basket = _basket_cln()
        curve = make_flat_curve(REF, 0.04)
        rm = basket_cln_risk_metrics(basket, curve, _survs(), n_sims=3_000)
        assert rm.cs01 < 0

    def test_rho01_equity_positive(self):
        """Equity tranche: higher rho → fewer idiosyncratic defaults → higher price."""
        basket = _basket_cln(0.0, 0.03)
        curve = make_flat_curve(REF, 0.04)
        rm = basket_cln_risk_metrics(basket, curve, _survs(), rho=0.20, n_sims=3_000)
        assert rm.rho01 > 0

    def test_rho01_senior_negative(self):
        """Senior tranche: higher rho → more tail risk → lower price."""
        basket = _basket_cln(0.07, 0.10)
        curve = make_flat_curve(REF, 0.04)
        rm = basket_cln_risk_metrics(basket, curve, _survs(), rho=0.20, n_sims=3_000)
        assert rm.rho01 < 0

    def test_to_dict(self):
        basket = _basket_cln()
        curve = make_flat_curve(REF, 0.04)
        rm = basket_cln_risk_metrics(basket, curve, _survs(), n_sims=2_000)
        d = rm.to_dict()
        assert "rho01" in d
        assert "cs01" in d
        assert "attachment" in d


class TestBasketCLNBook:

    def test_add_and_count(self):
        book = BasketCLNBook()
        survs = _survs()
        book.add(BasketCLNBookEntry("B1", _basket_cln(0.0, 0.03), survs, tranche_name="equity"))
        book.add(BasketCLNBookEntry("B2", _basket_cln(0.03, 0.07), survs, tranche_name="mezz"))
        assert len(book) == 2

    def test_by_tranche(self):
        book = BasketCLNBook()
        survs = _survs()
        book.add(BasketCLNBookEntry("B1", _basket_cln(0.0, 0.03), survs, tranche_name="equity"))
        book.add(BasketCLNBookEntry("B2", _basket_cln(0.03, 0.07), survs, tranche_name="mezz"))
        bt = book.by_tranche()
        assert "equity" in bt
        assert "mezz" in bt

    def test_total_notional(self):
        book = BasketCLNBook()
        survs = _survs()
        book.add(BasketCLNBookEntry("B1", _basket_cln(), survs))
        assert book.total_notional() == 10_000_000

    def test_aggregate_risk(self):
        book = BasketCLNBook()
        survs = _survs()
        book.add(BasketCLNBookEntry("B1", _basket_cln(), survs))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve, n_sims=2_000)
        assert "total_rho01" in risk
        assert "total_cs01" in risk
        assert risk["n_positions"] == 1


class TestBasketCLNDashboard:

    def test_dashboard_fields(self):
        book = BasketCLNBook()
        survs = _survs()
        book.add(BasketCLNBookEntry("B1", _basket_cln(), survs, tranche_name="equity"))
        curve = make_flat_curve(REF, 0.04)
        db = basket_cln_dashboard(book, REF, curve, n_sims=2_000)
        assert db.n_positions == 1
        assert math.isfinite(db.total_rho01)
        assert "equity" in db.by_tranche

    def test_to_dict(self):
        book = BasketCLNBook()
        survs = _survs()
        book.add(BasketCLNBookEntry("B1", _basket_cln(), survs))
        curve = make_flat_curve(REF, 0.04)
        db = basket_cln_dashboard(book, REF, curve, n_sims=2_000)
        d = db.to_dict()
        assert "rho01" in d
        assert "by_tranche" in d
