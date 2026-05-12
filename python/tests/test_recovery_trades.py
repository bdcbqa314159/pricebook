"""Tests for recovery trades: implied recovery, downturn LGD, recovery swaps, senior-sub basis."""

from __future__ import annotations

from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.cds_market import build_cds_curve
from pricebook.recovery_trades import (
    market_implied_recovery,
    recovery_by_spread_regime,
    downturn_lgd,
    portfolio_recovery_stress,
    recovery_swap_pv,
    recovery_lock_greeks,
    senior_sub_basis,
    cs01_neutral_ratio,
)

REF = date(2024, 7, 15)

def _ois():
    deposits = [(REF + relativedelta(months=6), 0.043)]
    swaps = [(REF + relativedelta(years=y), r) for y, r in [(1, 0.041), (5, 0.038), (10, 0.036)]]
    return bootstrap(REF, deposits, swaps)

CDS_SPREADS = {1: 0.0050, 3: 0.0080, 5: 0.0100, 10: 0.0120}


# ── Market-implied recovery ──

class TestMarketImpliedRecovery:

    def test_cds_only(self):
        ois = _ois()
        result = market_implied_recovery(0.0100, ois, REF)
        assert 0 < result.recovery_cds < 1
        assert result.method == "cds_only"

    def test_cds_bond(self):
        ois = _ois()
        result = market_implied_recovery(0.0100, ois, REF, bond_asw_spread=0.0130)
        assert result.recovery_bond is not None
        assert result.method == "cds_bond_average"

    def test_zero_spread_returns_default(self):
        ois = _ois()
        result = market_implied_recovery(0.0, ois, REF)
        assert result.method == "zero_spread"
        assert result.recovery_cds == 0.40

    def test_distressed_lower_recovery(self):
        ois = _ois()
        r_ig = market_implied_recovery(0.0050, ois, REF, initial_hazard_guess=0.015)
        r_hy = market_implied_recovery(0.0500, ois, REF, initial_hazard_guess=0.10)
        # Higher spread → same hazard → lower implied R
        assert r_hy.recovery_cds < r_ig.recovery_cds


class TestRecoveryBySpreadRegime:

    def test_tight_spread_near_base(self):
        R = recovery_by_spread_regime(50, "senior_unsecured")
        assert R == pytest.approx(0.45, abs=0.02)

    def test_distressed_lower(self):
        R_ig = recovery_by_spread_regime(80)
        R_hy = recovery_by_spread_regime(600)
        assert R_hy < R_ig

    def test_floor_respected(self):
        R = recovery_by_spread_regime(2000, "sub")
        assert R >= 0.05

    def test_seniority_matters(self):
        R_sec = recovery_by_spread_regime(200, "senior_secured")
        R_sub = recovery_by_spread_regime(200, "sub")
        assert R_sec > R_sub


# ── Downturn LGD ──

class TestDownturnLGD:

    def test_downturn_exceeds_base(self):
        result = downturn_lgd(base_lgd=0.45)
        assert result.downturn_lgd >= result.base_lgd

    def test_higher_stress_higher_lgd(self):
        r95 = downturn_lgd(base_lgd=0.45, stress_quantile=0.95)
        r99 = downturn_lgd(base_lgd=0.45, stress_quantile=0.99)
        assert r99.downturn_lgd >= r95.downturn_lgd

    def test_secured_less_stressed(self):
        r_sec = downturn_lgd(base_lgd=0.35, seniority="senior_secured")
        r_unsec = downturn_lgd(base_lgd=0.45, seniority="senior_unsecured")
        # Multiplier should be lower for secured
        assert r_sec.downturn_multiplier < r_unsec.downturn_multiplier

    def test_cap_at_95(self):
        result = downturn_lgd(base_lgd=0.90, stress_quantile=0.999)
        assert result.downturn_lgd <= 0.95


# ── Portfolio recovery stress ──

class TestPortfolioRecoveryStress:

    def test_recovery_shock_negative_pv(self):
        from pricebook.cln import CreditLinkedNote
        from pricebook.schedule import Frequency
        ois = _ois()
        cln = CreditLinkedNote(REF, REF + relativedelta(years=5),
                                coupon_rate=0.05, notional=10e6, recovery=0.40,
                                frequency=Frequency.QUARTERLY)
        result = portfolio_recovery_stress([cln], ois, CDS_SPREADS, REF,
                                            recovery_shock=-0.10)
        assert result.pv_change != 0
        assert result.n_positions == 1

    def test_combined_shock(self):
        from pricebook.cln import CreditLinkedNote
        from pricebook.schedule import Frequency
        ois = _ois()
        cln = CreditLinkedNote(REF, REF + relativedelta(years=5),
                                coupon_rate=0.05, notional=10e6, recovery=0.40,
                                frequency=Frequency.QUARTERLY)
        r_only = portfolio_recovery_stress([cln], ois, CDS_SPREADS, REF,
                                            recovery_shock=-0.10)
        r_both = portfolio_recovery_stress([cln], ois, CDS_SPREADS, REF,
                                            recovery_shock=-0.10, spread_shock_bps=100)
        # Combined shock should be worse
        assert abs(r_both.pv_change) > abs(r_only.pv_change) * 0.5


# ── Recovery swaps ──

class TestRecoverySwap:

    def test_pv_positive_when_expected_above_fixed(self):
        ois = _ois()
        surv = build_cds_curve(REF, CDS_SPREADS, ois, recovery=0.40)
        result = recovery_swap_pv(10e6, REF + relativedelta(years=5), REF,
                                   ois, surv, fixed_recovery=0.35,
                                   expected_recovery=0.40)
        assert result.pv > 0  # expect R=40%, fixed pays R=35%, net positive

    def test_pv_zero_at_par(self):
        ois = _ois()
        surv = build_cds_curve(REF, CDS_SPREADS, ois, recovery=0.40)
        result = recovery_swap_pv(10e6, REF + relativedelta(years=5), REF,
                                   ois, surv, fixed_recovery=0.40,
                                   expected_recovery=0.40)
        assert result.pv == pytest.approx(0, abs=1)

    def test_par_recovery_equals_expected(self):
        ois = _ois()
        surv = build_cds_curve(REF, CDS_SPREADS, ois, recovery=0.40)
        result = recovery_swap_pv(10e6, REF + relativedelta(years=5), REF,
                                   ois, surv, fixed_recovery=0.30,
                                   expected_recovery=0.40)
        assert result.par_fixed_recovery == pytest.approx(0.40)


class TestRecoveryLockGreeks:

    def test_delta_R_positive(self):
        ois = _ois()
        result = recovery_lock_greeks(10e6, REF + relativedelta(years=5), REF,
                                       ois, CDS_SPREADS, lock_strike=0.35)
        # Higher R → more payout on default → positive delta
        assert result.delta_R > 0

    def test_pv_positive_when_R_above_strike(self):
        ois = _ois()
        result = recovery_lock_greeks(10e6, REF + relativedelta(years=5), REF,
                                       ois, CDS_SPREADS, lock_strike=0.35,
                                       base_recovery=0.40)
        assert result.pv > 0


# ── Senior-sub basis ──

class TestSeniorSubBasis:

    def test_basis_negative(self):
        ois = _ois()
        result = senior_sub_basis(0.0080, 0.0200, ois, REF)
        assert result.basis_bps < 0  # senior tighter than sub

    def test_positive_carry(self):
        ois = _ois()
        result = senior_sub_basis(0.0080, 0.0200, ois, REF)
        # Short sub (receive 200bp) > long senior (pay 80bp) → positive carry
        assert result.carry_30d > 0

    def test_implied_senior_recovery(self):
        ois = _ois()
        result = senior_sub_basis(0.0080, 0.0200, ois, REF)
        assert 0 < result.implied_senior_recovery < 1

    def test_cs01_ratio(self):
        ratio = cs01_neutral_ratio(0.45, 0.25)
        # (1 - 0.45) / (1 - 0.25) = 0.55 / 0.75 ≈ 0.733
        assert ratio == pytest.approx(0.733, abs=0.01)

    def test_notional_ratio_reasonable(self):
        ois = _ois()
        result = senior_sub_basis(0.0080, 0.0200, ois, REF)
        assert 0.3 < result.notional_ratio < 3.0
