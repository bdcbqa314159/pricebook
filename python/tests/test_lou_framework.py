"""Tests for Lou Papers Framework (2015-2017): liability-side pricing, IRS XVA,
repo gap risk, non-cash collateral discounting."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from pricebook.xva import (
    total_xva_decomposition,
    irs_xva,
    repo_gap_risk,
    implied_repo_rate_from_gap,
    TotalXVAResult,
)
from pricebook.csa import (
    non_cash_collateral_discount_rate,
    NonCashCollateralAsset,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2026, 4, 27)


def _disc():
    return make_flat_curve(REF, 0.04)


def _surv(hazard=0.02):
    return make_flat_survival(REF, hazard)


# ---- Phase 5a: Total XVA decomposition ----

class TestTotalXVADecomposition:

    def test_basic_decomposition(self):
        """All components should be finite."""
        disc = _disc()
        cpty = _surv(0.02)
        own = _surv(0.01)
        time_grid = [0.25, 0.50, 0.75, 1.0]
        epe = np.array([50_000, 45_000, 40_000, 35_000], dtype=float)
        ene = np.array([10_000, 12_000, 15_000, 18_000], dtype=float)

        result = total_xva_decomposition(
            epe=epe, ene=ene, time_grid=time_grid,
            discount_curve=disc, cpty_survival=cpty, own_survival=own,
            s_b=0.01, s_c=0.02, funding_spread=0.005,
        )
        assert math.isfinite(result.total)
        assert result.cva > 0  # positive EPE → positive CVA

    def test_total_equals_sum(self):
        """total = CVA - DVA + CFA - DFA + ColVA + FVA + MVA + KVA."""
        disc = _disc()
        cpty = _surv(0.02)
        own = _surv(0.01)
        time_grid = [0.5, 1.0]
        epe = np.array([100_000, 80_000], dtype=float)
        ene = np.array([20_000, 25_000], dtype=float)

        result = total_xva_decomposition(
            epe=epe, ene=ene, time_grid=time_grid,
            discount_curve=disc, cpty_survival=cpty, own_survival=own,
        )
        manual = (result.cva - result.dva + result.cfa - result.dfa
                  + result.colva + result.fva_val + result.mva_val + result.kva_val)
        assert result.total == pytest.approx(manual)

    def test_with_im_and_capital(self):
        """MVA and KVA should be positive when profiles are provided."""
        disc = _disc()
        cpty = _surv(0.02)
        own = _surv(0.01)
        time_grid = [0.5, 1.0, 1.5, 2.0]
        epe = np.array([50_000, 45_000, 40_000, 35_000], dtype=float)
        ene = np.array([10_000, 12_000, 15_000, 18_000], dtype=float)
        im = np.array([200_000, 180_000, 160_000, 140_000], dtype=float)
        capital = np.array([80_000, 70_000, 60_000, 50_000], dtype=float)

        result = total_xva_decomposition(
            epe=epe, ene=ene, time_grid=time_grid,
            discount_curve=disc, cpty_survival=cpty, own_survival=own,
            im_profile=im, capital_profile=capital,
            funding_spread=0.005, hurdle_rate=0.10,
        )
        assert result.mva_val > 0
        assert result.kva_val > 0

    def test_to_dict(self):
        disc = _disc()
        cpty = _surv(0.02)
        own = _surv(0.01)
        time_grid = [0.5, 1.0]
        epe = np.array([50_000, 40_000], dtype=float)
        ene = np.array([10_000, 15_000], dtype=float)

        result = total_xva_decomposition(
            epe=epe, ene=ene, time_grid=time_grid,
            discount_curve=disc, cpty_survival=cpty, own_survival=own,
        )
        d = result.to_dict()
        assert d["type"] == "total_xva_result"
        assert "total" in d["params"]
        assert "bilateral_cva" in d["params"]
        assert "total_funding" in d["params"]

    def test_zero_exposure_zero_xva(self):
        disc = _disc()
        cpty = _surv(0.02)
        own = _surv(0.01)
        time_grid = [0.5, 1.0]
        epe = np.zeros(2)
        ene = np.zeros(2)

        result = total_xva_decomposition(
            epe=epe, ene=ene, time_grid=time_grid,
            discount_curve=disc, cpty_survival=cpty, own_survival=own,
        )
        assert result.total == pytest.approx(0.0, abs=1e-10)


# ---- Phase 5b: IRS XVA ----

class TestIRSXVA:

    def test_irs_xva_finite(self):
        """IRS XVA should produce finite results."""
        disc = _disc()
        cpty = _surv(0.02)
        own = _surv(0.01)

        result = irs_xva(
            swap_pv=500_000, swap_dv01=5_000,
            time_to_maturity=5.0,
            discount_curve=disc,
            cpty_survival=cpty, own_survival=own,
        )
        assert math.isfinite(result.total)
        assert result.cva > 0  # positive PV → positive CVA

    def test_atm_swap_smaller_cva(self):
        """ATM swap (PV≈0) should have smaller CVA than ITM swap."""
        disc = _disc()
        cpty = _surv(0.02)
        own = _surv(0.01)

        r_itm = irs_xva(
            swap_pv=1_000_000, swap_dv01=5_000,
            time_to_maturity=5.0,
            discount_curve=disc, cpty_survival=cpty, own_survival=own,
        )
        r_atm = irs_xva(
            swap_pv=0.0, swap_dv01=5_000,
            time_to_maturity=5.0,
            discount_curve=disc, cpty_survival=cpty, own_survival=own,
        )
        assert r_itm.cva > r_atm.cva

    def test_higher_hazard_higher_cva(self):
        """Higher counterparty hazard → higher CVA."""
        disc = _disc()
        own = _surv(0.01)

        r_safe = irs_xva(
            swap_pv=500_000, swap_dv01=5_000,
            time_to_maturity=5.0,
            discount_curve=disc,
            cpty_survival=_surv(0.01), own_survival=own,
        )
        r_risky = irs_xva(
            swap_pv=500_000, swap_dv01=5_000,
            time_to_maturity=5.0,
            discount_curve=disc,
            cpty_survival=_surv(0.05), own_survival=own,
        )
        assert r_risky.cva > r_safe.cva


# ---- Phase 5c: Repo gap risk ----

class TestRepoGapRisk:

    def test_full_coverage_no_gap(self):
        """Full collateral coverage → zero gap cost."""
        gap = repo_gap_risk(
            position_value=10_000_000, repo_rate=0.03,
            funding_rate=0.05, collateral_coverage=1.0,
        )
        assert gap == pytest.approx(0.0)

    def test_zero_coverage_full_gap(self):
        """Zero coverage → gap = position × funding_premium."""
        gap = repo_gap_risk(
            position_value=10_000_000, repo_rate=0.03,
            funding_rate=0.05, collateral_coverage=0.0,
        )
        expected = 10_000_000 * 0.02 * 1.0  # 200,000
        assert gap == pytest.approx(expected)

    def test_partial_coverage(self):
        """50% coverage → half the gap."""
        gap_full = repo_gap_risk(
            position_value=10_000_000, repo_rate=0.03,
            funding_rate=0.05, collateral_coverage=0.0,
        )
        gap_half = repo_gap_risk(
            position_value=10_000_000, repo_rate=0.03,
            funding_rate=0.05, collateral_coverage=0.5,
        )
        assert gap_half == pytest.approx(gap_full * 0.5)

    def test_implied_repo_rate(self):
        """Implied rate = repo + premium × (1-coverage)."""
        implied = implied_repo_rate_from_gap(
            repo_rate=0.03, funding_rate=0.05,
            collateral_coverage=0.8,
        )
        expected = 0.03 + 0.02 * 0.2  # 3.4%
        assert implied == pytest.approx(expected)

    def test_implied_at_full_coverage(self):
        """Full coverage → implied = repo rate."""
        implied = implied_repo_rate_from_gap(
            repo_rate=0.03, funding_rate=0.05,
            collateral_coverage=1.0,
        )
        assert implied == pytest.approx(0.03)

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError, match="collateral_coverage"):
            repo_gap_risk(10_000_000, 0.03, 0.05, collateral_coverage=1.5)

    def test_invalid_coverage_implied_raises(self):
        with pytest.raises(ValueError, match="collateral_coverage"):
            implied_repo_rate_from_gap(0.03, 0.05, collateral_coverage=-0.1)


# ---- Phase 5d: Non-cash collateral discounting ----

class TestNonCashCollateral:

    def test_empty_pool_returns_cash(self):
        result = non_cash_collateral_discount_rate([], funding_rate=0.05, cash_rate=0.04)
        assert result.effective_rate == pytest.approx(0.04)
        assert result.optimal_collateral == "cash"

    def test_cheapest_to_deliver(self):
        """Poster delivers the cheapest asset."""
        pool = [
            NonCashCollateralAsset("UST_10Y", yield_rate=0.035, haircut=0.02),
            NonCashCollateralAsset("Bund_10Y", yield_rate=0.02, haircut=0.03),
            NonCashCollateralAsset("JGB_10Y", yield_rate=0.005, haircut=0.05),
        ]
        result = non_cash_collateral_discount_rate(pool, funding_rate=0.05, cash_rate=0.04)
        # UST has highest yield → lowest haircut cost → should be cheapest
        assert result.optimal_collateral == "UST_10Y"

    def test_haircut_increases_cost(self):
        """Higher haircut → higher effective rate."""
        low_hc = [NonCashCollateralAsset("A", yield_rate=0.03, haircut=0.01)]
        high_hc = [NonCashCollateralAsset("A", yield_rate=0.03, haircut=0.10)]

        r_low = non_cash_collateral_discount_rate(low_hc, funding_rate=0.05, cash_rate=0.04)
        r_high = non_cash_collateral_discount_rate(high_hc, funding_rate=0.05, cash_rate=0.04)
        assert r_high.effective_rate > r_low.effective_rate

    def test_liquidity_premium(self):
        """Liquidity premium increases effective rate."""
        no_lp = [NonCashCollateralAsset("X", yield_rate=0.03, haircut=0.02, liquidity_premium=0.0)]
        with_lp = [NonCashCollateralAsset("X", yield_rate=0.03, haircut=0.02, liquidity_premium=0.005)]

        r_no = non_cash_collateral_discount_rate(no_lp, funding_rate=0.05, cash_rate=0.04)
        r_yes = non_cash_collateral_discount_rate(with_lp, funding_rate=0.05, cash_rate=0.04)
        assert r_yes.effective_rate > r_no.effective_rate

    def test_collateral_costs_dict(self):
        pool = [
            NonCashCollateralAsset("A", yield_rate=0.03, haircut=0.02),
            NonCashCollateralAsset("B", yield_rate=0.02, haircut=0.05),
        ]
        result = non_cash_collateral_discount_rate(pool, funding_rate=0.05, cash_rate=0.04)
        assert "A" in result.collateral_costs
        assert "B" in result.collateral_costs
