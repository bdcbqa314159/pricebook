"""Tests for repo-TRS unified view: RepoFinancedPosition, ReverseRepo, specialness."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.funded import Repo, ReverseRepo, RepoFinancedPosition
from tests.conftest import make_flat_curve


REF = date(2026, 4, 27)


def _disc():
    return make_flat_curve(REF, 0.04)


# ---- ReverseRepo ----

class TestReverseRepo:

    def test_cash_received(self):
        rr = ReverseRepo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25,
                         notional=1_000_000)
        assert rr.cash_received == pytest.approx(1_000_000)

    def test_cash_received_with_haircut(self):
        rr = ReverseRepo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25,
                         haircut=0.02, notional=1_000_000)
        assert rr.cash_received == pytest.approx(980_000)

    def test_cost(self):
        rr = ReverseRepo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25,
                         notional=1_000_000)
        assert rr.cost == pytest.approx(10_000)  # 1M × 4% × 0.25

    def test_total_repayment(self):
        rr = ReverseRepo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25,
                         notional=1_000_000)
        assert rr.total_repayment == pytest.approx(1_010_000)

    def test_pv_at_inception_near_zero(self):
        """At inception, reverse repo PV ≈ 0 if repo rate = risk-free."""
        disc = _disc()
        rr = ReverseRepo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25,
                         notional=1_000_000)
        mat = REF + timedelta(days=91)
        pv = rr.pv(disc, REF, mat)
        assert abs(pv) < 5_000  # near zero for short term

    def test_symmetry_with_repo(self):
        """ReverseRepo cost = Repo interest (same terms)."""
        repo = Repo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25,
                    notional=1_000_000)
        rr = ReverseRepo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25,
                         notional=1_000_000)
        # Repo repurchase = cash_lent × (1 + r × T)
        # ReverseRepo total_repayment = cash_received × (1 + r × T)
        assert repo.repurchase_price == pytest.approx(rr.total_repayment)


# ---- RepoFinancedPosition ----

class TestRepoFinancedPosition:

    def test_net_carry_positive(self):
        """Asset yield > financing cost → positive net carry."""
        pos = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.03,
            trs_spread=0.005, asset_yield=0.05, T=1.0,
            notional=1_000_000)
        nc = pos.net_carry()
        assert nc > 0  # 5% - 3% - 0.5% = 1.5% × 1M = 15,000

    def test_net_carry_negative(self):
        """Expensive financing → negative net carry."""
        pos = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.06,
            trs_spread=0.01, asset_yield=0.04, T=1.0,
            notional=1_000_000)
        assert pos.net_carry() < 0

    def test_breakeven_repo_rate(self):
        """Breakeven repo: net carry = 0."""
        pos = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.03,
            trs_spread=0.005, asset_yield=0.05, T=1.0,
            notional=1_000_000)
        be = pos.breakeven_repo_rate()
        # At breakeven, net carry should be ≈ 0
        check = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=be,
            trs_spread=0.005, asset_yield=0.05, T=1.0,
            notional=1_000_000)
        assert abs(check.net_carry()) < 1.0  # within $1

    def test_blended_financing_with_haircut(self):
        """Haircut blends repo and unsecured rates."""
        pos = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.03,
            funding_rate=0.05, haircut=0.10, notional=1_000_000)
        expected = 0.90 * 0.03 + 0.10 * 0.05  # 3.2%
        assert pos.blended_financing_rate == pytest.approx(expected)

    def test_implied_repo_from_trs(self):
        """Implied repo = asset_yield - trs_spread."""
        pos = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.03,
            trs_spread=0.015, asset_yield=0.05,
            notional=1_000_000)
        assert pos.implied_repo_from_trs_spread() == pytest.approx(0.035)

    def test_specialness_pricing(self):
        """Specialness reduces effective repo rate."""
        pos_gc = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.04,
            asset_yield=0.05, T=1.0, specialness=0.0,
            notional=1_000_000)
        pos_spec = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.04,
            asset_yield=0.05, T=1.0, specialness=0.01,
            notional=1_000_000)
        # Special collateral → lower effective repo → higher net carry
        assert pos_spec.net_carry() > pos_gc.net_carry()
        assert pos_spec.effective_repo_rate == pytest.approx(0.03)

    def test_implied_specialness(self):
        pos = RepoFinancedPosition(
            bond_dirty_price=100.0, repo_rate=0.03,
            trs_spread=0.015, asset_yield=0.05,
            notional=1_000_000)
        # Implied repo = 3.5%, GC = 4% → specialness = 0.5%
        assert pos.implied_specialness_from_trs(gc_rate=0.04) == pytest.approx(0.005)

    def test_repo_cost_scales_with_price(self):
        """Repo cost should scale with bond dirty price."""
        pos_low = RepoFinancedPosition(
            bond_dirty_price=95.0, repo_rate=0.04, T=1.0,
            notional=1_000_000)
        pos_high = RepoFinancedPosition(
            bond_dirty_price=105.0, repo_rate=0.04, T=1.0,
            notional=1_000_000)
        assert pos_high.repo_cost > pos_low.repo_cost
