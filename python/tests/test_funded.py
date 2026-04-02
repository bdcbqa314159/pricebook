"""Tests for funded structures: repo, TRS, participation."""

import pytest
import math
from datetime import date

from pricebook.funded import Repo, TotalReturnSwap, FundedParticipation
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestRepo:
    def test_pv_near_zero_at_inception(self):
        """Fair repo has PV ≈ 0."""
        curve = make_flat_curve(REF, 0.04)
        repo = Repo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25)
        mat = date(2024, 4, 15)
        pv = repo.pv(curve, REF, mat)
        assert pv == pytest.approx(0.0, abs=repo.notional * 0.001)

    def test_cash_lent(self):
        repo = Repo(bond_dirty_price=101.0, repo_rate=0.04, T=0.25, notional=1_000_000)
        assert repo.cash_lent == pytest.approx(1_010_000.0)

    def test_haircut_reduces_cash(self):
        repo_no = Repo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25, haircut=0.0)
        repo_hc = Repo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25, haircut=0.05)
        assert repo_hc.cash_lent < repo_no.cash_lent

    def test_effective_rate_higher_with_haircut(self):
        repo = Repo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25, haircut=0.05)
        assert repo.effective_funding_rate > 0.04

    def test_repurchase_price(self):
        repo = Repo(bond_dirty_price=100.0, repo_rate=0.04, T=0.25, notional=1_000_000)
        expected = 1_000_000 * (1 + 0.04 * 0.25)
        assert repo.repurchase_price == pytest.approx(expected)

    def test_implied_repo(self):
        """Implied repo from spot/forward."""
        rate = Repo.implied_repo_rate(
            bond_spot_price=100.0, bond_forward_price=101.0, T=0.25,
        )
        assert rate == pytest.approx(0.04)

    def test_implied_repo_with_coupon(self):
        rate = Repo.implied_repo_rate(
            bond_spot_price=100.0, bond_forward_price=99.0, T=0.25,
            coupon_income=2.0,
        )
        # (99 - 100 + 2) / (100 * 0.25) = 1/25 = 0.04
        assert rate == pytest.approx(0.04)


class TestTotalReturnSwap:
    def test_pv_zero_at_inception(self):
        """At inception ref_current = ref_start, total_return = 0."""
        trs = TotalReturnSwap(
            reference_pv_start=100, reference_pv_current=100,
            funding_rate=0.05, spread=0.0, T=0.0,
        )
        assert trs.pv() == pytest.approx(0.0)

    def test_positive_return(self):
        """Reference asset appreciated → positive PV for receiver."""
        trs = TotalReturnSwap(
            reference_pv_start=100, reference_pv_current=110,
            funding_rate=0.05, spread=0.01, T=1.0,
        )
        assert trs.pv() > 0

    def test_negative_return(self):
        """Reference depreciated → negative PV."""
        trs = TotalReturnSwap(
            reference_pv_start=100, reference_pv_current=90,
            funding_rate=0.05, spread=0.01, T=1.0,
        )
        assert trs.pv() < 0

    def test_fair_spread(self):
        spread = TotalReturnSwap.fair_spread(
            reference_yield=0.06, funding_rate=0.04,
        )
        assert spread == pytest.approx(0.02)

    def test_notional_scales(self):
        trs1 = TotalReturnSwap(100, 105, 0.05, 0.0, 1.0, notional=1_000_000)
        trs2 = TotalReturnSwap(100, 105, 0.05, 0.0, 1.0, notional=2_000_000)
        assert trs2.pv() == pytest.approx(2 * trs1.pv())


class TestFundedParticipation:
    def test_full_participation(self):
        fp = FundedParticipation(
            total_notional=1_000_000, participation_rate=1.0,
            asset_yield=0.06, funding_cost=0.04, T=1.0,
        )
        assert fp.funded_amount == 1_000_000
        assert fp.pv() == pytest.approx(1_000_000 * 0.02)

    def test_zero_participation(self):
        fp = FundedParticipation(
            total_notional=1_000_000, participation_rate=0.0,
            asset_yield=0.06, funding_cost=0.04,
        )
        assert fp.pv() == pytest.approx(0.0)

    def test_partial_participation(self):
        fp = FundedParticipation(
            total_notional=1_000_000, participation_rate=0.5,
            asset_yield=0.06, funding_cost=0.04, T=1.0,
        )
        assert fp.funded_amount == 500_000
        assert fp.pv() == pytest.approx(500_000 * 0.02)

    def test_expected_loss_reduces_pv(self):
        fp_no_loss = FundedParticipation(
            total_notional=1_000_000, participation_rate=1.0,
            asset_yield=0.06, funding_cost=0.04, expected_loss=0.0,
        )
        fp_loss = FundedParticipation(
            total_notional=1_000_000, participation_rate=1.0,
            asset_yield=0.06, funding_cost=0.04, expected_loss=0.01,
        )
        assert fp_loss.pv() < fp_no_loss.pv()

    def test_net_carry(self):
        fp = FundedParticipation(
            total_notional=1_000_000, participation_rate=1.0,
            asset_yield=0.06, funding_cost=0.04, expected_loss=0.005,
        )
        assert fp.net_carry == pytest.approx(0.015)

    def test_invalid_participation_raises(self):
        with pytest.raises(ValueError):
            FundedParticipation(1_000_000, 1.5, 0.06, 0.04)

    def test_cash_cds_basis(self):
        basis = FundedParticipation.cash_cds_basis(
            funded_spread=0.015, cds_spread=0.012,
        )
        assert basis == pytest.approx(0.003)

    def test_negative_basis(self):
        basis = FundedParticipation.cash_cds_basis(0.010, 0.015)
        assert basis < 0
