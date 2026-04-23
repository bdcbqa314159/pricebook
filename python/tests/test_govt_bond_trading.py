"""Tests for government bond trading."""

import pytest
from datetime import date

from pricebook.govt_bond_trading import (
    AuctionResult,
    BasisDecomposition,
    CTDSwitchEntry,
    OTROFRSpread,
    WhenIssuedEstimate,
    auction_analytics,
    basis_decomposition,
    ctd_switch_monitor,
    otr_ofr_spread,
    when_issued_price,
)


# ---- Step 1: OTR/OFR + WI + auction ----

class TestOTROFRSpread:
    def test_positive_spread_normal(self):
        """Step 1 test: OTR-OFR spread is positive (liquidity premium)."""
        result = otr_ofr_spread("10Y", otr_yield=0.0400, ofr_yield=0.0410)
        assert result.spread_bps == pytest.approx(10.0)
        assert result.spread_bps > 0

    def test_z_score_wide(self):
        history = [8.0, 9.0, 10.0, 11.0, 12.0] * 4
        result = otr_ofr_spread("10Y", 0.0400, 0.0430,
                                history_bps=history, threshold=2.0)
        # 30bp vs avg ~10bp → wide
        assert result.signal == "wide"

    def test_z_score_tight(self):
        history = [8.0, 9.0, 10.0, 11.0, 12.0] * 4
        result = otr_ofr_spread("10Y", 0.0400, 0.04005,
                                history_bps=history, threshold=2.0)
        # 0.5bp vs avg ~10bp → tight
        assert result.signal == "tight"

    def test_no_history_fair(self):
        result = otr_ofr_spread("10Y", 0.04, 0.041)
        assert result.signal == "fair"
        assert result.z_score is None

    def test_records_tenor(self):
        result = otr_ofr_spread("5Y", 0.03, 0.031)
        assert result.tenor_label == "5Y"


class TestWhenIssuedPrice:
    def test_interpolation(self):
        wi = when_issued_price(
            short_tenor_yield=0.04, long_tenor_yield=0.05,
            short_tenor_years=5, long_tenor_years=10,
            target_tenor_years=7, coupon_rate=0.045,
        )
        # 7Y is 40% between 5Y and 10Y → yield = 0.044
        assert wi.estimated_yield == pytest.approx(0.044)
        assert wi.estimated_price > 0

    def test_at_short_tenor(self):
        wi = when_issued_price(0.04, 0.05, 5, 10, 5, 0.04)
        assert wi.estimated_yield == pytest.approx(0.04)

    def test_at_long_tenor(self):
        wi = when_issued_price(0.04, 0.05, 5, 10, 10, 0.05)
        assert wi.estimated_yield == pytest.approx(0.05)

    def test_par_bond_near_100(self):
        """If coupon ≈ yield, price should be near par."""
        wi = when_issued_price(0.04, 0.04, 5, 10, 7, 0.04)
        assert wi.estimated_price == pytest.approx(100.0, rel=0.02)


class TestAuctionAnalytics:
    def test_well_received(self):
        result = auction_analytics(
            "10Y", high_yield=0.0400, when_issued_yield=0.0400,
            total_bids=70_000, accepted_amount=25_000,
            dealer_amount=10_000, indirect_amount=10_000, direct_amount=5_000,
        )
        assert result.bid_to_cover == pytest.approx(2.8)
        assert result.tail_bps == pytest.approx(0.0)
        assert result.well_received is True

    def test_tail_positive_when_auction_cheap(self):
        result = auction_analytics(
            "10Y", high_yield=0.0405, when_issued_yield=0.0400,
            total_bids=50_000, accepted_amount=25_000,
            dealer_amount=12_000, indirect_amount=8_000, direct_amount=5_000,
        )
        # tail = (0.0405 - 0.0400) × 10000 = 5bp
        assert result.tail_bps == pytest.approx(5.0)
        assert result.bid_to_cover == pytest.approx(2.0)
        assert result.well_received is False

    def test_allocation_percents(self):
        result = auction_analytics(
            "5Y", 0.03, 0.03, 60_000, 30_000,
            dealer_amount=15_000, indirect_amount=10_000, direct_amount=5_000,
        )
        assert result.dealer_pct == pytest.approx(50.0)
        assert result.indirect_pct == pytest.approx(100.0 / 3.0)
        assert result.direct_pct == pytest.approx(50.0 / 3.0)
        assert result.dealer_pct + result.indirect_pct + result.direct_pct == pytest.approx(100.0)


# ---- Step 2: basis trading ----

class TestBasisDecomposition:
    def test_gross_basis(self):
        bd = basis_decomposition(
            "UST_10Y", bond_price=99.0, futures_price=100.0, cf=0.98,
            coupon_rate=0.04, repo_rate=0.05, days_to_delivery=90,
        )
        # gross = 99 - 0.98 × 100 = 1.0
        assert bd.gross_basis == pytest.approx(1.0)

    def test_carry_formula(self):
        bd = basis_decomposition(
            "UST_10Y", bond_price=99.0, futures_price=100.0, cf=0.98,
            coupon_rate=0.04, repo_rate=0.05, days_to_delivery=360,
        )
        # carry = 0.04×100×(360/360) - 99×0.05×(360/360) = 4 - 4.95 = -0.95
        assert bd.carry == pytest.approx(-0.95)

    def test_net_basis_equals_gross_minus_carry(self):
        """Step 2 test: basis decomposes into carry + optionality."""
        bd = basis_decomposition(
            "UST_10Y", bond_price=99.0, futures_price=100.0, cf=0.98,
            coupon_rate=0.04, repo_rate=0.05, days_to_delivery=180,
        )
        assert bd.net_basis == pytest.approx(bd.gross_basis - bd.carry)

    def test_implied_repo(self):
        bd = basis_decomposition(
            "UST_10Y", bond_price=100.0, futures_price=100.0, cf=1.0,
            coupon_rate=0.04, repo_rate=0.04, days_to_delivery=365,
        )
        # gross = 0, carry = 4 - 4 = 0, implied_repo = (4 - 0) / (100 × 1) = 0.04
        assert bd.implied_repo == pytest.approx(0.04)


class TestCTDSwitchMonitor:
    def test_identifies_ctd(self):
        deliverables = [
            basis_decomposition("Bond_A", 98.0, 100.0, 0.97, 0.04, 0.045, 90),
            basis_decomposition("Bond_B", 99.0, 100.0, 0.98, 0.035, 0.045, 90),
            basis_decomposition("Bond_C", 100.0, 100.0, 0.99, 0.05, 0.045, 90),
        ]
        entries = ctd_switch_monitor(deliverables)
        assert len(entries) == 3
        # Exactly one is CTD
        ctd = [e for e in entries if e.is_ctd]
        assert len(ctd) == 1
        # First entry has highest implied repo
        assert entries[0].is_ctd is True
        assert entries[0].implied_repo >= entries[1].implied_repo
        assert entries[1].implied_repo >= entries[2].implied_repo

    def test_empty(self):
        assert ctd_switch_monitor([]) == []

    def test_sorted_by_implied_repo_descending(self):
        deliverables = [
            basis_decomposition("A", 98.0, 100.0, 0.97, 0.04, 0.04, 90),
            basis_decomposition("B", 100.0, 100.0, 0.99, 0.05, 0.04, 90),
        ]
        entries = ctd_switch_monitor(deliverables)
        assert entries[0].implied_repo >= entries[1].implied_repo
