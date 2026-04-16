"""Tests for sovereign bond trading."""

import math

import numpy as np
import pytest

from pricebook.sovereign_bond import (
    AuctionResult,
    CrossCountryRV,
    OTROFRResult,
    SovereignBasisResult,
    SovereignSpreadCurve,
    auction_analytics,
    build_spread_curve,
    cross_country_rv,
    otr_ofr_analysis,
    sovereign_basis,
)


# ---- Spread curve ----

class TestSovereignSpreadCurve:
    def test_build(self):
        curve = build_spread_curve(
            country="IT", benchmark="DE",
            tenors=[2, 5, 10, 30],
            country_yields_pct=[3.0, 3.5, 4.0, 4.5],
            benchmark_yields_pct=[2.5, 2.7, 2.9, 3.0],
        )
        assert isinstance(curve, SovereignSpreadCurve)
        # 2Y spread = (3.0 - 2.5) × 100 = 50 bps
        assert curve.spreads_bps[0] == pytest.approx(50)

    def test_spread_at(self):
        curve = build_spread_curve("IT", "DE", [2, 10], [3.0, 4.0], [2.5, 3.0])
        # 6Y: interpolate between 50 and 100 bps
        assert curve.spread_at(6) == pytest.approx(75, abs=1)

    def test_spread_extrapolation(self):
        curve = build_spread_curve("IT", "DE", [2, 10], [3.0, 4.0], [2.5, 3.0])
        assert curve.spread_at(1) == 50     # flat below
        assert curve.spread_at(20) == 100   # flat above

    def test_z_score(self):
        curve = build_spread_curve("IT", "DE", [5], [3.5], [2.7])
        # Current 80 bps; historical mean 60, std 20 → z = 1
        historical = [40, 50, 60, 70, 80]     # mean=60, std≈14.1
        z = curve.z_score(historical, 5.0)
        assert z > 0   # current is above mean


# ---- Sovereign basis ----

class TestSovereignBasis:
    def test_basic(self):
        result = sovereign_basis(
            bond_yield_pct=3.50,
            swap_rate_pct=3.40,
            cds_spread_pct=0.15,
            futures_implied_yield_pct=3.48,
        )
        assert isinstance(result, SovereignBasisResult)

    def test_bond_cds_basis(self):
        """Bond-CDS basis = ASW - CDS."""
        # ASW = 3.50 - 3.40 = 10 bps
        # CDS = 15 bps
        # Basis = 10 - 15 = -5 bps (negative, typical)
        result = sovereign_basis(3.50, 3.40, 0.15, 3.48)
        assert result.bond_cds_basis_bps == pytest.approx(-5)

    def test_bond_futures_basis(self):
        result = sovereign_basis(3.50, 3.40, 0.15, 3.48)
        # 3.50 - 3.48 = 0.02% = 2 bps
        assert result.bond_futures_basis_bps == pytest.approx(2)


# ---- Cross-country RV ----

class TestCrossCountryRV:
    def test_basic(self):
        result = cross_country_rv(
            country1="IT", country2="DE", tenor=10,
            yield1_pct=4.0, yield2_pct=2.5,
            historical_diffs_bps=[100, 120, 150, 130, 110],
        )
        assert isinstance(result, CrossCountryRV)
        assert result.yield_diff_bps == 150

    def test_z_score_high(self):
        """Current diff much higher than historical → high z-score."""
        result = cross_country_rv(
            "IT", "DE", 10, 5.0, 2.5,
            historical_diffs_bps=[100, 110, 120, 115, 105],
        )
        # current = 250, historical mean ≈ 110 → large positive z
        assert result.z_score > 2

    def test_z_score_historical_mean(self):
        result = cross_country_rv(
            "IT", "DE", 10, 3.6, 2.5,
            historical_diffs_bps=[100, 110, 120, 115, 105],
        )
        # Current = 110, historical mean ≈ 110 → z ≈ 0
        assert abs(result.z_score) < 1


# ---- Auction ----

class TestAuctionAnalytics:
    def test_good_auction(self):
        """High bid-cover, low tail, low concession → high quality."""
        result = auction_analytics(
            instrument="10Y UST",
            amount_offered=40_000_000_000,
            bids_received=120_000_000_000,
            stop_out_yield_pct=3.50,
            average_bid_yield_pct=3.49,
            wi_yield_pct=3.50,
            secondary_curve_yield_pct=3.50,
        )
        assert isinstance(result, AuctionResult)
        assert result.bid_cover_ratio == 3.0
        assert abs(result.tail_bps) < 2
        assert result.quality_score >= 7

    def test_poor_auction(self):
        """Low bid-cover, high tail → low quality."""
        result = auction_analytics(
            "10Y UST", 40e9, 50e9,
            stop_out_yield_pct=3.55,
            average_bid_yield_pct=3.48,     # 7 bp tail
            wi_yield_pct=3.55,
            secondary_curve_yield_pct=3.47,  # 8 bp concession
        )
        assert result.bid_cover_ratio == pytest.approx(1.25)
        assert result.tail_bps > 5
        assert result.quality_score <= 4


# ---- OTR/OFR ----

class TestOTROFR:
    def test_normal_otr_premium(self):
        result = otr_ofr_analysis(10, otr_yield_pct=3.50, ofr_yield_pct=3.52)
        assert result.otr_premium_bps == pytest.approx(2)
        assert not result.is_squeeze

    def test_squeeze_detected(self):
        result = otr_ofr_analysis(10, otr_yield_pct=3.50, ofr_yield_pct=3.65,
                                    squeeze_threshold_bps=10)
        assert result.otr_premium_bps == pytest.approx(15)
        assert result.is_squeeze

    def test_zero_premium(self):
        result = otr_ofr_analysis(10, 3.50, 3.50)
        assert result.otr_premium_bps == 0
        assert not result.is_squeeze

    def test_negative_premium(self):
        """OTR yield higher than OFR → unusual; not a squeeze but flagged."""
        result = otr_ofr_analysis(10, 3.52, 3.50)
        assert result.otr_premium_bps < 0
