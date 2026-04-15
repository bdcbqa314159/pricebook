"""Tests for leveraged credit structures."""

import math

import pytest

from pricebook.credit_leveraged import (
    CDSStraddleResult,
    CMCDSResult,
    CreditTRSResult,
    DigitalCLNResult,
    LeveragedCDSResult,
    cds_straddle,
    constant_maturity_cds,
    credit_trs,
    digital_cln_leveraged,
    leveraged_cds,
)


# ---- Leveraged CDS ----

class TestLeveragedCDS:
    def test_leveraged_spread_exceeds_standard(self):
        result = leveraged_cds(1_000_000, leverage=3.0, flat_hazard=0.02)
        assert result.leveraged_spread > result.standard_spread

    def test_leverage_scales_spread(self):
        """Leveraged spread ≈ leverage × standard spread."""
        result = leveraged_cds(1_000_000, leverage=3.0, flat_hazard=0.02)
        assert result.leveraged_spread == pytest.approx(
            result.standard_spread * result.leverage, rel=0.05)

    def test_leverage_one_matches_standard(self):
        result = leveraged_cds(1_000_000, leverage=1.0, flat_hazard=0.02)
        assert result.leveraged_spread == pytest.approx(
            result.standard_spread, rel=0.01)

    def test_max_loss_scales_with_leverage(self):
        r1 = leveraged_cds(1_000_000, 1.0, 0.02)
        r3 = leveraged_cds(1_000_000, 3.0, 0.02)
        assert r3.max_loss == pytest.approx(3 * r1.max_loss)


# ---- Digital CLN ----

class TestDigitalCLNLeveraged:
    def test_positive_price(self):
        result = digital_cln_leveraged(100, 0.05, leverage=1.0)
        assert result.price > 0

    def test_higher_leverage_lower_price(self):
        low = digital_cln_leveraged(100, 0.05, leverage=0.5)
        high = digital_cln_leveraged(100, 0.05, leverage=1.0)
        assert high.price < low.price

    def test_zero_hazard_near_par(self):
        result = digital_cln_leveraged(100, 0.05, 1.0, flat_hazard=0.0)
        # No default risk → price = PV of coupons + principal ≈ par
        assert result.price > 90


# ---- Constant-maturity CDS ----

class TestConstantMaturityCDS:
    def test_convexity_positive(self):
        """CMCDS has positive convexity adjustment."""
        result = constant_maturity_cds(5, flat_hazard=0.02, spread_vol=0.40)
        assert result.convexity_adjustment > 0

    def test_fair_exceeds_forward(self):
        """Fair CMCDS spread > forward spread (convexity bias)."""
        result = constant_maturity_cds(5, 0.02, spread_vol=0.40)
        assert result.fair_spread > result.forward_spread

    def test_zero_vol_no_convexity(self):
        result = constant_maturity_cds(5, 0.02, spread_vol=0.0)
        assert result.convexity_adjustment == pytest.approx(0.0)
        assert result.fair_spread == pytest.approx(result.forward_spread)

    def test_higher_vol_more_convexity(self):
        low = constant_maturity_cds(5, 0.02, spread_vol=0.20)
        high = constant_maturity_cds(5, 0.02, spread_vol=0.60)
        assert high.convexity_adjustment > low.convexity_adjustment


# ---- CDS straddle ----

class TestCDSStraddle:
    def test_positive_premium(self):
        result = cds_straddle(0.02, spread_vol=0.40)
        assert result.premium > 0

    def test_atm_payer_equals_receiver(self):
        """ATM straddle: payer ≈ receiver."""
        result = cds_straddle(0.02)  # ATM by default
        assert result.payer_premium == pytest.approx(
            result.receiver_premium, rel=0.05)

    def test_higher_vol_higher_premium(self):
        low = cds_straddle(0.02, spread_vol=0.20)
        high = cds_straddle(0.02, spread_vol=0.60)
        assert high.premium > low.premium

    def test_breakeven_positive(self):
        result = cds_straddle(0.02, spread_vol=0.40)
        assert result.breakeven_move > 0


# ---- Credit TRS ----

class TestCreditTRS:
    def test_spread_tightening_positive_pnl(self):
        """Large spread tightening → positive total return."""
        result = credit_trs(
            10_000_000, index_spread_start=200, index_spread_end=100,
            funding_rate=0.02, trs_spread=0.005,
        )
        assert result.total_return > 0
        assert result.trs_pv > 0

    def test_spread_widening_negative_pnl(self):
        """Spread widening → negative P&L for TR receiver."""
        result = credit_trs(
            10_000_000, 100, 130, 0.05, 0.005,
        )
        assert result.credit_pnl < 0

    def test_no_spread_move_carry_only(self):
        """No spread change → P&L = carry − funding."""
        result = credit_trs(
            10_000_000, 100, 100, 0.04, 0.005, period_years=0.25,
        )
        # carry = 100/10000 × 0.25 = 0.0025
        # funding = (0.04 + 0.005) × 0.25 = 0.01125
        assert result.total_return == pytest.approx(0.0025)
        assert result.funding_cost == pytest.approx(0.01125)

    def test_decomposition(self):
        result = credit_trs(10_000_000, 100, 90, 0.04, 0.005)
        assert result.credit_pnl == pytest.approx(
            result.total_return - result.funding_cost)
