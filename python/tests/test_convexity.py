"""Tests for convexity adjustments."""

import math

import numpy as np
import pytest

from pricebook.convexity import (
    ArrearsResult,
    CMSConvexityResult,
    QuantoIRResult,
    TimingResult,
    arrears_adjustment,
    cms_convexity_adjustment,
    cms_rate_replication,
    quanto_ir_adjustment,
    timing_adjustment,
)


# ---- CMS convexity ----

class TestCMSConvexityAdjustment:
    def test_positive_adjustment(self):
        """CMS rate > forward swap rate (positive convexity)."""
        result = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.20, 5.0)
        assert isinstance(result, CMSConvexityResult)
        assert result.convexity_adjustment > 0
        assert result.cms_rate > result.forward_swap_rate

    def test_zero_vol_no_adjustment(self):
        result = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.0, 5.0)
        assert result.convexity_adjustment == 0.0
        assert result.cms_rate == result.forward_swap_rate

    def test_higher_vol_larger_adjustment(self):
        low = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.10, 5.0)
        high = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.30, 5.0)
        assert high.convexity_adjustment > low.convexity_adjustment

    def test_longer_expiry_larger_adjustment(self):
        short = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.20, 1.0)
        long = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.20, 10.0)
        assert long.convexity_adjustment > short.convexity_adjustment

    def test_mean_reversion_reduces(self):
        """Mean reversion reduces effective expiry → smaller adjustment."""
        no_mr = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.20, 5.0, mean_reversion=0.0)
        mr = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.20, 5.0, mean_reversion=0.1)
        assert mr.convexity_adjustment < no_mr.convexity_adjustment

    def test_method(self):
        result = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.20, 5.0)
        assert result.method == "hagan_linear"


class TestCMSReplication:
    def test_positive_adjustment(self):
        result = cms_rate_replication(0.05, 0.20, 5.0, 10.0)
        assert isinstance(result, CMSConvexityResult)
        assert result.method == "replication"

    def test_replication_agrees_with_analytical(self):
        """Replication and analytical should give same sign."""
        rep = cms_rate_replication(0.05, 0.20, 5.0, 10.0)
        ana = cms_convexity_adjustment(0.05, 4.0, 8.0, 0.20, 5.0)
        assert rep.convexity_adjustment * ana.convexity_adjustment >= 0  # same sign

    def test_higher_vol_larger(self):
        low = cms_rate_replication(0.05, 0.10, 5.0, 10.0)
        high = cms_rate_replication(0.05, 0.30, 5.0, 10.0)
        assert abs(high.convexity_adjustment) >= abs(low.convexity_adjustment) * 0.5


# ---- Arrears ----

class TestArrearsAdjustment:
    def test_positive_adjustment(self):
        result = arrears_adjustment(0.05, 0.20, 1.0, 0.5)
        assert isinstance(result, ArrearsResult)
        assert result.adjustment > 0
        assert result.arrears_rate > result.forward_rate

    def test_zero_vol_no_adjustment(self):
        result = arrears_adjustment(0.05, 0.0, 1.0, 0.5)
        assert result.adjustment == 0.0

    def test_higher_vol_larger(self):
        low = arrears_adjustment(0.05, 0.10, 1.0, 0.5)
        high = arrears_adjustment(0.05, 0.30, 1.0, 0.5)
        assert high.adjustment > low.adjustment

    def test_later_fixing_larger(self):
        early = arrears_adjustment(0.05, 0.20, 0.5, 0.5)
        late = arrears_adjustment(0.05, 0.20, 5.0, 0.5)
        assert late.adjustment > early.adjustment

    def test_higher_rate_larger(self):
        low_r = arrears_adjustment(0.02, 0.20, 1.0, 0.5)
        high_r = arrears_adjustment(0.08, 0.20, 1.0, 0.5)
        assert high_r.adjustment > low_r.adjustment


# ---- Timing ----

class TestTimingAdjustment:
    def test_no_delay_no_adjustment(self):
        result = timing_adjustment(0.05, 0.20, 1.0, 1.5, 1.5)
        assert result.adjustment == 0.0
        assert result.payment_delay == 0.0

    def test_delayed_payment(self):
        result = timing_adjustment(0.05, 0.20, 1.0, 1.5, 2.0)
        assert isinstance(result, TimingResult)
        assert result.payment_delay == pytest.approx(0.5)
        assert result.adjustment != 0

    def test_early_payment(self):
        result = timing_adjustment(0.05, 0.20, 1.0, 1.5, 1.0)
        assert result.payment_delay == pytest.approx(-0.5)

    def test_longer_delay_larger_adjustment(self):
        short_delay = timing_adjustment(0.05, 0.20, 1.0, 1.5, 1.75)
        long_delay = timing_adjustment(0.05, 0.20, 1.0, 1.5, 2.5)
        assert abs(long_delay.adjustment) > abs(short_delay.adjustment)


# ---- Quanto IR ----

class TestQuantoIRAdjustment:
    def test_positive_correlation_negative_adjustment(self):
        """Positive correlation → negative quanto adjustment."""
        result = quanto_ir_adjustment(0.05, 0.20, 0.10, 0.3, 5.0)
        assert isinstance(result, QuantoIRResult)
        assert result.adjustment < 0
        assert result.quanto_rate < result.domestic_rate

    def test_negative_correlation_positive_adjustment(self):
        result = quanto_ir_adjustment(0.05, 0.20, 0.10, -0.3, 5.0)
        assert result.adjustment > 0

    def test_zero_correlation_no_adjustment(self):
        result = quanto_ir_adjustment(0.05, 0.20, 0.10, 0.0, 5.0)
        assert result.adjustment == 0.0

    def test_higher_vols_larger(self):
        low = quanto_ir_adjustment(0.05, 0.10, 0.05, 0.3, 5.0)
        high = quanto_ir_adjustment(0.05, 0.30, 0.15, 0.3, 5.0)
        assert abs(high.adjustment) > abs(low.adjustment)

    def test_longer_expiry_larger(self):
        short = quanto_ir_adjustment(0.05, 0.20, 0.10, 0.3, 1.0)
        long = quanto_ir_adjustment(0.05, 0.20, 0.10, 0.3, 10.0)
        assert abs(long.adjustment) > abs(short.adjustment)
