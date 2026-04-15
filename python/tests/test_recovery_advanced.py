"""Tests for advanced recovery models."""

import math

import pytest

from pricebook.recovery_advanced import (
    LGDCycleResult,
    StochasticRecoveryCDSResult,
    WaterfallResult,
    WrongWayRecoveryCVA,
    lgd_cycle,
    seniority_waterfall,
    stochastic_recovery_cds,
    waterfall_recovery_rates,
    wrong_way_recovery_cva,
)


# ---- Seniority waterfall ----

class TestSeniorityWaterfall:
    def test_full_recovery(self):
        """Full recovery: everyone gets their face value."""
        result = seniority_waterfall(100, 60, 30, 10)
        assert result.senior_recovery == pytest.approx(60)
        assert result.subordinated_recovery == pytest.approx(30)
        assert result.equity_recovery == pytest.approx(10)

    def test_partial_recovery_senior_first(self):
        """Partial: senior gets full, sub gets remainder."""
        result = seniority_waterfall(80, 60, 30, 10)
        assert result.senior_recovery == pytest.approx(60)
        assert result.subordinated_recovery == pytest.approx(20)
        assert result.equity_recovery == pytest.approx(0)

    def test_severe_loss(self):
        """Severe: even senior doesn't get full recovery."""
        result = seniority_waterfall(40, 60, 30, 10)
        assert result.senior_recovery == pytest.approx(40)
        assert result.subordinated_recovery == pytest.approx(0)
        assert result.equity_recovery == pytest.approx(0)

    def test_zero_recovery(self):
        result = seniority_waterfall(0, 60, 30, 10)
        assert result.total_recovery == 0.0

    def test_senior_recovery_exceeds_sub(self):
        """Senior always recovers more than subordinated."""
        for recovery in [20, 50, 70, 90]:
            result = seniority_waterfall(recovery, 60, 30, 10)
            assert result.senior_recovery >= result.subordinated_recovery
            assert result.subordinated_recovery >= result.equity_recovery


class TestWaterfallRecoveryRates:
    def test_senior_rate_highest(self):
        result = waterfall_recovery_rates(0.50)
        assert result.senior_recovery >= result.subordinated_recovery
        assert result.subordinated_recovery >= result.equity_recovery

    def test_full_recovery_all_100pct(self):
        result = waterfall_recovery_rates(1.0)
        assert result.senior_recovery == pytest.approx(1.0)
        assert result.subordinated_recovery == pytest.approx(1.0)
        assert result.equity_recovery == pytest.approx(1.0)


# ---- LGD cycle ----

class TestLGDCycle:
    def test_downturn_higher_lgd(self):
        normal = lgd_cycle(0.60, default_rate=0.02)
        downturn = lgd_cycle(0.60, default_rate=0.06)
        assert downturn.lgd > normal.lgd

    def test_benign_lower_lgd(self):
        normal = lgd_cycle(0.60, default_rate=0.02)
        benign = lgd_cycle(0.60, default_rate=0.005)
        assert benign.lgd < normal.lgd

    def test_recovery_bounded(self):
        extreme = lgd_cycle(0.60, default_rate=0.20, sensitivity=5.0)
        assert extreme.recovery >= 0.10  # floor
        assert extreme.recovery <= 0.80  # cap

    def test_regime_classification(self):
        assert lgd_cycle(0.60, 0.04).regime == "downturn"
        assert lgd_cycle(0.60, 0.005).regime == "benign"
        assert lgd_cycle(0.60, 0.02).regime == "normal"


# ---- Stochastic recovery CDS ----

class TestStochasticRecoveryCDS:
    def test_wrong_way_higher_spread(self):
        """Negative correlation → recovery drops in default → higher spread."""
        result = stochastic_recovery_cds(
            0.02, 0.40, recovery_vol=0.15,
            hazard_recovery_corr=-0.5, seed=42,
        )
        assert result.spread_stochastic_recovery > result.spread_fixed_recovery
        assert result.wrong_way_premium > 0

    def test_stochastic_spread_positive(self):
        """Stochastic recovery CDS spread should be positive."""
        result = stochastic_recovery_cds(
            0.02, 0.40, 0.15, hazard_recovery_corr=0.5, seed=42)
        assert result.spread_stochastic_recovery > 0

    def test_zero_corr_similar_spread(self):
        """Zero correlation → stochastic ≈ fixed (no wrong-way)."""
        result = stochastic_recovery_cds(
            0.02, 0.40, recovery_vol=0.10,
            hazard_recovery_corr=0.0, seed=42,
        )
        assert result.spread_stochastic_recovery == pytest.approx(
            result.spread_fixed_recovery, rel=0.05)

    def test_mean_recovery_near_base(self):
        result = stochastic_recovery_cds(
            0.02, 0.40, 0.10, 0.0, seed=42)
        assert result.mean_recovery == pytest.approx(0.40, rel=0.05)


# ---- Wrong-way recovery CVA ----

class TestWrongWayRecoveryCVA:
    def test_wrong_way_higher_cva(self):
        """Negative corr → recovery drops → higher CVA."""
        epe = [0, 100_000, 200_000, 150_000, 100_000, 50_000]
        times = [0, 1, 2, 3, 4, 5]
        result = wrong_way_recovery_cva(
            epe, times, 0.02, 0.05,
            hazard_recovery_corr=-0.30, seed=42,
        )
        assert result.cva_stochastic_recovery > result.cva_fixed_recovery
        assert result.wrong_way_adjustment > 0

    def test_positive_cva(self):
        epe = [0, 100_000, 100_000]
        times = [0, 1, 2]
        result = wrong_way_recovery_cva(epe, times, 0.02, 0.05, seed=42)
        assert result.cva_fixed_recovery > 0
        assert result.cva_stochastic_recovery > 0
