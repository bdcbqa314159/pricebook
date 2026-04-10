"""Tests for commodity rich/cheap analysis and roll strategies."""

import pytest
from datetime import date

from pricebook.commodity_rv import (
    RatioLevel,
    RollCandidate,
    SeasonalLevel,
    ZScoreSignal,
    optimal_roll_date,
    ratio_monitor,
    roll_cost_or_gain,
    roll_pnl,
    seasonality_monitor,
    spread_zscore,
    track_roll_pnl,
)


# ---- Step 1: RV analysis ----

class TestSpreadZscore:
    def test_extreme_high_signals_rich(self):
        history = [1.0, 1.5, 2.0, 2.5, 3.0] * 4
        sig = spread_zscore(10.0, history, threshold=2.0)
        assert sig.signal == "rich"
        assert sig.z_score > 2.0

    def test_extreme_low_signals_cheap(self):
        history = [1.0, 1.5, 2.0, 2.5, 3.0] * 4
        sig = spread_zscore(-5.0, history, threshold=2.0)
        assert sig.signal == "cheap"

    def test_mid_range_fair(self):
        history = [1.0, 1.5, 2.0, 2.5, 3.0] * 4
        sig = spread_zscore(2.0, history)
        assert sig.signal == "fair"

    def test_empty_history(self):
        sig = spread_zscore(2.0, [])
        assert sig.signal == "fair"
        assert sig.z_score is None


class TestRatioMonitor:
    def test_extreme_ratio_triggers_signal(self):
        """Step 1 test: extreme ratio triggers signal."""
        # Gold/silver ratio history around 80; current at 100 → rich
        history = [78.0, 80.0, 82.0, 79.0, 81.0] * 4
        result = ratio_monitor(
            "gold", "silver",
            numerator_price=2_000.0,
            denominator_price=20.0,  # ratio = 100
            history=history,
            threshold=2.0,
        )
        assert result.ratio == pytest.approx(100.0)
        assert result.signal == "rich"
        assert result.z_score > 2.0

    def test_normal_ratio_fair(self):
        history = [78.0, 80.0, 82.0, 79.0, 81.0] * 4
        result = ratio_monitor(
            "gold", "silver", 1_600.0, 20.0,  # ratio = 80
            history=history,
        )
        assert result.signal == "fair"

    def test_brent_wti_spread(self):
        history = [3.0, 3.5, 4.0, 3.2, 3.8] * 4
        result = ratio_monitor(
            "brent", "WTI", 78.0, 72.0,
            history=history,
        )
        assert result.ratio == pytest.approx(78.0 / 72.0)

    def test_zero_denominator(self):
        result = ratio_monitor("A", "B", 100.0, 0.0, history=[1.0, 2.0])
        assert result.ratio == 0.0


class TestSeasonalityMonitor:
    def test_above_seasonal_norm(self):
        norms = {1: 70.0, 2: 72.0, 3: 75.0, 7: 68.0, 12: 80.0}
        result = seasonality_monitor(
            "natgas", month=12, current_price=95.0,
            seasonal_norms=norms,
            history_deviations=[2.0, -3.0, 5.0, -1.0, 4.0] * 4,
            threshold=2.0,
        )
        assert result.deviation == pytest.approx(15.0)
        assert result.signal == "rich"

    def test_at_seasonal_norm(self):
        norms = {1: 70.0}
        result = seasonality_monitor(
            "natgas", month=1, current_price=70.0,
            seasonal_norms=norms,
        )
        assert result.deviation == pytest.approx(0.0)

    def test_missing_month_uses_current(self):
        result = seasonality_monitor(
            "natgas", month=6, current_price=72.0,
            seasonal_norms={1: 70.0},
        )
        assert result.seasonal_norm == 72.0
        assert result.deviation == 0.0


# ---- Step 2: roll strategies ----

class TestRollPnL:
    def test_backwardation_positive_pnl(self):
        """Step 2 test: backwardation roll → positive P&L."""
        # Long position: sell old at 76, buy new at 74 → gain 2 per unit
        pnl = roll_pnl(old_price=76.0, new_price=74.0, quantity=1_000)
        assert pnl == pytest.approx(2_000)

    def test_contango_negative_pnl(self):
        # Long: sell old at 72, buy new at 74 → loss 2 per unit
        pnl = roll_pnl(72.0, 74.0, 1_000)
        assert pnl == pytest.approx(-2_000)

    def test_short_direction_flips(self):
        long_pnl = roll_pnl(76.0, 74.0, 1_000, direction=1)
        short_pnl = roll_pnl(76.0, 74.0, 1_000, direction=-1)
        assert long_pnl == pytest.approx(-short_pnl)

    def test_flat_roll_zero(self):
        assert roll_pnl(72.0, 72.0, 1_000) == pytest.approx(0.0)


class TestRollCostOrGain:
    def test_contango(self):
        assert roll_cost_or_gain(72.0, 74.0) == "contango_cost"

    def test_backwardation(self):
        assert roll_cost_or_gain(76.0, 74.0) == "backwardation_gain"

    def test_flat(self):
        assert roll_cost_or_gain(72.0, 72.0) == "flat"


class TestOptimalRollDate:
    def test_picks_highest_spread(self):
        candidates = [
            RollCandidate(date(2024, 2, 10), date(2024, 3, 1), date(2024, 4, 1),
                          74.0, 75.0, -1.0, 1_000),
            RollCandidate(date(2024, 2, 15), date(2024, 3, 1), date(2024, 4, 1),
                          74.5, 74.0, 0.5, -500),
            RollCandidate(date(2024, 2, 20), date(2024, 3, 1), date(2024, 4, 1),
                          74.2, 74.8, -0.6, 600),
        ]
        best = optimal_roll_date(candidates)
        assert best is not None
        assert best.roll_date == date(2024, 2, 15)
        assert best.roll_spread == pytest.approx(0.5)

    def test_empty_returns_none(self):
        assert optimal_roll_date([]) is None


class TestTrackRollPnL:
    def test_cumulative(self):
        rolls = [
            (76.0, 74.0, 1_000),   # +2K
            (74.0, 75.0, 1_000),   # -1K
            (75.0, 73.0, 500),     # +1K
        ]
        assert track_roll_pnl(rolls) == pytest.approx(2_000)

    def test_empty(self):
        assert track_roll_pnl([]) == pytest.approx(0.0)

    def test_single_roll(self):
        assert track_roll_pnl([(80.0, 78.0, 100)]) == pytest.approx(200.0)
