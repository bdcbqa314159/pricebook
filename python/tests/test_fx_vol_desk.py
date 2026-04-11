"""Tests for FX vol desk."""

import pytest

from pricebook.fx_vol_desk import (
    FXButterfly,
    FXRiskReversal,
    FXSkewSignal,
    FXStraddle,
    FXStrangle,
    FXVegaBucket,
    FXVolRV,
    fx_skew_monitor,
    fx_vega_ladder,
    fx_vol_rv,
    total_fx_vega,
)

SPOT, RD, RF, VOL, T = 1.085, 0.05, 0.03, 0.08, 0.25


class TestFXStraddle:
    def test_positive_premium(self):
        s = FXStraddle("EUR/USD", 90, 1.085)
        assert s.premium(SPOT, RD, RF, VOL, T) > 0

    def test_vega_positive(self):
        s = FXStraddle("EUR/USD", 90, 1.085)
        assert s.vega(SPOT, RD, RF, VOL, T) > 0


class TestFXStrangle:
    def test_positive_premium(self):
        s = FXStrangle("EUR/USD", 1.12, 1.05)
        assert s.premium(SPOT, RD, RF, VOL, T) > 0


class TestFXRiskReversal:
    def test_direction_flips(self):
        long = FXRiskReversal("EUR/USD", 1.12, 1.05, direction=1)
        short = FXRiskReversal("EUR/USD", 1.12, 1.05, direction=-1)
        assert long.premium(SPOT, RD, RF, VOL, VOL, T) == pytest.approx(
            -short.premium(SPOT, RD, RF, VOL, VOL, T)
        )


class TestFXButterfly:
    def test_premium_sign(self):
        bf = FXButterfly("EUR/USD", 1.085, 1.12, 1.05)
        # Strangle - straddle: negative for flat vol (OTM cheaper than ATM pair)
        prem = bf.premium(SPOT, RD, RF, VOL, VOL, VOL, T)
        assert prem < 0


class TestFXVolRV:
    def test_rich_signal(self):
        history = [0.07, 0.075, 0.08, 0.065, 0.072] * 4
        rv = fx_vol_rv("EUR/USD", 0.15, history, threshold=2.0)
        assert rv.signal == "rich"

    def test_fair(self):
        history = [0.07, 0.075, 0.08, 0.065, 0.072] * 4
        rv = fx_vol_rv("EUR/USD", 0.074, history)
        assert rv.signal == "fair"


class TestFXSkewMonitor:
    def test_extreme_skew(self):
        history = [-0.5, -0.3, -0.4, -0.6, -0.2] * 4
        sig = fx_skew_monitor("EUR/USD", 1.0, history, threshold=2.0)
        assert sig.signal == "rich"


class TestFXVegaLadder:
    def test_aggregation(self):
        positions = [
            ("EUR/USD", "1M", 100), ("EUR/USD", "1M", 50),
            ("EUR/USD", "3M", 200), ("GBP/USD", "1M", 80),
        ]
        ladder = fx_vega_ladder(positions)
        assert len(ladder) == 3
        eur_1m = next(b for b in ladder if b.pair == "EUR/USD" and b.expiry == "1M")
        assert eur_1m.vega == pytest.approx(150)

    def test_total_vega(self):
        """Test: vega ladder sums to total vega."""
        positions = [("EUR/USD", "1M", 100), ("GBP/USD", "3M", 200)]
        ladder = fx_vega_ladder(positions)
        assert total_fx_vega(ladder) == pytest.approx(300)
