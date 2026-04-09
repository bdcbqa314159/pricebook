"""Tests for local volatility: Dupire surface and MC pricing."""

import math
import pytest
import numpy as np

from pricebook.local_vol import (
    LocalVolSurface, local_vol_mc, local_vol_mc_european,
)
from pricebook.black76 import black76_price, OptionType


SPOT = 100.0
RATE = 0.05
VOL = 0.20


def _flat_implied_surface():
    """Flat implied vol surface at 20%."""
    strikes = [80, 90, 95, 100, 105, 110, 120]
    times = [0.25, 0.5, 1.0, 2.0]
    vols = [[VOL] * len(strikes) for _ in times]
    return strikes, times, vols


def _skewed_implied_surface():
    """Implied vol surface with skew (higher vol at low strikes)."""
    strikes = [80, 90, 95, 100, 105, 110, 120]
    times = [0.25, 0.5, 1.0, 2.0]
    vols = []
    for _ in times:
        row = [0.28, 0.24, 0.22, 0.20, 0.19, 0.18, 0.17]
        vols.append(row)
    return strikes, times, vols


# ---- Local vol surface ----

class TestLocalVolSurface:
    def test_flat_vol_surface(self):
        """Flat implied vol → local vol ≈ implied vol."""
        strikes, times, vols = _flat_implied_surface()
        lv = LocalVolSurface.from_implied_vols(SPOT, RATE, strikes, times, vols)
        # At ATM, local vol should be close to implied vol
        v = lv.vol(100, 1.0)
        assert v == pytest.approx(VOL, rel=0.3)

    def test_interpolation(self):
        strikes, times, vols = _flat_implied_surface()
        lv = LocalVolSurface.from_implied_vols(SPOT, RATE, strikes, times, vols)
        # Should interpolate smoothly
        v1 = lv.vol(95, 0.5)
        v2 = lv.vol(105, 0.5)
        assert v1 > 0
        assert v2 > 0

    def test_skewed_surface(self):
        """Skewed implied vol → local vol varies by strike."""
        strikes, times, vols = _skewed_implied_surface()
        lv = LocalVolSurface.from_implied_vols(SPOT, RATE, strikes, times, vols)
        v_low = lv.vol(85, 1.0)
        v_high = lv.vol(115, 1.0)
        # Both should be positive
        assert v_low > 0
        assert v_high > 0

    def test_positive_vols(self):
        strikes, times, vols = _flat_implied_surface()
        lv = LocalVolSurface.from_implied_vols(SPOT, RATE, strikes, times, vols)
        for t in times:
            for k in strikes:
                assert lv.vol(k, t) > 0

    def test_direct_construction(self):
        lv = LocalVolSurface(
            np.array([90, 100, 110]),
            np.array([0.5, 1.0]),
            np.array([[0.20, 0.19, 0.18], [0.21, 0.20, 0.19]]),
        )
        assert lv.vol(100, 0.5) == pytest.approx(0.19)
        assert lv.vol(100, 1.0) == pytest.approx(0.20)


# ---- Local vol MC ----

class TestLocalVolMC:
    def test_terminal_spots_positive(self):
        lv = LocalVolSurface(
            np.array([80, 100, 120]),
            np.array([0.5, 1.0]),
            np.array([[VOL] * 3, [VOL] * 3]),
        )
        S_T = local_vol_mc(SPOT, RATE, lv, 1.0, n_steps=50, n_paths=1000)
        assert all(S_T > 0)

    def test_mean_approx_forward(self):
        """E[S_T] ≈ S × exp(rT) under risk-neutral."""
        lv = LocalVolSurface(
            np.array([80, 100, 120]),
            np.array([0.5, 1.0]),
            np.array([[VOL] * 3, [VOL] * 3]),
        )
        S_T = local_vol_mc(SPOT, RATE, lv, 1.0, n_steps=100, n_paths=50_000)
        fwd = SPOT * math.exp(RATE * 1.0)
        assert S_T.mean() == pytest.approx(fwd, rel=0.02)

    def test_european_call_vs_bs(self):
        """Local vol MC call ≈ BS call for flat vol."""
        lv = LocalVolSurface(
            np.array([80, 90, 100, 110, 120]),
            np.array([0.25, 0.5, 1.0]),
            np.array([[VOL] * 5] * 3),
        )
        mc_price = local_vol_mc_european(
            SPOT, RATE, lv, 100, 1.0,
            n_steps=100, n_paths=100_000,
        )
        bs_price = black76_price(
            SPOT * math.exp(RATE), 100, VOL, 1.0,
            math.exp(-RATE), OptionType.CALL,
        )
        assert mc_price == pytest.approx(bs_price, rel=0.05)

    def test_put_positive(self):
        lv = LocalVolSurface(
            np.array([80, 100, 120]),
            np.array([0.5, 1.0]),
            np.array([[VOL] * 3, [VOL] * 3]),
        )
        put = local_vol_mc_european(
            SPOT, RATE, lv, 100, 1.0,
            option_type=OptionType.PUT, n_paths=10_000,
        )
        assert put > 0

    def test_deterministic(self):
        lv = LocalVolSurface(
            np.array([80, 100, 120]),
            np.array([1.0]),
            np.array([[VOL] * 3]),
        )
        p1 = local_vol_mc_european(SPOT, RATE, lv, 100, 1.0, n_paths=5000, seed=123)
        p2 = local_vol_mc_european(SPOT, RATE, lv, 100, 1.0, n_paths=5000, seed=123)
        assert p1 == p2
