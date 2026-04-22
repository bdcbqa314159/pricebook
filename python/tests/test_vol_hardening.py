"""Tests for vol infrastructure hardening (VH5-VH12)."""

import math
from datetime import date

import numpy as np
import pytest

from pricebook.black76 import black76_price, OptionType
from pricebook.greeks import Greeks, bump_greeks
from pricebook.variance_swap import fair_variance_from_vols, variance_swap_pv
from pricebook.vol_surface import (
    FlatVol, VolTermStructure,
    check_calendar_arbitrage, check_butterfly_arbitrage, validate_vol_surface,
)
from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike


# ---- VH5: VolSurface.bumped() ----

class TestFlatVolBumped:
    def test_bump_up(self):
        fv = FlatVol(0.20)
        bumped = fv.bumped(0.01)
        assert bumped.vol() == pytest.approx(0.21)

    def test_bump_down(self):
        fv = FlatVol(0.20)
        bumped = fv.bumped(-0.05)
        assert bumped.vol() == pytest.approx(0.15)

    def test_bump_floor_at_zero(self):
        fv = FlatVol(0.02)
        bumped = fv.bumped(-0.10)
        assert bumped.vol() == 0.0


class TestVolTermStructureBumped:
    def test_bump_shifts_all(self):
        ref = date(2026, 4, 21)
        vts = VolTermStructure(ref, [date(2027, 4, 21), date(2028, 4, 21)], [0.20, 0.22])
        bumped = vts.bumped(0.01)
        assert bumped.vol(date(2027, 4, 21)) == pytest.approx(0.21, rel=1e-4)
        assert bumped.vol(date(2028, 4, 21)) == pytest.approx(0.23, rel=1e-4)


class TestVolSmileBumped:
    def test_bump_shifts_all(self):
        smile = VolSmile([90, 95, 100, 105, 110], [0.25, 0.22, 0.20, 0.22, 0.25])
        bumped = smile.bumped(0.02)
        assert bumped.vol(100) == pytest.approx(0.22)
        assert bumped.vol(90) == pytest.approx(0.27)


class TestVolSurfaceStrikeBumped:
    def test_bump_shifts_surface(self):
        ref = date(2026, 4, 21)
        smile1 = VolSmile([90, 100, 110], [0.25, 0.20, 0.25])
        smile2 = VolSmile([90, 100, 110], [0.27, 0.22, 0.27])
        surface = VolSurfaceStrike(ref, [date(2027, 4, 21), date(2028, 4, 21)],
                                   [smile1, smile2])
        bumped = surface.bumped(0.01)
        assert bumped.vol(date(2027, 4, 21), 100) == pytest.approx(0.21, rel=1e-4)


# ---- VH6: Arbitrage checks ----

class TestCalendarArbitrage:
    def test_no_violation_monotone(self):
        """Increasing total variance → no violation."""
        violations = check_calendar_arbitrage([0.5, 1.0, 2.0], [0.20, 0.20, 0.20])
        assert len(violations) == 0

    def test_violation_decreasing_total_var(self):
        """Decreasing total variance → calendar arbitrage."""
        # σ²T: 0.04*0.5=0.02, 0.01*1.0=0.01 → decreasing
        violations = check_calendar_arbitrage([0.5, 1.0], [0.20, 0.10])
        assert len(violations) == 1

    def test_flat_vol_no_violation(self):
        violations = check_calendar_arbitrage([0.25, 0.5, 1.0, 2.0], [0.20] * 4)
        assert len(violations) == 0


class TestButterflyArbitrage:
    def test_convex_prices_no_violation(self):
        """Convex call prices → no butterfly arbitrage."""
        # Call prices should be convex in strike
        strikes = [90, 95, 100, 105, 110]
        prices = [12.0, 8.0, 5.0, 3.0, 2.0]  # convex
        violations = check_butterfly_arbitrage(strikes, prices)
        assert len(violations) == 0

    def test_non_convex_violation(self):
        """Non-convex call prices → butterfly arbitrage."""
        strikes = [90, 95, 100, 105, 110]
        prices = [12.0, 8.0, 5.0, 6.0, 2.0]  # bump at 105 → non-convex
        violations = check_butterfly_arbitrage(strikes, prices)
        assert len(violations) > 0


class TestValidateVolSurface:
    def test_clean_surface(self):
        result = validate_vol_surface([0.5, 1.0, 2.0], [0.20, 0.20, 0.20])
        assert result.is_arbitrage_free
        assert result.total_variance_monotone

    def test_dirty_surface(self):
        result = validate_vol_surface([0.5, 1.0], [0.20, 0.10])
        assert not result.is_arbitrage_free


# ---- VH9: Unified Greeks ----

class TestGreeks:
    def test_dataclass(self):
        g = Greeks(price=5.0, delta=0.55, gamma=0.03, vega=15.0, theta=-0.05, rho=0.12)
        assert g.price == 5.0
        assert g.delta == 0.55

    def test_bump_greeks_call(self):
        """bump_greeks on Black-76 call should give reasonable Greeks."""
        def price_func(S, vol, r, T):
            F = S * math.exp(r * T)
            df = math.exp(-r * T)
            return black76_price(F, 100.0, vol, T, df, OptionType.CALL)

        g = bump_greeks(price_func, spot=100.0, vol=0.20, rate=0.04, T=1.0)
        assert g.price > 0
        assert 0.3 < g.delta < 0.8   # ATM-ish call delta
        assert g.gamma > 0
        assert g.vega > 0
        assert g.theta < 0           # time decay

    def test_bump_greeks_put(self):
        def price_func(S, vol, r, T):
            F = S * math.exp(r * T)
            df = math.exp(-r * T)
            return black76_price(F, 100.0, vol, T, df, OptionType.PUT)

        g = bump_greeks(price_func, spot=100.0, vol=0.20, rate=0.04, T=1.0)
        assert g.delta < 0  # put delta negative


# ---- VH10: Variance swap ----

class TestVarianceSwap:
    def test_fair_variance_flat_smile(self):
        """With flat vol, fair variance ≈ σ²."""
        F = 100.0
        T = 1.0
        df = 0.96
        vol = 0.20
        # Generate strikes around ATM
        strikes = np.linspace(70, 130, 61)
        vols = [vol] * len(strikes)
        result = fair_variance_from_vols(F, df, T, strikes, vols)
        # Fair variance should be close to σ² = 0.04
        assert result.fair_variance == pytest.approx(vol ** 2, rel=0.05)
        assert result.fair_vol == pytest.approx(vol, rel=0.05)

    def test_skewed_smile_higher_variance(self):
        """Skewed smile → higher fair variance than flat (convexity)."""
        F = 100.0
        T = 1.0
        df = 0.96
        strikes = np.linspace(70, 130, 61)
        flat_vols = [0.20] * len(strikes)
        skewed_vols = [0.20 + 0.05 * (100 - k) / 30 for k in strikes]  # put skew
        flat_result = fair_variance_from_vols(F, df, T, strikes, flat_vols)
        skew_result = fair_variance_from_vols(F, df, T, strikes, skewed_vols)
        assert skew_result.fair_variance > flat_result.fair_variance

    def test_variance_swap_pv(self):
        """Long variance at below fair → positive PV."""
        result = variance_swap_pv(0.04, 0.035, 100_000, 0.5, 0.98)
        assert result.pv > 0

    def test_variance_swap_pv_at_fair(self):
        """At fair strike, PV = 0."""
        result = variance_swap_pv(0.04, 0.04, 100_000, 0.5, 0.98)
        assert result.pv == pytest.approx(0.0)
