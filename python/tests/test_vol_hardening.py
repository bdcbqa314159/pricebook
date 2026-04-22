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


# ---- VH7: FX vol surface bumped ----

class TestFXVolSurfaceBumped:
    def test_fx_surface_builds(self):
        """FX vol surface builds from ATM/RR/BF quotes."""
        from pricebook.fx_vol_surface import FXVolSurface, FXVolQuote
        quotes = [
            FXVolQuote(date(2027, 4, 21), 0.08, 0.01, 0.005),
            FXVolQuote(date(2028, 4, 21), 0.09, 0.012, 0.006),
        ]
        surface = FXVolSurface(1.10, 0.04, 0.03, quotes,
                               reference_date=date(2026, 4, 21))
        v = surface.vol(date(2027, 4, 21), 1.10)
        assert v > 0

    def test_fx_surface_bumped(self):
        from pricebook.fx_vol_surface import FXVolSurface, FXVolQuote
        quotes = [FXVolQuote(date(2027, 4, 21), 0.08, 0.01, 0.005)]
        surface = FXVolSurface(1.10, 0.04, 0.03, quotes,
                               reference_date=date(2026, 4, 21))
        bumped = surface.bumped(0.01)
        v_base = surface.vol(date(2027, 4, 21), 1.10)
        v_bump = bumped.vol(date(2027, 4, 21), 1.10)
        assert v_bump > v_base


# ---- VH8: Swaption vol surface bumped ----

class TestSwaptionVolBumped:
    def test_swaption_vol_bumped(self):
        from pricebook.swaption_vol import SwaptionVolSurface
        svs = SwaptionVolSurface(
            date(2026, 4, 21),
            [date(2027, 4, 21), date(2028, 4, 21)],
            [5.0, 10.0],
            [[0.20, 0.19], [0.21, 0.20]],
        )
        bumped = svs.bumped(0.01)
        v_base = svs.vol_expiry_tenor(date(2027, 4, 21), 5.0)
        v_bump = bumped.vol_expiry_tenor(date(2027, 4, 21), 5.0)
        assert v_bump == pytest.approx(v_base + 0.01, rel=1e-4)


# ---- VH11: SABR edge cases ----

class TestSABREdgeCases:
    def test_atm_vol_positive(self):
        from pricebook.sabr import sabr_implied_vol
        # Standard params
        v = sabr_implied_vol(0.04, 0.04, 1.0, 0.03, 0.5, -0.3, 0.4)
        assert v > 0

    def test_beta_zero_normal(self):
        """β=0: normal SABR should give positive vol."""
        from pricebook.sabr import sabr_implied_vol
        v = sabr_implied_vol(0.04, 0.04, 1.0, 0.03, 0.0, -0.3, 0.4)
        assert v > 0

    def test_beta_one_lognormal(self):
        """β=1: lognormal SABR."""
        from pricebook.sabr import sabr_implied_vol
        v = sabr_implied_vol(0.04, 0.04, 1.0, 0.03, 1.0, -0.3, 0.4)
        assert v > 0

    def test_high_nu_no_crash(self):
        """High vol-of-vol should not crash."""
        from pricebook.sabr import sabr_implied_vol
        v = sabr_implied_vol(0.04, 0.04, 1.0, 0.03, 0.5, -0.3, 1.5)
        assert v > 0

    def test_extreme_rho_no_crash(self):
        """Near-extreme correlation should not crash."""
        from pricebook.sabr import sabr_implied_vol
        v1 = sabr_implied_vol(0.04, 0.04, 1.0, 0.03, 0.5, -0.95, 0.4)
        v2 = sabr_implied_vol(0.04, 0.04, 1.0, 0.03, 0.5, 0.95, 0.4)
        assert v1 > 0
        assert v2 > 0


# ---- VH12: Heston convergence ----

class TestHestonConvergence:
    def test_standard_params(self):
        from pricebook.heston import heston_price
        # heston_price(spot, strike, rate, T, v0, kappa, theta, xi, rho)
        price = heston_price(100, 100, 0.04, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7)
        assert price > 0
        assert price < 100

    def test_high_vol_of_vol(self):
        """High ξ should still converge."""
        from pricebook.heston import heston_price
        price = heston_price(100, 100, 0.04, 1.0, 0.04, 1.5, 0.04, 1.0, -0.7)
        assert price > 0

    def test_near_zero_kappa(self):
        """Low mean reversion should still converge."""
        from pricebook.heston import heston_price
        price = heston_price(100, 100, 0.04, 1.0, 0.04, 0.01, 0.04, 0.3, -0.7)
        assert price > 0

    def test_deep_otm(self):
        """Deep OTM option should be very small but positive."""
        from pricebook.heston import heston_price
        price = heston_price(100, 200, 0.04, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7)
        assert 0 <= price < 1.0
