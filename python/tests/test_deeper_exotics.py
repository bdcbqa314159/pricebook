"""Tests for deeper exotics (DE1-DE5)."""

import numpy as np
import pytest


# ---- DE1: PRDC ----

class TestPRDC:
    def test_prdc_positive(self):
        from pricebook.prdc import prdc_price
        result = prdc_price(
            spot_fx=110.0, rate_dom=0.04, rate_for=0.01,
            vol_fx=0.10, vol_dom=0.005, vol_for=0.005,
            rho_fx_dom=0.2, rho_fx_for=-0.1, rho_dom_for=0.3,
            notional=100.0, fixed_coupon=0.05,
            fx_participation=1.5, fx_strike=110.0,
            T=5.0, n_coupons=5, n_paths=5000, seed=42,
        )
        assert result.price > 0

    def test_callable_has_call_value(self):
        from pricebook.prdc import callable_prdc
        result = callable_prdc(
            spot_fx=110.0, rate_dom=0.04, rate_for=0.01,
            vol_fx=0.10, vol_dom=0.005, vol_for=0.005,
            rho_fx_dom=0.2, rho_fx_for=-0.1, rho_dom_for=0.3,
            notional=100.0, fixed_coupon=0.05,
            fx_participation=1.5, fx_strike=110.0,
            T=5.0, n_coupons=5, n_paths=5000, seed=42,
        )
        assert result.price > 0
        assert result.price <= result.price_no_call + 5.0  # callable ≤ non-callable


# ---- DE3: Multi-asset correlation ----

class TestMultiAssetCorrelation:
    def test_basket_option(self):
        from pricebook.equity_basket import equity_basket_mc
        result = equity_basket_mc(
            spots=[100.0, 100.0],
            weights=[0.5, 0.5],
            strike=100.0,
            rate=0.04,
            dividend_yields=[0.02, 0.02],
            vols=[0.20, 0.25],
            correlations=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=1.0, n_paths=10000, seed=42,
        )
        assert result.price > 0

    def test_worst_of(self):
        from pricebook.equity_structured import worst_of_autocallable
        result = worst_of_autocallable(
            spots=[100.0, 100.0],
            autocall_barrier_pct=1.0, coupon=5.0,
            rate=0.04, dividend_yields=[0.02, 0.02],
            vols=[0.20, 0.25],
            correlations=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=3.0, observation_dates=[1.0, 2.0, 3.0],
            n_paths=5000, seed=42,
        )
        assert result.price > 0


# ---- DE4: Callable bond ----

class TestCallableBond:
    def test_callable_bond(self):
        from pricebook.hull_white import HullWhite
        from pricebook.callable_bond import callable_bond_price
        from tests.conftest import make_flat_curve
        from datetime import date
        curve = make_flat_curve(date(2026, 4, 21), 0.04)
        hw = HullWhite(a=0.1, sigma=0.01, curve=curve)
        price = callable_bond_price(hw, 0.05, 10.0, n_steps=50)
        assert price > 0


# ---- DE5: Autocall deepening ----

class TestAutocall:
    def test_phoenix_autocall(self):
        from pricebook.equity_structured import equity_autocallable
        result = equity_autocallable(
            spot=100.0, autocall_barrier=110.0,
            coupon_barrier=90.0, protection_barrier=80.0,
            coupon=3.0, rate=0.04, dividend_yield=0.02, vol=0.20,
            T=3.0, observation_dates=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            notional=100.0, has_memory=True,
            n_paths=10000, seed=42,
        )
        assert result.price > 0
        assert 0 <= result.autocall_probability <= 1

    def test_higher_coupon_barrier_lower_price(self):
        from pricebook.equity_structured import equity_autocallable
        low = equity_autocallable(
            spot=100, autocall_barrier=120, coupon_barrier=80,
            protection_barrier=70, coupon=5,
            rate=0.04, dividend_yield=0.02, vol=0.20,
            T=2.0, observation_dates=[0.5, 1.0, 1.5, 2.0],
            n_paths=5000, seed=42,
        )
        high = equity_autocallable(
            spot=100, autocall_barrier=120, coupon_barrier=100,
            protection_barrier=70, coupon=5,
            rate=0.04, dividend_yield=0.02, vol=0.20,
            T=2.0, observation_dates=[0.5, 1.0, 1.5, 2.0],
            n_paths=5000, seed=42,
        )
        assert high.price <= low.price + 3.0  # within MC noise
