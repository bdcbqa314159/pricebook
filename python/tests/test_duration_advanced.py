"""Tests for advanced duration hedging."""
import math
import numpy as np
import pytest
from pricebook.duration_advanced import (
    key_rate_immunise, barbell_bullet_analysis, ldi_cashflow_match,
    cross_currency_duration_hedge,
)

class TestKeyRateImmunise:
    def test_basic(self):
        # 3 pillars, 2 instruments
        portfolio = [100, 50, 30]
        instruments = [[-80, 0, 0], [0, -50, -30]]
        r = key_rate_immunise(portfolio, instruments)
        assert r.n_instruments == 2
        assert r.max_residual < 50  # not perfect with 2 for 3

    def test_exact_solution(self):
        """3 instruments for 3 pillars → exact."""
        portfolio = [100, 50, 30]
        instruments = [[-100, 0, 0], [0, -50, 0], [0, 0, -30]]
        r = key_rate_immunise(portfolio, instruments)
        assert r.max_residual < 1e-6
        np.testing.assert_allclose(r.hedge_weights, [1, 1, 1], atol=1e-6)

    def test_residual_krd(self):
        portfolio = [100, 0]
        instruments = [[-80, 0]]
        r = key_rate_immunise(portfolio, instruments)
        # w ≈ 1.25, residual = 100 - 100 = 0 for pillar 1
        assert abs(r.residual_krd[0]) < 1e-6

class TestBarbellBullet:
    def test_convexity_pickup_positive(self):
        """Barbell typically has higher convexity than bullet."""
        r = barbell_bullet_analysis(
            short_dv01=30, long_dv01=120, short_convexity=50, long_convexity=800,
            short_yield_pct=4.0, long_yield_pct=4.5,
            bullet_dv01=70, bullet_convexity=300, bullet_yield_pct=4.3,
        )
        assert r.convexity_pickup > 0

    def test_yield_pickup_positive(self):
        """Bullet typically has higher yield than barbell (negative carry)."""
        r = barbell_bullet_analysis(
            30, 120, 50, 800, 4.0, 4.5, 70, 300, 4.35,
        )
        assert r.yield_pickup_bps > 0  # bullet yields more

    def test_dv01_match(self):
        r = barbell_bullet_analysis(30, 120, 50, 800, 4.0, 4.5, 70, 300, 4.3)
        assert r.barbell_dv01 == pytest.approx(70, rel=0.01)

class TestLDICashflowMatch:
    def test_matched(self):
        liab = [(1, 100), (2, 100), (3, 100)]
        asset = [(1, 100), (2, 100), (3, 100)]
        r = ldi_cashflow_match(liab, asset, 0.04)
        assert r.surplus == pytest.approx(0.0, abs=1e-6)
        assert r.max_mismatch == 0.0

    def test_surplus(self):
        liab = [(1, 100)]
        asset = [(1, 110)]
        r = ldi_cashflow_match(liab, asset, 0.04)
        assert r.surplus > 0

    def test_deficit(self):
        liab = [(1, 200)]
        asset = [(1, 100)]
        r = ldi_cashflow_match(liab, asset, 0.04)
        assert r.surplus < 0

    def test_mismatch_per_bucket(self):
        liab = [(1, 100), (2, 200)]
        asset = [(1, 80), (2, 250)]
        r = ldi_cashflow_match(liab, asset, 0.04)
        assert r.mismatch_per_bucket[0] == pytest.approx(-20)
        assert r.mismatch_per_bucket[1] == pytest.approx(50)

class TestCrossCurrencyDuration:
    def test_basic(self):
        r = cross_currency_duration_hedge(100, 50, fx_rate=0.85)
        assert r.hedge_ratio < 0  # short foreign to hedge long domestic

    def test_fx_adjusted(self):
        r = cross_currency_duration_hedge(100, 50, 0.85, fx_vol=0.10, correlation=0.3)
        # Adjustment should differ from raw
        assert r.fx_adjusted_hedge != r.hedge_ratio

    def test_zero_foreign_dv01(self):
        r = cross_currency_duration_hedge(100, 0, 0.85)
        assert r.hedge_ratio == 0.0
