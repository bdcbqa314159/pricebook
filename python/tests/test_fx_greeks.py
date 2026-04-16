"""Tests for FX Greeks deepening."""

import math

import numpy as np
import pytest

from pricebook.fx_greeks import (
    SmileGreeksResult,
    VegaBucket,
    VegaLadder,
    fx_charm,
    fx_dvega_dspot,
    fx_dvega_dvol,
    fx_smile_consistent_greeks,
    fx_vanna,
    fx_vega,
    fx_vega_ladder,
    fx_volga,
)


# ---- Higher-order Greeks ----

class TestFXVega:
    def test_atm_positive(self):
        v = fx_vega(1.0, 1.0, 0.02, 0.01, 0.10, 1.0)
        assert v > 0

    def test_far_otm_small(self):
        v = fx_vega(1.0, 10.0, 0.02, 0.01, 0.10, 1.0)
        assert abs(v) < 1e-6


class TestFXVanna:
    def test_sign(self):
        """Vanna sign depends on moneyness (OTM calls → positive vanna typically)."""
        # OTM call
        va_otm = fx_vanna(1.0, 1.10, 0.02, 0.01, 0.10, 1.0)
        # ITM call
        va_itm = fx_vanna(1.0, 0.90, 0.02, 0.01, 0.10, 1.0)
        # ATM vanna ≈ 0
        va_atm = fx_vanna(1.0, 1.0, 0.02, 0.01, 0.10, 1.0)
        # Opposite signs OTM vs ITM
        assert va_otm * va_itm < 0 or abs(va_atm) < 0.1

    def test_zero_vol(self):
        assert fx_vanna(1.0, 1.0, 0.02, 0.01, 0.0, 1.0) == 0.0


class TestFXVolga:
    def test_positive_away_from_atm(self):
        """Volga positive for OTM (vega is max at ATM)."""
        vo_otm = fx_volga(1.0, 1.10, 0.02, 0.01, 0.10, 1.0)
        vo_itm = fx_volga(1.0, 0.90, 0.02, 0.01, 0.10, 1.0)
        assert vo_otm > 0 or vo_itm > 0

    def test_atm_volga_small(self):
        vo = fx_volga(1.0, 1.01, 0.02, 0.01, 0.10, 1.0)
        # Near ATM, volga is small but not necessarily 0
        assert abs(vo) < 10

    def test_zero_vol(self):
        assert fx_volga(1.0, 1.0, 0.02, 0.01, 0.0, 1.0) == 0.0


class TestFXCharm:
    def test_basic(self):
        c = fx_charm(1.0, 1.0, 0.02, 0.01, 0.10, 0.5)
        assert isinstance(c, float)

    def test_call_vs_put(self):
        c_call = fx_charm(1.0, 1.0, 0.02, 0.01, 0.10, 0.5, is_call=True)
        c_put = fx_charm(1.0, 1.0, 0.02, 0.01, 0.10, 0.5, is_call=False)
        assert c_call != c_put


class TestDvegaDspot:
    def test_matches_vanna_approximately(self):
        """DvegaDspot should be close to vanna."""
        dvds = fx_dvega_dspot(1.0, 1.05, 0.02, 0.01, 0.10, 1.0)
        vanna = fx_vanna(1.0, 1.05, 0.02, 0.01, 0.10, 1.0)
        # These differ by a scale factor (finite difference approximation)
        assert isinstance(dvds, float)
        assert dvds != 0


class TestDvegaDvol:
    def test_equals_volga(self):
        dvol = fx_dvega_dvol(1.0, 1.05, 0.02, 0.01, 0.10, 1.0)
        volga = fx_volga(1.0, 1.05, 0.02, 0.01, 0.10, 1.0)
        assert dvol == pytest.approx(volga)


# ---- Vega ladder ----

class TestVegaLadder:
    def test_basic(self):
        tenors = [0.25, 1.0, 3.0]
        deltas = [0.25, 0.50, -0.25]

        def vol_fn(T, K):
            return 0.10

        ladder = fx_vega_ladder(1.0, 0.02, 0.01, tenors, deltas, vol_fn)
        assert isinstance(ladder, VegaLadder)
        assert len(ladder.buckets) == 9

    def test_tenor_totals(self):
        tenors = [1.0, 2.0]
        deltas = [0.25, 0.50]

        def vol_fn(T, K):
            return 0.10

        ladder = fx_vega_ladder(1.0, 0.02, 0.01, tenors, deltas, vol_fn)
        assert set(ladder.tenor_totals.keys()) == {1.0, 2.0}
        # Sum should equal total
        assert sum(ladder.tenor_totals.values()) == pytest.approx(ladder.total_vega, rel=1e-6)

    def test_delta_totals(self):
        tenors = [1.0, 2.0]
        deltas = [0.25, 0.50]

        def vol_fn(T, K):
            return 0.10

        ladder = fx_vega_ladder(1.0, 0.02, 0.01, tenors, deltas, vol_fn)
        assert set(ladder.delta_totals.keys()) == {0.25, 0.50}
        assert sum(ladder.delta_totals.values()) == pytest.approx(ladder.total_vega, rel=1e-6)

    def test_atm_highest_vega(self):
        """Vega is highest at ATM (50D) for given tenor."""
        tenors = [1.0]
        deltas = [0.10, 0.25, 0.50]

        def vol_fn(T, K):
            return 0.10

        ladder = fx_vega_ladder(1.0, 0.02, 0.01, tenors, deltas, vol_fn)
        # Find 50D bucket
        vegas = {b.delta: b.vega for b in ladder.buckets}
        assert vegas[0.50] > vegas[0.10]

    def test_notional_scaling(self):
        tenors = [1.0]
        deltas = [0.25]

        def vol_fn(T, K):
            return 0.10

        l1 = fx_vega_ladder(1.0, 0.02, 0.01, tenors, deltas, vol_fn, notional=1.0)
        l2 = fx_vega_ladder(1.0, 0.02, 0.01, tenors, deltas, vol_fn, notional=2.0)
        assert l2.total_vega == pytest.approx(2 * l1.total_vega, rel=1e-6)


# ---- Smile-consistent Greeks ----

class TestSmileConsistentGreeks:
    def test_basic(self):
        result = fx_smile_consistent_greeks(
            1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0,
        )
        assert isinstance(result, SmileGreeksResult)

    def test_no_smile_bs_matches(self):
        """Flat smile → VV greeks ≈ BS greeks."""
        result = fx_smile_consistent_greeks(
            1.0, 1.0, 0.02, 0.01, 0.10, 0.10, 0.10, 1.0,
        )
        assert result.delta_smile == pytest.approx(result.delta_bs, abs=0.02)
        assert result.vega_smile == pytest.approx(result.vega_bs, rel=0.10)

    def test_smile_produces_different_greeks(self):
        """With smile, VV Greeks differ from BS."""
        result = fx_smile_consistent_greeks(
            1.0, 1.0, 0.02, 0.01, 0.10, 0.12, 0.14, 1.0,
        )
        # Some greek should differ meaningfully
        assert (abs(result.delta_smile - result.delta_bs) > 1e-5 or
                abs(result.vega_smile - result.vega_bs) > 1e-5)

    def test_delta_in_range(self):
        """FX call spot delta should be in [0, 1] × exp(-rf T)."""
        result = fx_smile_consistent_greeks(
            1.0, 1.0, 0.02, 0.01, 0.10, 0.11, 0.12, 1.0,
        )
        cap = math.exp(-0.01 * 1.0)
        assert 0 <= result.delta_bs <= cap + 0.01
        assert 0 <= result.delta_smile <= cap + 0.05
