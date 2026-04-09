"""Tests for Stochastic Local Volatility (SLV)."""

import math
import pytest
import numpy as np

from pricebook.slv import SLVModel, HestonParams, slv_mc, slv_mc_european
from pricebook.local_vol import LocalVolSurface, local_vol_mc_european
from pricebook.black76 import OptionType


SPOT = 100.0
RATE = 0.05
VOL = 0.20


def _flat_lv():
    return LocalVolSurface(
        np.array([80, 90, 100, 110, 120]),
        np.array([0.25, 0.5, 1.0]),
        np.array([[VOL] * 5] * 3),
    )


def _heston():
    return HestonParams(v0=VOL ** 2, kappa=2.0, theta=VOL ** 2, xi=0.3, rho=-0.7)


# ---- SLV model ----

class TestSLVModel:
    def test_pure_local_vol(self):
        """mixing=1 → leverage = σ_loc / √v."""
        model = SLVModel(_flat_lv(), _heston(), mixing=1.0)
        L = model.leverage(100, 0.5, VOL ** 2)
        # σ_loc = 0.20, √v = 0.20 → L ≈ 1.0
        assert L == pytest.approx(1.0, rel=0.1)

    def test_pure_heston(self):
        """mixing=0 → leverage = 1."""
        model = SLVModel(_flat_lv(), _heston(), mixing=0.0)
        L = model.leverage(100, 0.5, VOL ** 2)
        assert L == 1.0

    def test_mixed(self):
        model = SLVModel(_flat_lv(), _heston(), mixing=0.5)
        L = model.leverage(100, 0.5, VOL ** 2)
        assert L > 0

    def test_clamps_mixing(self):
        model = SLVModel(_flat_lv(), _heston(), mixing=1.5)
        assert model.mixing == 1.0
        model2 = SLVModel(_flat_lv(), _heston(), mixing=-0.5)
        assert model2.mixing == 0.0


# ---- SLV MC ----

class TestSLVMC:
    def test_terminal_spots_positive(self):
        model = SLVModel(_flat_lv(), _heston(), mixing=0.5)
        S_T = slv_mc(SPOT, RATE, model, 1.0, n_steps=50, n_paths=1000)
        assert all(S_T > 0)

    def test_mean_approx_forward(self):
        model = SLVModel(_flat_lv(), _heston(), mixing=0.5)
        S_T = slv_mc(SPOT, RATE, model, 1.0, n_steps=50, n_paths=50_000)
        fwd = SPOT * math.exp(RATE * 1.0)
        assert S_T.mean() == pytest.approx(fwd, rel=0.03)

    def test_european_call_positive(self):
        model = SLVModel(_flat_lv(), _heston(), mixing=0.5)
        price = slv_mc_european(SPOT, RATE, model, 100, 1.0, n_paths=10_000)
        assert price > 0

    def test_pure_lv_matches_lv_mc(self):
        """mixing=1 SLV should give similar prices to pure local vol MC."""
        lv = _flat_lv()
        model = SLVModel(lv, _heston(), mixing=1.0)
        slv_price = slv_mc_european(SPOT, RATE, model, 100, 1.0, n_paths=50_000, seed=42)
        lv_price = local_vol_mc_european(SPOT, RATE, lv, 100, 1.0, n_paths=50_000, seed=42)
        # Won't be identical (different random structure), but should be in same ballpark
        assert slv_price == pytest.approx(lv_price, rel=0.15)

    def test_deterministic(self):
        model = SLVModel(_flat_lv(), _heston(), mixing=0.5)
        p1 = slv_mc_european(SPOT, RATE, model, 100, 1.0, n_paths=5000, seed=99)
        p2 = slv_mc_european(SPOT, RATE, model, 100, 1.0, n_paths=5000, seed=99)
        assert p1 == p2

    def test_put_positive(self):
        model = SLVModel(_flat_lv(), _heston(), mixing=0.5)
        put = slv_mc_european(SPOT, RATE, model, 100, 1.0,
                             option_type=OptionType.PUT, n_paths=10_000)
        assert put > 0
