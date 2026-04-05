"""Tests for FX barrier options and Vanna-Volga."""

import pytest

from pricebook.fx_barrier import (
    fx_barrier_pde,
    vanna_volga_barrier,
)
from pricebook.black76 import OptionType


S, K, RD, RF, VOL, T = 1.10, 1.10, 0.05, 0.03, 0.10, 1.0


class TestFXBarrierPDE:
    def test_knockout_positive(self):
        price = fx_barrier_pde(S, K, 0.95, RD, RF, VOL, T,
                               is_up=False, is_knock_in=False)
        assert price > 0

    def test_knockout_leq_vanilla(self):
        from pricebook.fx_option import fx_option_price
        vanilla = fx_option_price(S, K, RD, RF, VOL, T)
        ko = fx_barrier_pde(S, K, 0.90, RD, RF, VOL, T,
                            is_up=False, is_knock_in=False)
        assert ko <= vanilla * 1.05

    def test_knockin_positive(self):
        price = fx_barrier_pde(S, K, 0.95, RD, RF, VOL, T,
                               is_up=False, is_knock_in=True)
        assert price >= 0

    def test_in_out_parity(self):
        ko = fx_barrier_pde(S, K, 0.95, RD, RF, VOL, T,
                            is_up=False, is_knock_in=False)
        ki = fx_barrier_pde(S, K, 0.95, RD, RF, VOL, T,
                            is_up=False, is_knock_in=True)
        from pricebook.fx_option import fx_option_price
        vanilla = fx_option_price(S, K, RD, RF, VOL, T)
        assert ko + ki == pytest.approx(vanilla, rel=0.05)

    def test_up_barrier(self):
        price = fx_barrier_pde(S, K, 1.25, RD, RF, VOL, T,
                               is_up=True, is_knock_in=False)
        assert price >= 0

    def test_put(self):
        price = fx_barrier_pde(S, K, 0.95, RD, RF, VOL, T,
                               is_up=False, is_knock_in=False,
                               option_type=OptionType.PUT)
        assert price >= 0


class TestVannaVolga:
    def test_flat_smile_matches_pde(self):
        """With flat smile (RR=0, BF=0), VV = PDE."""
        pde = fx_barrier_pde(S, K, 0.95, RD, RF, VOL, T,
                             is_up=False, is_knock_in=False)
        vv = vanna_volga_barrier(S, K, 0.95, RD, RF, VOL, VOL, VOL, T,
                                 is_up=False, is_knock_in=False)
        assert vv == pytest.approx(pde, rel=0.05)

    def test_smile_adjusts_price(self):
        """Non-flat smile should give a different price."""
        pde = fx_barrier_pde(S, K, 0.95, RD, RF, VOL, T,
                             is_up=False, is_knock_in=False)
        vv = vanna_volga_barrier(S, K, 0.95, RD, RF,
                                 vol_atm=0.10, vol_25d_call=0.11, vol_25d_put=0.12,
                                 T=T, is_up=False, is_knock_in=False)
        # With smile, price should differ from flat
        assert vv != pytest.approx(pde, abs=0.0001)

    def test_positive(self):
        vv = vanna_volga_barrier(S, K, 0.95, RD, RF,
                                 vol_atm=0.10, vol_25d_call=0.11, vol_25d_put=0.12,
                                 T=T, is_up=False, is_knock_in=False)
        assert vv >= 0

    def test_knockin(self):
        vv = vanna_volga_barrier(S, K, 0.95, RD, RF,
                                 vol_atm=0.10, vol_25d_call=0.11, vol_25d_put=0.12,
                                 T=T, is_up=False, is_knock_in=True)
        assert vv >= 0
