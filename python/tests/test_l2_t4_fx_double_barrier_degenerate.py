"""Regression for L2 Wave-2 audit — `fx_double_barrier_option` returned
``vanilla`` unconditionally at vol=0/T>0, assuming no barrier breach.

At vol=0 the spot path is deterministic ``S_t = spot·exp((rd-rf)·t)``,
monotonic from spot to spot_T = forward.  If the forward exits the
corridor [L, U]:

- Knock-out is worthless (path will breach).
- Knock-in receives full vanilla (path will activate).

If the forward stays inside [L, U]:

- Knock-out receives full vanilla (path never breaches).
- Knock-in is worthless.

Pre-fix systematically over-priced KO / under-priced KI when the
deterministic forward drifted out of the corridor.
"""

from __future__ import annotations

import math

import pytest

from pricebook.fx.fx_exotic_extensions import fx_double_barrier_option
from pricebook.fx.fx_option import fx_option_price
from pricebook.models.black76 import OptionType


def _vanilla_price(spot, strike, rd, rf, vol, T):
    return fx_option_price(spot, strike, rd, rf, vol, T, OptionType.CALL)


class TestFxDoubleBarrierVolZeroKnockOut:
    def test_forward_inside_corridor_ko_pays_vanilla(self):
        # spot=100, fwd = 100·exp(0.02) ≈ 102, corridor [90, 110] → fwd inside.
        res = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=90.0, upper_barrier=110.0,
            r_d=0.05, r_f=0.03, vol=0.0, T=1.0,
            option_type="call", knock_type="out",
        )
        van = _vanilla_price(100.0, 100.0, 0.05, 0.03, 0.0, 1.0)
        assert res.price == pytest.approx(van, abs=1e-12)
        assert res.barrier_discount == pytest.approx(0.0, abs=1e-12)

    def test_forward_above_upper_ko_worthless(self):
        # spot=100, rd-rf=10% → fwd ≈ 110.5 ≥ U=105 → breach upper.
        # Pre-fix returned vanilla > 0; should be 0.
        res = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=85.0, upper_barrier=105.0,
            r_d=0.10, r_f=0.0, vol=0.0, T=1.0,
            option_type="call", knock_type="out",
        )
        assert res.price == pytest.approx(0.0, abs=1e-12)
        van = _vanilla_price(100.0, 100.0, 0.10, 0.0, 0.0, 1.0)
        assert res.barrier_discount == pytest.approx(van, abs=1e-12)

    def test_forward_below_lower_ko_worthless(self):
        # spot=100, rd-rf=-10% → fwd ≈ 90.5, lower=92 → breach lower.
        res = fx_double_barrier_option(
            spot=100.0, strike=95.0, lower_barrier=92.0, upper_barrier=115.0,
            r_d=0.0, r_f=0.10, vol=0.0, T=1.0,
            option_type="put", knock_type="out",
        )
        assert res.price == pytest.approx(0.0, abs=1e-12)


class TestFxDoubleBarrierVolZeroKnockIn:
    def test_forward_inside_corridor_ki_worthless(self):
        # No breach → KI never activates → 0.
        res = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=90.0, upper_barrier=110.0,
            r_d=0.05, r_f=0.03, vol=0.0, T=1.0,
            option_type="call", knock_type="in",
        )
        assert res.price == pytest.approx(0.0, abs=1e-12)

    def test_forward_above_upper_ki_pays_full_vanilla(self):
        # Breach upper → KI activates → receives full vanilla.
        res = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=85.0, upper_barrier=105.0,
            r_d=0.10, r_f=0.0, vol=0.0, T=1.0,
            option_type="call", knock_type="in",
        )
        van = _vanilla_price(100.0, 100.0, 0.10, 0.0, 0.0, 1.0)
        assert res.price == pytest.approx(van, abs=1e-12)


class TestFxDoubleBarrierTZero:
    def test_T_zero_no_path_traversal(self):
        # T=0: no time to breach (and spot was checked already in corridor).
        # KO should return vanilla (= 0 at T=0 since intrinsic is computed).
        res = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=90.0, upper_barrier=110.0,
            r_d=0.05, r_f=0.03, vol=0.2, T=0.0,
            option_type="call", knock_type="out",
        )
        van = _vanilla_price(100.0, 100.0, 0.05, 0.03, 0.2, 0.0)
        assert res.price == pytest.approx(van, abs=1e-12)


class TestParity:
    """KO + KI = vanilla at vol=0 (should always hold by construction)."""

    def test_parity_inside(self):
        ko = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=90.0, upper_barrier=110.0,
            r_d=0.05, r_f=0.03, vol=0.0, T=1.0,
            option_type="call", knock_type="out",
        )
        ki = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=90.0, upper_barrier=110.0,
            r_d=0.05, r_f=0.03, vol=0.0, T=1.0,
            option_type="call", knock_type="in",
        )
        assert ko.price + ki.price == pytest.approx(ko.vanilla_price, abs=1e-12)

    def test_parity_breach(self):
        ko = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=85.0, upper_barrier=105.0,
            r_d=0.10, r_f=0.0, vol=0.0, T=1.0,
            option_type="call", knock_type="out",
        )
        ki = fx_double_barrier_option(
            spot=100.0, strike=100.0, lower_barrier=85.0, upper_barrier=105.0,
            r_d=0.10, r_f=0.0, vol=0.0, T=1.0,
            option_type="call", knock_type="in",
        )
        assert ko.price + ki.price == pytest.approx(ko.vanilla_price, abs=1e-12)
