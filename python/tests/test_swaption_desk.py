"""Tests for swaption desk tools."""

import pytest
from datetime import date

from pricebook.swaption_desk import (
    VolCube, SABRCell,
    straddle, strangle, risk_reversal,
    delta_hedge, vega_hedge,
    _swaption_greeks,
)
from pricebook.swaption import Swaption, SwaptionType
from pricebook.swaption_vol import SwaptionVolSurface
from pricebook.discount_curve import DiscountCurve
from pricebook.vol_surface import FlatVol
from pricebook.sabr import sabr_implied_vol


REF = date(2024, 1, 15)


def _curve(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _flat_vol(v=0.20):
    return FlatVol(v)


def _atm_surface():
    expiries = [date(2025, 1, 15), date(2026, 1, 15), date(2029, 1, 15)]
    tenors = [2.0, 5.0, 10.0]
    vols = [
        [0.22, 0.20, 0.18],
        [0.21, 0.19, 0.17],
        [0.20, 0.18, 0.16],
    ]
    return SwaptionVolSurface(REF, expiries, tenors, vols)


def _swn(strike=0.05, typ=SwaptionType.PAYER, notional=1_000_000):
    return Swaption(date(2025, 1, 15), date(2030, 1, 15), strike, typ, notional)


# ---- Vol Cube ----

class TestVolCube:
    def test_atm_vol(self):
        cube = VolCube(_atm_surface())
        v = cube.vol(date(2025, 1, 15))
        assert 0.15 < v < 0.25

    def test_set_get_sabr(self):
        cube = VolCube(_atm_surface())
        cell = SABRCell(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        cube.set_sabr(0, 1, cell)
        assert cube.get_sabr(0, 1) is cell
        assert cube.get_sabr(1, 1) is None

    def test_sabr_vol(self):
        cube = VolCube(_atm_surface())
        cell = SABRCell(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        cube.set_sabr(0, 1, cell)
        v = cube.vol_sabr(0, 1, forward=0.05, strike=0.05, T=1.0)
        expected = sabr_implied_vol(0.05, 0.05, 1.0, 0.2, 0.5, -0.3, 0.4)
        assert v == pytest.approx(expected)

    def test_sabr_smile(self):
        """OTM strikes should have different vol than ATM."""
        cube = VolCube(_atm_surface())
        cell = SABRCell(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        cube.set_sabr(0, 1, cell)
        v_atm = cube.vol_sabr(0, 1, forward=0.05, strike=0.05, T=1.0)
        v_otm = cube.vol_sabr(0, 1, forward=0.05, strike=0.06, T=1.0)
        assert v_atm != pytest.approx(v_otm, abs=1e-6)

    def test_shifted_sabr(self):
        cube = VolCube(_atm_surface())
        cell = SABRCell(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4, shift=0.03)
        cube.set_sabr(0, 1, cell)
        v = cube.vol_sabr(0, 1, forward=0.01, strike=0.01, T=1.0)
        assert v > 0

    def test_bump_parallel(self):
        cube = VolCube(_atm_surface())
        bumped = cube.bump_parallel(0.01)
        v_base = cube.vol(date(2025, 1, 15))
        v_bumped = bumped.vol(date(2025, 1, 15))
        assert v_bumped == pytest.approx(v_base + 0.01, abs=0.005)

    def test_bump_term(self):
        cube = VolCube(_atm_surface())
        bumped = cube.bump_term(0, 0.02)
        # First expiry bumped
        v0 = bumped.atm._vols[0, 1]
        v0_base = cube.atm._vols[0, 1]
        assert v0 == pytest.approx(v0_base + 0.02)
        # Second expiry unchanged
        v1 = bumped.atm._vols[1, 1]
        v1_base = cube.atm._vols[1, 1]
        assert v1 == pytest.approx(v1_base)


# ---- Greeks ----

class TestGreeks:
    def test_payer_positive_delta(self):
        g = _swaption_greeks(_swn(), _curve(), _flat_vol())
        assert g["delta"] > 0  # payer gains when rates rise

    def test_receiver_negative_delta(self):
        g = _swaption_greeks(_swn(typ=SwaptionType.RECEIVER), _curve(), _flat_vol())
        assert g["delta"] < 0

    def test_positive_vega(self):
        g = _swaption_greeks(_swn(), _curve(), _flat_vol())
        assert g["vega"] > 0  # long option → positive vega

    def test_positive_gamma(self):
        g = _swaption_greeks(_swn(), _curve(), _flat_vol())
        assert g["gamma"] > 0

    def test_negative_theta(self):
        g = _swaption_greeks(_swn(), _curve(), _flat_vol())
        assert g["theta"] < 0  # time decay


# ---- Straddle ----

class TestStraddle:
    def test_straddle_delta_smaller_than_legs(self):
        """ATM straddle has smaller delta than individual legs."""
        curve = _curve()
        vol = _flat_vol()
        fwd = _swn().forward_swap_rate(curve)
        combo = straddle(date(2025, 1, 15), date(2030, 1, 15), fwd, curve, vol)
        payer_delta = _swaption_greeks(
            Swaption(date(2025, 1, 15), date(2030, 1, 15), fwd, SwaptionType.PAYER),
            curve, vol,
        )["delta"]
        # Straddle rate delta should be smaller than single payer delta
        assert abs(combo.delta) < abs(payer_delta) * 1.5

    def test_straddle_positive_pv(self):
        combo = straddle(date(2025, 1, 15), date(2030, 1, 15), 0.05, _curve(), _flat_vol())
        assert combo.pv > 0

    def test_straddle_positive_vega(self):
        combo = straddle(date(2025, 1, 15), date(2030, 1, 15), 0.05, _curve(), _flat_vol())
        assert combo.vega > 0

    def test_straddle_positive_gamma(self):
        combo = straddle(date(2025, 1, 15), date(2030, 1, 15), 0.05, _curve(), _flat_vol())
        assert combo.gamma > 0


# ---- Strangle ----

class TestStrangle:
    def test_strangle_positive_pv(self):
        combo = strangle(
            date(2025, 1, 15), date(2030, 1, 15),
            0.04, 0.06, _curve(), _flat_vol(),
        )
        assert combo.pv > 0

    def test_strangle_cheaper_than_straddle(self):
        curve = _curve()
        vol = _flat_vol()
        s = straddle(date(2025, 1, 15), date(2030, 1, 15), 0.05, curve, vol)
        g = strangle(date(2025, 1, 15), date(2030, 1, 15), 0.04, 0.06, curve, vol)
        assert g.pv < s.pv


# ---- Risk reversal ----

class TestRiskReversal:
    def test_directional_delta(self):
        """Risk reversal has significant delta (directional bet)."""
        combo = risk_reversal(
            date(2025, 1, 15), date(2030, 1, 15),
            0.04, 0.06, _curve(), _flat_vol(),
        )
        assert abs(combo.delta) > 0

    def test_low_vega(self):
        """RR has lower vega than straddle (long + short cancel partially)."""
        curve = _curve()
        vol = _flat_vol()
        rr = risk_reversal(date(2025, 1, 15), date(2030, 1, 15), 0.04, 0.06, curve, vol)
        s = straddle(date(2025, 1, 15), date(2030, 1, 15), 0.05, curve, vol)
        assert abs(rr.vega) < abs(s.vega)


# ---- Hedging ----

class TestDeltaHedge:
    def test_residual_delta_near_zero(self):
        curve = _curve()
        vol = _flat_vol()
        swns = [(_swn(notional=10_000_000), 1)]
        result = delta_hedge(swns, curve, vol)
        assert abs(result["residual_delta"]) < abs(result["portfolio_delta"]) * 0.05

    def test_hedge_notional_nonzero(self):
        curve = _curve()
        vol = _flat_vol()
        swns = [(_swn(notional=10_000_000), 1)]
        result = delta_hedge(swns, curve, vol)
        assert result["hedge_notional"] != 0.0

    def test_multi_swaption_hedge(self):
        curve = _curve()
        vol = _flat_vol()
        swns = [
            (_swn(strike=0.05, notional=5_000_000), 1),
            (_swn(strike=0.04, typ=SwaptionType.RECEIVER, notional=3_000_000), -1),
        ]
        result = delta_hedge(swns, curve, vol)
        assert abs(result["residual_delta"]) < abs(result["portfolio_delta"]) * 0.05


class TestVegaHedge:
    def test_residual_vega_near_zero(self):
        curve = _curve()
        vol = _flat_vol()
        swns = [(_swn(notional=10_000_000), 1)]
        hedge = Swaption(date(2026, 1, 15), date(2031, 1, 15), 0.05,
                         SwaptionType.PAYER, 1_000_000)
        result = vega_hedge(swns, hedge, curve, vol)
        assert abs(result["residual_vega"]) < 1e-6

    def test_hedge_ratio(self):
        curve = _curve()
        vol = _flat_vol()
        swns = [(_swn(notional=10_000_000), 1)]
        hedge = Swaption(date(2026, 1, 15), date(2031, 1, 15), 0.05,
                         SwaptionType.PAYER, 1_000_000)
        result = vega_hedge(swns, hedge, curve, vol)
        assert result["hedge_ratio"] != 0.0
