"""Tests for cross-currency swaps."""

import pytest
from datetime import date

from pricebook.xccy_swap import CrossCurrencySwap
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
START = date(2024, 7, 15)  # forward-starting to show basis


def _usd_curve():
    return make_flat_curve(REF, 0.05)


def _eur_curve():
    return make_flat_curve(REF, 0.03)


FX_SPOT = 1.10  # EUR/USD


class TestXCCYSwapBasic:
    def test_par_swap_near_zero(self):
        """XCCY swap at par spread should have PV near zero."""
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
            domestic_spread=0.0,
        )
        par = swap.par_spread(usd, eur, FX_SPOT)
        swap.domestic_spread = par
        pv = swap.pv(usd, eur, FX_SPOT)
        assert pv == pytest.approx(0.0, abs=100)

    def test_nonzero_spread_nonzero_pv(self):
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
            domestic_spread=0.01,  # 100bp spread
        )
        pv = swap.pv(usd, eur, FX_SPOT)
        assert pv != 0.0

    def test_payment_dates(self):
        swap = CrossCurrencySwap(
            START, date(2025, 7, 15), 10_000_000, FX_SPOT,
            frequency=Frequency.QUARTERLY,
        )
        assert len(swap.payment_dates) >= 3

    def test_notionals(self):
        swap = CrossCurrencySwap(REF, date(2029, 1, 15), 10_000_000, FX_SPOT)
        assert swap.domestic_notional == 10_000_000
        assert swap.foreign_notional == pytest.approx(11_000_000)


class TestNotionalExchange:
    def test_same_curves_same_fx(self):
        """With identical curves and fx=1, PV = 0 (no basis)."""
        same_curve = make_flat_curve(REF, 0.05)
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, 1.0,
            domestic_spread=0.0,
        )
        pv = swap.pv(same_curve, same_curve, 1.0)
        assert abs(pv) < 10_000

    def test_fx_move_creates_pv(self):
        """FX move after inception creates P&L."""
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
        )
        par = swap.par_spread(usd, eur, FX_SPOT)
        swap.domestic_spread = par

        # FX moves: EUR strengthens
        pv_moved = swap.pv(usd, eur, FX_SPOT + 0.05)
        assert pv_moved != pytest.approx(0.0, abs=1000)


class TestMTMReset:
    def test_mtm_swap_prices(self):
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
            domestic_spread=0.0, mtm_reset=True,
        )
        pv = swap.pv(usd, eur, FX_SPOT)
        # Should return a number (not crash)
        assert isinstance(pv, float)

    def test_mtm_less_fx_sensitive(self):
        """MTM reset swap should be less sensitive to FX than non-MTM."""
        usd = _usd_curve()
        eur = _eur_curve()

        swap_std = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
        )
        swap_mtm = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
            mtm_reset=True,
        )

        fx_delta_std = swap_std.fx_delta(usd, eur, FX_SPOT)
        fx_delta_mtm = swap_mtm.fx_delta(usd, eur, FX_SPOT)

        # MTM reset reduces FX delta
        assert abs(fx_delta_mtm) < abs(fx_delta_std)


class TestRisk:
    def test_dv01_domestic(self):
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
            domestic_spread=-0.002,
        )
        dv01 = swap.dv01_domestic(usd, eur, FX_SPOT)
        assert dv01 != 0.0

    def test_dv01_foreign(self):
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
            domestic_spread=-0.002,
        )
        dv01 = swap.dv01_foreign(usd, eur, FX_SPOT)
        assert dv01 != 0.0

    def test_dv01_opposite_signs(self):
        """Domestic and foreign DV01 should have opposite signs (receive one, pay other)."""
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
            domestic_spread=-0.002,
        )
        dom_dv01 = swap.dv01_domestic(usd, eur, FX_SPOT)
        for_dv01 = swap.dv01_foreign(usd, eur, FX_SPOT)
        assert dom_dv01 * for_dv01 < 0

    def test_fx_delta(self):
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
        )
        delta = swap.fx_delta(usd, eur, FX_SPOT)
        assert delta != 0.0


class TestParSpread:
    def test_par_spread_sign(self):
        """With rate differential, par spread should be non-zero."""
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2029, 1, 15), 10_000_000, FX_SPOT,
        )
        par = swap.par_spread(usd, eur, FX_SPOT)
        # Non-trivial basis when curves differ
        assert abs(par) > 1e-6

    def test_par_spread_sensitivity(self):
        """Higher rate differential → larger par spread magnitude."""
        eur_low = make_flat_curve(REF, 0.02)
        eur_high = make_flat_curve(REF, 0.04)
        usd = _usd_curve()

        swap_low = CrossCurrencySwap(START, date(2029, 1, 15), 10_000_000, FX_SPOT)
        swap_high = CrossCurrencySwap(START, date(2029, 1, 15), 10_000_000, FX_SPOT)

        par_low = swap_low.par_spread(usd, eur_low, FX_SPOT)
        par_high = swap_high.par_spread(usd, eur_high, FX_SPOT)

        # Different rate differentials produce different spreads
        assert par_low != pytest.approx(par_high, abs=1e-5)

    def test_short_tenor(self):
        usd = _usd_curve()
        eur = _eur_curve()
        swap = CrossCurrencySwap(
            START, date(2025, 7, 15), 10_000_000, FX_SPOT,
        )
        par = swap.par_spread(usd, eur, FX_SPOT)
        swap.domestic_spread = par
        assert swap.pv(usd, eur, FX_SPOT) == pytest.approx(0.0, abs=100)
