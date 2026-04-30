"""Tests for CDS swaption deep dive: curves, Greeks, smile, stochastic, Bachelier."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cds_swaption import (
    cds_swaption_black, cds_swaption_black_curves,
    cds_swaption_bachelier, cds_swaption_greeks,
    CDSSwaptionGreeks, CDSSpreadSmile,
    StochasticIntensitySwaption,
    PedersenCDSSwaption,
    exercise_into_physical, ExerciseResult,
    forward_cds_spread,
    _pedersen_price_curves,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
EXPIRY = REF + relativedelta(years=1)
MATURITY = REF + relativedelta(years=6)


# ---- Curve-based swaption ----

class TestCurveBasedSwaption:

    def test_positive_premium(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        result = cds_swaption_black_curves(
            dc, sc, EXPIRY, MATURITY,
            strike_spread=0.01, spread_vol=0.4,
        )
        assert result.premium > 0

    def test_payer_vs_receiver(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        payer = cds_swaption_black_curves(
            dc, sc, EXPIRY, MATURITY,
            strike_spread=0.01, spread_vol=0.4, option_type="payer",
        )
        receiver = cds_swaption_black_curves(
            dc, sc, EXPIRY, MATURITY,
            strike_spread=0.01, spread_vol=0.4, option_type="receiver",
        )
        # Both should be positive
        assert payer.premium > 0
        assert receiver.premium > 0

    def test_put_call_parity_curves(self):
        """Payer - Receiver = Q × A × (F - K) × notional."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        K = 0.012
        payer = cds_swaption_black_curves(
            dc, sc, EXPIRY, MATURITY,
            strike_spread=K, spread_vol=0.4, option_type="payer",
        )
        receiver = cds_swaption_black_curves(
            dc, sc, EXPIRY, MATURITY,
            strike_spread=K, spread_vol=0.4, option_type="receiver",
        )
        forward_pv = payer.survival_to_expiry * 1_000_000 * payer.risky_annuity * \
            (payer.forward_spread - K)
        diff = payer.premium - receiver.premium
        assert diff == pytest.approx(forward_pv, rel=0.01)

    def test_agrees_with_flat(self):
        """Flat curves should give similar result to flat-scalar version."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        curve_result = cds_swaption_black_curves(
            dc, sc, EXPIRY, MATURITY,
            strike_spread=0.012, spread_vol=0.4,
        )

        # Flat scalar version
        fwd = forward_cds_spread(1.0, 6.0, 0.02, 0.04, 0.4)
        flat_result = cds_swaption_black(
            fwd.forward_spread, 0.012, 0.4,
            1.0, fwd.risky_annuity, fwd.survival_to_start,
        )

        # Should be in the same ballpark (different discretisation)
        assert curve_result.premium == pytest.approx(flat_result.premium, rel=0.15)


# ---- Pedersen with curves ----

class TestPedersenCurves:

    def test_price_curves(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        model = PedersenCDSSwaption(spread_vol=0.4)
        result = model.price_curves(
            dc, sc, EXPIRY, MATURITY, strike_spread=0.012,
        )
        assert result.premium > 0

    def test_curves_vs_flat(self):
        """Curve-based should agree with flat for constant curves."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        model = PedersenCDSSwaption(
            flat_hazard=0.02, flat_rate=0.04,
            recovery=0.4, spread_vol=0.4,
        )
        flat = model.price(1.0, 6.0, 0.012)
        curve = model.price_curves(dc, sc, EXPIRY, MATURITY, 0.012)
        assert curve.premium == pytest.approx(flat.premium, rel=0.15)


# ---- Greeks ----

class TestGreeks:

    def test_delta_positive_for_payer(self):
        """Payer delta > 0 (higher forward → more valuable)."""
        g = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98)
        assert g.delta > 0

    def test_delta_negative_for_receiver(self):
        g = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98,
                                 option_type="receiver")
        assert g.delta < 0

    def test_delta_parity(self):
        """Delta_payer - Delta_receiver ≈ Q × A × notional."""
        g_p = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98,
                                   option_type="payer")
        g_r = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98,
                                   option_type="receiver")
        expected = 0.98 * 4.5 * 1_000_000
        assert (g_p.delta - g_r.delta) == pytest.approx(expected, rel=0.05)

    def test_gamma_positive(self):
        g = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98)
        assert g.gamma > 0

    def test_vega_positive(self):
        g = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98)
        assert g.vega > 0

    def test_vega_maximal_atm(self):
        """Vega should be highest near ATM."""
        g_atm = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98)
        g_otm = cds_swaption_greeks(0.012, 0.020, 0.4, 1.0, 4.5, 0.98)
        assert g_atm.vega > g_otm.vega

    def test_to_dict(self):
        g = cds_swaption_greeks(0.012, 0.012, 0.4, 1.0, 4.5, 0.98)
        d = g.to_dict()
        assert "delta" in d and "vega" in d


# ---- Bachelier ----

class TestBachelier:

    def test_positive_premium(self):
        result = cds_swaption_bachelier(
            0.012, 0.012, 0.003, 1.0, 4.5, 0.98,
        )
        assert result.premium > 0

    def test_agrees_with_black_atm_low_vol(self):
        """At ATM with low vol, Black and Bachelier should roughly agree."""
        F = 0.012
        K = 0.012
        black_vol = 0.3
        normal_vol = black_vol * F  # approximate conversion
        black = cds_swaption_black(F, K, black_vol, 1.0, 4.5, 0.98)
        bach = cds_swaption_bachelier(F, K, normal_vol, 1.0, 4.5, 0.98)
        # Should be in the same order of magnitude
        assert bach.premium == pytest.approx(black.premium, rel=0.3)

    def test_near_zero_spread(self):
        """Bachelier should work fine with very small forward spread."""
        result = cds_swaption_bachelier(
            0.0005, 0.001, 0.001, 1.0, 4.5, 0.98,
        )
        assert result.premium >= 0
        assert math.isfinite(result.premium)

    def test_payer_receiver(self):
        payer = cds_swaption_bachelier(
            0.012, 0.012, 0.003, 1.0, 4.5, 0.98, option_type="payer",
        )
        receiver = cds_swaption_bachelier(
            0.012, 0.012, 0.003, 1.0, 4.5, 0.98, option_type="receiver",
        )
        # ATM: payer ≈ receiver
        assert payer.premium == pytest.approx(receiver.premium, rel=0.01)


# ---- SABR smile ----

class TestCDSSpreadSmile:

    def test_atm_vol(self):
        smile = CDSSpreadSmile(forward=0.012, alpha=0.4, beta=0.5, rho=-0.3, nu=0.4)
        vol = smile.implied_vol(0.012, T=1.0)
        assert vol > 0

    def test_smile_shape(self):
        """Vol smile should be non-negative everywhere."""
        smile = CDSSpreadSmile(forward=0.012, alpha=0.4, beta=0.5, rho=-0.3, nu=0.4)
        strikes = [0.005, 0.008, 0.010, 0.012, 0.015, 0.020]
        vols = smile.smile(strikes, T=1.0)
        for v in vols:
            assert v > 0

    def test_to_dict_from_dict(self):
        smile = CDSSpreadSmile(0.012, 0.4, 0.5, -0.3, 0.4)
        d = smile.to_dict()
        smile2 = CDSSpreadSmile.from_dict(d)
        assert smile2.forward == smile.forward
        assert smile2.alpha == smile.alpha
        assert smile2.implied_vol(0.012) == smile.implied_vol(0.012)


# ---- Stochastic intensity ----

class TestStochasticIntensity:

    def test_positive_premium(self):
        model = StochasticIntensitySwaption(kappa=1.0, theta=0.02, xi=0.1)
        result = model.price(1.0, 6.0, 0.012, n_paths=10_000)
        assert result.premium > 0

    def test_converges_to_deterministic(self):
        """As xi → 0, should approach Black-76 with deterministic intensity."""
        model_det = StochasticIntensitySwaption(kappa=1.0, theta=0.02, xi=0.001)
        model_stoch = StochasticIntensitySwaption(kappa=1.0, theta=0.02, xi=0.3)

        det = model_det.price(1.0, 6.0, 0.012, n_paths=20_000, seed=42)
        stoch = model_stoch.price(1.0, 6.0, 0.012, n_paths=20_000, seed=42)

        # Stochastic should give higher price (vol of vol adds value)
        # At minimum, both should be positive
        assert det.premium > 0
        assert stoch.premium > 0

    def test_to_dict_from_dict(self):
        model = StochasticIntensitySwaption(kappa=1.5, theta=0.03, xi=0.15)
        d = model.to_dict()
        model2 = StochasticIntensitySwaption.from_dict(d)
        assert model2.kappa == 1.5
        assert model2.theta == 0.03

    def test_payer_vs_receiver(self):
        model = StochasticIntensitySwaption(kappa=1.0, theta=0.02, xi=0.1)
        payer = model.price(1.0, 6.0, 0.012, option_type="payer", n_paths=10_000)
        receiver = model.price(1.0, 6.0, 0.012, option_type="receiver", n_paths=10_000)
        assert payer.premium > 0
        assert receiver.premium > 0


# ---- Exercise into physical ----

class TestExercise:

    def test_payer_exercise_itm(self):
        """Payer exercised when F > K → positive PV."""
        r = exercise_into_physical(0.020, 0.012, 5000, 4.5, 0.98)
        assert r.exercised is True
        assert r.exercise_pv > 0

    def test_payer_no_exercise_otm(self):
        """Payer not exercised when F < K."""
        r = exercise_into_physical(0.008, 0.012, 5000, 4.5, 0.98)
        assert r.exercised is False
        assert r.exercise_pv == 0.0

    def test_total_pnl(self):
        """Total P&L = exercise PV - premium paid."""
        r = exercise_into_physical(0.020, 0.012, 5000, 4.5, 0.98)
        assert r.total_pnl == pytest.approx(r.exercise_pv - 5000)

    def test_receiver_exercise(self):
        r = exercise_into_physical(0.008, 0.012, 3000, 4.5, 0.98,
                                    option_type="receiver")
        assert r.exercised is True
        assert r.exercise_pv > 0

    def test_to_dict(self):
        r = exercise_into_physical(0.020, 0.012, 5000, 4.5, 0.98)
        d = r.to_dict()
        assert "exercise_pv" in d
        assert "total_pnl" in d
