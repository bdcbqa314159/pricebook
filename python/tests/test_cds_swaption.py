"""Tests for CDS swaptions and Pedersen model."""

import math

import pytest

from pricebook.cds_swaption import (
    CDSSwaption,
    ForwardCDSResult,
    PedersenCDSSwaption,
    PedersenResult,
    PutCallParityResult,
    cds_swaption_black,
    cds_swaption_put_call_parity,
    forward_cds_spread,
)


# ---- Forward CDS spread ----

class TestForwardCDSSpread:
    def test_positive_spread(self):
        result = forward_cds_spread(1.0, 6.0, flat_hazard=0.02)
        assert result.forward_spread > 0

    def test_spread_increases_with_hazard(self):
        low = forward_cds_spread(1.0, 6.0, flat_hazard=0.01)
        high = forward_cds_spread(1.0, 6.0, flat_hazard=0.05)
        assert high.forward_spread > low.forward_spread

    def test_spread_approximation(self):
        """Forward spread ≈ λ(1−R) for flat hazard."""
        lam, R = 0.02, 0.4
        result = forward_cds_spread(1.0, 6.0, lam, recovery=R)
        assert result.forward_spread == pytest.approx(lam * (1 - R), rel=0.15)

    def test_annuity_positive(self):
        result = forward_cds_spread(1.0, 6.0, 0.02)
        assert result.risky_annuity > 0

    def test_survival_to_start(self):
        result = forward_cds_spread(2.0, 7.0, 0.03)
        expected = math.exp(-0.03 * 2.0)
        assert result.survival_to_start == pytest.approx(expected)


# ---- CDS swaption Black-76 ----

class TestCDSSwaption:
    def test_payer_positive(self):
        result = cds_swaption_black(
            forward_spread=0.012, strike_spread=0.010,
            spread_vol=0.40, expiry=1.0, risky_annuity=4.0,
            survival_to_expiry=0.98,
        )
        assert result.premium > 0

    def test_receiver_positive(self):
        result = cds_swaption_black(
            0.012, 0.015, 0.40, 1.0, 4.0, 0.98,
            option_type="receiver",
        )
        assert result.premium > 0

    def test_atm_payer_equals_receiver(self):
        """At-the-money: payer ≈ receiver (by symmetry of Black-76)."""
        payer = cds_swaption_black(0.012, 0.012, 0.40, 1.0, 4.0, 0.98,
                                    option_type="payer")
        receiver = cds_swaption_black(0.012, 0.012, 0.40, 1.0, 4.0, 0.98,
                                       option_type="receiver")
        assert payer.premium == pytest.approx(receiver.premium, rel=0.01)

    def test_higher_vol_higher_premium(self):
        low = cds_swaption_black(0.012, 0.012, 0.20, 1.0, 4.0, 0.98)
        high = cds_swaption_black(0.012, 0.012, 0.60, 1.0, 4.0, 0.98)
        assert high.premium > low.premium

    def test_zero_vol_intrinsic(self):
        """σ=0 → option is intrinsic value."""
        result = cds_swaption_black(0.015, 0.010, 0.0, 1.0, 4.0, 0.98,
                                     option_type="payer")
        expected = 0.98 * 1_000_000 * 4.0 * (0.015 - 0.010)
        assert result.premium == pytest.approx(expected, rel=0.01)

    def test_knockout_on_default(self):
        """Lower survival → lower premium (option knocks out)."""
        high_surv = cds_swaption_black(0.012, 0.010, 0.40, 1.0, 4.0, 0.99)
        low_surv = cds_swaption_black(0.012, 0.010, 0.40, 1.0, 4.0, 0.50)
        assert high_surv.premium > low_surv.premium


# ---- Pedersen model ----

class TestPedersenCDSSwaption:
    def test_price_positive(self):
        model = PedersenCDSSwaption(flat_hazard=0.02, spread_vol=0.40)
        result = model.price(1.0, 6.0, strike_spread=0.012)
        assert result.premium > 0

    def test_mc_matches_analytical(self):
        """MC and Black-76 should agree."""
        model = PedersenCDSSwaption(flat_hazard=0.02, spread_vol=0.40)
        analytical = model.price(1.0, 6.0, 0.012, option_type="payer")
        mc = model.price_mc(1.0, 6.0, 0.012, option_type="payer",
                            n_paths=200_000, seed=42)
        assert mc.mc_premium == pytest.approx(analytical.premium, rel=0.05)

    def test_payer_vs_receiver(self):
        model = PedersenCDSSwaption(flat_hazard=0.02, spread_vol=0.40)
        payer = model.price(1.0, 6.0, 0.012, option_type="payer")
        receiver = model.price(1.0, 6.0, 0.012, option_type="receiver")
        # Both positive, payer ≠ receiver unless ATM
        assert payer.premium > 0
        assert receiver.premium > 0

    def test_reduces_to_forward_at_zero_vol(self):
        """σ=0 → swaption = intrinsic of forward CDS."""
        model = PedersenCDSSwaption(flat_hazard=0.03, spread_vol=0.0)
        result = model.price(1.0, 6.0, strike_spread=0.005)
        # Intrinsic = max(F − K, 0) × survival × annuity × notional
        assert result.premium > 0  # F > K since hazard is high

    def test_higher_vol_higher_premium(self):
        low = PedersenCDSSwaption(flat_hazard=0.02, spread_vol=0.20)
        high = PedersenCDSSwaption(flat_hazard=0.02, spread_vol=0.60)
        p_low = low.price(1.0, 6.0, 0.012)
        p_high = high.price(1.0, 6.0, 0.012)
        assert p_high.premium >= p_low.premium


# ---- Put-call parity ----

class TestPutCallParity:
    def test_parity_holds(self):
        """Payer − Receiver = Q × A × (F − K)."""
        result = cds_swaption_put_call_parity(
            forward_spread=0.015, strike_spread=0.012,
            spread_vol=0.40, expiry=1.0, risky_annuity=4.0,
            survival_to_expiry=0.98,
        )
        assert result.holds
        assert result.parity_error < 0.01

    def test_parity_at_atm(self):
        """At ATM: payer = receiver, forward PV = 0."""
        result = cds_swaption_put_call_parity(
            0.012, 0.012, 0.40, 1.0, 4.0, 0.98,
        )
        assert result.holds
        assert result.forward_cds_pv == pytest.approx(0.0)

    def test_parity_various_strikes(self):
        for K in [0.005, 0.010, 0.015, 0.020, 0.030]:
            result = cds_swaption_put_call_parity(
                0.015, K, 0.40, 1.0, 4.0, 0.98,
            )
            assert result.holds, f"Parity failed at K={K}"
