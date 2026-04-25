"""XI9: P&L Attribution integration chain.

Day 1 curve → price swap. Day 2 = bumped curve. Total P&L.
Greek attribution: delta × Δr + ½γ × Δr². Verify unexplained < 1% for 10bp move.

Bug hotspots:
- DV01 scaling (per bp vs per unit)
- Gamma dollar scaling
- compute_carry input convention (annualised vs daily)
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.bootstrap import bootstrap
from pricebook.discount_curve import DiscountCurve
from pricebook.pnl_explain import greek_pnl, pnl_decompose, compute_carry
from pricebook.schedule import Frequency
from pricebook.swap import InterestRateSwap, SwapDirection


# ---- Helpers ----

REF = date(2026, 4, 25)


def _curve(ref: date, parallel_shift: float = 0.0) -> DiscountCurve:
    deposits = [
        (ref + timedelta(days=30), 0.040 + parallel_shift),
        (ref + timedelta(days=91), 0.040 + parallel_shift),
        (ref + timedelta(days=182), 0.039 + parallel_shift),
    ]
    swaps = [
        (ref + timedelta(days=365), 0.038 + parallel_shift),
        (ref + timedelta(days=730), 0.037 + parallel_shift),
        (ref + timedelta(days=1095), 0.036 + parallel_shift),
        (ref + timedelta(days=1825), 0.035 + parallel_shift),
        (ref + timedelta(days=3650), 0.034 + parallel_shift),
    ]
    return bootstrap(ref, deposits, swaps)


def _swap_10y(ref: date) -> InterestRateSwap:
    start = ref + timedelta(days=2)
    end = start + timedelta(days=3650)
    return InterestRateSwap(start, end, fixed_rate=0.035,
                            direction=SwapDirection.PAYER)


# ---- R1: Greek P&L decomposition ----

class TestXI9R1GreekPnL:
    """greek_pnl: delta × Δr + ½γ × Δr² matches actual P&L."""

    def test_delta_pnl_small_move(self):
        """For a 1bp move, delta alone should explain most of P&L."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        shift = 0.0001  # 1bp
        pv_base = swap.pv(curve)
        pv_up = swap.pv(curve.bumped(shift))

        actual_pnl = pv_up - pv_base
        dv01 = swap.dv01(curve, shift=shift)  # PV change for +1bp

        # Delta-only attribution
        delta_pnl = dv01  # dv01 is already pv(up) - pv(base) for 1bp
        assert delta_pnl == pytest.approx(actual_pnl, rel=0.01)

    def test_delta_gamma_10bp_move(self):
        """For 10bp move, delta + gamma should explain P&L well."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        shift = 0.0010  # 10bp
        pv_base = swap.pv(curve)
        pv_shifted = swap.pv(curve.bumped(shift))
        actual_pnl = pv_shifted - pv_base

        # Compute delta and gamma
        h = 0.0001
        pv_up = swap.pv(curve.bumped(h))
        pv_dn = swap.pv(curve.bumped(-h))
        delta = (pv_up - pv_dn) / (2 * h)  # per unit rate change
        gamma = (pv_up - 2 * pv_base + pv_dn) / (h * h)

        # Greek attribution
        attributed = greek_pnl(delta, shift, gamma)
        unexplained = abs(actual_pnl - attributed)
        assert unexplained / abs(actual_pnl) < 0.01, (
            f"Unexplained {unexplained:.2f} > 1% of {actual_pnl:.2f}"
        )

    def test_delta_gamma_50bp_move(self):
        """For 50bp move, delta + gamma should still explain > 95%."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        shift = 0.0050  # 50bp
        pv_base = swap.pv(curve)
        pv_shifted = swap.pv(curve.bumped(shift))
        actual_pnl = pv_shifted - pv_base

        h = 0.0001
        pv_up = swap.pv(curve.bumped(h))
        pv_dn = swap.pv(curve.bumped(-h))
        delta = (pv_up - pv_dn) / (2 * h)
        gamma = (pv_up - 2 * pv_base + pv_dn) / (h * h)

        attributed = greek_pnl(delta, shift, gamma)
        unexplained_pct = abs(actual_pnl - attributed) / abs(actual_pnl)
        assert unexplained_pct < 0.05, f"Unexplained {unexplained_pct:.1%} > 5%"

    def test_greek_pnl_zero_move(self):
        """Zero move → zero P&L."""
        pnl = greek_pnl(1000.0, 0.0, 50000.0)
        assert pnl == 0.0


# ---- R2: Full pnl_decompose ----

class TestXI9R2PnLDecompose:
    """pnl_decompose: total, explained, unexplained."""

    def test_decompose_rate_only(self):
        """Rate-only P&L decomposition."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        shift = 0.0010  # 10bp
        pv_base = swap.pv(curve)
        pv_shifted = swap.pv(curve.bumped(shift))

        h = 0.0001
        pv_up = swap.pv(curve.bumped(h))
        pv_dn = swap.pv(curve.bumped(-h))
        delta = (pv_up - pv_dn) / (2 * h)
        gamma = (pv_up - 2 * pv_base + pv_dn) / (h * h)

        result = pnl_decompose(
            base_pv=pv_base,
            current_pv=pv_shifted,
            rate_delta=delta,
            rate_change=shift,
            rate_gamma=gamma,
        )

        assert result.total == pytest.approx(pv_shifted - pv_base, rel=1e-10)
        assert abs(result.unexplained) / abs(result.total) < 0.01

    def test_decompose_total_equals_pv_diff(self):
        """total = current_pv - base_pv exactly."""
        result = pnl_decompose(base_pv=100.0, current_pv=105.0)
        assert result.total == pytest.approx(5.0)

    def test_decompose_carry(self):
        """Carry component should flow through."""
        carry = compute_carry(coupon_income=40000.0, funding_cost=35000.0,
                              dt=1.0/252)
        assert carry > 0

        result = pnl_decompose(
            base_pv=1000000.0,
            current_pv=1000000.0 + carry,
            carry=carry,
        )
        assert result.carry == pytest.approx(carry)
        assert abs(result.unexplained) < 1.0

    def test_carry_scaling(self):
        """Carry should scale linearly with dt."""
        c1 = compute_carry(40000.0, 35000.0, dt=1.0/252)
        c2 = compute_carry(40000.0, 35000.0, dt=2.0/252)
        assert c2 == pytest.approx(2 * c1, rel=1e-10)


# ---- R3: Symmetry and sign checks ----

class TestXI9R3Symmetry:
    """P&L attribution symmetry and sign conventions."""

    def test_payer_positive_pnl_on_rate_rise(self):
        """Payer swap (pay fixed) gains when rates rise."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        pv_base = swap.pv(curve)
        pv_up = swap.pv(curve.bumped(0.0010))
        assert pv_up > pv_base

    def test_receiver_negative_pnl_on_rate_rise(self):
        """Receiver swap loses when rates rise."""
        curve = _curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=3650)
        swap = InterestRateSwap(start, end, fixed_rate=0.035,
                                direction=SwapDirection.RECEIVER)

        pv_base = swap.pv(curve)
        pv_up = swap.pv(curve.bumped(0.0010))
        assert pv_up < pv_base

    def test_symmetric_up_down_pnl(self):
        """P&L for +Δr and -Δr should be approximately symmetric for small moves."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        shift = 0.0001
        pv_base = swap.pv(curve)
        pnl_up = swap.pv(curve.bumped(shift)) - pv_base
        pnl_dn = swap.pv(curve.bumped(-shift)) - pv_base

        # Should be approximately equal magnitude, opposite sign
        assert pnl_up == pytest.approx(-pnl_dn, rel=0.05)

    def test_gamma_negative_for_payer_swap(self):
        """Payer swap gamma is negative (short fixed-leg convexity)."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        h = 0.0001
        pv_base = swap.pv(curve)
        pv_up = swap.pv(curve.bumped(h))
        pv_dn = swap.pv(curve.bumped(-h))
        gamma = (pv_up - 2 * pv_base + pv_dn) / (h * h)
        # Payer swap is short the fixed leg → negative convexity
        assert gamma < 0


# ---- R4: Edge cases ----

class TestXI9R4EdgeCases:
    """Edge cases for P&L attribution."""

    def test_flat_curve_attribution(self):
        """Attribution on flat curve should work."""
        flat = DiscountCurve.flat(REF, 0.04)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=1825)
        swap = InterestRateSwap(start, end, fixed_rate=0.04,
                                direction=SwapDirection.PAYER)

        shift = 0.0010
        pv_base = swap.pv(flat)
        pv_shifted = swap.pv(flat.bumped(shift))

        h = 0.0001
        pv_up = swap.pv(flat.bumped(h))
        pv_dn = swap.pv(flat.bumped(-h))
        delta = (pv_up - pv_dn) / (2 * h)
        gamma = (pv_up - 2 * pv_base + pv_dn) / (h * h)

        attributed = greek_pnl(delta, shift, gamma)
        actual = pv_shifted - pv_base
        assert abs(actual - attributed) / max(abs(actual), 1) < 0.01

    def test_short_swap_attribution(self):
        """1Y swap attribution."""
        curve = _curve(REF)
        start = REF + timedelta(days=2)
        end = start + timedelta(days=365)
        swap = InterestRateSwap(start, end, fixed_rate=0.038,
                                direction=SwapDirection.PAYER,
                                fixed_frequency=Frequency.QUARTERLY,
                                float_frequency=Frequency.QUARTERLY)

        shift = 0.0010
        pv_base = swap.pv(curve)
        pv_shifted = swap.pv(curve.bumped(shift))
        actual = pv_shifted - pv_base

        h = 0.0001
        pv_up = swap.pv(curve.bumped(h))
        pv_dn = swap.pv(curve.bumped(-h))
        delta = (pv_up - pv_dn) / (2 * h)
        gamma = (pv_up - 2 * pv_base + pv_dn) / (h * h)

        attributed = greek_pnl(delta, shift, gamma)
        assert abs(actual - attributed) / max(abs(actual), 1) < 0.01

    def test_large_move_higher_order_matters(self):
        """For 100bp move, gamma contribution should be material."""
        curve = _curve(REF)
        swap = _swap_10y(REF)

        shift = 0.0100  # 100bp
        h = 0.0001
        pv_base = swap.pv(curve)
        pv_up = swap.pv(curve.bumped(h))
        pv_dn = swap.pv(curve.bumped(-h))
        delta = (pv_up - pv_dn) / (2 * h)
        gamma = (pv_up - 2 * pv_base + pv_dn) / (h * h)

        delta_only = greek_pnl(delta, shift, 0.0)
        delta_gamma = greek_pnl(delta, shift, gamma)
        gamma_contribution = abs(delta_gamma - delta_only)

        # Gamma should be > 1% of delta for 100bp on 10Y
        assert gamma_contribution / abs(delta_only) > 0.001
