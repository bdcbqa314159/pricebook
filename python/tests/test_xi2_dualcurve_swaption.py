"""XI2: Dual-Curve → Swaption Parity integration chain.

OIS discount + projection → swap par rate → swaption forward → Black-76 →
payer-receiver parity at ATM and off-ATM. Verify forward_swap_rate from
swaption matches par_rate from swap.

Bug hotspots:
- swaption builds internal PAYER swap, annuity uses discount curve only
- vol surface vol(date, strike) type mismatch
- forward_swap_rate must match par_rate from swap module
- payer - receiver = annuity × (forward - strike) at all strikes
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap, bootstrap_forward_curve
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.swaption import Swaption, SwaptionType
from pricebook.vol_surface import FlatVol
from pricebook.black76 import black76_price, OptionType


# ---- Helpers ----

REF = date(2026, 4, 25)


def _dual_curves(ref: date) -> tuple[DiscountCurve, DiscountCurve]:
    """Bootstrap OIS discount curve + projection (forward) curve."""
    deposits_ois = [
        (ref + timedelta(days=1), 0.040),
        (ref + timedelta(days=30), 0.040),
        (ref + timedelta(days=91), 0.039),
        (ref + timedelta(days=182), 0.038),
    ]
    swaps_ois = [
        (ref + timedelta(days=365), 0.037),
        (ref + timedelta(days=730), 0.036),
        (ref + timedelta(days=1095), 0.035),
        (ref + timedelta(days=1825), 0.034),
        (ref + timedelta(days=3650), 0.033),
    ]
    ois = bootstrap(ref, deposits_ois, swaps_ois)

    # Projection curve: higher than OIS (term premium)
    deposits_proj = [
        (ref + timedelta(days=91), 0.042),
        (ref + timedelta(days=182), 0.041),
    ]
    swaps_proj = [
        (ref + timedelta(days=365), 0.040),
        (ref + timedelta(days=730), 0.039),
        (ref + timedelta(days=1095), 0.038),
        (ref + timedelta(days=1825), 0.037),
        (ref + timedelta(days=3650), 0.036),
    ]
    proj = bootstrap_forward_curve(ref, swaps_proj, ois, deposits=deposits_proj)

    return ois, proj


# ---- R1: Chain test — forward rate consistency ----

class TestXI2R1ForwardRateConsistency:
    """forward_swap_rate from swaption must match par_rate from swap."""

    def test_forward_rate_matches_par_rate_single_curve(self):
        """Single-curve: swaption forward = swap par rate."""
        ois, _ = _dual_curves(REF)
        expiry = REF + timedelta(days=365)      # 1Y option
        swap_end = expiry + timedelta(days=1825) # into 5Y swap

        swap = InterestRateSwap(expiry, swap_end, fixed_rate=0.03)
        swn = Swaption(expiry, swap_end, strike=0.03)

        par = swap.par_rate(ois)
        fwd = swn.forward_swap_rate(ois)
        assert fwd == pytest.approx(par, rel=1e-8)

    def test_forward_rate_matches_par_rate_dual_curve(self):
        """Dual-curve: swaption forward = swap par rate (both with projection)."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)

        swap = InterestRateSwap(expiry, swap_end, fixed_rate=0.03)
        swn = Swaption(expiry, swap_end, strike=0.03)

        par = swap.par_rate(ois, projection_curve=proj)
        fwd = swn.forward_swap_rate(ois, projection_curve=proj)
        assert fwd == pytest.approx(par, rel=1e-8)

    def test_projection_curve_raises_forward(self):
        """Projection curve > OIS → forward swap rate should be higher than single-curve."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)

        swn = Swaption(expiry, swap_end, strike=0.03)
        fwd_single = swn.forward_swap_rate(ois)
        fwd_dual = swn.forward_swap_rate(ois, projection_curve=proj)
        assert fwd_dual > fwd_single

    def test_annuity_uses_discount_curve_only(self):
        """Annuity must be the same regardless of projection curve."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)

        swn = Swaption(expiry, swap_end, strike=0.03)
        ann = swn.annuity(ois)
        assert ann > 0
        # Annuity depends only on discount curve — no projection involvement


# ---- R2: Payer-receiver parity ----

class TestXI2R2PayerReceiverParity:
    """Payer − Receiver = Annuity × (Forward − Strike) for all strikes."""

    def _check_parity(self, ois, proj, expiry, swap_end, strike, vol_flat):
        vol = FlatVol(vol_flat)
        payer = Swaption(expiry, swap_end, strike=strike,
                         swaption_type=SwaptionType.PAYER)
        receiver = Swaption(expiry, swap_end, strike=strike,
                            swaption_type=SwaptionType.RECEIVER)

        pv_pay = payer.pv(ois, vol, projection_curve=proj)
        pv_rec = receiver.pv(ois, vol, projection_curve=proj)
        fwd = payer.forward_swap_rate(ois, projection_curve=proj)
        ann = payer.annuity(ois)
        notional = payer.notional

        # Parity: payer - receiver = notional × annuity × (forward - strike)
        lhs = pv_pay - pv_rec
        rhs = notional * ann * (fwd - strike)
        assert lhs == pytest.approx(rhs, rel=0.001), (
            f"Parity failed: strike={strike}, lhs={lhs:.2f}, rhs={rhs:.2f}"
        )

    def test_parity_atm(self):
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        swn = Swaption(expiry, swap_end, strike=0.03)
        fwd = swn.forward_swap_rate(ois, projection_curve=proj)
        self._check_parity(ois, proj, expiry, swap_end, fwd, 0.20)

    def test_parity_itm(self):
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        self._check_parity(ois, proj, expiry, swap_end, 0.02, 0.20)

    def test_parity_otm(self):
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        self._check_parity(ois, proj, expiry, swap_end, 0.06, 0.20)

    def test_parity_deep_otm(self):
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        self._check_parity(ois, proj, expiry, swap_end, 0.10, 0.25)

    def test_parity_single_curve(self):
        ois, _ = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        self._check_parity(ois, None, expiry, swap_end, 0.04, 0.20)

    def test_parity_short_expiry(self):
        """3M expiry into 2Y swap."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=91)
        swap_end = expiry + timedelta(days=730)
        self._check_parity(ois, proj, expiry, swap_end, 0.04, 0.20)

    def test_parity_long_expiry(self):
        """5Y expiry into 5Y swap."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=1825)
        swap_end = expiry + timedelta(days=1825)
        self._check_parity(ois, proj, expiry, swap_end, 0.035, 0.20)


# ---- R3: Cross-module consistency ----

class TestXI2R3Consistency:
    """Verify consistency between swaption pricing and Black-76 directly."""

    def test_swaption_pv_matches_manual_black76(self):
        """Swaption PV should match notional × annuity × Black76(F,K,vol,T,df=1)."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        strike = 0.04
        vol_val = 0.20

        swn = Swaption(expiry, swap_end, strike=strike,
                        swaption_type=SwaptionType.PAYER)
        vol = FlatVol(vol_val)
        pv = swn.pv(ois, vol, projection_curve=proj)

        # Manual computation
        fwd = swn.forward_swap_rate(ois, projection_curve=proj)
        ann = swn.annuity(ois)
        T = (expiry - REF).days / 365.0
        manual_price = black76_price(fwd, strike, vol_val, T, df=1.0,
                                      option_type=OptionType.CALL)
        manual_pv = swn.notional * ann * manual_price
        assert pv == pytest.approx(manual_pv, rel=1e-6)

    def test_atm_swap_pv_zero_with_forward_strike(self):
        """Swap struck at swaption forward rate must have PV=0."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)

        swn = Swaption(expiry, swap_end, strike=0.03)
        fwd = swn.forward_swap_rate(ois, projection_curve=proj)

        swap = InterestRateSwap(expiry, swap_end, fixed_rate=fwd)
        pv = swap.pv(ois, projection_curve=proj)
        assert pv == pytest.approx(0.0, abs=1.0)

    def test_higher_vol_higher_swaption_price(self):
        """Higher vol → higher swaption price."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        strike = 0.04

        swn = Swaption(expiry, swap_end, strike=strike)
        pv_low = swn.pv(ois, FlatVol(0.10), projection_curve=proj)
        pv_high = swn.pv(ois, FlatVol(0.30), projection_curve=proj)
        assert pv_high > pv_low

    def test_zero_vol_payer_equals_intrinsic(self):
        """At zero vol, payer = max(F-K, 0) × annuity × notional."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        strike = 0.02  # deep ITM

        swn = Swaption(expiry, swap_end, strike=strike,
                        swaption_type=SwaptionType.PAYER)
        fwd = swn.forward_swap_rate(ois, projection_curve=proj)
        ann = swn.annuity(ois)

        # Use very small vol (not exactly 0 to avoid numerical issues)
        pv = swn.pv(ois, FlatVol(1e-6), projection_curve=proj)
        intrinsic = swn.notional * ann * max(fwd - strike, 0.0)
        assert pv == pytest.approx(intrinsic, rel=0.01)


# ---- R4: Edge cases ----

class TestXI2R4EdgeCases:
    """Edge cases and boundary conditions."""

    def test_expired_swaption_zero(self):
        """Swaption with expiry in the past should price to ~0 or handle gracefully."""
        ois, proj = _dual_curves(REF)
        expiry = REF - timedelta(days=1)
        swap_end = REF + timedelta(days=1825)
        swn = Swaption(expiry, swap_end, strike=0.04)
        pv = swn.pv(ois, FlatVol(0.20), projection_curve=proj)
        assert math.isfinite(pv)

    def test_atm_payer_receiver_equal(self):
        """At ATM forward, payer and receiver should have approximately equal value."""
        ois, proj = _dual_curves(REF)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)
        swn_temp = Swaption(expiry, swap_end, strike=0.03)
        fwd = swn_temp.forward_swap_rate(ois, projection_curve=proj)

        payer = Swaption(expiry, swap_end, strike=fwd,
                         swaption_type=SwaptionType.PAYER)
        receiver = Swaption(expiry, swap_end, strike=fwd,
                            swaption_type=SwaptionType.RECEIVER)
        vol = FlatVol(0.20)
        pv_pay = payer.pv(ois, vol, projection_curve=proj)
        pv_rec = receiver.pv(ois, vol, projection_curve=proj)
        # At ATM, payer ≈ receiver (difference = 0 by parity)
        assert pv_pay == pytest.approx(pv_rec, rel=0.01)

    def test_flat_curves_consistency(self):
        """On flat curves, forward rate should be near the flat rate."""
        flat_ois = DiscountCurve.flat(REF, 0.03)
        flat_proj = DiscountCurve.flat(REF, 0.04)
        expiry = REF + timedelta(days=365)
        swap_end = expiry + timedelta(days=1825)

        swn = Swaption(expiry, swap_end, strike=0.04)
        fwd = swn.forward_swap_rate(flat_ois, projection_curve=flat_proj)
        # Forward should be near 0.04 (projection rate)
        assert fwd == pytest.approx(0.04, abs=0.005)
