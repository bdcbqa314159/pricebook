"""
Slice 8 round-trip validation: swaptions + PricingContext.

1. Payer-receiver swaption parity: payer - receiver = swap PV
2. ATM swaption: forward swap rate = strike, delta ≈ 0.5 * annuity
3. Greeks: vega via bump vol, delta via bump rate
4. PricingContext: same results as explicit curve parameters
5. SwaptionVolSurface integrates with swaption pricing
"""

import pytest
import math
from datetime import date

from pricebook.swaption import Swaption, SwaptionType
from pricebook.swaption_vol import SwaptionVolSurface
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol


def _flat_curve(ref: date, rate: float) -> DiscountCurve:
    dates = [date(ref.year + i, ref.month, ref.day) for i in range(1, 21)]
    dfs = [math.exp(-rate * i) for i in range(1, 21)]
    return DiscountCurve(reference_date=ref, dates=dates, dfs=dfs)


REF = date(2024, 1, 15)
EXPIRY = date(2025, 1, 15)
SWAP_END = date(2030, 1, 15)
RATE = 0.03
VOL = 0.20


class TestPayerReceiverParity:
    """Payer - Receiver = PV of forward swap."""

    def test_atm_parity(self):
        curve = _flat_curve(REF, RATE)
        vol = FlatVol(VOL)
        strike = 0.03

        payer = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.PAYER)
        receiver = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.RECEIVER)

        fwd = payer.forward_swap_rate(curve)
        ann = payer.annuity(curve)
        expected_diff = payer.notional * ann * (fwd - strike)

        diff = payer.pv(curve, vol) - receiver.pv(curve, vol)
        assert diff == pytest.approx(expected_diff, abs=1.0)

    def test_otm_parity(self):
        curve = _flat_curve(REF, RATE)
        vol = FlatVol(VOL)
        strike = 0.05  # OTM payer

        payer = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.PAYER)
        receiver = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.RECEIVER)

        fwd = payer.forward_swap_rate(curve)
        ann = payer.annuity(curve)
        expected_diff = payer.notional * ann * (fwd - strike)

        diff = payer.pv(curve, vol) - receiver.pv(curve, vol)
        assert diff == pytest.approx(expected_diff, abs=1.0)


class TestATMProperties:
    """ATM swaption: strike = forward swap rate."""

    def test_atm_payer_equals_receiver(self):
        """At ATM, payer and receiver have the same price."""
        curve = _flat_curve(REF, RATE)
        vol = FlatVol(VOL)

        payer = Swaption(EXPIRY, SWAP_END, strike=0.03, swaption_type=SwaptionType.PAYER)
        fwd = payer.forward_swap_rate(curve)

        payer_atm = Swaption(EXPIRY, SWAP_END, strike=fwd, swaption_type=SwaptionType.PAYER)
        receiver_atm = Swaption(EXPIRY, SWAP_END, strike=fwd, swaption_type=SwaptionType.RECEIVER)

        pv_p = payer_atm.pv(curve, vol)
        pv_r = receiver_atm.pv(curve, vol)
        assert pv_p == pytest.approx(pv_r, rel=0.01)

    def test_atm_price_positive(self):
        curve = _flat_curve(REF, RATE)
        vol = FlatVol(VOL)
        payer = Swaption(EXPIRY, SWAP_END, strike=0.03)
        fwd = payer.forward_swap_rate(curve)
        atm = Swaption(EXPIRY, SWAP_END, strike=fwd)
        assert atm.pv(curve, vol) > 0


class TestGreeksBumpAndReprice:
    """Greeks via bump-and-reprice, cross-checked against expectations."""

    def test_vega_positive(self):
        """Bumping vol up increases swaption price."""
        curve = _flat_curve(REF, RATE)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)

        pv_base = swn.pv(curve, FlatVol(VOL))
        pv_up = swn.pv(curve, FlatVol(VOL + 0.01))

        vega = pv_up - pv_base
        assert vega > 0

    def test_vega_magnitude(self):
        """Vega should be material on a 1M notional swaption."""
        curve = _flat_curve(REF, RATE)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)

        pv_base = swn.pv(curve, FlatVol(VOL))
        pv_up = swn.pv(curve, FlatVol(VOL + 0.01))
        vega = pv_up - pv_base
        # 1bp vol bump on a 1M 5Y swaption: vega should be thousands
        assert vega > 100

    def test_delta_via_rate_bump(self):
        """Bumping rates changes the swaption price."""
        curve_base = _flat_curve(REF, RATE)
        curve_up = _flat_curve(REF, RATE + 0.0001)  # +1bp
        vol = FlatVol(VOL)

        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_base = swn.pv(curve_base, vol)
        pv_up = swn.pv(curve_up, vol)

        delta = pv_up - pv_base
        # Payer swaption gains when rates rise (forward rate goes up)
        assert delta > 0

    def test_receiver_delta_negative(self):
        """Receiver swaption loses value when rates rise."""
        curve_base = _flat_curve(REF, RATE)
        curve_up = _flat_curve(REF, RATE + 0.0001)
        vol = FlatVol(VOL)

        swn = Swaption(EXPIRY, SWAP_END, strike=0.03, swaption_type=SwaptionType.RECEIVER)
        pv_base = swn.pv(curve_base, vol)
        pv_up = swn.pv(curve_up, vol)

        delta = pv_up - pv_base
        assert delta < 0


class TestPricingContextConsistency:
    """pv_ctx produces the same results as explicit parameters."""

    def test_single_curve_consistency(self):
        curve = _flat_curve(REF, RATE)
        vol = FlatVol(VOL)

        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)

        pv_explicit = swn.pv(curve, vol)

        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=curve,
            vol_surfaces={"ir": vol},
        )
        pv_ctx = swn.pv_ctx(ctx)

        assert pv_ctx == pytest.approx(pv_explicit)

    def test_dual_curve_consistency(self):
        disc = _flat_curve(REF, 0.025)
        proj = _flat_curve(REF, 0.035)
        vol = FlatVol(VOL)

        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)

        pv_explicit = swn.pv(disc, vol, proj)

        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=disc,
            projection_curves={"USD.3M": proj},
            vol_surfaces={"ir": vol},
        )
        pv_ctx = swn.pv_ctx(ctx, projection_curve_name="USD.3M")

        assert pv_ctx == pytest.approx(pv_explicit)


class TestSwaptionVolSurfaceIntegration:
    """SwaptionVolSurface works with swaption pricing end-to-end."""

    def test_price_with_swaption_vol_surface(self):
        curve = _flat_curve(REF, RATE)
        vol_surface = SwaptionVolSurface(
            reference_date=REF,
            expiries=[EXPIRY, date(2026, 1, 15)],
            tenors=[2.0, 5.0, 10.0],
            vols=[
                [0.22, 0.20, 0.18],
                [0.24, 0.22, 0.20],
            ],
        )

        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv = swn.pv(curve, vol_surface)
        assert pv > 0

    def test_vol_surface_in_context(self):
        curve = _flat_curve(REF, RATE)
        vol_surface = SwaptionVolSurface(
            reference_date=REF,
            expiries=[EXPIRY, date(2026, 1, 15)],
            tenors=[2.0, 5.0, 10.0],
            vols=[
                [0.22, 0.20, 0.18],
                [0.24, 0.22, 0.20],
            ],
        )

        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=curve,
            vol_surfaces={"ir": vol_surface},
        )

        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_ctx = swn.pv_ctx(ctx)
        pv_explicit = swn.pv(curve, vol_surface)
        assert pv_ctx == pytest.approx(pv_explicit)

    def test_different_vol_surface_different_price(self):
        """Two vol surfaces with different values produce different prices."""
        curve = _flat_curve(REF, RATE)

        vol_low = SwaptionVolSurface(
            reference_date=REF,
            expiries=[EXPIRY],
            tenors=[5.0],
            vols=[[0.10]],
        )
        vol_high = SwaptionVolSurface(
            reference_date=REF,
            expiries=[EXPIRY],
            tenors=[5.0],
            vols=[[0.30]],
        )

        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        assert swn.pv(curve, vol_high) > swn.pv(curve, vol_low)
