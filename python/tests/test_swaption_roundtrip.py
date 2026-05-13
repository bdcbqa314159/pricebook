"""
Slice 8 round-trip validation: swaptions + PricingContext.

1. Payer-receiver swaption parity: payer - receiver = swap PV
2. ATM swaption: forward swap rate = strike, delta ~ 0.5 * annuity
3. Greeks: vega via bump vol, delta via bump rate
4. PricingContext: same results as explicit .price(model, curve)
5. SwaptionVolSurface integrates with swaption pricing
"""

import pytest
from datetime import date

from pricebook.swaption import Swaption, SwaptionType
from pricebook.swaption_vol import SwaptionVolSurface
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol
from pricebook.models import Black76Model
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
EXPIRY = date(2025, 1, 15)
SWAP_END = date(2030, 1, 15)
RATE = 0.03
VOL = 0.20

_model = Black76Model(vol=VOL)


class TestPayerReceiverParity:

    def test_atm_parity(self):
        curve = make_flat_curve(REF, RATE)
        strike = 0.03
        payer = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.PAYER)
        receiver = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.RECEIVER)
        fwd = payer.forward_swap_rate(curve)
        ann = payer.annuity(curve)
        expected_diff = payer.notional * ann * (fwd - strike)
        diff = payer.price(_model, curve) - receiver.price(_model, curve)
        assert diff == pytest.approx(expected_diff, abs=1.0)

    def test_otm_parity(self):
        curve = make_flat_curve(REF, RATE)
        strike = 0.05
        payer = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.PAYER)
        receiver = Swaption(EXPIRY, SWAP_END, strike, SwaptionType.RECEIVER)
        fwd = payer.forward_swap_rate(curve)
        ann = payer.annuity(curve)
        expected_diff = payer.notional * ann * (fwd - strike)
        diff = payer.price(_model, curve) - receiver.price(_model, curve)
        assert diff == pytest.approx(expected_diff, abs=1.0)


class TestATMProperties:

    def test_atm_payer_equals_receiver(self):
        curve = make_flat_curve(REF, RATE)
        payer = Swaption(EXPIRY, SWAP_END, strike=0.03, swaption_type=SwaptionType.PAYER)
        fwd = payer.forward_swap_rate(curve)
        payer_atm = Swaption(EXPIRY, SWAP_END, strike=fwd, swaption_type=SwaptionType.PAYER)
        receiver_atm = Swaption(EXPIRY, SWAP_END, strike=fwd, swaption_type=SwaptionType.RECEIVER)
        pv_p = payer_atm.price(_model, curve)
        pv_r = receiver_atm.price(_model, curve)
        assert pv_p == pytest.approx(pv_r, rel=0.01)

    def test_atm_price_positive(self):
        curve = make_flat_curve(REF, RATE)
        payer = Swaption(EXPIRY, SWAP_END, strike=0.03)
        fwd = payer.forward_swap_rate(curve)
        atm = Swaption(EXPIRY, SWAP_END, strike=fwd)
        assert atm.price(_model, curve) > 0


class TestGreeksBumpAndReprice:

    def test_vega_positive(self):
        curve = make_flat_curve(REF, RATE)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_base = swn.price(Black76Model(vol=VOL), curve)
        pv_up = swn.price(Black76Model(vol=VOL + 0.01), curve)
        assert pv_up - pv_base > 0

    def test_vega_magnitude(self):
        curve = make_flat_curve(REF, RATE)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_base = swn.price(Black76Model(vol=VOL), curve)
        pv_up = swn.price(Black76Model(vol=VOL + 0.01), curve)
        assert pv_up - pv_base > 100

    def test_delta_via_rate_bump(self):
        curve_base = make_flat_curve(REF, RATE)
        curve_up = make_flat_curve(REF, RATE + 0.0001)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_base = swn.price(_model, curve_base)
        pv_up = swn.price(_model, curve_up)
        assert pv_up - pv_base > 0

    def test_receiver_delta_negative(self):
        curve_base = make_flat_curve(REF, RATE)
        curve_up = make_flat_curve(REF, RATE + 0.0001)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03, swaption_type=SwaptionType.RECEIVER)
        pv_base = swn.price(_model, curve_base)
        pv_up = swn.price(_model, curve_up)
        assert pv_up - pv_base < 0


class TestPricingContextConsistency:

    def test_single_curve_consistency(self):
        curve = make_flat_curve(REF, RATE)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_price = swn.price(_model, curve)
        ctx = PricingContext(
            valuation_date=REF, discount_curve=curve,
            vol_surfaces={"ir": FlatVol(VOL)},
        )
        pv_ctx = swn.pv_ctx(ctx)
        assert pv_ctx == pytest.approx(pv_price)

    def test_dual_curve_consistency(self):
        disc = make_flat_curve(REF, 0.025)
        proj = make_flat_curve(REF, 0.035)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_price = swn.price(_model, disc, proj)
        ctx = PricingContext(
            valuation_date=REF, discount_curve=disc,
            projection_curves={"USD.3M": proj},
            vol_surfaces={"ir": FlatVol(VOL)},
        )
        pv_ctx = swn.pv_ctx(ctx, projection_curve_name="USD.3M")
        assert pv_ctx == pytest.approx(pv_price)


class TestSwaptionVolSurfaceIntegration:

    def test_price_with_swaption_vol_surface(self):
        curve = make_flat_curve(REF, RATE)
        vol_surface = SwaptionVolSurface(
            reference_date=REF,
            expiries=[EXPIRY, date(2026, 1, 15)],
            tenors=[2.0, 5.0, 10.0],
            vols=[[0.22, 0.20, 0.18], [0.24, 0.22, 0.20]],
        )
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        # Use vol from surface to create model
        vol = vol_surface.vol(EXPIRY, 0.03)
        pv = swn.price(Black76Model(vol=vol), curve)
        assert pv > 0

    def test_vol_surface_in_context(self):
        curve = make_flat_curve(REF, RATE)
        vol_surface = SwaptionVolSurface(
            reference_date=REF,
            expiries=[EXPIRY, date(2026, 1, 15)],
            tenors=[2.0, 5.0, 10.0],
            vols=[[0.22, 0.20, 0.18], [0.24, 0.22, 0.20]],
        )
        ctx = PricingContext(
            valuation_date=REF, discount_curve=curve,
            vol_surfaces={"ir": vol_surface},
        )
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        pv_ctx = swn.pv_ctx(ctx)
        vol = vol_surface.vol(EXPIRY, 0.03)
        pv_price = swn.price(Black76Model(vol=vol), curve)
        assert pv_ctx == pytest.approx(pv_price)

    def test_different_vol_surface_different_price(self):
        curve = make_flat_curve(REF, RATE)
        swn = Swaption(EXPIRY, SWAP_END, strike=0.03)
        assert swn.price(Black76Model(vol=0.30), curve) > swn.price(Black76Model(vol=0.10), curve)
