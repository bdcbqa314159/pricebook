"""Tests for European swaption pricing."""

import pytest
from datetime import date

from pricebook.options.swaption import Swaption, SwaptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.options.vol_surface import FlatVol
from pricebook.models.models import Black76Model
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from tests.conftest import make_flat_curve
import math


@pytest.fixture
def flat_curve():
    return make_flat_curve(date(2024, 1, 15), 0.03)


@pytest.fixture
def model():
    return Black76Model(vol=0.20)


class TestSwaptionConstruction:
    def test_basic_construction(self):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        assert swn.expiry == date(2025, 1, 15)
        assert swn.swaption_type == SwaptionType.PAYER
        assert swn.notional == 1_000_000.0

    def test_receiver_swaption(self):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15),
                       strike=0.03, swaption_type=SwaptionType.RECEIVER)
        assert swn.swaption_type == SwaptionType.RECEIVER

    def test_expiry_after_swap_end_raises(self):
        with pytest.raises(ValueError, match="expiry must be before swap_end"):
            Swaption(expiry=date(2030, 1, 15), swap_end=date(2025, 1, 15), strike=0.03)

    def test_expiry_equals_swap_end_raises(self):
        with pytest.raises(ValueError, match="expiry must be before swap_end"):
            Swaption(expiry=date(2025, 1, 15), swap_end=date(2025, 1, 15), strike=0.03)


class TestSwaptionForwardRate:
    def test_forward_swap_rate_flat_curve(self, flat_curve):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        fwd = swn.forward_swap_rate(flat_curve)
        assert fwd == pytest.approx(0.03, abs=0.002)

    def test_annuity_positive(self, flat_curve):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        assert swn.annuity(flat_curve) > 0


class TestSwaptionPricing:
    def test_payer_price_positive(self, flat_curve, model):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        assert swn.price(model, flat_curve) > 0

    def test_receiver_price_positive(self, flat_curve, model):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15),
                       strike=0.03, swaption_type=SwaptionType.RECEIVER)
        assert swn.price(model, flat_curve) > 0

    def test_payer_receiver_parity(self, flat_curve, model):
        strike = 0.03
        payer = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15),
                         strike=strike, swaption_type=SwaptionType.PAYER)
        receiver = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15),
                            strike=strike, swaption_type=SwaptionType.RECEIVER)

        pv_payer = payer.price(model, flat_curve)
        pv_receiver = receiver.price(model, flat_curve)

        fwd = payer.forward_swap_rate(flat_curve)
        ann = payer.annuity(flat_curve)
        expected_diff = payer.notional * ann * (fwd - strike)
        assert pv_payer - pv_receiver == pytest.approx(expected_diff, abs=1.0)

    def test_higher_vol_higher_price(self, flat_curve):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        assert swn.price(Black76Model(vol=0.30), flat_curve) > swn.price(Black76Model(vol=0.10), flat_curve)

    def test_deep_itm_payer_approx_intrinsic(self, flat_curve):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.01)
        fwd = swn.forward_swap_rate(flat_curve)
        ann = swn.annuity(flat_curve)
        intrinsic = swn.notional * ann * max(fwd - 0.01, 0)
        pv = swn.price(Black76Model(vol=0.01), flat_curve)
        assert pv == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_payer_near_zero(self, flat_curve):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.10)
        pv = swn.price(Black76Model(vol=0.01), flat_curve)
        assert pv < 100

    def test_longer_expiry_higher_price(self, flat_curve, model):
        short = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        long = Swaption(expiry=date(2026, 1, 15), swap_end=date(2031, 1, 15), strike=0.03)
        assert long.price(model, flat_curve) > short.price(model, flat_curve)


class TestSwaptionPricingContext:
    def test_pv_ctx_matches_price(self, flat_curve, model):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        ctx = PricingContext(
            valuation_date=flat_curve.reference_date,
            discount_curve=flat_curve,
            vol_surfaces={"ir": FlatVol(0.20)},
        )
        pv_price = swn.price(model, flat_curve)
        pv_ctx = swn.pv_ctx(ctx)
        assert pv_ctx == pytest.approx(pv_price)

    def test_pv_ctx_with_projection_curve(self, flat_curve, model):
        ref = date(2024, 1, 15)
        r_proj = 0.035
        proj_dates = [date(2024 + i, 1, 15) for i in range(1, 21)]
        proj_dfs = [math.exp(-r_proj * i) for i in range(1, 21)]
        proj_curve = DiscountCurve(reference_date=ref, dates=proj_dates, dfs=proj_dfs)

        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        ctx = PricingContext(
            valuation_date=ref,
            discount_curve=flat_curve,
            projection_curves={"USD.3M": proj_curve},
            vol_surfaces={"ir": FlatVol(0.20)},
        )

        pv_ctx = swn.pv_ctx(ctx, projection_curve_name="USD.3M")
        pv_price = swn.price(model, flat_curve, proj_curve)
        assert pv_ctx == pytest.approx(pv_price)

    def test_pv_ctx_missing_vol_raises(self, flat_curve):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        ctx = PricingContext(valuation_date=flat_curve.reference_date, discount_curve=flat_curve)
        with pytest.raises(KeyError, match="Vol surface"):
            swn.pv_ctx(ctx)

    def test_pv_ctx_missing_discount_raises(self):
        swn = Swaption(expiry=date(2025, 1, 15), swap_end=date(2030, 1, 15), strike=0.03)
        ctx = PricingContext(valuation_date=date(2024, 1, 15), vol_surfaces={"ir": FlatVol(0.20)})
        with pytest.raises(ValueError, match="discount_curve"):
            swn.pv_ctx(ctx)
