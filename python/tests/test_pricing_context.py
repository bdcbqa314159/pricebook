"""Tests for PricingContext."""

import pytest
from datetime import date

from pricebook.pricing_context import PricingContext
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.vol_surface import FlatVol
from pricebook.day_count import DayCountConvention


def _make_curve(ref: date) -> DiscountCurve:
    """Helper: simple flat 5% discount curve."""
    return DiscountCurve(
        reference_date=ref,
        dates=[date(2025, 1, 15)],
        dfs=[0.95],
    )


def _make_survival(ref: date) -> SurvivalCurve:
    """Helper: simple survival curve."""
    return SurvivalCurve(
        reference_date=ref,
        dates=[date(2025, 1, 15)],
        survival_probs=[0.98],
    )


class TestPricingContextConstruction:
    def test_minimal_context(self):
        ctx = PricingContext(valuation_date=date(2024, 1, 15))
        assert ctx.valuation_date == date(2024, 1, 15)
        assert ctx.discount_curve is None
        assert ctx.projection_curves == {}
        assert ctx.vol_surfaces == {}
        assert ctx.credit_curves == {}
        assert ctx.fx_spots == {}

    def test_full_context(self):
        ref = date(2024, 1, 15)
        disc = _make_curve(ref)
        proj = _make_curve(ref)
        vol = FlatVol(0.20)
        credit = _make_survival(ref)

        ctx = PricingContext(
            valuation_date=ref,
            discount_curve=disc,
            projection_curves={"USD.3M": proj},
            vol_surfaces={"ir": vol},
            credit_curves={"ACME": credit},
            fx_spots={("EUR", "USD"): 1.0850},
        )

        assert ctx.discount_curve is disc
        assert ctx.projection_curves["USD.3M"] is proj
        assert ctx.vol_surfaces["ir"] is vol
        assert ctx.credit_curves["ACME"] is credit
        assert ctx.fx_spots[("EUR", "USD")] == 1.0850


class TestPricingContextAccessors:
    @pytest.fixture
    def ctx(self):
        ref = date(2024, 1, 15)
        return PricingContext(
            valuation_date=ref,
            discount_curve=_make_curve(ref),
            projection_curves={"USD.3M": _make_curve(ref)},
            vol_surfaces={"ir": FlatVol(0.20)},
            credit_curves={"ACME": _make_survival(ref)},
            fx_spots={("EUR", "USD"): 1.0850},
        )

    def test_get_projection_curve(self, ctx):
        curve = ctx.get_projection_curve("USD.3M")
        assert isinstance(curve, DiscountCurve)

    def test_get_projection_curve_missing(self, ctx):
        with pytest.raises(KeyError, match="Projection curve 'GBP.6M'"):
            ctx.get_projection_curve("GBP.6M")

    def test_get_vol_surface(self, ctx):
        vol = ctx.get_vol_surface("ir")
        assert vol.vol() == pytest.approx(0.20)

    def test_get_vol_surface_missing(self, ctx):
        with pytest.raises(KeyError, match="Vol surface 'fx'"):
            ctx.get_vol_surface("fx")

    def test_get_credit_curve(self, ctx):
        curve = ctx.get_credit_curve("ACME")
        assert isinstance(curve, SurvivalCurve)

    def test_get_credit_curve_missing(self, ctx):
        with pytest.raises(KeyError, match="Credit curve 'BETA'"):
            ctx.get_credit_curve("BETA")

    def test_get_fx_spot(self, ctx):
        assert ctx.get_fx_spot("EUR", "USD") == 1.0850

    def test_get_fx_spot_missing(self, ctx):
        with pytest.raises(KeyError, match="FX spot 'GBP/USD'"):
            ctx.get_fx_spot("GBP", "USD")

    def test_multiple_projection_curves(self):
        ref = date(2024, 1, 15)
        c1 = _make_curve(ref)
        c2 = _make_curve(ref)
        ctx = PricingContext(
            valuation_date=ref,
            projection_curves={"USD.3M": c1, "USD.6M": c2},
        )
        assert ctx.get_projection_curve("USD.3M") is c1
        assert ctx.get_projection_curve("USD.6M") is c2
