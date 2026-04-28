"""Tests for MultiCurrencyCurveSet: build, templates, PricingContext."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.ibor_curve import bootstrap_ibor, EURIBOR_3M_CONVENTIONS
from pricebook.multi_currency_curves import (
    MultiCurrencyCurveSet,
    CurrencyCurveSetSpec,
    IBORCurveSpec,
)
from pricebook.ois import bootstrap_ois
from pricebook.swap import InterestRateSwap


REF = date(2026, 4, 27)


def _estr_rates():
    return [
        (REF + timedelta(days=365), 0.030),
        (REF + timedelta(days=730), 0.031),
        (REF + timedelta(days=1825), 0.033),
        (REF + timedelta(days=3650), 0.035),
    ]


def _euribor_3m_swaps():
    return [
        (REF + timedelta(days=365), 0.032),
        (REF + timedelta(days=730), 0.033),
        (REF + timedelta(days=1825), 0.035),
        (REF + timedelta(days=3650), 0.037),
    ]


def _euribor_6m_swaps():
    return [
        (REF + timedelta(days=365), 0.033),
        (REF + timedelta(days=730), 0.034),
        (REF + timedelta(days=1825), 0.036),
    ]


def _basis_3m_6m():
    return [
        (REF + timedelta(days=730), 0.0005),
        (REF + timedelta(days=1825), 0.0010),
    ]


def _usd_ois_rates():
    return [
        (REF + timedelta(days=365), 0.040),
        (REF + timedelta(days=730), 0.041),
        (REF + timedelta(days=1825), 0.042),
    ]


# ---- USD post-LIBOR ----

class TestUSDPostLibor:

    def test_single_curve(self):
        curves = MultiCurrencyCurveSet.usd_post_libor(REF, _usd_ois_rates())
        assert "USD" in curves.currencies
        ois = curves.ois("USD")
        assert ois.reference_date == REF

    def test_no_ibor_raises(self):
        curves = MultiCurrencyCurveSet.usd_post_libor(REF, _usd_ois_rates())
        with pytest.raises(KeyError, match="EURIBOR"):
            curves.ibor("EURIBOR_3M")

    def test_to_pricing_context(self):
        curves = MultiCurrencyCurveSet.usd_post_libor(REF, _usd_ois_rates())
        ctx = curves.to_pricing_context(reporting_currency="USD")
        assert ctx.valuation_date == REF
        assert ctx.discount_curve is not None


# ---- EUR with EURIBOR ----

class TestEURWithEuribor:

    def test_3m_only(self):
        curves = MultiCurrencyCurveSet.eur_with_euribor(
            REF, _estr_rates(), _euribor_3m_swaps(),
        )
        assert "EUR" in curves.currencies
        ibor = curves.ibor("EURIBOR_3M")
        assert ibor.tenor_months == 3

    def test_3m_plus_6m_direct(self):
        """6M bootstrapped directly from swap rates."""
        curves = MultiCurrencyCurveSet.eur_with_euribor(
            REF, _estr_rates(), _euribor_3m_swaps(),
            euribor_6m_swaps=_euribor_6m_swaps(),
        )
        ibor_3m = curves.ibor("EURIBOR_3M")
        ibor_6m = curves.ibor("EURIBOR_6M")
        assert ibor_6m.tenor_months == 6

    def test_3m_plus_6m_via_basis(self):
        """6M bootstrapped from 3M + tenor basis."""
        curves = MultiCurrencyCurveSet.eur_with_euribor(
            REF, _estr_rates(), _euribor_3m_swaps(),
            basis_3m_6m=_basis_3m_6m(),
        )
        ibor_6m = curves.ibor("EURIBOR_6M")
        assert ibor_6m.tenor_months == 6
        basis = curves.tenor_basis_for("EUR", "3M_6M")
        assert len(basis.dates) > 0

    def test_6m_above_3m(self):
        """6M forwards should be above 3M (positive tenor basis)."""
        curves = MultiCurrencyCurveSet.eur_with_euribor(
            REF, _estr_rates(), _euribor_3m_swaps(),
            basis_3m_6m=_basis_3m_6m(),
        )
        d1 = REF + timedelta(days=730)
        d2 = REF + timedelta(days=912)
        fwd_3m = curves.ibor("EURIBOR_3M").forward_rate(d1, d2)
        fwd_6m = curves.ibor("EURIBOR_6M").forward_rate(d1, d2)
        assert fwd_6m > fwd_3m

    def test_pricing_context_has_projection(self):
        curves = MultiCurrencyCurveSet.eur_with_euribor(
            REF, _estr_rates(), _euribor_3m_swaps(),
        )
        ctx = curves.to_pricing_context(reporting_currency="EUR")
        assert ctx.projection_curves is not None
        assert "EURIBOR_3M" in ctx.projection_curves


# ---- Manual construction ----

class TestManualConstruction:

    def test_add_currency(self):
        ois = DiscountCurve.flat(REF, 0.03)
        curves = MultiCurrencyCurveSet()
        curves.add_currency("GBP", ois)
        assert "GBP" in curves.currencies
        assert curves.ois("GBP") is ois

    def test_add_xccy(self):
        ois_usd = DiscountCurve.flat(REF, 0.04)
        ois_eur = DiscountCurve.flat(REF, 0.03)
        xccy = DiscountCurve.flat(REF, 0.028)

        curves = MultiCurrencyCurveSet()
        curves.add_currency("USD", ois_usd)
        curves.add_currency("EUR", ois_eur)
        curves.add_xccy_basis("USD", "EUR", xccy)

        assert curves.xccy_curve("USD", "EUR").reference_date == REF

    def test_missing_currency_raises(self):
        curves = MultiCurrencyCurveSet()
        with pytest.raises(KeyError, match="GBP"):
            curves.ois("GBP")


# ---- Build from specs ----

class TestBuildFromSpecs:

    def test_single_currency(self):
        spec = CurrencyCurveSetSpec(
            currency="EUR",
            ois_rates=_estr_rates(),
            ibor_specs=[
                IBORCurveSpec(
                    conventions=EURIBOR_3M_CONVENTIONS,
                    swaps=_euribor_3m_swaps(),
                ),
            ],
        )
        curves = MultiCurrencyCurveSet.build(REF, [spec])
        assert "EUR" in curves.currencies
        assert curves.ibor("EURIBOR_3M").tenor_months == 3

    def test_price_irs(self):
        """Full round-trip: build curves → PricingContext → price IRS."""
        curves = MultiCurrencyCurveSet.eur_with_euribor(
            REF, _estr_rates(), _euribor_3m_swaps(),
        )
        ctx = curves.to_pricing_context(reporting_currency="EUR")
        swap = InterestRateSwap(
            start=REF, end=REF + timedelta(days=1825),
            fixed_rate=0.035, notional=10_000_000,
        )
        pv = swap.pv(ctx.discount_curve)
        assert math.isfinite(pv)
