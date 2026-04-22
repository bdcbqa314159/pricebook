"""Tests for collateralisation (COL1-COL12)."""

import math
from datetime import date

import pytest

from pricebook.csa import CSA, csa_discount_curve, colva
from pricebook.simm import SIMMCalculator, SIMMSensitivity
from tests.conftest import make_flat_curve


# ---- COL1: CSA-aware discounting ----

class TestCSADiscounting:
    def test_same_currency(self):
        """CSA in same currency → use that currency's OIS."""
        csa = CSA(currency="USD")
        curves = {"USD": make_flat_curve(date(2026, 4, 21), 0.04)}
        curve = csa_discount_curve(csa, "USD", curves)
        assert curve is curves["USD"]

    def test_cross_currency_fallback(self):
        """CSA in EUR on USD trade → use EUR OIS if no basis curve."""
        csa = CSA(currency="EUR")
        curves = {
            "USD": make_flat_curve(date(2026, 4, 21), 0.04),
            "EUR": make_flat_curve(date(2026, 4, 21), 0.03),
        }
        curve = csa_discount_curve(csa, "USD", curves)
        assert curve is curves["EUR"]

    def test_cross_currency_with_basis(self):
        """CSA in EUR on USD trade with basis curve → use basis."""
        csa = CSA(currency="EUR")
        usd = make_flat_curve(date(2026, 4, 21), 0.04)
        eur = make_flat_curve(date(2026, 4, 21), 0.03)
        basis = make_flat_curve(date(2026, 4, 21), 0.035)
        curve = csa_discount_curve(csa, "USD", {"USD": usd, "EUR": eur},
                                    xccy_basis_curves={"USD_EUR": basis})
        assert curve is basis

    def test_missing_both_currencies_raises(self):
        csa = CSA(currency="JPY")
        with pytest.raises(ValueError):
            csa_discount_curve(csa, "CHF", {"USD": make_flat_curve(date(2026, 4, 21), 0.04)})


# ---- COL7: ColVA ----

class TestColVA:
    def test_zero_spread_zero_colva(self):
        """Same collateral and discount rate → ColVA = 0."""
        cv = colva([100_000] * 10, [80_000] * 10, 0.04, 0.04, 0.25)
        assert cv == pytest.approx(0.0)

    def test_positive_spread_positive_colva(self):
        """Discount > collateral rate → positive ColVA (cost)."""
        cv = colva([100_000] * 4, [80_000] * 4, 0.03, 0.04, 0.25)
        assert cv > 0

    def test_negative_spread_negative_colva(self):
        """Discount < collateral rate → negative ColVA (benefit)."""
        cv = colva([100_000] * 4, [80_000] * 4, 0.05, 0.04, 0.25)
        assert cv < 0


# ---- COL4-COL6: ISDA SIMM ----

class TestSIMM:
    def test_single_girr_sensitivity(self):
        """Single GIRR sensitivity → margin = |delta × RW|."""
        calc = SIMMCalculator()
        sensitivities = [SIMMSensitivity("GIRR", "USD", "10Y", delta=1_000_000)]
        result = calc.compute(sensitivities)
        assert result.total_margin > 0
        assert result.n_sensitivities == 1

    def test_two_tenors_netting(self):
        """Two same-currency GIRR sensitivities should partially net."""
        calc = SIMMCalculator()
        single = calc.compute([SIMMSensitivity("GIRR", "USD", "10Y", delta=1_000_000)])
        double = calc.compute([
            SIMMSensitivity("GIRR", "USD", "10Y", delta=1_000_000),
            SIMMSensitivity("GIRR", "USD", "2Y", delta=-500_000),
        ])
        # With netting, two-tenor margin < sum of individual margins
        assert double.total_margin < single.total_margin * 1.5

    def test_two_currencies_diversification(self):
        """Two currencies → diversification benefit (inter-bucket corr < 1)."""
        calc = SIMMCalculator()
        usd = calc.compute([SIMMSensitivity("GIRR", "USD", "10Y", delta=1_000_000)])
        eur = calc.compute([SIMMSensitivity("GIRR", "EUR", "10Y", delta=1_000_000)])
        combined = calc.compute([
            SIMMSensitivity("GIRR", "USD", "10Y", delta=1_000_000),
            SIMMSensitivity("GIRR", "EUR", "10Y", delta=1_000_000),
        ])
        # Diversification: combined < USD + EUR
        assert combined.total_margin < usd.total_margin + eur.total_margin

    def test_fx_margin(self):
        calc = SIMMCalculator()
        result = calc.compute([SIMMSensitivity("FX", "EUR/USD", "spot", delta=10_000_000)])
        assert result.total_margin > 0

    def test_multi_risk_class(self):
        """GIRR + FX → across-risk-class aggregation."""
        calc = SIMMCalculator()
        result = calc.compute([
            SIMMSensitivity("GIRR", "USD", "10Y", delta=1_000_000),
            SIMMSensitivity("FX", "EUR/USD", "spot", delta=5_000_000),
        ])
        assert len(result.risk_classes) == 2
        assert result.total_margin > 0

    def test_empty(self):
        calc = SIMMCalculator()
        result = calc.compute([])
        assert result.total_margin == 0.0
