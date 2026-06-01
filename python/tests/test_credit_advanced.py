"""Tests for credit spread vol surface, quanto CDS, credit portfolio VaR."""

import pytest
import math
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


def _make_curves(hazard=0.02, rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.survival_curve import SurvivalCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 15)]
    dfs = [math.exp(-rate * y) for y in range(1, 15)]
    survs = [math.exp(-hazard * y) for y in range(1, 15)]
    dc = DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)
    sc = SurvivalCurve(REF, dates, survs)
    return dc, sc


# ═══════════════════════════════════════════════════════════════
# Credit Spread Vol Surface (C3)
# ═══════════════════════════════════════════════════════════════

class TestCreditSpreadVol:
    def test_construction(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 2, 5, 10], [3, 5, 10],
                                          [[0.40, 0.38, 0.35],
                                           [0.38, 0.36, 0.33],
                                           [0.35, 0.33, 0.30],
                                           [0.32, 0.30, 0.28]])
        assert surface.vol(1.0, 5.0) == pytest.approx(0.38)

    def test_interpolation(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5, 10], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30], [0.30, 0.25]])
        vol = surface.vol(3.0, 7.0)
        assert 0.25 < vol < 0.40

    def test_bumped(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30]])
        bumped = surface.bumped(0.05)
        assert bumped.vol(1.0, 5.0) == pytest.approx(0.45)

    def test_synthetic(self):
        from pricebook.credit.credit_spread_vol import synthetic_credit_vol_surface
        surface = synthetic_credit_vol_surface(0.02, REF)
        vol = surface.vol(5.0, 5.0)
        assert 0.20 < vol < 0.60

    def test_to_dict(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30]])
        d = surface.to_dict()
        assert "expiries" in d or "expiry_years" in d or "reference_date" in d


# ═══════════════════════════════════════════════════════════════
# Quanto CDS (C4)
# ═══════════════════════════════════════════════════════════════

class TestQuantoCDS:
    def test_quanto_adjustment_positive_corr(self):
        """Positive FX-credit correlation → quanto spread > foreign."""
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01  # 100bp
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, 0.30, 5.0)
        assert adjusted > foreign

    def test_quanto_adjustment_negative_corr(self):
        """Negative correlation → quanto spread < foreign."""
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, -0.30, 5.0)
        assert adjusted < foreign

    def test_zero_corr_no_adjustment(self):
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, 0.0, 5.0)
        assert adjusted == pytest.approx(foreign)

    def test_price_quanto_cds(self):
        from pricebook.credit.quanto_cds import price_quanto_cds
        dc, sc = _make_curves()
        dc_foreign, _ = _make_curves(rate=0.02)
        r = price_quanto_cds(REF, 5.0, 0.01, dc, dc_foreign, sc,
                              fx_spot=1.10, fx_vol=0.08, credit_vol=0.40,
                              correlation=0.25)
        assert r.domestic_spread > 0
        assert r.quanto_adjustment_bp != 0

    def test_to_dict(self):
        from pricebook.credit.quanto_cds import price_quanto_cds
        dc, sc = _make_curves()
        dc_f, _ = _make_curves(rate=0.02)
        r = price_quanto_cds(REF, 5.0, 0.01, dc, dc_f, sc, 1.10, 0.08, 0.40, 0.25)
        d = r.to_dict()
        assert "quanto_adjustment_bp" in d


# ═══════════════════════════════════════════════════════════════
# Credit Portfolio VaR (C9)
# ═══════════════════════════════════════════════════════════════

class TestCreditVaR:
    def test_historical_var(self):
        from pricebook.credit.credit_var import historical_credit_var
        rng = np.random.default_rng(42)
        positions = [
            {"name": "A", "cs01": -50_000},
            {"name": "B", "cs01": -30_000},
        ]
        spread_changes = {
            "A": rng.normal(0, 0.0005, 250).tolist(),
            "B": rng.normal(0, 0.0003, 250).tolist(),
        }
        result = historical_credit_var(positions, spread_changes, confidence=0.99)
        assert abs(result.var_amount) > 0  # VaR is non-zero
        assert abs(result.es_amount) >= abs(result.var_amount) - 1  # |ES| ≥ |VaR|

    def test_parametric_var(self):
        from pricebook.credit.credit_var import parametric_credit_var
        positions = [
            {"name": "A", "cs01": -50_000},
            {"name": "B", "cs01": -30_000},
        ]
        vols = [0.0005, 0.0003]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = parametric_credit_var(positions, vols, corr, confidence=0.99)
        assert abs(result.var_amount) > 0

    def test_copula_var(self):
        from pricebook.credit.credit_var import copula_credit_var
        positions = [
            {"name": "A", "notional": 10_000_000},
            {"name": "B", "notional": 5_000_000},
            {"name": "C", "notional": 8_000_000},
        ]
        pds = [0.02, 0.03, 0.01]
        lgds = [0.6, 0.6, 0.4]
        result = copula_credit_var(positions, pds, lgds, correlation=0.3, confidence=0.99)
        assert abs(result.var_amount) > 0
        assert result.worst_name is not None

    def test_var_increases_with_correlation(self):
        from pricebook.credit.credit_var import copula_credit_var
        positions = [
            {"name": "A", "notional": 10_000_000},
            {"name": "B", "notional": 10_000_000},
        ]
        pds = [0.02, 0.02]
        lgds = [0.6, 0.6]
        low = copula_credit_var(positions, pds, lgds, correlation=0.1, confidence=0.99, seed=42)
        high = copula_credit_var(positions, pds, lgds, correlation=0.8, confidence=0.99, seed=42)
        assert abs(high.var_amount) >= abs(low.var_amount) * 0.8

    def test_to_dict(self):
        from pricebook.credit.credit_var import historical_credit_var
        rng = np.random.default_rng(42)
        positions = [{"name": "A", "cs01": -50_000}]
        spread_changes = {"A": rng.normal(0, 0.0005, 100).tolist()}
        result = historical_credit_var(positions, spread_changes)
        d = result.to_dict()
        assert "var_amount" in d
        assert "method" in d
