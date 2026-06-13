"""Regression for L2 phase-2 audit of `risk.simm`:

(a) Cross-risk-class aggregation: pre-fix used zero correlation
    (`sqrt(Σ M_i²)`) despite ISDA SIMM v2.6 specifying explicit cross-
    class correlations (e.g. GIRR-FX = 0.20).  Now uses the Table-21
    correlation matrix.

(b) Vega and curvature silently ignored: the SIMMSensitivity dataclass
    has vega and curvature fields, but pre-fix `_compute_bucket` only
    read `s.delta`.  Options books were materially under-margined.
    Now aggregates each component separately and combines via sum-of-
    squares.
"""

from __future__ import annotations

import math

import pytest

from pricebook.risk.simm import SIMMCalculator, SIMMSensitivity


class TestCrossRiskClassCorrelation:
    def test_correlated_margin_above_independent(self):
        """Diversified GIRR+FX book should give margin > zero-corr aggregation."""
        # Two risk classes with comparable margin contributions.
        sens = [
            SIMMSensitivity("GIRR", "USD", "5Y", delta=100_000),
            SIMMSensitivity("FX", "EUR/USD", "spot", delta=200_000),
        ]
        result = SIMMCalculator().compute(sens)
        # Zero-corr would give sqrt(M_GIRR² + M_FX²).
        # Post-fix with ρ=0.20: sqrt(M_GIRR² + M_FX² + 2·0.20·M_GIRR·M_FX) > that.
        m_girr = result.risk_classes[0].total
        m_fx = result.risk_classes[1].total
        zero_corr = math.sqrt(m_girr ** 2 + m_fx ** 2)
        assert result.total_margin > zero_corr - 1e-6
        # And specifically with ρ=0.20:
        expected = math.sqrt(m_girr ** 2 + m_fx ** 2 + 2 * 0.20 * m_girr * m_fx)
        assert result.total_margin == pytest.approx(expected, rel=1e-9)


class TestVegaCurvatureIncluded:
    def test_vega_increases_margin(self):
        """Adding vega to the same trade should increase margin."""
        delta_only = SIMMCalculator().compute([
            SIMMSensitivity("GIRR", "USD", "5Y", delta=100_000, vega=0, curvature=0),
        ]).total_margin
        delta_and_vega = SIMMCalculator().compute([
            SIMMSensitivity("GIRR", "USD", "5Y", delta=100_000, vega=50_000, curvature=0),
        ]).total_margin
        assert delta_and_vega > delta_only

    def test_curvature_increases_margin(self):
        no_curv = SIMMCalculator().compute([
            SIMMSensitivity("EQ", "AAPL", "spot", delta=100_000),
        ]).total_margin
        with_curv = SIMMCalculator().compute([
            SIMMSensitivity("EQ", "AAPL", "spot", delta=100_000, curvature=20_000),
        ]).total_margin
        assert with_curv > no_curv

    def test_components_sum_of_squares(self):
        """For a single sensitivity with delta=d, vega=v: margin = rw·sqrt(d²+v²)."""
        rw = 0.05  # GIRR 5Y is 56bp = 0.0056; let's use a clear case
        # GIRR 5Y rw = 56bp = 0.0056.
        d, v = 100_000, 60_000
        result = SIMMCalculator().compute([
            SIMMSensitivity("GIRR", "USD", "5Y", delta=d, vega=v, curvature=0),
        ])
        rw_5y = 56 / 10000
        expected = rw_5y * math.sqrt(d ** 2 + v ** 2)
        assert result.total_margin == pytest.approx(expected, rel=1e-9)


class TestDeltaOnlyUnchanged:
    """Existing delta-only callers shouldn't see numerical changes within a single risk class."""

    def test_single_risk_class_delta_only(self):
        sens = [
            SIMMSensitivity("GIRR", "USD", "5Y", delta=100_000),
            SIMMSensitivity("GIRR", "USD", "10Y", delta=80_000),
        ]
        result = SIMMCalculator().compute(sens)
        # Single risk class → cross-class correction has no effect.
        # Within-bucket aggregation of two same-bucket sensitivities.
        rw_5y, rw_10y = 56 / 10000, 56 / 10000
        d1, d2 = 100_000 * rw_5y, 80_000 * rw_10y
        intra = 0.98
        expected = math.sqrt(d1 ** 2 + d2 ** 2 + 2 * intra * d1 * d2)
        assert result.total_margin == pytest.approx(expected, rel=1e-9)
