"""Regression for L2 phase-2 audit of `regulatory.market_risk_sa.calculate_curvature_capital`:

Pre-fix used ``sum(bucket_caps.values())`` for cross-bucket aggregation
— equivalent to γ=1 for all bucket pairs (full positive correlation).
FRTB MAR21.14 specifies
    K_curvature = sqrt[max(0, Σ K_b² + Σ_{b≠c} γ_bc · S_b · S_c · ψ)]
with γ in {0.15, 0.20, 0.25, 0.50, 0.60} by risk class and ψ=0 when
both bucket S_b, S_c are negative.

Fix: implement the FRTB cross-bucket aggregation with γ and ψ.
"""

from __future__ import annotations

import math

import pytest

from pricebook.regulatory.market_risk_sa import calculate_curvature_capital


class TestCurvatureCrossBucketCorrelation:
    def test_single_bucket_unchanged(self):
        """With only one bucket, no cross-bucket aggregation needed; K = K_b."""
        positions = [
            {"bucket": "A", "cvr_up": 100, "cvr_down": -50},
            {"bucket": "A", "cvr_up": 80, "cvr_down": -30},
        ]
        result = calculate_curvature_capital(positions, risk_class="EQ")
        # K_b^up = 180 (sum of cvr_ups, simplified within-bucket).
        # K_b = max(180, -80, 0) = 180.  K = sqrt(180²) = 180.
        assert result["capital"] == pytest.approx(180.0, abs=1e-6)

    def test_two_buckets_with_correlation(self):
        """Two buckets with positive S → sqrt(K1² + K2² + 2γ·S1·S2)."""
        # γ_EQ = 0.15.
        positions = [
            {"bucket": "A", "cvr_up": 100, "cvr_down": 50},
            {"bucket": "B", "cvr_up": 80, "cvr_down": 40},
        ]
        result = calculate_curvature_capital(positions, risk_class="EQ")
        # K_A = max(100, 50) = 100, S_A = 100.
        # K_B = max(80, 40) = 80, S_B = 80.
        # K² = 100² + 80² + 2 · 0.15 · 100 · 80 · 1 (both positive → ψ=1)
        #    = 10000 + 6400 + 2400 = 18800.
        expected = math.sqrt(18800)
        assert result["capital"] == pytest.approx(expected, abs=1e-6)

    def test_negative_buckets_psi_zero(self):
        """Two buckets where both worst-cases are negative → ψ=0 → cross term drops."""
        positions = [
            {"bucket": "A", "cvr_up": -100, "cvr_down": -50},  # worst (max-side) = -50
            {"bucket": "B", "cvr_up": -80, "cvr_down": -40},   # worst = -40
        ]
        # Both K_b^up and K_b^down → max(·, 0) = 0 (since all cvrs negative).
        # So K_b = 0 for both; K = 0.
        result = calculate_curvature_capital(positions, risk_class="EQ")
        assert result["capital"] == pytest.approx(0.0, abs=1e-6)

    def test_lower_correlation_lower_capital(self):
        """Cross-asset (EQ γ=0.15) gives lower K than full sum (pre-fix equivalent)."""
        positions = [
            {"bucket": "A", "cvr_up": 100, "cvr_down": 50},
            {"bucket": "B", "cvr_up": 100, "cvr_down": 50},
        ]
        eq_result = calculate_curvature_capital(positions, risk_class="EQ")
        # Plain sum (pre-fix): 100 + 100 = 200.
        # Post-fix γ=0.15: sqrt(2·100² + 2·0.15·100·100) = sqrt(23000) ≈ 151.66.
        assert eq_result["capital"] == pytest.approx(math.sqrt(23000), rel=1e-6)
        assert eq_result["capital"] < 200.0  # less than plain sum


class TestRiskClassCorrelations:
    def test_fx_has_higher_correlation(self):
        positions = [
            {"bucket": "A", "cvr_up": 100, "cvr_down": 0},
            {"bucket": "B", "cvr_up": 100, "cvr_down": 0},
        ]
        eq_k = calculate_curvature_capital(positions, risk_class="EQ")["capital"]  # γ=0.15
        fx_k = calculate_curvature_capital(positions, risk_class="FX")["capital"]  # γ=0.60
        # FX higher correlation → higher K (cross term larger).
        assert fx_k > eq_k
