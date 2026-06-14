"""Regression for L2 T4 audit of `regulatory.stress_irrbb.calculate_eve_impact`:

Pre-fix used ``ΔEVE ≈ -DurationGap × Equity × Δr`` which is off by a
factor of A/E vs the textbook derivation:

    ΔE = ΔA − ΔL
       = −D_A·A·Δr + D_L·L·Δr
       = −A·(D_A − (L/A)·D_L)·Δr
       = −A · DurationGap · Δr

The same function also returns ``eve_change_pv01 = -net_pv01 × bps``,
where ``net_pv01 = A·D_A − L·D_L = A·DurationGap``.  That output was
already correct, so the two values in the same dict silently disagreed
by exactly the A/E ratio.  ``calculate_irrbb_capital`` consumes the
broken ``worst_eve_change``, so banks under-stated SOT outlier capital
by ~10× for typical (10:1) A/E ratios.
"""

from __future__ import annotations

import pytest

from pricebook.regulatory.stress_irrbb import (
    calculate_duration_gap, calculate_eve_impact, calculate_eve_all_scenarios,
    calculate_irrbb_capital,
)


class TestEVEMatchesPV01:
    def test_eve_change_equals_pv01_form(self):
        """Both outputs must equal -net_pv01 × bps = -A · DurGap · Δr."""
        assets = [{"notional": 100.0, "duration": 5.0}]
        liabs = [{"notional": 90.0, "duration": 2.0}]
        gap = calculate_duration_gap(assets, liabs)
        for bps in (100, 200, 250, -200):
            impact = calculate_eve_impact(gap, rate_shock_bps=bps)
            assert impact["eve_change"] == pytest.approx(
                impact["eve_change_pv01"], rel=1e-9
            )

    def test_textbook_formula(self):
        """Numerical check vs textbook: ΔE = -A · DurGap · Δr."""
        assets = [{"notional": 100.0, "duration": 5.0}]
        liabs = [{"notional": 90.0, "duration": 2.0}]
        # D_A=5, D_L=2, L/A=0.9, DurGap = 5 - 0.9·2 = 3.2.
        # ΔE @ 200bp = -100 · 3.2 · 0.02 = -6.4.  (Pre-fix gave -0.64.)
        gap = calculate_duration_gap(assets, liabs)
        impact = calculate_eve_impact(gap, rate_shock_bps=200)
        assert impact["eve_change"] == pytest.approx(-6.4, rel=1e-9)


class TestIRRBBOutlierCapitalDoesNotUnderstate:
    """SOT outlier test: bank is outlier if max EVE loss > 15% of Tier1."""

    def test_realistic_bank_correctly_flagged(self):
        """Bank with assets=$10B, equity=$1B (10:1) and duration gap of 2y:
        ΔEVE @ 200bp = -10B · 2 · 0.02 = -$400M loss.
        15% of $1B Tier1 = $150M → must be outlier.
        Pre-fix code: -1B · 2 · 0.02 = -$40M loss → wrongly not outlier."""
        assets = [{"notional": 10_000_000_000.0, "duration": 5.0}]
        liabs = [{"notional": 9_000_000_000.0, "duration": 3.0}]
        # D_A=5, D_L=3, L/A=0.9, DurGap = 5 - 0.9·3 = 2.3.
        # ΔEVE @ 200bp = -10e9 · 2.3 · 0.02 = -460M.
        result = calculate_eve_all_scenarios(assets, liabs, "USD")
        assert result["worst_eve_change"] == pytest.approx(-460_000_000.0, rel=1e-6)

        # 15% of $1B Tier1 = $150M; loss $460M → outlier with $310M charge.
        cap = calculate_irrbb_capital(result, tier1_capital=1_000_000_000.0)
        assert cap["is_outlier"]
        assert cap["capital_charge"] == pytest.approx(310_000_000.0, rel=1e-6)


class TestEVEDirection:
    def test_asset_sensitive_book_loses_when_rates_up(self):
        assets = [{"notional": 100.0, "duration": 5.0}]
        liabs = [{"notional": 80.0, "duration": 2.0}]
        gap = calculate_duration_gap(assets, liabs)
        up = calculate_eve_impact(gap, rate_shock_bps=200)
        down = calculate_eve_impact(gap, rate_shock_bps=-200)
        assert up["eve_change"] < 0
        assert down["eve_change"] > 0
        assert up["eve_change"] == pytest.approx(-down["eve_change"], rel=1e-9)
