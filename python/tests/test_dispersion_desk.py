"""Tests for dispersion trading desk."""

import math

import pytest
from datetime import date

from pricebook.dispersion_desk import (
    CorrelationTermStructure,
    DispersionTrade,
    historical_correlation,
    implied_correlation,
    index_variance,
    index_vol,
)


# ---- Step 1: implied correlation + dispersion ----

class TestIndexVariance:
    def test_zero_correlation(self):
        # All idiosyncratic: σ²_idx = Σ w_i² σ_i²
        weights = [0.5, 0.5]
        vols = [0.20, 0.30]
        var = index_variance(weights, vols, correlation=0.0)
        assert var == pytest.approx(0.25 * 0.04 + 0.25 * 0.09)

    def test_full_correlation(self):
        # Perfectly correlated: σ_idx = Σ w_i σ_i, var = (Σ w_i σ_i)²
        weights = [0.5, 0.5]
        vols = [0.20, 0.30]
        var = index_variance(weights, vols, correlation=1.0)
        expected = (0.5 * 0.20 + 0.5 * 0.30) ** 2
        assert var == pytest.approx(expected)

    def test_monotone_in_correlation(self):
        weights = [0.4, 0.3, 0.3]
        vols = [0.25, 0.30, 0.20]
        v0 = index_variance(weights, vols, 0.0)
        v_mid = index_variance(weights, vols, 0.5)
        v1 = index_variance(weights, vols, 1.0)
        assert v0 < v_mid < v1


class TestImpliedCorrelation:
    def test_round_trip(self):
        """Build index vol from a known ρ; recover the same ρ."""
        weights = [0.4, 0.3, 0.3]
        vols = [0.25, 0.30, 0.20]
        rho_in = 0.45
        idx_v = index_vol(weights, vols, rho_in)
        rho_out = implied_correlation(weights, vols, idx_v)
        assert rho_out == pytest.approx(rho_in, rel=1e-9)

    def test_zero_when_idx_var_equals_diag(self):
        weights = [0.5, 0.5]
        vols = [0.20, 0.30]
        sum_w2 = 0.25 * 0.04 + 0.25 * 0.09
        idx_v = math.sqrt(sum_w2)
        assert implied_correlation(weights, vols, idx_v) == pytest.approx(0.0)

    def test_one_at_full_correlation(self):
        weights = [0.5, 0.5]
        vols = [0.20, 0.30]
        idx_v = index_vol(weights, vols, 1.0)
        assert implied_correlation(weights, vols, idx_v) == pytest.approx(1.0)

    def test_single_name_returns_zero(self):
        # Degenerate: all weight on one name → denom = 0 → fall through to 0
        assert implied_correlation([1.0], [0.20], 0.20) == 0.0


class TestDispersionTrade:
    def _trade(self, direction: int = 1):
        return DispersionTrade(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.3, 0.3],
            single_strikes=[0.04, 0.04, 0.04],
            single_notionals=[1_000_000, 1_000_000, 1_000_000],
            index_strike=0.04,
            index_notional=1_000_000,
            direction=direction,
        )

    def test_construction_validates_lengths(self):
        with pytest.raises(ValueError):
            DispersionTrade(
                tickers=["AAPL", "MSFT"],
                weights=[0.5],
                single_strikes=[0.04, 0.04],
                single_notionals=[1.0, 1.0],
                index_strike=0.04,
                index_notional=1.0,
            )

    def test_n_names(self):
        assert self._trade().n_names == 3

    def test_dispersion_max_at_zero_correlation(self):
        """Step 1 test: dispersion P&L is maximised at ρ = 0."""
        trade = self._trade()
        vols = [0.25, 0.30, 0.20]
        d0 = trade.dispersion_value(vols, 0.0)
        d_mid = trade.dispersion_value(vols, 0.5)
        d1 = trade.dispersion_value(vols, 1.0)
        assert d0 > d_mid > d1
        # Long dispersion is non-negative when ρ ∈ [0, 1] (Jensen on σ²).
        assert d0 > 0
        assert d1 >= 0

    def test_dispersion_short_flips_sign(self):
        long = self._trade(direction=1)
        short = self._trade(direction=-1)
        vols = [0.25, 0.30, 0.20]
        assert long.dispersion_value(vols, 0.3) == pytest.approx(
            -short.dispersion_value(vols, 0.3)
        )

    def test_pv_at_strike_correlation(self):
        """If single vols realise their strikes and idx vol realises its
        strike, PV = 0."""
        trade = DispersionTrade(
            tickers=["A", "B"],
            weights=[0.5, 0.5],
            single_strikes=[0.04, 0.04],
            single_notionals=[1.0, 1.0],
            index_strike=0.04,
            index_notional=1.0,
            direction=1,
        )
        # σ_i = 0.20 → var = 0.04 (at strike)
        # σ²_idx with ρ such that σ²_idx = 0.04: ρ = 1 here
        pv = trade.pv([0.20, 0.20], correlation=1.0)
        assert pv == pytest.approx(0.0)

    def test_pv_vol_mismatch_raises(self):
        trade = self._trade()
        with pytest.raises(ValueError):
            trade.pv([0.20, 0.20], correlation=0.5)


# ---- Step 2: correlation risk + term structure ----

class TestCorrelationRisk:
    def _trade(self, direction: int = 1):
        return DispersionTrade(
            tickers=["A", "B", "C"],
            weights=[0.4, 0.3, 0.3],
            single_strikes=[0.04, 0.04, 0.04],
            single_notionals=[1_000_000, 1_000_000, 1_000_000],
            index_strike=0.04,
            index_notional=1_000_000,
            direction=direction,
        )

    def test_long_dispersion_negative_corr_sensitivity(self):
        trade = self._trade(direction=1)
        sens = trade.correlation_sensitivity([0.25, 0.30, 0.20])
        assert sens < 0.0  # higher ρ → long dispersion loses

    def test_short_dispersion_positive_corr_sensitivity(self):
        trade = self._trade(direction=-1)
        sens = trade.correlation_sensitivity([0.25, 0.30, 0.20])
        assert sens > 0.0

    def test_correlation_rises_long_dispersion_loses(self):
        """Step 2 test: correlation rises → dispersion trade loses."""
        trade = self._trade(direction=1)
        vols = [0.25, 0.30, 0.20]
        pv_low = trade.pv(vols, correlation=0.20)
        pv_high = trade.pv(vols, correlation=0.80)
        assert pv_high < pv_low

    def test_analytic_sensitivity_matches_finite_difference(self):
        trade = self._trade(direction=1)
        vols = [0.25, 0.30, 0.20]
        rho = 0.50
        d_rho = 1e-5
        analytic = trade.correlation_sensitivity(vols)
        numeric = (
            trade.pv(vols, rho + d_rho) - trade.pv(vols, rho - d_rho)
        ) / (2 * d_rho)
        assert analytic == pytest.approx(numeric, rel=1e-6)


class TestHistoricalCorrelation:
    def test_perfect_correlation(self):
        r1 = [0.01, -0.02, 0.03, -0.01, 0.02]
        r2 = [0.02, -0.04, 0.06, -0.02, 0.04]
        assert historical_correlation([r1, r2]) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_anti_correlation(self):
        r1 = [0.01, -0.02, 0.03, -0.01]
        r2 = [-0.01, 0.02, -0.03, 0.01]
        assert historical_correlation([r1, r2]) == pytest.approx(-1.0, abs=1e-9)

    def test_three_series_average(self):
        # All identical → average pairwise correlation = 1
        r = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert historical_correlation([r, r, r]) == pytest.approx(1.0, abs=1e-9)

    def test_single_series_returns_zero(self):
        assert historical_correlation([[0.01, 0.02]]) == 0.0


class TestCorrelationTermStructure:
    def test_pillar_lookup(self):
        ref = date(2024, 1, 15)
        ts = CorrelationTermStructure(
            reference_date=ref,
            expiries=[date(2024, 7, 15), date(2025, 1, 15)],
            correlations=[0.40, 0.55],
        )
        assert ts.correlation(date(2024, 7, 15)) == pytest.approx(0.40)
        assert ts.correlation(date(2025, 1, 15)) == pytest.approx(0.55)

    def test_flat_extrapolation(self):
        ref = date(2024, 1, 15)
        ts = CorrelationTermStructure(
            ref, [date(2024, 7, 15), date(2025, 1, 15)], [0.40, 0.55],
        )
        assert ts.correlation(date(2024, 1, 1)) == pytest.approx(0.40)
        assert ts.correlation(date(2030, 1, 1)) == pytest.approx(0.55)

    def test_linear_interpolation(self):
        ref = date(2024, 1, 15)
        ts = CorrelationTermStructure(
            ref,
            [date(2024, 1, 15), date(2025, 1, 14)],  # ~365 days apart
            [0.20, 0.60],
        )
        # Halfway through ~ avg of pillars
        mid = date(2024, 7, 15)
        result = ts.correlation(mid)
        assert 0.20 < result < 0.60

    def test_unsorted_input_sorted_internally(self):
        ref = date(2024, 1, 15)
        ts = CorrelationTermStructure(
            ref,
            [date(2025, 1, 15), date(2024, 7, 15)],
            [0.55, 0.40],
        )
        assert ts.expiries[0] == date(2024, 7, 15)
        assert ts.correlations[0] == 0.40

    def test_n_pillars(self):
        ref = date(2024, 1, 15)
        ts = CorrelationTermStructure(
            ref, [date(2024, 7, 15), date(2025, 1, 15)], [0.40, 0.55],
        )
        assert ts.n_pillars == 2

    def test_validation(self):
        ref = date(2024, 1, 15)
        with pytest.raises(ValueError):
            CorrelationTermStructure(ref, [date(2024, 7, 15)], [0.4, 0.5])
        with pytest.raises(ValueError):
            CorrelationTermStructure(ref, [], [])
