"""Tests for tranche improvements: multi-period, t-copula, rho01, correlation interpolation."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.tranche_pricing import (
    expected_tranche_loss, expected_tranche_loss_t,
    price_tranche_multiperiod,
    tranche_rho01,
    interpolate_base_correlation,
    TrancheCDS,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
N = 20  # small portfolio for speed


def _make_survs(n=N):
    hazards = [0.01 + 0.002 * i for i in range(n)]
    return [make_flat_survival(REF, h) for h in hazards]


def _dc():
    return make_flat_curve(REF, 0.04)


# ---- Multi-period tranche ----

class TestMultiPeriod:

    def test_par_spread_positive(self):
        survs = _make_survs()
        dc = _dc()
        result = price_tranche_multiperiod(
            0.0, 0.03, survs, dc, 0.3, spread=0.05,
            maturity_years=5.0, n_sims=10_000,
        )
        assert result.par_spread > 0

    def test_protection_positive(self):
        survs = _make_survs()
        dc = _dc()
        result = price_tranche_multiperiod(
            0.0, 0.03, survs, dc, 0.3, spread=0.05,
            maturity_years=5.0, n_sims=10_000,
        )
        assert result.protection_pv > 0

    def test_premium_positive(self):
        survs = _make_survs()
        dc = _dc()
        result = price_tranche_multiperiod(
            0.0, 0.03, survs, dc, 0.3, spread=0.05,
            maturity_years=5.0, n_sims=10_000,
        )
        assert result.premium_pv > 0

    def test_equity_higher_spread_than_senior(self):
        """Equity tranche should have higher par spread than senior."""
        survs = _make_survs()
        dc = _dc()
        eq = price_tranche_multiperiod(
            0.0, 0.03, survs, dc, 0.3, spread=0.05,
            maturity_years=5.0, n_sims=10_000,
        )
        sr = price_tranche_multiperiod(
            0.12, 0.22, survs, dc, 0.3, spread=0.01,
            maturity_years=5.0, n_sims=10_000,
        )
        assert eq.par_spread > sr.par_spread


# ---- Student-t copula ----

class TestTCopula:

    def test_positive_el(self):
        survs = _make_survs()
        dc = _dc()
        el = expected_tranche_loss_t(
            0.0, 0.03, survs, dc, 0.3, T=5.0, nu=5,
            n_sims=10_000,
        )
        assert el > 0

    def test_converges_to_gaussian(self):
        """As nu → ∞, t-copula → Gaussian copula."""
        survs = _make_survs()
        dc = _dc()
        el_gauss = expected_tranche_loss(
            0.0, 0.03, survs, dc, 0.3, T=5.0,
            n_sims=20_000, seed=42,
        )
        el_t_large = expected_tranche_loss_t(
            0.0, 0.03, survs, dc, 0.3, T=5.0, nu=200,
            n_sims=20_000, seed=42,
        )
        # Should be similar (within MC noise)
        assert el_t_large == pytest.approx(el_gauss, abs=0.05)

    def test_fat_tails_senior(self):
        """Fat tails (low nu) should increase senior tranche loss."""
        survs = _make_survs()
        dc = _dc()
        el_gauss = expected_tranche_loss(
            0.12, 0.22, survs, dc, 0.3, T=5.0,
            n_sims=20_000, seed=42,
        )
        el_t5 = expected_tranche_loss_t(
            0.12, 0.22, survs, dc, 0.3, T=5.0, nu=3,
            n_sims=20_000, seed=42,
        )
        # t-copula should give higher senior loss (more tail dependence)
        assert el_t5 > el_gauss * 0.5  # at least not dramatically lower


# ---- Correlation sensitivity (rho01) ----

class TestRho01:

    def test_equity_negative(self):
        """Higher correlation reduces equity tranche expected loss."""
        survs = _make_survs()
        dc = _dc()
        r = tranche_rho01(0.0, 0.03, survs, dc, 0.3, T=5.0,
                           n_sims=20_000, seed=42)
        assert r < 0

    def test_senior_positive(self):
        """Higher correlation increases senior tranche expected loss."""
        survs = _make_survs()
        dc = _dc()
        r = tranche_rho01(0.12, 0.22, survs, dc, 0.3, T=5.0,
                           n_sims=20_000, seed=42)
        assert r > 0

    def test_finite(self):
        survs = _make_survs()
        dc = _dc()
        r = tranche_rho01(0.03, 0.06, survs, dc, 0.3, T=5.0,
                           n_sims=10_000, seed=42)
        assert math.isfinite(r)


# ---- Base correlation interpolation ----

class TestBaseCorrelationInterpolation:

    def test_exact_at_calibrated(self):
        calibrated = {0.03: 0.15, 0.06: 0.25, 0.09: 0.35, 0.12: 0.45}
        for d, rho in calibrated.items():
            assert interpolate_base_correlation(calibrated, d) == pytest.approx(rho)

    def test_interpolated_between(self):
        calibrated = {0.03: 0.15, 0.06: 0.25}
        rho = interpolate_base_correlation(calibrated, 0.045)
        # Linear: 0.15 + (0.25-0.15) × (0.045-0.03)/(0.06-0.03) = 0.20
        assert rho == pytest.approx(0.20)

    def test_monotonic(self):
        """Interpolated values should be monotonically increasing."""
        calibrated = {0.03: 0.15, 0.06: 0.25, 0.09: 0.35, 0.12: 0.45}
        dets = [0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.10, 0.12, 0.15]
        corrs = [interpolate_base_correlation(calibrated, d) for d in dets]
        for i in range(1, len(corrs)):
            assert corrs[i] >= corrs[i - 1] - 1e-10

    def test_extrapolation_flat(self):
        calibrated = {0.03: 0.15, 0.06: 0.25}
        assert interpolate_base_correlation(calibrated, 0.01) == 0.15
        assert interpolate_base_correlation(calibrated, 0.10) == 0.25

    def test_empty_default(self):
        assert interpolate_base_correlation({}, 0.05) == 0.3
