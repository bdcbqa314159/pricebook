"""Tests for tranche correlation trading."""

import math
import pytest

from pricebook.tranche_trading import (
    tranche_delta, tranche_cs01, correlation_sensitivity,
    correlation_skew, skew_bump,
)
from pricebook.cdo import portfolio_loss_distribution, tranche_spread


PD = 0.02
RHO = 0.3
LGD = 0.6
T = 5.0
RATE = 0.05

# Standard tranche points
EQ_ATTACH, EQ_DETACH = 0.0, 0.03
MEZZ_ATTACH, MEZZ_DETACH = 0.03, 0.07
SR_ATTACH, SR_DETACH = 0.07, 0.15


# ---- Tranche delta ----

class TestTrancheDelta:
    def test_equity_positive_delta(self):
        """Equity tranche: spread widening increases equity tranche spread."""
        d = tranche_delta(PD, RHO, LGD, EQ_ATTACH, EQ_DETACH, T, RATE)
        assert d > 0

    def test_delta_scales_with_seniority(self):
        """More senior tranches have smaller delta."""
        d_eq = tranche_delta(PD, RHO, LGD, EQ_ATTACH, EQ_DETACH, T, RATE)
        d_mezz = tranche_delta(PD, RHO, LGD, MEZZ_ATTACH, MEZZ_DETACH, T, RATE)
        assert abs(d_eq) > abs(d_mezz)

    def test_cs01_nonzero(self):
        cs = tranche_cs01(PD, RHO, LGD, EQ_ATTACH, EQ_DETACH, notional=10_000_000)
        assert cs != 0.0

    def test_cs01_scales_with_notional(self):
        cs1 = tranche_cs01(PD, RHO, LGD, EQ_ATTACH, EQ_DETACH, notional=10_000_000)
        cs2 = tranche_cs01(PD, RHO, LGD, EQ_ATTACH, EQ_DETACH, notional=20_000_000)
        assert cs2 == pytest.approx(2 * cs1, rel=0.01)


# ---- Correlation sensitivity ----

class TestCorrelationSensitivity:
    def test_equity_corr_sens_nonzero(self):
        """Equity tranche spread is sensitive to correlation."""
        sens = correlation_sensitivity(PD, RHO, LGD, EQ_ATTACH, EQ_DETACH, T, RATE)
        assert sens != 0.0

    def test_senior_positive_corr_sens(self):
        """Senior tranche spread increases when correlation rises."""
        sens = correlation_sensitivity(PD, RHO, LGD, SR_ATTACH, SR_DETACH, T, RATE)
        assert sens > 0  # higher corr → fatter tail for senior → higher spread

    def test_nonzero(self):
        sens = correlation_sensitivity(PD, RHO, LGD, MEZZ_ATTACH, MEZZ_DETACH, T, RATE)
        assert sens != 0.0


# ---- Correlation skew ----

class TestCorrelationSkew:
    def test_base_corr_roundtrip(self):
        """Base correlation should be in valid range."""
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        eq_spread = tranche_spread(loss_grid, density, 0.0, 0.03, T, RATE)
        skew = correlation_skew({0.03: eq_spread}, PD, LGD, T, RATE)
        assert len(skew.points) == 1
        assert 0.0 < skew.points[0].base_corr < 1.0

    def test_skew_structure(self):
        """Senior base corr > equity base corr (typical skew)."""
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        eq_spread = tranche_spread(loss_grid, density, 0.0, 0.03, T, RATE)
        sr_spread = tranche_spread(loss_grid, density, 0.0, 0.15, T, RATE)
        skew = correlation_skew({0.03: eq_spread, 0.15: sr_spread}, PD, LGD, T, RATE)
        assert len(skew.points) == 2
        # Senior base corr should be higher than equity
        assert skew.points[1].base_corr > skew.points[0].base_corr

    def test_multiple_points(self):
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        spreads = {}
        for d in [0.03, 0.07, 0.15]:
            spreads[d] = tranche_spread(loss_grid, density, 0.0, d, T, RATE)
        skew = correlation_skew(spreads, PD, LGD, T, RATE)
        assert len(skew.points) == 3


# ---- Skew bump ----

class TestSkewBump:
    def test_parallel_bump(self):
        results = skew_bump(PD, RHO, LGD, [0.03, 0.07, 0.15], "parallel", 0.05, T, RATE)
        assert len(results) == 3
        for detach, (base, bumped) in results.items():
            assert base >= 0
            assert bumped >= 0
            # Bumped and base should differ
            assert base != bumped or base == 0

    def test_tilt_bump(self):
        results = skew_bump(PD, RHO, LGD, [0.03, 0.07, 0.15], "tilt", 0.05, T, RATE)
        assert len(results) == 3
        # First point (equity) gets no tilt bump → base ≈ bumped
        eq_base, eq_bumped = results[0.03]
        assert eq_base == pytest.approx(eq_bumped, rel=0.01)

    def test_larger_bump_more_impact(self):
        r1 = skew_bump(PD, RHO, LGD, [0.03], "parallel", 0.01, T, RATE)
        r2 = skew_bump(PD, RHO, LGD, [0.03], "parallel", 0.10, T, RATE)
        diff1 = abs(r1[0.03][1] - r1[0.03][0])
        diff2 = abs(r2[0.03][1] - r2[0.03][0])
        assert diff2 > diff1
