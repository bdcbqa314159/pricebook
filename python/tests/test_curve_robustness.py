"""Tests for curve robustness: round-trip, negative rates, edge cases."""

import math
import warnings
from datetime import date

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap
from pricebook.curve_builder import build_curves
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.multicurve_solver import validate_curve


# ---- Round-trip repricing ----

class TestRoundTrip:
    def test_deposits_reprice(self):
        """Bootstrapped curve should reprice all input deposits exactly."""
        ref = date(2026, 4, 21)
        deposits = [
            (date(2026, 5, 21), 0.043),
            (date(2026, 7, 21), 0.042),
            (date(2026, 10, 21), 0.041),
        ]
        curve = bootstrap(ref, deposits, [])
        for mat, rate in deposits:
            tau = year_fraction(ref, mat, DayCountConvention.ACT_360)
            model = (1.0 / curve.df(mat) - 1.0) / tau
            assert model == pytest.approx(rate, abs=1e-10)

    def test_swaps_reprice(self):
        """Bootstrapped curve should reprice all input swaps to < 1e-8."""
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 10, 21), 0.040)]
        swaps = [
            (date(2027, 4, 21), 0.0395),
            (date(2028, 4, 21), 0.0390),
            (date(2031, 4, 21), 0.0385),
            (date(2036, 4, 21), 0.0400),
        ]
        # Should not raise any round-trip warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve = bootstrap(ref, deposits, swaps)
        assert curve.df(date(2036, 4, 21)) > 0

    def test_fras_reprice(self):
        """FRA forward rates should match input."""
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 7, 21), 0.042)]
        fras = [
            (date(2026, 7, 21), date(2026, 10, 21), 0.041),
            (date(2026, 10, 21), date(2027, 1, 21), 0.040),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve = bootstrap(ref, deposits, [], fras=fras)
        # Verify FRA forward rates
        for start, end, rate in fras:
            tau = year_fraction(start, end, DayCountConvention.ACT_360)
            model_fwd = (curve.df(start) / curve.df(end) - 1.0) / tau
            assert model_fwd == pytest.approx(rate, abs=1e-6)

    def test_full_pipeline_round_trip(self):
        """Full pipeline: deposits + FRAs + swaps all reprice."""
        ref = date(2026, 4, 21)
        deposits = [
            (date(2026, 5, 21), 0.043),
            (date(2026, 7, 21), 0.042),
        ]
        fras = [
            (date(2026, 7, 21), date(2026, 10, 21), 0.041),
        ]
        swaps = [
            (date(2028, 4, 21), 0.039),
            (date(2031, 4, 21), 0.038),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve = bootstrap(ref, deposits, swaps, fras=fras)
        # If we get here without RuntimeWarning, round-trip passed
        assert curve.df(date(2031, 4, 21)) > 0


# ---- Negative rates ----

class TestNegativeRates:
    def test_negative_deposit_rates(self):
        """EUR-style negative deposit rates should bootstrap correctly."""
        ref = date(2026, 4, 21)
        deposits = [
            (date(2026, 5, 21), -0.005),   # -50bp O/N
            (date(2026, 7, 21), -0.004),   # -40bp 3M
            (date(2026, 10, 21), -0.003),  # -30bp 6M
        ]
        curve = bootstrap(ref, deposits, [])
        # DFs should be > 1 for negative rates
        for mat, rate in deposits:
            assert curve.df(mat) > 1.0, f"DF at {mat} should be > 1 for negative rate"

    def test_negative_swap_rates(self):
        """Negative swap rates (EUR 2020-2022 regime) should bootstrap."""
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 10, 21), -0.004)]
        swaps = [
            (date(2027, 4, 21), -0.003),
            (date(2028, 4, 21), -0.002),
            (date(2031, 4, 21), 0.001),    # rates turn positive at 5Y
            (date(2036, 4, 21), 0.005),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve = bootstrap(ref, deposits, swaps)

        # Short end: DFs > 1 (negative rates)
        assert curve.df(date(2027, 4, 21)) > 1.0
        # Long end: DFs < 1 (positive rates)
        assert curve.df(date(2036, 4, 21)) < 1.0

    def test_negative_rates_round_trip(self):
        """Negative rate instruments should reprice correctly."""
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 7, 21), -0.005)]
        swaps = [(date(2028, 4, 21), -0.003)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve = bootstrap(ref, deposits, swaps)

        # Deposit reprices
        tau = year_fraction(ref, date(2026, 7, 21), DayCountConvention.ACT_360)
        model = (1.0 / curve.df(date(2026, 7, 21)) - 1.0) / tau
        assert model == pytest.approx(-0.005, abs=1e-8)

    def test_mixed_positive_negative(self):
        """Curve crossing zero should work (inverted to normal)."""
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 7, 21), -0.002)]
        swaps = [
            (date(2027, 4, 21), -0.001),
            (date(2028, 4, 21), 0.002),
            (date(2031, 4, 21), 0.010),
            (date(2036, 4, 21), 0.015),
        ]
        curve = bootstrap(ref, deposits, swaps)
        validation = validate_curve(curve, min_forward_rate=-0.02)
        # Should have no extreme forward warnings
        assert not any("above 0.2" in w for w in validation.warnings)


# ---- Curve validation ----

class TestCurveValidationIntegration:
    def test_normal_curve_valid(self):
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 7, 21), 0.042)]
        swaps = [
            (date(2028, 4, 21), 0.039),
            (date(2031, 4, 21), 0.038),
            (date(2033, 4, 21), 0.039),
            (date(2036, 4, 21), 0.040),
        ]
        curve = bootstrap(ref, deposits, swaps)
        result = validate_curve(curve)
        assert result.is_valid
        assert result.n_pillars >= 4

    def test_inverted_curve_valid(self):
        """Inverted curve (short > long) is mathematically valid."""
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 7, 21), 0.055)]
        swaps = [
            (date(2028, 4, 21), 0.045),
            (date(2031, 4, 21), 0.035),
        ]
        curve = bootstrap(ref, deposits, swaps)
        result = validate_curve(curve)
        # Inverted is valid as long as forwards are non-negative
        assert result.min_forward_rate >= -0.01


# ---- build_curves unified entry ----

class TestBuildCurves:
    def test_usd_basic(self):
        ref = date(2026, 4, 21)
        deposits = [
            (date(2026, 5, 21), 0.043),
            (date(2026, 7, 21), 0.042),
        ]
        swaps = [
            (date(2028, 4, 21), 0.039),
            (date(2031, 4, 21), 0.038),
        ]
        result = build_curves("USD", ref, deposits, swaps)
        assert result.currency == "USD"
        assert result.ois is not None
        assert result.projection is None  # no projection quotes

    def test_eur_negative(self):
        ref = date(2026, 4, 21)
        deposits = [(date(2026, 7, 21), -0.004)]
        swaps = [(date(2031, 4, 21), -0.001)]
        result = build_curves("EUR", ref, deposits, swaps)
        assert result.ois.df(date(2031, 4, 21)) > 0


# ---- Date precision ----

class TestDatePrecision:
    def test_pillar_dates_exact(self):
        """pillar_dates should return exact input dates, not approximations."""
        ref = date(2026, 4, 21)
        exact_dates = [date(2026, 7, 21), date(2028, 4, 21), date(2036, 4, 21)]
        dfs = [0.99, 0.92, 0.67]
        curve = DiscountCurve(ref, exact_dates, dfs)
        returned = curve.pillar_dates
        for original, returned_date in zip(exact_dates, returned):
            assert original == returned_date, f"Expected {original}, got {returned_date}"
