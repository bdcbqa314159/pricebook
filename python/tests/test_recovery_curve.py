"""Tests for term structure of recovery rates."""

import pytest
import numpy as np

from pricebook.credit.recovery_curve import (
    RecoveryCurve, StochasticRecovery, RecoverySeniority,
    recovery_by_seniority, recovery_vol_by_seniority,
)


class TestRecoveryCurve:
    def test_flat(self):
        c = RecoveryCurve.flat(0.40)
        assert c.recovery(1.0) == 0.40
        assert c.recovery(10.0) == 0.40

    def test_linear(self):
        c = RecoveryCurve.linear(0.50, 0.35, pivot_years=5.0)
        assert c.recovery(0.0) == 0.50
        assert c.recovery(5.0) == 0.35
        assert c.recovery(2.5) == pytest.approx(0.425)
        # Flat extrapolation beyond pivot
        assert c.recovery(10.0) == 0.35

    def test_piecewise(self):
        c = RecoveryCurve([1, 3, 5, 10], [0.50, 0.45, 0.40, 0.30])
        assert c.recovery(1.0) == 0.50
        assert c.recovery(10.0) == 0.30
        assert c.recovery(2.0) == pytest.approx(0.475)

    def test_from_seniority(self):
        c = RecoveryCurve.from_seniority(RecoverySeniority.SENIOR_UNSECURED)
        r_short = c.recovery(0.0)
        r_long = c.recovery(5.0)
        assert r_short > r_long  # declines with maturity
        assert r_short == pytest.approx(0.40)

    def test_average(self):
        c = RecoveryCurve.linear(0.50, 0.30, 10.0)
        avg = c.average(0.0, 10.0)
        assert avg == pytest.approx(0.40, abs=0.02)

    def test_to_dict(self):
        c = RecoveryCurve.flat(0.40)
        d = c.to_dict()
        assert "tenors" in d
        assert "recovery_rates" in d

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            RecoveryCurve([], [])


class TestSeniorityLookup:
    def test_senior_secured(self):
        r = recovery_by_seniority(RecoverySeniority.SENIOR_SECURED)
        assert r == 0.53

    def test_subordinated(self):
        r = recovery_by_seniority(RecoverySeniority.SUBORDINATED)
        assert r == 0.28

    def test_ordering(self):
        """Higher seniority → higher recovery."""
        ss = recovery_by_seniority(RecoverySeniority.SENIOR_SECURED)
        su = recovery_by_seniority(RecoverySeniority.SENIOR_UNSECURED)
        sub = recovery_by_seniority(RecoverySeniority.SUBORDINATED)
        jsub = recovery_by_seniority(RecoverySeniority.JUNIOR_SUBORDINATED)
        assert ss > su > sub > jsub

    def test_vol_positive(self):
        for s in RecoverySeniority:
            v = recovery_vol_by_seniority(s)
            assert v > 0


class TestStochasticRecovery:
    def test_sample_shape(self):
        sr = StochasticRecovery(0.40, 0.25)
        samples = sr.sample(1000, rng=np.random.default_rng(42))
        assert samples.shape == (1000,)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_sample_mean(self):
        sr = StochasticRecovery(0.40, 0.20)
        samples = sr.sample(10000, rng=np.random.default_rng(42))
        assert abs(np.mean(samples) - 0.40) < 0.02

    def test_from_seniority(self):
        sr = StochasticRecovery.from_seniority(RecoverySeniority.SENIOR_UNSECURED)
        assert sr.mean == 0.40
        assert sr.vol == 0.25
        assert sr.seniority == RecoverySeniority.SENIOR_UNSECURED

    def test_percentile(self):
        sr = StochasticRecovery(0.40, 0.20)
        p25 = sr.percentile(0.25)
        p50 = sr.percentile(0.50)
        p75 = sr.percentile(0.75)
        assert p25 < p50 < p75
        assert abs(p50 - 0.40) < 0.05  # median near mean

    def test_to_dict(self):
        sr = StochasticRecovery.from_seniority(RecoverySeniority.SUBORDINATED)
        d = sr.to_dict()
        assert d["mean"] == 0.28
        assert d["seniority"] == "subordinated"
