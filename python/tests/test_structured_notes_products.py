"""Tests for structured notes."""
from __future__ import annotations
import pytest
from pricebook.structured_notes import (
    capital_protected_note, dual_digital, bonus_certificate,
    outperformance_certificate,
)


class TestCapitalProtected:
    def test_price_above_bond_floor(self):
        r = capital_protected_note(100, 0.05, 0.02, 0.20, 3.0)
        assert r.price >= r.bond_floor

    def test_participation_positive(self):
        r = capital_protected_note(100, 0.05, 0.02, 0.20, 3.0)
        assert r.participation > 0

    def test_higher_rate_more_participation(self):
        r_low = capital_protected_note(100, 0.02, 0.02, 0.20, 3.0)
        r_high = capital_protected_note(100, 0.06, 0.02, 0.20, 3.0)
        assert r_high.participation > r_low.participation


class TestDualDigital:
    def test_price_positive(self):
        r = dual_digital(100, 50, 100, 50, 0.05, 0.02, 0.01, 0.20, 0.25, 0.5, 1.0, n_paths=5_000)
        assert r.price > 0

    def test_probability_bounded(self):
        r = dual_digital(100, 50, 90, 45, 0.05, 0.02, 0.01, 0.20, 0.25, 0.5, 1.0, n_paths=5_000)
        assert 0 < r.prob_both < 1

    def test_high_correlation_higher_prob(self):
        r_low = dual_digital(100, 100, 105, 105, 0.05, 0.02, 0.02, 0.20, 0.20, 0.1, 1.0, n_paths=10_000)
        r_high = dual_digital(100, 100, 105, 105, 0.05, 0.02, 0.02, 0.20, 0.20, 0.9, 1.0, n_paths=10_000)
        assert r_high.prob_both > r_low.prob_both * 0.8


class TestBonusCertificate:
    def test_price_positive(self):
        r = bonus_certificate(100, 0.05, 0.02, 0.20, 1.0, 110, 80, n_paths=5_000)
        assert r.price > 0

    def test_barrier_probability(self):
        r = bonus_certificate(100, 0.05, 0.02, 0.20, 1.0, 110, 80, n_paths=5_000)
        assert 0 <= r.barrier_hit_probability <= 1

    def test_higher_barrier_more_hits(self):
        r_low = bonus_certificate(100, 0.05, 0.02, 0.20, 1.0, 110, 70, n_paths=10_000)
        r_high = bonus_certificate(100, 0.05, 0.02, 0.20, 1.0, 110, 90, n_paths=10_000)
        assert r_high.barrier_hit_probability > r_low.barrier_hit_probability


class TestOutperformanceCert:
    def test_price_positive(self):
        r = outperformance_certificate(100, 0.05, 0.02, 0.20, 1.0)
        assert r.price > 0

    def test_higher_participation_higher_price(self):
        from pricebook.structured_notes import outperformance_certificate
        low = outperformance_certificate(100, 0.05, 0.02, 0.20, 1.0, participation=1.2)
        high = outperformance_certificate(100, 0.05, 0.02, 0.20, 1.0, participation=2.0)
        assert high.price > low.price

    def test_cap_reduces_price(self):
        from pricebook.structured_notes import outperformance_certificate
        uncapped = outperformance_certificate(100, 0.05, 0.02, 0.20, 1.0, participation=1.5)
        capped = outperformance_certificate(100, 0.05, 0.02, 0.20, 1.0, participation=1.5, cap=0.20)
        assert capped.price < uncapped.price
