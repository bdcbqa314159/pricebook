"""Tests for regulatory ratings module."""

import pytest

from pricebook.regulatory.ratings import (
    RATING_TO_PD, normalize_rating, get_rating_from_pd, get_rating_from_pd_log,
    get_pd_range, resolve_pd, resolve_rating,
    is_investment_grade, is_high_yield,
)


class TestRatingToPD:
    def test_aaa_lowest(self):
        assert RATING_TO_PD["AAA"] < RATING_TO_PD["AA"]

    def test_d_is_one(self):
        assert RATING_TO_PD["D"] == 1.0

    def test_monotonic(self):
        ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
        pds = [RATING_TO_PD[r] for r in ratings]
        assert pds == sorted(pds)


class TestNormalize:
    def test_notched(self):
        assert normalize_rating("AA+") == "AA"
        assert normalize_rating("BBB-") == "BBB"

    def test_base_unchanged(self):
        assert normalize_rating("A") == "A"
        assert normalize_rating("BBB") == "BBB"

    def test_empty(self):
        assert normalize_rating("") == "BBB"


class TestPDConversion:
    def test_round_trip(self):
        for rating, pd in RATING_TO_PD.items():
            recovered = get_rating_from_pd(pd)
            assert recovered == rating

    def test_zero_pd(self):
        assert get_rating_from_pd(0) == "AAA"

    def test_high_pd(self):
        assert get_rating_from_pd(1.0) == "D"

    def test_log_scale(self):
        r = get_rating_from_pd_log(0.005)
        assert r in RATING_TO_PD

    def test_pd_range(self):
        low, high = get_pd_range("BBB")
        assert low < RATING_TO_PD["BBB"] < high


class TestResolve:
    def test_resolve_pd_explicit(self):
        assert resolve_pd(pd=0.02) == 0.02

    def test_resolve_pd_from_rating(self):
        assert resolve_pd(rating="BB") == RATING_TO_PD["BB"]

    def test_resolve_pd_default(self):
        assert resolve_pd() == 0.004

    def test_resolve_rating_explicit(self):
        assert resolve_rating(rating="A") == "A"

    def test_resolve_rating_from_pd(self):
        assert resolve_rating(pd=0.02) == "BB"

    def test_resolve_rating_default(self):
        assert resolve_rating() == "BBB"


class TestIG:
    def test_ig(self):
        assert is_investment_grade("AAA")
        assert is_investment_grade("BBB-")

    def test_hy(self):
        assert is_high_yield("BB+")
        assert is_high_yield("CCC")

    def test_boundary(self):
        assert is_investment_grade("BBB-")
        assert not is_investment_grade("BB+")
