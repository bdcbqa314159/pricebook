"""Tests for multi-currency dealer funding curve."""

import pytest
from datetime import date

from pricebook.fixed_income.repo_funding_curve import (
    build_dealer_funding_curve, DealerFundingCurve,
    get_repo_conventions, list_repo_currencies, RepoMarketConventions,
)

REF = date(2024, 1, 15)

SEC = {1: 0.053, 7: 0.0525, 30: 0.052, 90: 0.051, 180: 0.050, 360: 0.049}
UNSEC = {1: 0.056, 7: 0.0555, 30: 0.055, 90: 0.054, 180: 0.053, 360: 0.052}


class TestConventions:
    def test_usd(self):
        c = get_repo_conventions("USD")
        assert c.benchmark_index == "SOFR"
        assert c.settlement_days == 1
        assert "UST" in c.gc_collateral

    def test_eur_t0(self):
        c = get_repo_conventions("EUR")
        assert c.settlement_days == 0  # EUR GC is T+0

    def test_gbp(self):
        c = get_repo_conventions("GBP")
        assert c.benchmark_index == "SONIA"
        assert "Gilt" in c.gc_collateral

    def test_brl(self):
        c = get_repo_conventions("BRL")
        assert c.benchmark_index == "CDI"
        assert c.day_count.value == "BUS/252"

    def test_list_currencies(self):
        ccys = list_repo_currencies()
        assert len(ccys) == 11
        assert "USD" in ccys
        assert "EUR" in ccys
        assert "BRL" in ccys
        assert "ZAR" in ccys

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_repo_conventions("XXX")

    def test_to_dict(self):
        d = get_repo_conventions("USD").to_dict()
        assert d["currency"] == "USD"


class TestBuildCurve:
    def test_basic(self):
        curve = build_dealer_funding_curve("USD", REF, SEC, UNSEC)
        assert isinstance(curve, DealerFundingCurve)
        assert curve.currency == "USD"

    def test_secured_rate_interpolation(self):
        curve = build_dealer_funding_curve("USD", REF, SEC, UNSEC)
        r_15d = curve.secured_rate(15)
        assert SEC[7] >= r_15d >= SEC[30]  # between 1W and 1M

    def test_funding_spread_positive(self):
        """Unsecured > secured → positive funding spread."""
        curve = build_dealer_funding_curve("USD", REF, SEC, UNSEC)
        for days in [1, 30, 90, 360]:
            assert curve.funding_spread(days) > 0

    def test_blended_rate(self):
        curve = build_dealer_funding_curve("USD", REF, SEC, UNSEC)
        r_sec = curve.secured_rate(30)
        r_unsec = curve.unsecured_rate(30)
        r_blend = curve.blended_rate(30, haircut=0.02)
        assert r_sec < r_blend < r_unsec

    def test_to_discount_curve(self):
        curve = build_dealer_funding_curve("USD", REF, SEC, UNSEC)
        dc = curve.to_discount_curve("secured")
        assert dc.df(date(2024, 4, 15)) < 1.0

    def test_multi_currency(self):
        for ccy in ["USD", "EUR", "GBP", "JPY"]:
            curve = build_dealer_funding_curve(ccy, REF, SEC, UNSEC)
            assert curve.currency == ccy

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_dealer_funding_curve("USD", REF, {}, UNSEC)

    def test_to_dict(self):
        curve = build_dealer_funding_curve("USD", REF, SEC, UNSEC)
        d = curve.to_dict()
        assert "funding_spread_1m_bp" in d
        assert d["funding_spread_1m_bp"] > 0
