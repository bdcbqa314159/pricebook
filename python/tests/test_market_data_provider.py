"""Tests for market data providers (sample data only — no network)."""

import pytest
from datetime import date

from pricebook.market_data_provider import (
    RateSeries, SampleProvider, build_curve_from_yields,
    FRED_SERIES, ECB_SERIES, FREDProvider, ECBProvider,
)
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 6, 15)


# ---- RateSeries ----

class TestRateSeries:
    def test_latest(self):
        s = RateSeries("test", "USD", "rate",
                       [date(2024, 1, 1), date(2024, 1, 2)], [0.04, 0.05])
        d, v = s.latest()
        assert d == date(2024, 1, 2)
        assert v == 0.05

    def test_on_date(self):
        s = RateSeries("test", "USD", "rate",
                       [date(2024, 1, 1), date(2024, 1, 2)], [0.04, 0.05])
        assert s.on_date(date(2024, 1, 1)) == 0.04
        assert s.on_date(date(2024, 1, 3)) is None

    def test_between(self):
        s = RateSeries("test", "USD", "rate",
                       [date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 1)],
                       [0.04, 0.05, 0.06])
        filtered = s.between(date(2024, 1, 10), date(2024, 1, 20))
        assert len(filtered.dates) == 1
        assert filtered.values[0] == 0.05

    def test_empty_latest_raises(self):
        s = RateSeries("test", "USD", "rate", [], [])
        with pytest.raises(ValueError):
            s.latest()


# ---- Sample provider ----

class TestSampleProvider:
    def test_available_series(self):
        p = SampleProvider(REF)
        series = p.available_series()
        assert "SOFR" in series
        assert "ESTR" in series
        assert "UST_10Y" in series

    def test_fetch_sofr(self):
        p = SampleProvider(REF)
        s = p.fetch("SOFR")
        assert s.name == "SOFR"
        assert s.currency == "USD"
        assert len(s.dates) > 0
        assert all(v > 0 for v in s.values)

    def test_fetch_estr(self):
        p = SampleProvider(REF)
        s = p.fetch("ESTR")
        assert s.currency == "EUR"
        assert len(s.dates) > 0

    def test_fetch_all_series(self):
        p = SampleProvider(REF)
        for name in p.available_series():
            s = p.fetch(name)
            assert len(s.dates) > 0
            assert len(s.dates) == len(s.values)

    def test_unknown_raises(self):
        p = SampleProvider(REF)
        with pytest.raises(KeyError):
            p.fetch("UNKNOWN")

    def test_date_filter(self):
        p = SampleProvider(REF, history_days=500)
        s = p.fetch("SOFR", start=date(2024, 1, 1), end=date(2024, 3, 1))
        for d in s.dates:
            assert date(2024, 1, 1) <= d <= date(2024, 3, 1)

    def test_deterministic(self):
        """Same name → same data."""
        p = SampleProvider(REF)
        s1 = p.fetch("SOFR")
        s2 = p.fetch("SOFR")
        assert s1.values == s2.values

    def test_reasonable_rates(self):
        p = SampleProvider(REF)
        s = p.fetch("UST_10Y")
        _, latest = s.latest()
        assert 0.0 < latest < 0.20  # between 0% and 20%


# ---- FRED/ECB series registration ----

class TestProviderRegistration:
    def test_fred_series_defined(self):
        assert "SOFR" in FRED_SERIES
        assert "UST_10Y" in FRED_SERIES
        assert "CPI" in FRED_SERIES

    def test_ecb_series_defined(self):
        assert "ESTR" in ECB_SERIES
        assert "EURIBOR_3M" in ECB_SERIES

    def test_fred_available(self):
        """FREDProvider lists series without needing a key."""
        p = FREDProvider(api_key="dummy")
        assert "SOFR" in p.available_series()

    def test_ecb_available(self):
        p = ECBProvider()
        assert "ESTR" in p.available_series()


# ---- Curve building ----

class TestBuildCurve:
    def test_from_yields(self):
        curve = build_curve_from_yields(REF, {2: 0.04, 5: 0.039, 10: 0.041})
        assert isinstance(curve, DiscountCurve)
        # DF should be decreasing
        d2 = date(2026, 6, 15)
        d10 = date(2034, 6, 15)
        assert curve.df(d2) > curve.df(d10)

    def test_single_tenor(self):
        curve = build_curve_from_yields(REF, {5: 0.04})
        assert curve.df(date(2029, 6, 15)) > 0

    def test_reasonable_dfs(self):
        curve = build_curve_from_yields(REF, {1: 0.05, 10: 0.05})
        import math
        assert curve.df(date(2025, 6, 15)) == pytest.approx(math.exp(-0.05), rel=0.05)
