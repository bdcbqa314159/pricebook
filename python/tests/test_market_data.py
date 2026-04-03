"""Tests for market data snapshots, curve building pipeline, and historical data."""

import json
import math
import pytest
from datetime import date

from pricebook.market_data import (
    Quote,
    QuoteType,
    MarketDataSnapshot,
    HistoricalData,
    CurveConfig,
    PipelineConfig,
    MissingQuoteError,
    build_context,
    tenor_to_years,
    tenor_to_date,
)


REF = date(2024, 1, 15)


# ---------------------------------------------------------------------------
# Step 1 — Market data container
# ---------------------------------------------------------------------------


class TestQuote:
    def test_roundtrip(self):
        q = Quote(QuoteType.DEPOSIT_RATE, "3M", 0.05, "USD")
        d = q.to_dict()
        q2 = Quote.from_dict(d)
        assert q2.quote_type == q.quote_type
        assert q2.tenor == q.tenor
        assert q2.value == q.value
        assert q2.currency == q.currency

    def test_all_types(self):
        for qt in QuoteType:
            q = Quote(qt, "1Y", 0.01)
            assert q.quote_type == qt


class TestSnapshot:
    def _sample(self):
        snap = MarketDataSnapshot(REF)
        snap.add(Quote(QuoteType.DEPOSIT_RATE, "3M", 0.05))
        snap.add(Quote(QuoteType.DEPOSIT_RATE, "6M", 0.051))
        snap.add(Quote(QuoteType.SWAP_RATE, "5Y", 0.04))
        snap.add(Quote(QuoteType.CDS_SPREAD, "5Y", 0.01, name="ACME"))
        snap.add(Quote(QuoteType.VOL_POINT, "1Y", 0.20, name="ir"))
        snap.add(Quote(QuoteType.FX_SPOT, "SPOT", 1.085, name="EUR/USD"))
        return snap

    def test_add_and_count(self):
        snap = self._sample()
        assert len(snap.quotes) == 6

    def test_filter_by_type(self):
        snap = self._sample()
        deposits = snap.get_quotes(QuoteType.DEPOSIT_RATE)
        assert len(deposits) == 2

    def test_filter_by_name(self):
        snap = self._sample()
        cds = snap.get_quotes(QuoteType.CDS_SPREAD, name="ACME")
        assert len(cds) == 1
        assert cds[0].value == 0.01

    def test_json_roundtrip(self):
        snap = self._sample()
        j = snap.to_json()
        snap2 = MarketDataSnapshot.from_json(j)
        assert snap2.snapshot_date == snap.snapshot_date
        assert len(snap2.quotes) == len(snap.quotes)
        for q1, q2 in zip(snap.quotes, snap2.quotes):
            assert q1.quote_type == q2.quote_type
            assert q1.tenor == q2.tenor
            assert q1.value == q2.value

    def test_dict_roundtrip(self):
        snap = self._sample()
        d = snap.to_dict()
        snap2 = MarketDataSnapshot.from_dict(d)
        assert snap2.snapshot_date == REF
        assert len(snap2.quotes) == 6


class TestTenorParsing:
    def test_common_tenors(self):
        assert tenor_to_years("1Y") == 1.0
        assert tenor_to_years("5Y") == 5.0
        assert tenor_to_years("3M") == pytest.approx(0.25)
        assert tenor_to_years("6M") == pytest.approx(0.5)

    def test_day_tenor(self):
        assert tenor_to_years("1D") == pytest.approx(1 / 365)

    def test_week_tenor(self):
        assert tenor_to_years("1W") == pytest.approx(7 / 365)

    def test_custom_tenor(self):
        assert tenor_to_years("8Y") == 8.0
        assert tenor_to_years("18M") == pytest.approx(1.5)

    def test_invalid_tenor(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            tenor_to_years("xyz")

    def test_tenor_to_date(self):
        d = tenor_to_date(REF, "1Y")
        assert d > REF


# ---------------------------------------------------------------------------
# Step 2 — Curve building pipeline
# ---------------------------------------------------------------------------


class TestBuildContext:
    def _rate_snapshot(self):
        snap = MarketDataSnapshot(REF)
        for tenor, rate in [("1Y", 0.05), ("2Y", 0.048), ("5Y", 0.045), ("10Y", 0.043)]:
            snap.add(Quote(QuoteType.SWAP_RATE, tenor, rate))
        return snap

    def test_discount_curve(self):
        snap = self._rate_snapshot()
        cfg = PipelineConfig(
            discount_config=CurveConfig("ois", QuoteType.SWAP_RATE),
        )
        ctx = build_context(snap, cfg)
        assert ctx.discount_curve is not None
        assert ctx.valuation_date == REF
        # Discount factor at 5Y should be reasonable
        d5 = date.fromordinal(REF.toordinal() + int(5 * 365))
        df5 = ctx.discount_curve.df(d5)
        assert 0.7 < df5 < 1.0

    def test_missing_quotes_raises(self):
        snap = MarketDataSnapshot(REF)  # empty
        cfg = PipelineConfig(
            discount_config=CurveConfig("ois", QuoteType.SWAP_RATE),
        )
        with pytest.raises(MissingQuoteError):
            build_context(snap, cfg)

    def test_credit_curve(self):
        snap = MarketDataSnapshot(REF)
        for tenor, spread in [("1Y", 0.01), ("3Y", 0.012), ("5Y", 0.015)]:
            snap.add(Quote(QuoteType.CDS_SPREAD, tenor, spread, name="ACME"))
        cfg = PipelineConfig(
            credit_configs={"ACME": CurveConfig("cds", QuoteType.CDS_SPREAD, name="ACME")},
        )
        ctx = build_context(snap, cfg)
        assert "ACME" in ctx.credit_curves
        d3 = date.fromordinal(REF.toordinal() + int(3 * 365))
        sp = ctx.credit_curves["ACME"].survival(d3)
        assert 0.9 < sp < 1.0

    def test_vol_surface(self):
        snap = MarketDataSnapshot(REF)
        snap.add(Quote(QuoteType.VOL_POINT, "1Y", 0.20, name="ir"))
        snap.add(Quote(QuoteType.VOL_POINT, "5Y", 0.22, name="ir"))
        cfg = PipelineConfig(vol_config={"ir": "ir"})
        ctx = build_context(snap, cfg)
        assert "ir" in ctx.vol_surfaces
        # FlatVol at average = 0.21
        vol = ctx.vol_surfaces["ir"].vol(1.0, 0.05)
        assert vol == pytest.approx(0.21)

    def test_fx_spot(self):
        snap = MarketDataSnapshot(REF)
        snap.add(Quote(QuoteType.FX_SPOT, "SPOT", 1.085, name="EUR/USD"))
        cfg = PipelineConfig(fx_pairs=[("EUR", "USD")])
        ctx = build_context(snap, cfg)
        assert ctx.fx_spots[("EUR", "USD")] == pytest.approx(1.085)

    def test_full_pipeline(self):
        snap = MarketDataSnapshot(REF)
        # Rates
        for tenor, rate in [("1Y", 0.05), ("5Y", 0.045), ("10Y", 0.043)]:
            snap.add(Quote(QuoteType.SWAP_RATE, tenor, rate))
        # CDS
        snap.add(Quote(QuoteType.CDS_SPREAD, "5Y", 0.015, name="ACME"))
        # Vol
        snap.add(Quote(QuoteType.VOL_POINT, "1Y", 0.20, name="ir"))
        # FX
        snap.add(Quote(QuoteType.FX_SPOT, "SPOT", 1.085, name="EUR/USD"))

        cfg = PipelineConfig(
            discount_config=CurveConfig("ois", QuoteType.SWAP_RATE),
            credit_configs={"ACME": CurveConfig("cds", QuoteType.CDS_SPREAD, name="ACME")},
            vol_config={"ir": "ir"},
            fx_pairs=[("EUR", "USD")],
        )
        ctx = build_context(snap, cfg)
        assert ctx.discount_curve is not None
        assert "ACME" in ctx.credit_curves
        assert "ir" in ctx.vol_surfaces
        assert ("EUR", "USD") in ctx.fx_spots


# ---------------------------------------------------------------------------
# Step 3 — Historical data
# ---------------------------------------------------------------------------


class TestHistoricalData:
    def _sample_history(self):
        hd = HistoricalData()
        for i, d in enumerate([date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]):
            snap = MarketDataSnapshot(d)
            snap.add(Quote(QuoteType.SWAP_RATE, "5Y", 0.045 + i * 0.001))
            hd.add(snap)
        return hd

    def test_dates_sorted(self):
        hd = self._sample_history()
        dates = hd.dates
        assert dates == sorted(dates)
        assert len(dates) == 3

    def test_size(self):
        hd = self._sample_history()
        assert hd.size == 3

    def test_get(self):
        hd = self._sample_history()
        snap = hd.get(date(2024, 1, 16))
        assert snap.snapshot_date == date(2024, 1, 16)

    def test_get_missing_raises(self):
        hd = self._sample_history()
        with pytest.raises(KeyError):
            hd.get(date(2024, 1, 20))

    def test_json_roundtrip(self):
        hd = self._sample_history()
        j = hd.to_json()
        hd2 = HistoricalData.from_json(j)
        assert hd2.size == 3
        assert hd2.dates == hd.dates

    def test_dict_roundtrip(self):
        hd = self._sample_history()
        dl = hd.to_dict_list()
        hd2 = HistoricalData.from_dict_list(dl)
        assert hd2.size == hd.size

    def test_overwrite_same_date(self):
        hd = HistoricalData()
        snap1 = MarketDataSnapshot(REF)
        snap1.add(Quote(QuoteType.SWAP_RATE, "5Y", 0.04))
        hd.add(snap1)

        snap2 = MarketDataSnapshot(REF)
        snap2.add(Quote(QuoteType.SWAP_RATE, "5Y", 0.05))
        hd.add(snap2)

        assert hd.size == 1  # replaced, not duplicated
        assert hd.get(REF).quotes[0].value == 0.05
