"""Tests for `MarketSnapshot` linkage on curve bootstrap (G1 P2 Slice 2).

The snapshot is purely a *provenance pointer* in this slice: deposits/swaps
arguments remain authoritative for the actual numbers. The snapshot id is
stamped onto the resulting `CalibrationResult.market_snapshot_id` so that
the audit chain extends one step further (price -> calibration -> snapshot).

A future slice will let bootstrap derive the deposit/swap lists *from* the
snapshot's quotes.
"""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from pricebook.calibration import CalibrationResult
from pricebook.curves.bootstrap import bootstrap
from pricebook.curves.global_solver import global_bootstrap
from pricebook.market_data import MarketSnapshot, Quote, QuoteId, QuoteKind


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture
def ref_date() -> date:
    return date(2026, 6, 11)


@pytest.fixture
def deposits():
    return [
        (date(2026, 9, 11), 0.04),
        (date(2026, 12, 11), 0.041),
    ]


@pytest.fixture
def swaps():
    return [
        (date(2028, 6, 11), 0.045),
        (date(2031, 6, 11), 0.047),
    ]


@pytest.fixture
def snapshot(deposits, swaps) -> MarketSnapshot:
    """A snapshot whose quotes mirror the deposits + swaps in the test."""
    quotes: list[Quote] = []
    for mat, rate in deposits:
        # tenor label is informal here — exact convention is not what we test
        days = (mat - date(2026, 6, 11)).days
        tenor = f"{days // 30}M"
        quotes.append(
            Quote(QuoteId(QuoteKind.DEPOSIT_RATE, tenor, "USD"), value=rate)
        )
    for mat, rate in swaps:
        years = (mat - date(2026, 6, 11)).days // 365
        quotes.append(
            Quote(QuoteId(QuoteKind.SWAP_RATE, f"{years}Y", "USD"), value=rate)
        )
    return MarketSnapshot.new(quotes=quotes, label="EOD-test")


# ============================================================
# `bootstrap` — sequential brentq bootstrap
# ============================================================

class TestBootstrapWithSnapshot:
    def test_without_snapshot_id_is_none(self, ref_date, deposits, swaps):
        curve = bootstrap(ref_date, deposits, swaps)
        cr = curve.calibration_result
        assert isinstance(cr, CalibrationResult)
        assert cr.provenance.market_snapshot_id is None

    def test_with_snapshot_links_id(self, ref_date, deposits, swaps, snapshot):
        curve = bootstrap(ref_date, deposits, swaps, market_snapshot=snapshot)
        cr = curve.calibration_result
        assert cr is not None
        assert cr.provenance.market_snapshot_id == snapshot.id

    def test_snapshot_is_keyword_only(self):
        sig = inspect.signature(bootstrap)
        param = sig.parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_snapshot_does_not_alter_numerics(self, ref_date, deposits, swaps, snapshot):
        """Adding a snapshot must not change pricing / pillars / DFs."""
        curve_a = bootstrap(ref_date, deposits, swaps)
        curve_b = bootstrap(ref_date, deposits, swaps, market_snapshot=snapshot)
        # Pillar dates identical
        assert curve_a.pillar_dates == curve_b.pillar_dates
        # DFs identical (to machine precision)
        for d in curve_a.pillar_dates:
            assert curve_a.df(d) == pytest.approx(curve_b.df(d), rel=1e-15, abs=1e-15)

    def test_two_snapshots_yield_distinct_results(
        self, ref_date, deposits, swaps, snapshot
    ):
        """Different snapshots -> different calibration results (different ids)."""
        snap2 = MarketSnapshot.new(quotes=snapshot.quotes, label="EOD-test-2")
        assert snap2.id != snapshot.id
        curve1 = bootstrap(ref_date, deposits, swaps, market_snapshot=snapshot)
        curve2 = bootstrap(ref_date, deposits, swaps, market_snapshot=snap2)
        assert curve1.calibration_result.provenance.market_snapshot_id == snapshot.id
        assert curve2.calibration_result.provenance.market_snapshot_id == snap2.id


# ============================================================
# `global_bootstrap` — simultaneous Newton
# ============================================================

class TestGlobalBootstrapWithSnapshot:
    def test_without_snapshot_id_is_none(self, ref_date, deposits, swaps):
        curve = global_bootstrap(ref_date, deposits, swaps)
        cr = curve.calibration_result
        assert isinstance(cr, CalibrationResult)
        assert cr.provenance.market_snapshot_id is None

    def test_with_snapshot_links_id(self, ref_date, deposits, swaps, snapshot):
        curve = global_bootstrap(ref_date, deposits, swaps, market_snapshot=snapshot)
        cr = curve.calibration_result
        assert cr is not None
        assert cr.provenance.market_snapshot_id == snapshot.id

    def test_snapshot_is_keyword_only(self):
        sig = inspect.signature(global_bootstrap)
        param = sig.parameters["market_snapshot"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_snapshot_does_not_alter_numerics(self, ref_date, deposits, swaps, snapshot):
        curve_a = global_bootstrap(ref_date, deposits, swaps)
        curve_b = global_bootstrap(ref_date, deposits, swaps, market_snapshot=snapshot)
        assert curve_a.pillar_dates == curve_b.pillar_dates
        for d in curve_a.pillar_dates:
            assert curve_a.df(d) == pytest.approx(curve_b.df(d), rel=1e-15, abs=1e-15)
