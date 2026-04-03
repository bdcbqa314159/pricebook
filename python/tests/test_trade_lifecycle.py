"""Tests for trade lifecycle: amendments, exercises, novations, history."""

import pytest
from datetime import date

from pricebook.trade import Trade
from pricebook.trade_lifecycle import (
    ManagedTrade,
    EventType,
    LifecycleEvent,
)
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol
from pricebook.swaption import Swaption
from pricebook.swap import InterestRateSwap
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class FakeInstrument:
    """Simple instrument that returns a fixed PV."""

    def __init__(self, pv: float):
        self._pv = pv

    def pv_ctx(self, ctx):
        return self._pv


def _make_ctx():
    curve = make_flat_curve(REF, 0.05)
    return PricingContext(
        valuation_date=REF,
        discount_curve=curve,
        vol_surfaces={"ir": FlatVol(0.20)},
    )


# ---------------------------------------------------------------------------
# Step 1 — Amendments
# ---------------------------------------------------------------------------


class TestAmendments:
    def test_amend_notional(self):
        t = Trade(FakeInstrument(100.0), notional_scale=1.0, trade_id="t1")
        mt = ManagedTrade(t, REF)

        mt.amend(REF, notional_scale=2.0)
        assert mt.current.notional_scale == 2.0
        assert mt.version == 1

    def test_original_preserved(self):
        t = Trade(FakeInstrument(100.0), notional_scale=1.0, trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.amend(REF, notional_scale=2.0)

        assert mt.get_version(0).notional_scale == 1.0
        assert mt.get_version(1).notional_scale == 2.0

    def test_amend_direction(self):
        t = Trade(FakeInstrument(100.0), direction=1, trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.amend(REF, direction=-1)
        assert mt.current.direction == -1

    def test_amend_counterparty(self):
        t = Trade(FakeInstrument(100.0), counterparty="ACME", trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.amend(REF, counterparty="GLOBEX")
        assert mt.current.counterparty == "GLOBEX"

    def test_amend_instrument(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.amend(REF, instrument=FakeInstrument(200.0))
        ctx = _make_ctx()
        assert mt.current.pv(ctx) == 200.0

    def test_amended_trade_pv_differs(self):
        t = Trade(FakeInstrument(100.0), notional_scale=1.0, trade_id="t1")
        mt = ManagedTrade(t, REF)
        ctx = _make_ctx()
        pv_before = mt.current.pv(ctx)
        mt.amend(REF, notional_scale=2.0)
        pv_after = mt.current.pv(ctx)
        assert pv_after != pv_before
        assert pv_after == pytest.approx(2.0 * pv_before)

    def test_no_amendment_raises(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, REF)
        with pytest.raises(ValueError, match="No amendments"):
            mt.amend(REF)

    def test_multiple_amendments(self):
        t = Trade(FakeInstrument(100.0), notional_scale=1.0, trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.amend(REF, notional_scale=2.0)
        mt.amend(REF, notional_scale=3.0)
        assert mt.version == 2
        assert mt.current.notional_scale == 3.0


# ---------------------------------------------------------------------------
# Step 2 — Option exercise
# ---------------------------------------------------------------------------


class TestExercise:
    def test_exercise_swaption(self):
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)
        swap = InterestRateSwap(date(2025, 1, 15), date(2030, 1, 15), fixed_rate=0.05)

        t = Trade(swn, trade_id="swn1")
        mt = ManagedTrade(t, REF)
        mt.exercise(date(2025, 1, 15), swap)

        assert isinstance(mt.current.instrument, InterestRateSwap)
        assert mt.is_exercised

    def test_exercise_records_event(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.exercise(REF, FakeInstrument(50.0))

        ex_events = [e for e in mt.history if e.event_type == EventType.EXERCISED]
        assert len(ex_events) == 1
        assert ex_events[0].details["old_instrument"] == "FakeInstrument"
        assert ex_events[0].details["new_instrument"] == "FakeInstrument"

    def test_double_exercise_raises(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.exercise(REF, FakeInstrument(50.0))
        with pytest.raises(ValueError, match="already exercised"):
            mt.exercise(REF, FakeInstrument(25.0))

    def test_exercise_preserves_direction(self):
        t = Trade(FakeInstrument(100.0), direction=-1, trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.exercise(REF, FakeInstrument(50.0))
        assert mt.current.direction == -1


# ---------------------------------------------------------------------------
# Step 3 — Novation
# ---------------------------------------------------------------------------


class TestNovation:
    def test_novate(self):
        t = Trade(FakeInstrument(100.0), counterparty="ACME", trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.novate(REF, "GLOBEX")
        assert mt.current.counterparty == "GLOBEX"

    def test_novation_preserves_economics(self):
        t = Trade(FakeInstrument(100.0), counterparty="ACME",
                  notional_scale=2.0, direction=-1, trade_id="t1")
        mt = ManagedTrade(t, REF)
        ctx = _make_ctx()
        pv_before = mt.current.pv(ctx)
        mt.novate(REF, "GLOBEX")
        pv_after = mt.current.pv(ctx)
        assert pv_after == pv_before

    def test_novation_event(self):
        t = Trade(FakeInstrument(100.0), counterparty="ACME", trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.novate(REF, "GLOBEX")

        nov = [e for e in mt.history if e.event_type == EventType.NOVATED]
        assert len(nov) == 1
        assert nov[0].details["old_counterparty"] == "ACME"
        assert nov[0].details["new_counterparty"] == "GLOBEX"


# ---------------------------------------------------------------------------
# Step 4 — Trade history / audit trail
# ---------------------------------------------------------------------------


class TestHistory:
    def test_creation_event(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, REF)
        assert len(mt.history) == 1
        assert mt.history[0].event_type == EventType.CREATED

    def test_full_lifecycle(self):
        t = Trade(FakeInstrument(100.0), counterparty="ACME",
                  notional_scale=1.0, trade_id="t1")
        mt = ManagedTrade(t, REF)

        # Amend
        mt.amend(date(2024, 2, 1), notional_scale=2.0)
        # Novate
        mt.novate(date(2024, 3, 1), "GLOBEX")
        # Exercise
        mt.exercise(date(2024, 6, 1), FakeInstrument(50.0))

        history = mt.history
        assert len(history) == 4
        types = [e.event_type for e in history]
        assert types == [
            EventType.CREATED,
            EventType.AMENDED,
            EventType.NOVATED,
            EventType.EXERCISED,
        ]

    def test_history_chronological(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, date(2024, 1, 1))
        mt.amend(date(2024, 2, 1), notional_scale=2.0)
        mt.amend(date(2024, 3, 1), notional_scale=3.0)

        dates = [e.event_date for e in mt.history]
        assert dates == sorted(dates)

    def test_all_versions_queryable(self):
        t = Trade(FakeInstrument(100.0), notional_scale=1.0, trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.amend(REF, notional_scale=2.0)
        mt.amend(REF, notional_scale=3.0)

        assert mt.get_version(0).notional_scale == 1.0
        assert mt.get_version(1).notional_scale == 2.0
        assert mt.get_version(2).notional_scale == 3.0

    def test_invalid_version_raises(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, REF)
        with pytest.raises(IndexError):
            mt.get_version(5)

    def test_version_numbers_in_events(self):
        t = Trade(FakeInstrument(100.0), trade_id="t1")
        mt = ManagedTrade(t, REF)
        mt.amend(REF, notional_scale=2.0)
        mt.novate(REF, "GLOBEX")

        versions = [e.version for e in mt.history]
        assert versions == [0, 1, 2]
