"""Tests for equity rich/cheap analysis and delta/vega hedging."""

import pytest
from datetime import date

from pricebook.equity_rv import (
    BookExposure,
    CalendarLevel,
    HedgeAllocation,
    HedgeInstrument,
    SkewLevel,
    ZScoreSignal,
    calendar_monitor,
    delta_hedge,
    hedged_exposure,
    implied_vs_historical_vol,
    optimal_delta_vega_hedge,
    skew_monitor,
    vega_hedge,
)


# ---- Step 1: rich/cheap monitors ----

class TestImpliedVsHistoricalVol:
    def test_extreme_high_signals_rich(self):
        history = [0.18, 0.19, 0.20, 0.21, 0.22] * 4
        sig = implied_vs_historical_vol(0.40, history, threshold=2.0)
        assert sig.signal == "rich"
        assert sig.z_score is not None
        assert sig.z_score > 2.0

    def test_extreme_low_signals_cheap(self):
        history = [0.18, 0.19, 0.20, 0.21, 0.22] * 4
        sig = implied_vs_historical_vol(0.05, history, threshold=2.0)
        assert sig.signal == "cheap"
        assert sig.z_score < -2.0

    def test_mid_range_signals_fair(self):
        history = [0.18, 0.19, 0.20, 0.21, 0.22] * 4
        sig = implied_vs_historical_vol(0.20, history, threshold=2.0)
        assert sig.signal == "fair"
        # Within ±2σ
        assert sig.z_score is not None
        assert abs(sig.z_score) < 2.0

    def test_no_history_returns_fair(self):
        sig = implied_vs_historical_vol(0.20, [], threshold=2.0)
        assert sig.signal == "fair"
        assert sig.z_score is None

    def test_single_observation_returns_fair(self):
        sig = implied_vs_historical_vol(0.20, [0.18], threshold=2.0)
        assert sig.signal == "fair"
        assert sig.z_score is None

    def test_percentile_inside_range(self):
        history = [0.18, 0.19, 0.20, 0.21, 0.22]
        sig = implied_vs_historical_vol(0.205, history)
        assert sig.percentile is not None
        assert 0.0 <= sig.percentile <= 100.0

    def test_threshold_controls_sensitivity(self):
        history = [0.18, 0.19, 0.20, 0.21, 0.22]
        sig_strict = implied_vs_historical_vol(0.23, history, threshold=3.0)
        sig_loose = implied_vs_historical_vol(0.23, history, threshold=1.5)
        # 0.23 is ~2σ above mean — strict threshold says fair, loose says rich
        assert sig_strict.signal == "fair"
        assert sig_loose.signal == "rich"


class TestSkewMonitor:
    def test_extreme_skew_signals(self):
        history = [-0.02, -0.015, -0.01, -0.005, 0.0]
        sig = skew_monitor(date(2025, 1, 15), rr_level=0.05,
                           history=history, threshold=2.0)
        assert isinstance(sig, SkewLevel)
        assert sig.signal == "rich"

    def test_mid_range_fair(self):
        history = [-0.02, -0.015, -0.01, -0.005, 0.0]
        sig = skew_monitor(date(2025, 1, 15), rr_level=-0.012,
                           history=history)
        assert sig.signal == "fair"

    def test_records_expiry_and_level(self):
        sig = skew_monitor(date(2025, 1, 15), 0.0, history=[0.0, 0.0])
        assert sig.expiry == date(2025, 1, 15)
        assert sig.rr_level == 0.0


class TestCalendarMonitor:
    def test_backwardation_recorded(self):
        history = [-0.02, -0.015, -0.01, -0.005, 0.0]
        sig = calendar_monitor(
            short_expiry=date(2024, 4, 15),
            long_expiry=date(2025, 1, 15),
            short_vol=0.30, long_vol=0.20,  # front above back
            history=history, threshold=2.0,
        )
        assert isinstance(sig, CalendarLevel)
        assert sig.spread == pytest.approx(0.10)
        assert sig.signal == "rich"  # large positive spread vs history

    def test_normal_term_structure(self):
        history = [-0.02, -0.015, -0.01, -0.005, 0.0]
        sig = calendar_monitor(
            date(2024, 4, 15), date(2025, 1, 15),
            short_vol=0.20, long_vol=0.215, history=history,
        )
        assert sig.spread == pytest.approx(-0.015)
        # Within range → fair
        assert sig.signal == "fair"

    def test_records_both_vols(self):
        sig = calendar_monitor(
            date(2024, 4, 15), date(2025, 1, 15),
            short_vol=0.25, long_vol=0.20, history=[],
        )
        assert sig.short_vol == 0.25
        assert sig.long_vol == 0.20


# ---- Step 2: delta + vega hedging ----

class TestSingleInstrumentHedges:
    def test_delta_hedge_with_underlying(self):
        # Underlying has delta = 1.0 per share
        qty = delta_hedge(book_delta=10_000.0, hedge_delta_per_unit=1.0)
        assert qty == pytest.approx(-10_000.0)

    def test_delta_hedge_with_futures(self):
        # Future has delta = 50 (S&P futures multiplier)
        qty = delta_hedge(book_delta=10_000.0, hedge_delta_per_unit=50.0)
        assert qty == pytest.approx(-200.0)

    def test_delta_hedge_zero_per_unit(self):
        assert delta_hedge(10_000.0, 0.0) == 0.0

    def test_vega_hedge_basic(self):
        # Hedge option vega = 100 per contract; book vega = 5000
        qty = vega_hedge(book_vega=5_000.0, hedge_vega_per_unit=100.0)
        assert qty == pytest.approx(-50.0)

    def test_vega_hedge_zero_per_unit(self):
        assert vega_hedge(5_000.0, 0.0) == 0.0


class TestOptimalDeltaVegaHedge:
    def test_two_options_flatten_both(self):
        """Slice 139 step 2 test: hedged book has near-zero delta and vega."""
        book = BookExposure(delta=1_000.0, vega=5_000.0)
        # Two hedge options with distinct (delta, vega) profiles.
        h1 = HedgeInstrument("ATM_call", delta=0.55, vega=120.0)
        h2 = HedgeInstrument("OTM_put", delta=-0.30, vega=80.0)

        a1, a2 = optimal_delta_vega_hedge(book, h1, h2)

        hedged = hedged_exposure(book, [a1, a2])
        assert hedged.delta == pytest.approx(0.0, abs=1e-9)
        assert hedged.vega == pytest.approx(0.0, abs=1e-9)

    def test_returns_two_allocations(self):
        book = BookExposure(delta=100.0, vega=1_000.0)
        h1 = HedgeInstrument("a", 1.0, 0.0)
        h2 = HedgeInstrument("b", 0.0, 1.0)
        a1, a2 = optimal_delta_vega_hedge(book, h1, h2)
        assert isinstance(a1, HedgeAllocation)
        assert isinstance(a2, HedgeAllocation)
        assert a1.instrument.name == "a"
        assert a2.instrument.name == "b"
        # Direct: a1 = -100 (kill delta), a2 = -1000 (kill vega)
        assert a1.quantity == pytest.approx(-100.0)
        assert a2.quantity == pytest.approx(-1000.0)

    def test_linear_dependence_raises(self):
        book = BookExposure(delta=100.0, vega=1_000.0)
        h1 = HedgeInstrument("a", 0.5, 100.0)
        h2 = HedgeInstrument("b", 1.0, 200.0)  # parallel to h1
        with pytest.raises(ValueError):
            optimal_delta_vega_hedge(book, h1, h2)

    def test_zero_book_returns_zero_quantities(self):
        book = BookExposure(delta=0.0, vega=0.0)
        h1 = HedgeInstrument("a", 0.5, 100.0)
        h2 = HedgeInstrument("b", -0.3, 80.0)
        a1, a2 = optimal_delta_vega_hedge(book, h1, h2)
        assert a1.quantity == pytest.approx(0.0)
        assert a2.quantity == pytest.approx(0.0)


class TestHedgedExposure:
    def test_no_allocations(self):
        book = BookExposure(delta=10.0, vega=200.0)
        result = hedged_exposure(book, [])
        assert result.delta == 10.0
        assert result.vega == 200.0

    def test_single_allocation_subtracts(self):
        book = BookExposure(delta=10.0, vega=200.0)
        h = HedgeInstrument("hedge", delta=1.0, vega=50.0)
        alloc = HedgeAllocation(h, quantity=-10.0)
        result = hedged_exposure(book, [alloc])
        # delta: 10 + (-10 × 1) = 0
        # vega: 200 + (-10 × 50) = -300
        assert result.delta == pytest.approx(0.0)
        assert result.vega == pytest.approx(-300.0)

    def test_multiple_allocations(self):
        book = BookExposure(delta=100.0, vega=1_000.0)
        h1 = HedgeInstrument("a", delta=1.0, vega=10.0)
        h2 = HedgeInstrument("b", delta=0.0, vega=50.0)
        allocs = [
            HedgeAllocation(h1, quantity=-100.0),  # kills delta
            HedgeAllocation(h2, quantity=-20.0),    # kills 1000 vega
        ]
        result = hedged_exposure(book, allocs)
        assert result.delta == pytest.approx(0.0)
        # 1000 + (-100 × 10) + (-20 × 50) = 1000 - 1000 - 1000 = -1000
        assert result.vega == pytest.approx(-1000.0)
