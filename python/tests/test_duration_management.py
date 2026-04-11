"""Tests for duration management."""

import pytest

from pricebook.duration_management import (
    BarbellVsBullet,
    CurveDV01,
    DV01Ladder,
    DurationTarget,
    HedgeInstrumentKRD,
    KRDHedgeAllocation,
    barbell_vs_bullet,
    build_dv01_ladder,
    curve_dv01,
    duration_target_tracking,
    hedged_ladder,
    optimal_krd_hedge,
)


# ---- Step 1: DV01 ladder ----

class TestDV01Ladder:
    def test_build(self):
        ladder = build_dv01_ladder("test", {"2Y": 100, "5Y": 200, "10Y": 300})
        assert ladder.book_name == "test"
        assert len(ladder.rungs) == 3

    def test_total_dv01_equals_sum(self):
        """Step 1 test: DV01 ladder sums to total DV01."""
        ladder = build_dv01_ladder("test", {"2Y": 100, "5Y": 200, "10Y": 300})
        assert ladder.total_dv01 == pytest.approx(600)

    def test_dv01_at(self):
        ladder = build_dv01_ladder("test", {"2Y": 100, "5Y": 200})
        assert ladder.dv01_at("5Y") == pytest.approx(200)
        assert ladder.dv01_at("30Y") == 0.0

    def test_as_dict(self):
        ladder = build_dv01_ladder("test", {"2Y": 100, "5Y": 200})
        d = ladder.as_dict()
        assert d["2Y"] == 100
        assert d["5Y"] == 200

    def test_empty_ladder(self):
        ladder = build_dv01_ladder("test", {})
        assert ladder.total_dv01 == 0.0


class TestCurveDV01:
    def test_parallel(self):
        result = curve_dv01(100, 200, 300)
        assert result.parallel_dv01 == pytest.approx(600)

    def test_steepener_2s10s(self):
        result = curve_dv01(100, 200, 300)
        # 10Y - 2Y = 200
        assert result.steepener_2s10s == pytest.approx(200)

    def test_butterfly_2s5s10s(self):
        result = curve_dv01(100, 200, 300)
        # 100 + 300 - 2×200 = 0
        assert result.butterfly_2s5s10s == pytest.approx(0)

    def test_butterfly_nonzero(self):
        result = curve_dv01(100, 150, 300)
        # 100 + 300 - 2×150 = 100
        assert result.butterfly_2s5s10s == pytest.approx(100)

    def test_flat_dv01(self):
        result = curve_dv01(100, 100, 100)
        assert result.steepener_2s10s == pytest.approx(0)
        assert result.butterfly_2s5s10s == pytest.approx(0)


class TestDurationTargetTracking:
    def test_within_band(self):
        result = duration_target_tracking(5.2, 5.0, band=0.5)
        assert result.within_band is True
        assert result.deviation == pytest.approx(0.2)

    def test_outside_band(self):
        result = duration_target_tracking(6.0, 5.0, band=0.5)
        assert result.within_band is False
        assert result.deviation == pytest.approx(1.0)

    def test_exact_match(self):
        result = duration_target_tracking(5.0, 5.0)
        assert result.within_band is True
        assert result.deviation == pytest.approx(0.0)

    def test_underweight_duration(self):
        result = duration_target_tracking(4.0, 5.0)
        assert result.deviation == pytest.approx(-1.0)


# ---- Step 2: hedging ----

class TestOptimalKRDHedge:
    def test_single_tenor_single_instrument(self):
        book = {"2Y": 100.0}
        inst = [HedgeInstrumentKRD("2Y_swap", {"2Y": 50.0})]
        allocs = optimal_krd_hedge(book, inst)
        assert len(allocs) == 1
        # Need -100/50 = -2 units
        assert allocs[0].quantity == pytest.approx(-2.0)

    def test_two_tenor_two_instrument(self):
        """Step 2 test: hedged book has near-zero key rate DV01s."""
        book = {"2Y": 100.0, "10Y": 300.0}
        instruments = [
            HedgeInstrumentKRD("2Y_swap", {"2Y": 50.0, "10Y": 0.0}),
            HedgeInstrumentKRD("10Y_swap", {"2Y": 0.0, "10Y": 100.0}),
        ]
        allocs = optimal_krd_hedge(book, instruments)
        residual = hedged_ladder(book, allocs)
        assert residual["2Y"] == pytest.approx(0.0, abs=1e-9)
        assert residual["10Y"] == pytest.approx(0.0, abs=1e-9)

    def test_overdetermined_lstsq(self):
        """More tenors than instruments → least-squares best fit."""
        book = {"2Y": 100.0, "5Y": 200.0, "10Y": 300.0}
        instruments = [
            HedgeInstrumentKRD("5Y_swap", {"2Y": 10.0, "5Y": 50.0, "10Y": 20.0}),
        ]
        allocs = optimal_krd_hedge(book, instruments)
        assert len(allocs) == 1
        # Residual won't be zero with 1 instrument and 3 tenors
        # but the hedge should reduce total DV01
        residual = hedged_ladder(book, allocs)
        total_before = sum(book.values())
        total_after = sum(abs(v) for v in residual.values())
        assert total_after < total_before

    def test_empty_instruments(self):
        assert optimal_krd_hedge({"2Y": 100}, []) == []

    def test_empty_book(self):
        assert optimal_krd_hedge({}, [HedgeInstrumentKRD("a", {"2Y": 50})]) == []

    def test_multi_tenor_hedge_flattens(self):
        book = {"2Y": 100.0, "5Y": 200.0, "10Y": 300.0}
        instruments = [
            HedgeInstrumentKRD("2Y_swap", {"2Y": 50.0, "5Y": 5.0, "10Y": 0.0}),
            HedgeInstrumentKRD("5Y_swap", {"2Y": 5.0, "5Y": 80.0, "10Y": 10.0}),
            HedgeInstrumentKRD("10Y_swap", {"2Y": 0.0, "5Y": 10.0, "10Y": 120.0}),
        ]
        allocs = optimal_krd_hedge(book, instruments)
        residual = hedged_ladder(book, allocs)
        for tenor, dv01 in residual.items():
            assert abs(dv01) < 1.0  # near-zero after hedge


class TestHedgedLadder:
    def test_no_allocations(self):
        book = {"2Y": 100.0, "10Y": 300.0}
        assert hedged_ladder(book, []) == book

    def test_single_allocation(self):
        book = {"2Y": 100.0}
        alloc = KRDHedgeAllocation(
            HedgeInstrumentKRD("swap", {"2Y": 50.0}), quantity=-2.0,
        )
        residual = hedged_ladder(book, [alloc])
        assert residual["2Y"] == pytest.approx(0.0)


# ---- Barbell vs bullet ----

class TestBarbellVsBullet:
    def test_barbell_higher_convexity(self):
        result = barbell_vs_bullet(
            short_duration=2.0, short_convexity=8.0, short_weight=0.5,
            long_duration=10.0, long_convexity=120.0, long_weight=0.5,
            bullet_duration=6.0, bullet_convexity=40.0,
        )
        # Barbell duration = 0.5×2 + 0.5×10 = 6 (matches bullet)
        assert result.barbell_duration == pytest.approx(6.0)
        # Barbell convexity = 0.5×8 + 0.5×120 = 64 > 40
        assert result.barbell_convexity == pytest.approx(64.0)
        assert result.convexity_advantage == pytest.approx(24.0)
        assert result.recommendation == "barbell"

    def test_bullet_higher_convexity(self):
        result = barbell_vs_bullet(
            short_duration=2.0, short_convexity=5.0, short_weight=0.5,
            long_duration=10.0, long_convexity=50.0, long_weight=0.5,
            bullet_duration=6.0, bullet_convexity=40.0,
        )
        # Barbell convexity = 0.5×5 + 0.5×50 = 27.5 < 40
        assert result.recommendation == "bullet"
        assert result.convexity_advantage < 0

    def test_indifferent(self):
        result = barbell_vs_bullet(
            short_duration=2.0, short_convexity=20.0, short_weight=0.5,
            long_duration=10.0, long_convexity=60.0, long_weight=0.5,
            bullet_duration=6.0, bullet_convexity=40.0,
        )
        # 0.5×20 + 0.5×60 = 40 = bullet
        assert result.recommendation == "indifferent"
