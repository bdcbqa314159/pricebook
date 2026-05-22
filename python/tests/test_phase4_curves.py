"""Tests for Phase 4 curve modules: blending, seasonal, diffusion, storage."""

import math
import pytest
import numpy as np
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.curve_blending import (
    splice_curves, blend_curves, BlendMethod,
)
from pricebook.curves.seasonal_curve import (
    SeasonalPattern, SeasonalCurve, strip_seasonal,
    USD_SEASONAL, extract_seasonal_pattern,
)
from pricebook.curves.curve_diffusion import (
    CurveDiffusionEngine, CurveDiffusionConfig, CurveDiffusionResult,
)
from pricebook.curves.curve_storage import (
    CurveSnapshot, CurveDelta, CurveStore,
    compress_curve, decompress_curve,
)

REF = date(2024, 1, 15)


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.04)


@pytest.fixture
def steep_curve():
    """Steeper curve: 3% short, 5% long."""
    dates = [date(2025, 1, 15), date(2027, 1, 15), date(2029, 1, 15),
             date(2034, 1, 15), date(2044, 1, 15)]
    dfs = [math.exp(-r * t) for r, t in zip([0.03, 0.035, 0.04, 0.045, 0.05],
                                              [1, 3, 5, 10, 20])]
    return DiscountCurve(REF, dates, dfs)


# ═══════════════════════════════════════════════════════════════
# 4.1: Curve Blending
# ═══════════════════════════════════════════════════════════════

class TestCurveBlending:
    def test_splice_basic(self, flat_curve, steep_curve):
        spliced = splice_curves(flat_curve, steep_curve, splice_tenor=5.0)
        assert isinstance(spliced, DiscountCurve)
        # Should be close to flat_curve at short end
        df_1y = spliced.df(date(2025, 1, 15))
        assert abs(df_1y - flat_curve.df(date(2025, 1, 15))) < 0.01

    def test_splice_sigmoid(self, flat_curve, steep_curve):
        spliced = splice_curves(flat_curve, steep_curve, 5.0, method=BlendMethod.SIGMOID)
        assert spliced.df(date(2034, 1, 15)) > 0

    def test_splice_step(self, flat_curve, steep_curve):
        spliced = splice_curves(flat_curve, steep_curve, 5.0, method=BlendMethod.STEP)
        assert spliced.df(date(2034, 1, 15)) > 0

    def test_blend_equal_weights(self, flat_curve, steep_curve):
        blended = blend_curves([flat_curve, steep_curve], [0.5, 0.5])
        df_5y = blended.df(date(2029, 1, 15))
        # Should be between the two
        df_flat = flat_curve.df(date(2029, 1, 15))
        df_steep = steep_curve.df(date(2029, 1, 15))
        assert min(df_flat, df_steep) <= df_5y <= max(df_flat, df_steep)

    def test_blend_single(self, flat_curve):
        blended = blend_curves([flat_curve], [1.0])
        df = blended.df(date(2029, 1, 15))
        assert abs(df - flat_curve.df(date(2029, 1, 15))) < 0.01

    def test_blend_empty_raises(self):
        with pytest.raises(ValueError):
            blend_curves([], [])


# ═══════════════════════════════════════════════════════════════
# 4.2: Seasonal Curve
# ═══════════════════════════════════════════════════════════════

class TestSeasonalCurve:
    def test_year_end_premium(self, flat_curve):
        sc = SeasonalCurve(flat_curve, USD_SEASONAL)
        # Year-end should have spread
        spread = sc.seasonal_spread(date(2024, 12, 31))
        assert spread > 0

    def test_mid_month_no_spread(self, flat_curve):
        sc = SeasonalCurve(flat_curve, USD_SEASONAL)
        spread = sc.seasonal_spread(date(2024, 7, 15))
        assert spread == 0.0

    def test_df_lower_than_base(self, flat_curve):
        """Seasonal curve should have lower DFs (higher rates near period-ends)."""
        sc = SeasonalCurve(flat_curve, USD_SEASONAL)
        # Over a year, seasonal adds some spread → slightly lower DF
        df_base = flat_curve.df(date(2025, 1, 15))
        df_seasonal = sc.df(date(2025, 1, 15))
        assert df_seasonal <= df_base

    def test_strip_seasonal(self, flat_curve):
        """Stripping seasonal should recover approximately the base curve."""
        stripped = strip_seasonal(flat_curve, USD_SEASONAL)
        # Should be close to original (flat curve has no seasonal to strip)
        df_orig = flat_curve.df(date(2029, 1, 15))
        df_strip = stripped.df(date(2029, 1, 15))
        assert abs(df_orig - df_strip) < 0.01

    def test_pattern_to_dict(self):
        d = USD_SEASONAL.to_dict()
        assert "year_end_spread_bp" in d

    def test_extract_pattern(self):
        """Extract from synthetic fixings."""
        fixings = {}
        for i in range(365):
            d = date(2023, 1, 1) + __import__("datetime").timedelta(days=i)
            base = 0.05
            if d.month == 12 and d.day >= 27:
                base += 0.001  # year-end premium
            fixings[d] = base
        pattern = extract_seasonal_pattern(fixings)
        assert pattern.year_end_spread_bp > 0


# ═══════════════════════════════════════════════════════════════
# 4.3: Curve Diffusion
# ═══════════════════════════════════════════════════════════════

class TestCurveDiffusion:
    def test_basic_simulation(self, flat_curve):
        config = CurveDiffusionConfig(
            n_factors=2, n_paths=100, n_steps=4,
            horizon_years=1.0, seed=42,
        )
        engine = CurveDiffusionEngine(flat_curve, config)
        result = engine.simulate()
        assert isinstance(result, CurveDiffusionResult)
        assert result.n_paths == 100
        assert result.n_steps == 4
        assert len(result.curves) == 5  # n_steps + 1

    def test_curves_are_valid(self, flat_curve):
        config = CurveDiffusionConfig(n_paths=50, n_steps=4, seed=42)
        result = CurveDiffusionEngine(flat_curve, config).simulate()
        # Each path at each step should be a valid curve
        for curve in result.scenario_curves(2):
            df = curve.df(date(2029, 1, 15))
            assert 0 < df < 2  # reasonable range

    def test_mean_near_initial(self, flat_curve):
        """Mean forward rate should be near initial (risk-neutral drift ≈ 0)."""
        config = CurveDiffusionConfig(n_paths=500, n_steps=4, seed=42)
        result = CurveDiffusionEngine(flat_curve, config).simulate()
        # Mean forward at t=0 should match initial
        initial_mean = result.forward_rate_mean[0, :]
        for f in initial_mean:
            assert abs(f - 0.04) < 0.01

    def test_std_increases(self, flat_curve):
        """Forward rate uncertainty should increase with time."""
        config = CurveDiffusionConfig(n_paths=200, n_steps=8, seed=42)
        result = CurveDiffusionEngine(flat_curve, config).simulate()
        std_early = result.forward_rate_std[2, 0]
        std_late = result.forward_rate_std[-1, 0]
        assert std_late > std_early

    def test_to_dict(self, flat_curve):
        config = CurveDiffusionConfig(n_paths=20, n_steps=2, seed=42)
        d = CurveDiffusionEngine(flat_curve, config).simulate().to_dict()
        assert "n_paths" in d
        assert "forward_rate_mean" in d


# ═══════════════════════════════════════════════════════════════
# 4.4: Curve Storage
# ═══════════════════════════════════════════════════════════════

class TestCurveStorage:
    def test_snapshot_roundtrip(self, flat_curve):
        snap = CurveSnapshot.from_curve(flat_curve, "USD_OIS")
        reconstructed = snap.to_curve()
        df_orig = flat_curve.df(date(2029, 1, 15))
        df_recon = reconstructed.df(date(2029, 1, 15))
        assert abs(df_orig - df_recon) < 0.001

    def test_compress_decompress(self, flat_curve, steep_curve):
        s1 = CurveSnapshot.from_curve(flat_curve, "c1", "2024-01-15T00:00:00")
        s2 = CurveSnapshot.from_curve(flat_curve, "c2", "2024-01-16T00:00:00")
        # Same curve → zero delta
        delta = compress_curve(s2, s1)
        assert max(abs(s) for s in delta.pillar_shifts) < 0.1  # < 0.1bp

    def test_store_save_load(self, flat_curve):
        store = CurveStore()
        snap = CurveSnapshot.from_curve(flat_curve, "USD_OIS", "2024-01-15T18:00:00")
        snap_id = store.save(snap)
        loaded = store.load(snap_id)
        assert loaded.curve_id == "USD_OIS"

    def test_store_history(self, flat_curve):
        store = CurveStore()
        for i in range(3):
            snap = CurveSnapshot.from_curve(flat_curve, "USD_OIS", f"2024-01-{15+i}T18:00:00")
            store.save(snap)
        history = store.history("USD_OIS")
        assert len(history) == 3

    def test_store_diff(self, flat_curve):
        """Diff between two snapshots of the same curve structure."""
        store = CurveStore()
        bumped = flat_curve.bumped(0.005)  # 50bp shift
        s1 = CurveSnapshot.from_curve(flat_curve, "c1", "t1")
        s2 = CurveSnapshot.from_curve(bumped, "c2", "t2")
        id1 = store.save(s1)
        id2 = store.save(s2)
        delta = store.diff(id1, id2)
        assert max(abs(s) for s in delta.pillar_shifts) > 1  # > 1bp shift

    def test_snapshot_to_dict(self, flat_curve):
        d = CurveSnapshot.from_curve(flat_curve, "test").to_dict()
        assert "curve_id" in d
        assert "zero_rates" in d

    def test_delta_to_dict(self, flat_curve):
        s = CurveSnapshot.from_curve(flat_curve, "c1", "t1")
        delta = compress_curve(s, s)
        d = delta.to_dict()
        assert "max_shift_bp" in d
