"""Tests for caching, memoisation, and lazy evaluation."""

import pytest

from pricebook.caching import CurveCache, CalibrationCache, LazyValue


class TestCurveCache:
    def test_cache_hit(self):
        cache = CurveCache()
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return 0.95

        v1 = cache.get_or_compute("ois", "5y", compute)
        v2 = cache.get_or_compute("ois", "5y", compute)

        assert v1 == v2 == 0.95
        assert call_count == 1  # computed only once

    def test_different_keys_different_values(self):
        cache = CurveCache()
        v1 = cache.get_or_compute("ois", "5y", lambda: 0.95)
        v2 = cache.get_or_compute("ois", "10y", lambda: 0.90)
        assert v1 == 0.95
        assert v2 == 0.90
        assert cache.size == 2

    def test_different_curves(self):
        cache = CurveCache()
        cache.get_or_compute("ois", "5y", lambda: 0.95)
        cache.get_or_compute("sofr", "5y", lambda: 0.94)
        assert cache.size == 2

    def test_invalidate(self):
        cache = CurveCache()
        cache.get_or_compute("ois", "5y", lambda: 0.95)
        cache.get_or_compute("ois", "10y", lambda: 0.90)
        cache.get_or_compute("sofr", "5y", lambda: 0.94)

        removed = cache.invalidate("ois")
        assert removed == 2
        assert cache.size == 1  # only sofr left

    def test_lru_eviction(self):
        cache = CurveCache(maxsize=3)
        cache.get_or_compute("c", "1", lambda: 1.0)
        cache.get_or_compute("c", "2", lambda: 2.0)
        cache.get_or_compute("c", "3", lambda: 3.0)
        assert cache.size == 3

        cache.get_or_compute("c", "4", lambda: 4.0)
        assert cache.size == 3  # oldest evicted

    def test_hit_rate(self):
        cache = CurveCache()
        cache.get_or_compute("c", "x", lambda: 1.0)  # miss
        cache.get_or_compute("c", "x", lambda: 1.0)  # hit
        cache.get_or_compute("c", "x", lambda: 1.0)  # hit
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_clear(self):
        cache = CurveCache()
        cache.get_or_compute("c", "x", lambda: 1.0)
        cache.clear()
        assert cache.size == 0
        assert cache.hit_rate == 0.0

    def test_stats(self):
        cache = CurveCache()
        cache.get_or_compute("c", "x", lambda: 1.0)
        cache.get_or_compute("c", "x", lambda: 1.0)
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestCalibrationCache:
    def test_cache_hit(self):
        cache = CalibrationCache()
        call_count = 0

        def calibrate():
            nonlocal call_count
            call_count += 1
            return {"alpha": 0.2, "rho": -0.3}

        p1 = cache.get_or_calibrate("sabr", "hash_v1", calibrate)
        p2 = cache.get_or_calibrate("sabr", "hash_v1", calibrate)

        assert p1 == p2
        assert call_count == 1

    def test_different_hash_recalibrates(self):
        cache = CalibrationCache()
        call_count = 0

        def calibrate():
            nonlocal call_count
            call_count += 1
            return {"alpha": 0.2 + call_count * 0.01}

        cache.get_or_calibrate("sabr", "hash_v1", calibrate)
        cache.get_or_calibrate("sabr", "hash_v2", calibrate)
        assert call_count == 2

    def test_invalidate(self):
        cache = CalibrationCache()
        cache.get_or_calibrate("sabr", "h1", lambda: {"a": 1})
        cache.get_or_calibrate("heston", "h1", lambda: {"b": 2})
        cache.invalidate("sabr")
        assert cache.size == 1

    def test_clear(self):
        cache = CalibrationCache()
        cache.get_or_calibrate("m", "h", lambda: {})
        cache.clear()
        assert cache.size == 0


class TestLazyValue:
    def test_deferred(self):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return 42

        lazy = LazyValue(compute)
        assert not lazy.is_computed
        assert call_count == 0

        val = lazy.value
        assert val == 42
        assert lazy.is_computed
        assert call_count == 1

    def test_computed_once(self):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "expensive"

        lazy = LazyValue(compute)
        _ = lazy.value
        _ = lazy.value
        _ = lazy.value
        assert call_count == 1

    def test_reset(self):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return call_count

        lazy = LazyValue(compute)
        assert lazy.value == 1
        lazy.reset()
        assert not lazy.is_computed
        assert lazy.value == 2

    def test_with_curve_bootstrap(self):
        """Simulate lazy curve building."""
        from tests.conftest import make_flat_curve
        from datetime import date

        built = []

        def build_curve():
            curve = make_flat_curve(date(2024, 1, 15), 0.05)
            built.append(True)
            return curve

        lazy = LazyValue(build_curve)
        assert len(built) == 0

        # Access triggers build
        df = lazy.value.df(date(2025, 1, 15))
        assert len(built) == 1
        assert df > 0
