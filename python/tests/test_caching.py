"""Tests for the caching contract, the null baseline, and reference impls."""

import pytest

from pricebook.core.caching import Cache, DictCache, LazyValue, LRUCache, NullCache


class TestCacheProtocol:
    @pytest.mark.parametrize("cache", [NullCache(), DictCache(), LRUCache()])
    def test_every_impl_satisfies_the_protocol(self, cache):
        assert isinstance(cache, Cache)

    @pytest.mark.parametrize("cache", [NullCache(), DictCache(), LRUCache()])
    def test_returns_the_computed_value(self, cache):
        assert cache.get_or_compute(("ois", "5y"), lambda: 0.95) == 0.95


class TestNoCacheConfrontation:
    """The reason the abstraction exists: caching must not change the answer.
    The same computations through a real cache and through NullCache must give
    identical results — any divergence is a cache bug, not a computation bug."""

    def test_lru_matches_null_baseline(self):
        keys = [("df", d) for d in range(20)] * 3  # repeats ⇒ cache hits
        value_for = lambda k: k[1] ** 2 + 0.5  # noqa: E731  deterministic
        lru, null = LRUCache(maxsize=8), NullCache()
        lru_vals = [lru.get_or_compute(k, lambda k=k: value_for(k)) for k in keys]
        null_vals = [null.get_or_compute(k, lambda k=k: value_for(k)) for k in keys]
        assert lru_vals == null_vals  # the cache is honest

    def test_dict_matches_null_baseline(self):
        value_for = lambda k: hash(k) % 97  # noqa: E731
        keys = ["a", "b", "c", "a", "b", "a"]
        d, null = DictCache(), NullCache()
        assert [d.get_or_compute(k, lambda k=k: value_for(k)) for k in keys] == [
            null.get_or_compute(k, lambda k=k: value_for(k)) for k in keys
        ]


class TestNullCache:
    def test_always_computes_never_stores(self):
        n = 0

        def compute():
            nonlocal n
            n += 1
            return n

        c = NullCache()
        assert c.get_or_compute("k", compute) == 1
        assert c.get_or_compute("k", compute) == 2  # recomputed, not cached
        assert n == 2

    def test_invalidate_and_clear_are_noops(self):
        c = NullCache()
        assert c.invalidate(lambda k: True) == 0
        c.clear()  # must not raise


class TestDictCache:
    def test_memoises(self):
        n = 0

        def compute():
            nonlocal n
            n += 1
            return 0.95

        c = DictCache()
        assert c.get_or_compute("k", compute) == 0.95
        assert c.get_or_compute("k", compute) == 0.95
        assert n == 1  # computed once

    def test_invalidate_predicate(self):
        c = DictCache()
        c.get_or_compute(("ois", "5y"), lambda: 1.0)
        c.get_or_compute(("ois", "10y"), lambda: 2.0)
        c.get_or_compute(("sofr", "5y"), lambda: 3.0)
        assert c.invalidate(lambda k: k[0] == "ois") == 2
        assert c.size == 1

    def test_clear(self):
        c = DictCache()
        c.get_or_compute("k", lambda: 1.0)
        c.clear()
        assert c.size == 0


class TestLRUCache:
    def test_cache_hit(self):
        n = 0

        def compute():
            nonlocal n
            n += 1
            return 0.95

        c = LRUCache()
        assert c.get_or_compute(("ois", "5y"), compute) == 0.95
        assert c.get_or_compute(("ois", "5y"), compute) == 0.95
        assert n == 1

    def test_lru_eviction(self):
        c = LRUCache(maxsize=3)
        for i in range(1, 4):
            c.get_or_compute(("c", i), lambda i=i: float(i))
        assert c.size == 3
        c.get_or_compute(("c", 4), lambda: 4.0)
        assert c.size == 3  # oldest evicted

    def test_namespaced_invalidation(self):
        c = LRUCache()
        c.get_or_compute(("ois", "5y"), lambda: 0.95)
        c.get_or_compute(("ois", "10y"), lambda: 0.90)
        c.get_or_compute(("sofr", "5y"), lambda: 0.94)
        assert c.invalidate(lambda k: k[0] == "ois") == 2
        assert c.size == 1  # only sofr left

    def test_hit_rate_and_stats(self):
        c = LRUCache()
        c.get_or_compute("x", lambda: 1.0)  # miss
        c.get_or_compute("x", lambda: 1.0)  # hit
        c.get_or_compute("x", lambda: 1.0)  # hit
        assert c.hit_rate == pytest.approx(2 / 3)
        assert c.stats == {"hits": 2, "misses": 1, "size": 1, "hit_rate": pytest.approx(2 / 3)}

    def test_clear(self):
        c = LRUCache()
        c.get_or_compute("x", lambda: 1.0)
        c.clear()
        assert c.size == 0
        assert c.hit_rate == 0.0


class TestLazyValue:
    def test_deferred(self):
        n = 0

        def compute():
            nonlocal n
            n += 1
            return 42

        lazy = LazyValue(compute)
        assert not lazy.is_computed and n == 0
        assert lazy.value == 42
        assert lazy.is_computed and n == 1

    def test_computed_once(self):
        n = 0

        def compute():
            nonlocal n
            n += 1
            return "expensive"

        lazy = LazyValue(compute)
        _ = lazy.value, lazy.value, lazy.value
        assert n == 1

    def test_reset(self):
        n = 0

        def compute():
            nonlocal n
            n += 1
            return n

        lazy = LazyValue(compute)
        assert lazy.value == 1
        lazy.reset()
        assert not lazy.is_computed
        assert lazy.value == 2

    def test_with_curve_bootstrap(self):
        """Simulate lazy curve building."""
        from datetime import date

        from tests.conftest import make_flat_curve

        built = []

        def build_curve():
            curve = make_flat_curve(date(2024, 1, 15), 0.05)
            built.append(True)
            return curve

        lazy = LazyValue(build_curve)
        assert len(built) == 0
        df = lazy.value.df(date(2025, 1, 15))
        assert len(built) == 1
        assert df > 0
