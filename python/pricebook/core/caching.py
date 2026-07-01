"""Caching contract, a null baseline, and reference implementations.

The module defines the *action* — ``Cache.get_or_compute(key, compute)`` — as a
minimal Protocol, plus a :class:`NullCache` that computes every time and stores
nothing. Any component that caches takes a ``Cache``; injecting ``NullCache``
runs the identical code path with caching disabled, so a cache-induced
discrepancy (staleness, wrong key, aliasing) can always be **isolated** by
comparing against the no-cache baseline — the computation is confronted
independently of the cache.

    from pricebook.core.caching import Cache, NullCache, LRUCache, DictCache

    def price(..., cache: Cache = NullCache()):
        return cache.get_or_compute(("df", d), lambda: curve.df(d))
    # inject LRUCache() in production; NullCache() to prove the cache is honest.

Implementations:
* :class:`NullCache` — pass-through; the "no caching" baseline.
* :class:`DictCache` — unbounded memo, for immutable-keyed data.
* :class:`LRUCache`  — bounded LRU with hit/miss stats and predicate invalidation.
* :class:`LazyValue` — compute-once lazy thunk (a single value, not keyed).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Generic, Hashable, Protocol, TypeVar, runtime_checkable

V = TypeVar("V")
T = TypeVar("T")

Predicate = Callable[[Hashable], bool]


@runtime_checkable
class Cache(Protocol):
    """The caching action: return the cached value for ``key``, or compute+store it.

    The single seam every cache exposes. Swap :class:`NullCache` in anywhere to
    run the same path without caching — if the result changes, the *cache* is at
    fault, not the computation.
    """

    def get_or_compute(self, key: Hashable, compute: Callable[[], V]) -> V: ...


class NullCache:
    """Pass-through cache: always computes, never stores.

    The baseline for confronting any caching solution — inject this to disable
    caching without touching the call site, then compare results against a real
    cache. Any difference is a cache bug (staleness, wrong key, aliasing), never
    a computation bug. ``invalidate``/``clear`` are no-ops so it drops in cleanly.
    """

    def get_or_compute(self, key: Hashable, compute: Callable[[], V]) -> V:
        return compute()

    def invalidate(self, predicate: Predicate) -> int:
        return 0

    def clear(self) -> None:
        pass


class DictCache:
    """Unbounded memo — for immutable-keyed data (e.g. holidays per year).

    No eviction: use only where the key space is naturally bounded and the value
    for a key never changes. For anything unbounded/mutable, use :class:`LRUCache`.
    """

    def __init__(self) -> None:
        self._cache: dict[Hashable, object] = {}

    def get_or_compute(self, key: Hashable, compute: Callable[[], V]) -> V:
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]  # type: ignore[return-value]

    def invalidate(self, predicate: Predicate) -> int:
        """Remove every key for which predicate(key) is true. Returns count."""
        keys = [k for k in self._cache if predicate(k)]
        for k in keys:
            del self._cache[k]
        return len(keys)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class LRUCache:
    """Bounded LRU cache with hit/miss stats and predicate invalidation.

    Least-recently-used eviction once ``maxsize`` is exceeded. Structured keys
    give namespaced invalidation, e.g. ``invalidate(lambda k: k[0] == "ois")``.

    Not thread-safe — guard externally if shared across threads.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[Hashable, object] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], V]) -> V:
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]  # type: ignore[return-value]

        self._misses += 1
        value = compute()
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return value

    def invalidate(self, predicate: Predicate) -> int:
        """Remove every key for which predicate(key) is true. Returns count."""
        keys = [k for k in self._cache if predicate(k)]
        for k in keys:
            del self._cache[k]
        return len(keys)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self.size,
            "hit_rate": self.hit_rate,
        }


class LazyValue(Generic[T]):
    """Lazy evaluation: defers computation until ``.value`` is accessed.

    The compute function is called at most once; subsequent accesses return the
    cached result. Not thread-safe (a race can compute twice). Distinct from the
    keyed caches above — this memoises a single value.

    Args:
        compute_fn: callable that produces the value.
    """

    def __init__(self, compute_fn: Callable[[], T]) -> None:
        self._compute = compute_fn
        self._value: T | None = None
        self._computed = False

    @property
    def value(self) -> T:
        if not self._computed:
            self._value = self._compute()
            self._computed = True
        return self._value  # type: ignore[return-value]

    @property
    def is_computed(self) -> bool:
        return self._computed

    def reset(self) -> None:
        """Force recomputation on next access."""
        self._computed = False
        self._value = None
