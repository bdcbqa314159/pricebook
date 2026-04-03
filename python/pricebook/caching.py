"""
Caching, memoisation, and lazy evaluation for the pricing stack.

CurveCache: LRU cache for discount factor lookups with invalidation.
CalibrationCache: stores calibrated model params, invalidates on input change.
LazyValue: defers computation until first access.

    from pricebook.caching import CurveCache, CalibrationCache, LazyValue

    cache = CurveCache(maxsize=1024)
    df = cache.get_or_compute("ois", date(2029,1,15), lambda: curve.df(d))

    lazy_curve = LazyValue(lambda: bootstrap(ref, deposits, swaps))
    # curve not built yet
    pv = instrument.pv(lazy_curve.value)  # now it's built
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar, Generic

T = TypeVar("T")


class CurveCache:
    """LRU cache for curve query results (e.g. df, forward_rate).

    Keys are (curve_name, query_key) tuples. Invalidation clears
    all entries for a given curve_name.

    Args:
        maxsize: maximum number of cached entries.
    """

    def __init__(self, maxsize: int = 4096):
        self._maxsize = maxsize
        self._cache: OrderedDict[tuple, float] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get_or_compute(
        self, curve_name: str, query_key, compute_fn,
    ) -> float:
        """Return cached value or compute and cache it.

        Args:
            curve_name: identifier for the curve (e.g. "ois").
            query_key: hashable key (e.g. a date or (date1, date2)).
            compute_fn: callable that returns the value if not cached.
        """
        key = (curve_name, query_key)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        value = compute_fn()
        self._cache[key] = value

        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

        return value

    def invalidate(self, curve_name: str) -> int:
        """Remove all cached entries for a curve. Returns count removed."""
        keys_to_remove = [k for k in self._cache if k[0] == curve_name]
        for k in keys_to_remove:
            del self._cache[k]
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear entire cache."""
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


class CalibrationCache:
    """Cache for calibrated model parameters.

    Stores (inputs_hash → calibrated_params). Invalidates when inputs change.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def get_or_calibrate(
        self, model_name: str, inputs_hash: str, calibrate_fn,
    ) -> dict:
        """Return cached params or calibrate.

        Args:
            model_name: e.g. "sabr_5y".
            inputs_hash: hash of the calibration inputs.
            calibrate_fn: callable that returns the calibrated params dict.
        """
        key = (model_name, inputs_hash)
        if key in self._cache:
            return self._cache[key]

        params = calibrate_fn()
        self._cache[key] = params
        return params

    def invalidate(self, model_name: str) -> None:
        """Remove all cached params for a model."""
        keys_to_remove = [k for k in self._cache if k[0] == model_name]
        for k in keys_to_remove:
            del self._cache[k]

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class LazyValue(Generic[T]):
    """Lazy evaluation: defers computation until .value is accessed.

    The compute function is called at most once. Subsequent accesses
    return the cached result.

    Args:
        compute_fn: callable that produces the value.
    """

    def __init__(self, compute_fn):
        self._compute = compute_fn
        self._value: T | None = None
        self._computed = False

    @property
    def value(self) -> T:
        if not self._computed:
            self._value = self._compute()
            self._computed = True
        return self._value

    @property
    def is_computed(self) -> bool:
        return self._computed

    def reset(self) -> None:
        """Force recomputation on next access."""
        self._computed = False
        self._value = None
