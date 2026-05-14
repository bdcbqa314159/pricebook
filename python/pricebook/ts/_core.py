"""Core TimeSeries class: construction, arithmetic, alignment, filtering."""

from __future__ import annotations

from datetime import date
from typing import Iterator

import numpy as np


class TimeSeries:
    """Date-indexed numerical series backed by numpy.

    Construction::

        ts = TimeSeries.from_lists([date(2024,1,1), date(2024,1,2)], [1.0, 2.0])
        ts = TimeSeries.from_dict({"2024-01-01": 1.0, "2024-01-02": 2.0})
    """

    __slots__ = ("dates", "values", "name")

    def __init__(self, dates: np.ndarray, values: np.ndarray, name: str = ""):
        self.dates = dates
        self.values = values
        self.name = name

    # ── Construction ──

    @classmethod
    def from_lists(
        cls,
        dates: list[date | str],
        values: list[float],
        name: str = "",
    ) -> TimeSeries:
        if len(dates) != len(values):
            raise ValueError(f"dates ({len(dates)}) and values ({len(values)}) must have same length")
        d_arr = np.array([str(d) for d in dates], dtype="datetime64[D]")
        v_arr = np.asarray(values, dtype=np.float64)
        order = np.argsort(d_arr)
        return cls(d_arr[order], v_arr[order], name)

    @classmethod
    def from_dict(cls, d: dict[str, float], name: str = "") -> TimeSeries:
        keys = sorted(d.keys())
        return cls.from_lists(keys, [d[k] for k in keys], name)

    @classmethod
    def empty(cls, name: str = "") -> TimeSeries:
        return cls(np.array([], dtype="datetime64[D]"),
                   np.array([], dtype=np.float64), name)

    # ── Serialisation ──

    def to_dict(self) -> dict:
        """Serialise to a dict {name, dates, values}. NaN becomes None (JSON-safe)."""
        return {
            "name": self.name,
            "dates": [str(d) for d in self.dates],
            "values": [None if np.isnan(v) else float(v) for v in self.values],
        }

    @classmethod
    def from_serialised(cls, d: dict) -> "TimeSeries":
        """Deserialise from a dict produced by to_dict(). None becomes NaN."""
        values = [float('nan') if v is None else v for v in d["values"]]
        return cls.from_lists(d["dates"], values, d.get("name", ""))

    # ── Properties ──

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return float(self.values[key])
        if isinstance(key, (date, str)):
            d = np.datetime64(str(key), "D")
            idx = np.searchsorted(self.dates, d)
            if idx < len(self.dates) and self.dates[idx] == d:
                return float(self.values[idx])
            raise KeyError(f"date {key} not found")
        if isinstance(key, slice):
            return TimeSeries(self.dates[key], self.values[key], self.name)
        raise TypeError(f"unsupported key type: {type(key)}")

    def __iter__(self) -> Iterator[tuple[np.datetime64, float]]:
        for d, v in zip(self.dates, self.values):
            yield d, float(v)

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return f"TimeSeries('{self.name}', empty)"
        return (f"TimeSeries('{self.name}', n={n}, "
                f"{self.dates[0]}..{self.dates[-1]})")

    # ── Arithmetic ──

    def _binop(self, other, op):
        if isinstance(other, TimeSeries):
            a, b = _align_intersect(self, other)
            return TimeSeries(a.dates, op(a.values, b.values), self.name)
        return TimeSeries(self.dates.copy(), op(self.values, float(other)), self.name)

    def __add__(self, other):
        return self._binop(other, np.add)

    def __sub__(self, other):
        return self._binop(other, np.subtract)

    def __mul__(self, other):
        return self._binop(other, np.multiply)

    def __truediv__(self, other):
        return self._binop(other, np.divide)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return TimeSeries(self.dates.copy(), float(other) - self.values, self.name)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return TimeSeries(self.dates.copy(), -self.values, self.name)

    # ── Cumulative ──

    def cumsum(self) -> TimeSeries:
        return TimeSeries(self.dates.copy(), np.cumsum(self.values), self.name)

    def cumprod(self) -> TimeSeries:
        return TimeSeries(self.dates.copy(), np.cumprod(self.values), self.name)

    def shift(self, n: int = 1) -> TimeSeries:
        v = np.empty_like(self.values)
        v[:] = np.nan
        if n > 0:
            v[n:] = self.values[:-n]
        elif n < 0:
            v[:n] = self.values[-n:]
        else:
            v[:] = self.values
        return TimeSeries(self.dates.copy(), v, self.name)

    # ── Filtering ──

    def between(self, start: date | str, end: date | str) -> TimeSeries:
        d_start = np.datetime64(str(start), "D")
        d_end = np.datetime64(str(end), "D")
        mask = (self.dates >= d_start) & (self.dates <= d_end)
        return TimeSeries(self.dates[mask], self.values[mask], self.name)

    def business_days_only(self) -> TimeSeries:
        mask = np.is_busday(self.dates)
        return TimeSeries(self.dates[mask], self.values[mask], self.name)

    def dropna(self) -> TimeSeries:
        mask = ~np.isnan(self.values)
        return TimeSeries(self.dates[mask], self.values[mask], self.name)

    # ── Alignment ──

    def align(self, other: TimeSeries, fill: str = "ffill") -> tuple[TimeSeries, TimeSeries]:
        """Align two series to their date union."""
        union = np.union1d(self.dates, other.dates)
        a_vals = _reindex(self.dates, self.values, union, fill)
        b_vals = _reindex(other.dates, other.values, union, fill)
        return (TimeSeries(union, a_vals, self.name),
                TimeSeries(union, b_vals, other.name))

    # ── Resample ──

    def resample(self, freq: str = "M") -> TimeSeries:
        """Resample to lower frequency, taking last value per period.

        freq: 'W' (weekly), 'M' (monthly), 'Q' (quarterly).
        """
        if len(self) == 0:
            return TimeSeries.empty(self.name)

        if freq == "W":
            keys = (self.dates - self.dates.astype("datetime64[D]").astype(int) % 7).astype("datetime64[W]")
        elif freq == "M":
            keys = self.dates.astype("datetime64[M]")
        elif freq == "Q":
            months = self.dates.astype("datetime64[M]").astype(int)
            keys = ((months // 3) * 3).astype("datetime64[M]")
        else:
            raise ValueError(f"unsupported freq: {freq!r}")

        # Last value per period
        unique_keys, indices = np.unique(keys, return_index=True)
        # For "last" we need the last index per group
        result_dates = []
        result_values = []
        boundaries = list(indices) + [len(keys)]
        for i in range(len(boundaries) - 1):
            end_idx = boundaries[i + 1] - 1
            result_dates.append(self.dates[end_idx])
            result_values.append(self.values[end_idx])

        return TimeSeries(
            np.array(result_dates, dtype="datetime64[D]"),
            np.array(result_values, dtype=np.float64),
            self.name,
        )

    # ── Methods bound from other modules (set in __init__.py) ──
    # simple_returns, log_returns, sharpe, max_drawdown, etc.


# ── Helpers ──

def _align_intersect(a: TimeSeries, b: TimeSeries) -> tuple[TimeSeries, TimeSeries]:
    """Align two series to their date intersection."""
    common = np.intersect1d(a.dates, b.dates)
    a_idx = np.searchsorted(a.dates, common)
    b_idx = np.searchsorted(b.dates, common)
    return (TimeSeries(common, a.values[a_idx], a.name),
            TimeSeries(common, b.values[b_idx], b.name))


def _reindex(
    src_dates: np.ndarray,
    src_values: np.ndarray,
    target_dates: np.ndarray,
    fill: str,
) -> np.ndarray:
    """Reindex values to target dates with fill strategy."""
    result = np.full(len(target_dates), np.nan)
    # Direct matches
    idx_in_src = np.searchsorted(src_dates, target_dates)
    for i, idx in enumerate(idx_in_src):
        if idx < len(src_dates) and src_dates[idx] == target_dates[i]:
            result[i] = src_values[idx]

    if fill == "ffill":
        # Forward fill NaNs
        last = np.nan
        for i in range(len(result)):
            if np.isnan(result[i]):
                result[i] = last
            else:
                last = result[i]
    elif fill == "zero":
        result = np.nan_to_num(result, nan=0.0)
    # "nan" → leave NaNs
    return result
