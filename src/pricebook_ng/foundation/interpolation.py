"""Interpolation mechanism — finance-free numerics (L0).

Just the mechanism: given sample `(xs, ys)`, the value at `x` inside the range. The
**curve extrapolation policy is L1** (C4), so this raises outside the range rather than
guessing — a curve decides how to extend itself.

Provenance:
  quarry: python/pricebook/core/ (interpolation)
  source: linear; log-linear (linear in log y — discount factors)
  oracle: linear midpoint; log-linear on exp series; out-of-range raises
  slice:  numerics-config (Topic 0 S6)
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from enum import Enum


class Interpolation(Enum):
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"   # linear in log(y) — discount factors


def interpolate(xs: Sequence[float], ys: Sequence[float], x: float, method: Interpolation) -> float:
    """Interpolate `y` at `x` from ascending `xs`. Raises outside ``[xs[0], xs[-1]]``
    (extrapolation is a curve/L1 policy, not a numeric here)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need matching xs, ys of length >= 2")
    if x < xs[0] or x > xs[-1]:
        raise ValueError(f"{x} outside [{xs[0]}, {xs[-1]}] — extrapolation is an L1 curve policy")
    i = min(bisect_right(xs, x) - 1, len(xs) - 2)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    w = (x - x0) / (x1 - x0)
    if method is Interpolation.LINEAR:
        return y0 + w * (y1 - y0)
    return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))
