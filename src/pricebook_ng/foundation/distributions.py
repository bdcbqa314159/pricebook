"""Probability distributions (L0 numerical toolkit — migrated on demand).

Only `norm_cdf` so far: the standard-normal CDF via `math.erf`, exact to libm
precision and dependency-free. The full quarry distributions module (scipy-backed
Normal/StudentT/... ) migrates piecewise as products need it (CLAUDE.md 6b).

Provenance:
  quarry: python/pricebook/numerical/_distributions.py
  source: standard-normal CDF, N(x) = (1 + erf(x/sqrt(2))) / 2
  oracle: exercised by the Hull-White ZCB-option oracle (S07)
  slice:  S07
"""

from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    """Standard-normal cumulative distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))
