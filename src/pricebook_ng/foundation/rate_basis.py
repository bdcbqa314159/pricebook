"""Rate-quotation basis and conversion — finance-free numerics + convention (L0).

A rate is meaningless without its **basis** (compounding + day count). Market quotes carry
theirs — USD fixed 30/360 semi-annual; a bond yield is semi-annual bond-equivalent — while
curve internals are continuous, and par/zero/forward rates are all "rates" on *different*
bases. Converting is pure math; getting it wrong (semi treated as continuous) is off by
~r²/2 — numerically valid, silently incorrect.

**Recorded invariant (S12):** internal curve rates are **continuously compounded on the
curve's day count**; *quotes carry their own basis and convert at the boundary*. That keeps
`Rate` a plain float without the ambiguity.

`Compounding` (the quotation basis) is a *different* concept from the index `AccrualMethod`
(compounded/exponential/averaging) — the two were colliding, hence the index rename.

Provenance:
  quarry: — (absent from both trees; the biggest gap the gate audit found, S12)
  source: standard rate-basis conversion
  oracle: semi→annual exact; semi→continuous ≠ the raw number; round-trip; identity
  slice:  rate-basis (Topic 0 gate rework, S12)
"""

from __future__ import annotations

import math
from enum import Enum


class Compounding(Enum):
    """How a *quoted* rate compounds — its basis. A finite standard set (enum)."""

    SIMPLE = "simple"
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    CONTINUOUS = "continuous"


_PERIODS = {
    Compounding.ANNUAL: 1,
    Compounding.SEMI_ANNUAL: 2,
    Compounding.QUARTERLY: 4,
    Compounding.MONTHLY: 12,
}


def growth_factor(rate: float, t: float, basis: Compounding) -> float:
    """The growth of one unit over `t` years at `rate` quoted on `basis`."""
    if basis is Compounding.CONTINUOUS:
        return math.exp(rate * t)
    if basis is Compounding.SIMPLE:
        base = 1.0 + rate * t
    else:
        base = 1.0 + rate / _PERIODS[basis]
    if base <= 0.0:
        # a growth factor ≤ 0 means a loss of ≥ 100% — undefined, and it would otherwise
        # surface downstream as a bare `math domain error` or a silent complex (AC-T4.11).
        raise ValueError(
            f"growth_factor: rate {rate} on {basis.value} over t={t} implies a non-positive "
            f"growth factor (1 + rate{'·t' if basis is Compounding.SIMPLE else '/m'} = {base}); "
            f"a loss of 100% or more has no equivalent rate on another basis"
        )
    if basis is Compounding.SIMPLE:
        return base
    return base ** (_PERIODS[basis] * t)


def _rate_from_factor(factor: float, t: float, basis: Compounding) -> float:
    if basis is Compounding.CONTINUOUS:
        return math.log(factor) / t
    if basis is Compounding.SIMPLE:
        return (factor - 1.0) / t
    m = _PERIODS[basis]
    return m * (factor ** (1.0 / (m * t)) - 1.0)


def convert_rate(
    rate: float, t: float, from_basis: Compounding, to_basis: Compounding
) -> float:
    """The equivalent `rate` on `to_basis` — same growth over `t` years — from `from_basis`."""
    if t <= 0.0:
        raise ValueError(f"convert_rate needs t > 0, got {t}")
    if from_basis is to_basis:
        return rate
    return _rate_from_factor(growth_factor(rate, t, from_basis), t, to_basis)
