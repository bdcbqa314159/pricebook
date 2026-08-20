"""Calibration quotes — market data, distinct from products (L1, doc 18 §1).

Pure data: each quote is what the market prints, keyed to the curve segment it builds
(doc 18 §1). The calibrator turns a quote into a `CalibrationInstrument` (L3) and the
matching tradeable product (L2). A future is quoted as a PRICE; its implied forward rate
is `1 − price` (the convexity correction is deferred to the models topic).

Provenance:
  quarry: python/pricebook/curves/ (DepositQuote/FRAQuote/FutureQuote/ParSwapQuote)
  source: doc 18 §1 (quotes are market data); §2 (futures: rate = 1 − price)
  oracle: each quote's instrument reprices to par; future reproduces 1 − price at IMM
  slice:  cash-instruments (T1 slice 3, Step B)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation import Tenor


@dataclass(frozen=True)
class ParSwapQuote:
    """A par swap of `tenor` at its market par `rate` — the mid/long pillar."""

    tenor: Tenor
    rate: float


@dataclass(frozen=True)
class DepositQuote:
    """A cash deposit from spot to `tenor` at `rate` — the discount-curve front."""

    tenor: Tenor
    rate: float


@dataclass(frozen=True)
class FRAQuote:
    """A forward-rate agreement over ``[start, end]`` at `rate` — a projection-curve
    forward segment. `start=None` means spot (the front-anchoring FRA)."""

    start: Tenor | None
    end: Tenor
    rate: float


@dataclass(frozen=True)
class FutureQuote:
    """An IMM-dated futures quote: `price` (decimal, ~0.965) at IMM date `imm_start`; the
    implied forward rate is ``1 − price`` over ``[imm_start, imm_start + index.tenor]``."""

    imm_start: date
    price: float


@dataclass(frozen=True)
class CapQuote:
    """A flat (quoted) cap volatility: the single lognormal vol that prices the whole cap
    maturing at `maturity`, struck at `strike` — the vol-strip's pillar (doc 18 §1 shape, a
    vol quote). The caplet vols are STRIPPED from a ladder of these (C2 slice 3)."""

    maturity: date
    strike: float
    flat_vol: float
