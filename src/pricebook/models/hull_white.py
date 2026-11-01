"""The Hull-White one-factor core (L3). BERHAN-2012 §2.4; standard Hull-White.

`HullWhiteParams` (mean reversion and volatility) and `β(t,T) = (1 − e^{−κ(T−t)})/κ` — eqs 2.4.3
and 2.4.16 — the two pieces every Hull-White use in the library shares.

**Extracted from `models/jarrow_yildirim.py` at its second consumer**, exactly as Black was in
tranche 9. They lived inside JY while JY was the only user; the nominal-rates work is the second, so
§3d and the rule of two say extract now — one definition, so a Hull-White convention cannot diverge
between the inflation and the rates uses. The nominal Hull-White model *is* JY's nominal factor
standing alone.

**The move is verbatim and bit-identical.** The class body and `hw_beta` were lifted unchanged
rather than retyped: `β`'s `κ → 0` limit and the sign of the exponent are the kind of thing a
re-transcription gets subtly wrong, and tranche 4's oracles are the guard. 15 reference values
(`β` across three `κ` and three intervals, plus `C` and `V²` at three configurations) were captured
as `repr()` before the move and compared after.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py — the parameter bundle and `β`
          were purified into this tree in tranche 4; this module is their relocation, not a
          re-import
  source: BERHAN-2012 (docs/references.md) §2.4 eqs 2.4.3 / 2.4.16 — the Hull-White specification
          of the JY framework; standard Hull-White for the one-factor form
  oracle: tests/L3/test_jarrow_yildirim.py (tranche 4, unchanged and re-run green) — `β`'s `κ → 0`
          limit is `T − t`, which catches every misplaced-`κ` variant; the relocation additionally
          proved bit-identical across 15 captured values
  crosscheck: quantlib HullWhite / HullWhiteModel — deferred to the optional QuantLib tier
  slice:  tranche 10(a) — the Hull-White core extracted to its own module
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HullWhiteParams:
    """One Hull-White factor: mean reversion `kappa` and volatility `sigma`.

    A bundle rather than four loose floats on the model, because JY has *two* such factors and
    naming them `kappa_n, sigma_n, kappa_r, sigma_r` invites passing the nominal pair where the
    real belongs. The grouping is the model's own structure -- the thesis derives it as two
    factors plus the index."""

    kappa: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError(f"mean reversion must be positive, got {self.kappa}")
        if self.sigma < 0.0:
            raise ValueError(f"volatility cannot be negative, got {self.sigma}")


def hw_beta(kappa: float, t: float, maturity: float) -> float:
    """`beta(t,T) = (1 - e^{-kappa(T-t)})/kappa` -- eqs 2.4.3 / 2.4.16.

    Defined once and shared by both factors, so the nominal and real betas cannot drift. As
    `kappa -> 0` it tends to `T - t`, which is the limit that catches a misplaced `kappa`: every
    wrong-denominator variant still looks plausible at a realistic `kappa`."""
    return (1.0 - math.exp(-kappa * (maturity - t))) / kappa
