"""Model capability protocols (L3) — the closed-form building blocks a model exposes.

Promoted from doc 22 (F2), which *designed* these but deferred the code to the second model
(rule of two). C2 is that second model, so the protocols land now and the engine depends on the
CAPABILITY, not the concrete type (CLAUDE.md §1; doc 22 Q1/Q3′). `DiscountingModel` satisfies
`CalibratedModel` structurally (it carries `.market`); `BlackModel` adds `BlackVol`.

A capability ships WITH its semantic contract (measure/numeraire/units) or it is not ratified
(doc 22 Q3′a) — otherwise "shared capability" degrades to shared signature + per-model semantics,
which is `isinstance` with extra steps.

Provenance:
  quarry: (none — new protocol surface; doc 22 designed it)
  source: redesign/22 Q1 (CalibratedModel) · Q3′a (a capability carries its measure/units contract)
  oracle: the linear reprice-to-par suite prices through the protocol-typed engine unchanged
  slice:  black-caplet (C2 slice 1)
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from pricebook_ng.foundation import RateIndex
from pricebook_ng.market.snapshot import MarketSnapshot


@runtime_checkable
class CalibratedModel(Protocol):
    """A model bound to the `MarketSnapshot` it was calibrated to (A1). The engine reaches the
    market ONLY through `model.market`, never as a peer argument. Concrete models add one or more
    capability protocols; higher layers depend on the capability, not the concrete type."""

    @property
    def market(self) -> MarketSnapshot: ...


@runtime_checkable
class BlackVol(Protocol):
    """The lognormal-volatility capability.

    **Semantic contract (doc 22 Q3′a).** `black_vol(index, expiry, strike)` is the LOGNORMAL
    (Black-76) implied volatility of the `index` forward rate over its accrual, for an optionlet
    expiring at `expiry` struck at `strike`, quoted under the **T-FORWARD MEASURE** (T = the
    optionlet payment date) — where the forward rate is a martingale and Black-76 returns the
    undiscounted `E[(F − K)+]`. **Units:** annualized lognormal (`0.20` = 20%). Normal/Bachelier
    volatility is a DISTINCT capability (deferred to its second consumer), never this one."""

    def black_vol(self, index: RateIndex, expiry: date, strike: float) -> float: ...
