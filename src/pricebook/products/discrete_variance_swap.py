"""The discretely-monitored variance swap (L2) — frozen data, prices nothing.

Distinct from `VarianceSwap` (T31), which pays on **continuous** realized variance. The difference
is not cosmetic: the discrete contract is worth strictly more, by `O(1/N)`, and §6 keeps them as
**separate types rather than one with an optional count** — a `None` monitoring count would make the
payoff's identity depend on a field being absent, and the two are priced by different engines.

Provenance:
  stock:  none
  source: KWOK-ZHENG (docs/references.md) §4.4.2 / eq (4.4) — the discrete realized-variance swap
          and its fair strike as a sum over monitoring intervals
  oracle: tests/L4/test_time_changed_levy.py — priced through the registry as the continuous swap
          plus the §4.4.2 adjustment, and shown to converge to the continuous price as N grows
  crosscheck: none — a frozen product description
  slice:  tranche 78 — variance swaps under time-changed Lévy processes
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Money
from pricebook.market.equity import EquityAsset


@dataclass(frozen=True)
class DiscreteVarianceSwap:
    """Pays `notional · (realized variance over N monitorings − variance_strike)` at `expiry`.

    `monitoring_count` is **constitutive of the payoff**, not an implementation detail (§6): two
    otherwise-identical swaps monitored daily and monthly are different contracts worth different
    amounts, which is the entire subject of this tranche.
    """

    asset: EquityAsset
    expiry: date
    variance_strike: float
    variance_notional: Money
    monitoring_count: int

    def __post_init__(self) -> None:
        if self.variance_strike < 0.0:
            raise ValueError(f"a variance strike may not be negative, got {self.variance_strike}")
        if not self.monitoring_count > 0:
            raise ValueError(
                f"a discrete variance swap needs at least one monitoring, got "
                f"{self.monitoring_count}"
            )
