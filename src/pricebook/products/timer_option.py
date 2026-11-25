"""The perpetual timer option (L2) — frozen data, prices nothing.

A call or put that exercises when the underlying's **accumulated variance** `∫₀^τ V_s ds` first
reaches a **variance budget** `B`. The buyer fixes the variance they pay for rather than the
calendar time: a vanilla option's cost is driven by `σ²T`, and a timer makes that product the
contract term instead of leaving it to realised vol.

**Perpetual, so there is no expiry field** — the exercise date is a random first-passage time, and
a `None` expiry would be a field whose absence carried meaning. The finite-maturity variant is a
different product (and a PDE problem, `to_go_deeper`), not this one with a date hung on it (§6).

**`variance_budget` is in the same units as a total variance** — `0.08` is what a 20% vol
accumulates in two years — matching `VarianceSwap.variance_strike` (variance, not volatility) and
`models/black.py`'s `w = σ²T`, so no conversion happens at any boundary (C14's habit).

Provenance:
  stock:  none — the migration tree carried no timer option
  source: KWOK-ZHENG (docs/references.md) §1.3.3 — the timer-option payoff and the variance budget;
          §6.2.1 — the perpetual case as a first-passage problem on accumulated variance
  oracle: tests/L4/test_timer_option.py — priced through the registry under Heston by MC, pinned by
          the exact `ρ=0`/zero-carry Black–Scholes reduction and the §6.2.3 perturbation
  crosscheck: none — a frozen product description
  slice:  tranche 79 — perpetual timer options
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.money import Money
from pricebook.market.equity import EquityAsset
from pricebook.products.equity_option import EquityOptionKind


@dataclass(frozen=True)
class TimerOption:
    """Pays `notional · (S_τ − K)⁺` (call) at the random time `τ` when `∫V dt` reaches the budget."""

    asset: EquityAsset
    strike: float
    variance_budget: float
    notional: Money
    kind: EquityOptionKind

    def __post_init__(self) -> None:
        if not self.strike > 0.0:
            raise ValueError(f"a timer option's strike must be positive, got {self.strike}")
        if not self.variance_budget > 0.0:
            raise ValueError(
                f"a timer option's variance budget must be positive, got {self.variance_budget}"
            )
