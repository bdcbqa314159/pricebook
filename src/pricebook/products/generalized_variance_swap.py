"""Generalized variance swaps: the gamma swap and the corridor variance swap (L2).

Both re-weight the vanilla variance swap's floating leg (KWOK-ZHENG §1.3's `I_w = (A/N)Σ w_k r_k²`,
where `w_k` is a weight process) rather than changing what is being measured. Pure data: they
describe the contract and price nothing.

**Siblings of `VarianceSwap`, not subclasses** (§6). They share the words "variance swap" and not the
accrual: a gamma swap weights each squared return by the prevailing spot level `S_k/S_0`, and a
corridor swap includes a return only while the spot sits in `(L,U]`. Those are different payoffs, so
a base class would invite one to be passed where the other is meant — the real-leg / year-on-year-leg
line the charter draws.

**The strike stays in VARIANCE units** for the same reason T31's does: the payoff is linear in
variance and converting at the boundary invites the squaring to happen twice or not at all.

Provenance:
  stock:  none — the migration tree carried no generalized variance swap
  source: KWOK-ZHENG (docs/references.md) §1.3 — the gamma swap's `w_k = S_tk/S_t0` weight and its
          terminal payoff, and the corridor swap's `1{L<S≤U}` indicator on the *old* time level
  oracle: tests/L2/test_generalized_variance_swap.py — the round trip, the refusals, and that the
          two are distinct types with no hierarchy against `VarianceSwap`
  crosscheck: none — pure data, no numerical convention to cross-check
  slice:  tranche 74 — generalized variance swaps
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.money import Money
from pricebook.market.equity import EquityAsset


@dataclass(frozen=True)
class GammaSwap:
    """Pays `notional · (gamma-weighted realized variance − variance_strike)` at `expiry`.

    The weight is the prevailing spot over the initial spot, which **damps the downside**: a crash
    multiplies each squared return by a small `S/S_0`, so the seller is not exposed to the unbounded
    variance a collapsing stock produces. That is the motivation KWOK-ZHENG §1.3 gives, and it is
    also why the replication re-weights by `1/K` rather than `1/K²` — one power of the strike comes
    out of the spot weighting.
    """

    asset: EquityAsset
    expiry: date
    variance_strike: float
    variance_notional: Money

    def __post_init__(self) -> None:
        if self.variance_strike < 0.0:
            raise ValueError(f"a variance strike cannot be negative, got {self.variance_strike}")

    def to_dict(self) -> dict:
        return {
            "kind_": "gamma_swap",
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset.ticker,
            "expiry": self.expiry.isoformat(),
            "variance_strike": self.variance_strike,
            "variance_notional": self.variance_notional.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GammaSwap:
        check_schema_version(d)
        if d.get("kind_") != "gamma_swap":
            raise ValueError(f"not a gamma_swap payload: {d.get('kind_')!r}")
        return cls(
            EquityAsset(d["asset"]),
            date.fromisoformat(d["expiry"]),
            float(d["variance_strike"]),
            Money.from_dict(d["variance_notional"]),
        )


@dataclass(frozen=True)
class CorridorVarianceSwap:
    """Pays on variance accrued only while the spot sits inside the corridor `(lower, upper]`.

    **The corridor is a CONTRACT TERM and is deliberately not the replication grid's bounds**, which
    are a numerical truncation choice. Conflating them would make a wider quadrature grid change the
    payoff, which is the kind of silent modelling error §7's convention register exists to stop: one
    is what the trade says, the other is how accurately we integrate it.

    The source's interval is half-open — `(L,U]`, `L ≥ 0` and `U < ∞` — and monitored on the *old*
    time level for each squared return (§1.3). The half-openness is carried faithfully rather than
    rounded to a closed interval, because it is what the contract says.
    """

    asset: EquityAsset
    expiry: date
    variance_strike: float
    variance_notional: Money
    corridor: tuple[float, float]

    def __post_init__(self) -> None:
        if self.variance_strike < 0.0:
            raise ValueError(f"a variance strike cannot be negative, got {self.variance_strike}")
        lower, upper = self.corridor
        if lower < 0.0:
            raise ValueError(f"a corridor's lower bound cannot be negative, got {lower}")
        if upper <= lower:
            raise ValueError(
                f"a corridor's upper bound must sit above its lower bound, got {lower}..{upper}"
            )

    def to_dict(self) -> dict:
        return {
            "kind_": "corridor_variance_swap",
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset.ticker,
            "expiry": self.expiry.isoformat(),
            "variance_strike": self.variance_strike,
            "variance_notional": self.variance_notional.to_dict(),
            "corridor": list(self.corridor),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CorridorVarianceSwap:
        check_schema_version(d)
        if d.get("kind_") != "corridor_variance_swap":
            raise ValueError(f"not a corridor_variance_swap payload: {d.get('kind_')!r}")
        lower, upper = d["corridor"]
        return cls(
            EquityAsset(d["asset"]),
            date.fromisoformat(d["expiry"]),
            float(d["variance_strike"]),
            Money.from_dict(d["variance_notional"]),
            (float(lower), float(upper)),
        )
