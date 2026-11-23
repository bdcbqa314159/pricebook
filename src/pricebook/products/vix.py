"""VIX futures and options (L2). KWOK-ZHENG §3.1.2.

Pure data. Both name the **VIX window** `τ` as well as the maturity, because a VIX derivative is
written on the index's own averaging window — the CBOE's is 30 calendar days — and a payload without
it cannot say which index the trade is on. Carried and exercised by the round trip (§6: constitutive
of identity).

**A VIX future's price is an EXPECTATION, not a discounted payoff** (eq 3.6: `F_t = E_t[VIX_T]`), and
the option's is discounted (eq 3.7a). That asymmetry lives in the engine; the products only record
which contract they are.

Provenance:
  stock:  none — the migration tree carried no VIX derivative
  source: KWOK-ZHENG (docs/references.md) §3.1.2 — eq (3.6) the futures price, eq (3.7a) the
          European call; §1.5.2 for what the index itself is
  oracle: tests/L4/test_vix.py — the round trips, the refusals, and the discounted/undiscounted
          split asserted through the registered pricers
  crosscheck: none — pure data, no numerical convention to cross-check
  slice:  tranche 76 — VIX derivatives
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money


def _checked_window(window: float) -> float:
    if not window > 0.0:
        raise ValueError(f"a VIX window must be positive, got {window}")
    return window


@dataclass(frozen=True)
class VixFuture:
    """`E_t[VIX_T(window)]` per unit of notional — eq (3.6)."""

    maturity: date
    window: float
    notional: Money

    def __post_init__(self) -> None:
        _checked_window(self.window)

    def to_dict(self) -> dict:
        return {
            "kind_": "vix_future",
            "schema_version": SCHEMA_VERSION,
            "maturity": self.maturity.isoformat(),
            "window": self.window,
            "notional": self.notional.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> VixFuture:
        check_schema_version(d)
        if d.get("kind_") != "vix_future":
            raise ValueError(f"not a vix_future payload: {d.get('kind_')!r}")
        return cls(
            date.fromisoformat(d["maturity"]), float(d["window"]), Money.from_dict(d["notional"])
        )


@dataclass(frozen=True)
class VixOption:
    """`e^{-r(T-t)} E_t[(ω(VIX_T(window) − strike))⁺]` per unit of notional — eq (3.7a).

    The strike is in **volatility points** (the index's own units), not variance: VIX is quoted as a
    volatility, and the payoff is linear in it.
    """

    maturity: date
    window: float
    strike: float
    notional: Money
    kind: CapFloorKind

    def __post_init__(self) -> None:
        _checked_window(self.window)
        if self.strike < 0.0:
            raise ValueError(f"a VIX strike cannot be negative, got {self.strike}")

    def to_dict(self) -> dict:
        return {
            "kind_": "vix_option",
            "schema_version": SCHEMA_VERSION,
            "maturity": self.maturity.isoformat(),
            "window": self.window,
            "strike": self.strike,
            "notional": self.notional.to_dict(),
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VixOption:
        check_schema_version(d)
        if d.get("kind_") != "vix_option":
            raise ValueError(f"not a vix_option payload: {d.get('kind_')!r}")
        return cls(
            date.fromisoformat(d["maturity"]),
            float(d["window"]),
            float(d["strike"]),
            Money.from_dict(d["notional"]),
            CapFloorKind(d["kind"]),
        )
