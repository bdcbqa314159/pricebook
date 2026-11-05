"""The European swaption (L2). MCINERNEY-ZASTAWNIAK §5.9-5.10.

The right, at `T_first`, to enter the interest-rate swap spanning the LMM tenor's forwards
`first … first+count−1`. Payer or receiver. Pure frozen data — the pricing is L4's business and the
swap-rate volatility is L3's.

Payoff at expiry: `N · A(T_first) · [ω·(S(T_first) − K)]⁺`, with `A` the swap annuity (eq 5.32), `S`
the forward swap rate (eq 5.31) and `ω = +1` for a payer, `−1` for a receiver.

**The span is `(first, count)` rather than dates, and that follows the tenor it lives on.** The LMM
tenor (tranche 16) is an index structure whose `T_i` are the cumulative accruals, so an index pair
names the swap exactly; carrying dates as well would be a second source of truth that can disagree
with the tenor, which is worse than carrying none. The expiry is `T_first` and is *derived*, never
stored — §6's carried-but-unread rule, read the other way: a field that must agree with another is
not constitutive of identity, it is a liability.

**`SwaptionKind` is an enum, not a boolean**, for the same reason `CapFloorKind` is: a payer and a
receiver passed the wrong way round is a plausible wrong number with no error anywhere.

**Deliberately not sharing a type with `NominalCap` or `InflationCapFloor`.** All three carry a
strike, a notional and a direction flag, and that is the shared *word* §6 warns about — an option on
a *basket* of forwards (whose correlation is the whole modelling problem) is not the same shape as a
strip of options on single forwards. The rule of two does not fire on a coincidence of fields.

Provenance:
  stock:  parked/ng/.../swaption_mc.py — read for SHAPE only via docs/stock/mc_base_inventory.md;
          never imported (§5)
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §5.9 (the forward swap and its annuity) and
          §5.10 (the payer/receiver swaption payoff); JAECKEL-2002 §12.3 for the coterminal-span
          convention the Bermudan (tranche 18) will exercise
  oracle: tests/L4/test_swaption_mc.py — the dict round-trip, the payload-kind refusal, and the
          payer/receiver parity that pins `ω` against the finance
  crosscheck: none — a frozen swaption description, no numerics
  slice:  tranche 17 — the European swaption under the LMM
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.money import Money


class SwaptionKind(Enum):
    """`ω = +1` (payer, pays the fixed leg) or `ω = −1` (receiver)."""

    PAYER = "payer"
    RECEIVER = "receiver"

    @property
    def omega(self) -> float:
        return 1.0 if self is SwaptionKind.PAYER else -1.0


@dataclass(frozen=True)
class SwapSpan:
    """Which of the tenor's forwards the underlying swap spans: `count` of them, from `first`.

    A bundle rather than two loose ints on the swaption, because they are only meaningful together
    and a caller who swapped them would get a different, plausible swap (§6: compelled bundling is
    vocabulary). Bounds are checked against the tenor at pricing time, since the product does not
    know which tenor it will be priced on."""

    first: int
    count: int

    def __post_init__(self) -> None:
        if self.first < 0:
            raise ValueError(f"a swap span starts at a non-negative index, got {self.first}")
        if self.count < 1:
            raise ValueError(f"a swap span covers at least one period, got {self.count}")

    @property
    def end(self) -> int:
        """One past the last forward — the index of the bond the swap matures on."""
        return self.first + self.count

    def to_dict(self) -> dict:
        return {"first": self.first, "count": self.count}

    @classmethod
    def from_dict(cls, d: dict) -> SwapSpan:
        return cls(int(d["first"]), int(d["count"]))


@dataclass(frozen=True)
class EuropeanSwaption:
    """A European swaption: its underlying `span`, `strike` rate, `notional` and `kind`."""

    span: SwapSpan
    strike: float
    notional: Money
    kind: SwaptionKind

    def to_dict(self) -> dict:
        return {
            "kind_": "european_swaption",
            "schema_version": SCHEMA_VERSION,
            "span": self.span.to_dict(),
            "strike": self.strike,
            "notional": self.notional.to_dict(),
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EuropeanSwaption:
        check_schema_version(d)
        if d.get("kind_") != "european_swaption":
            raise ValueError(f"not a european_swaption payload: {d.get('kind_')!r}")
        return cls(
            SwapSpan.from_dict(d["span"]),
            float(d["strike"]),
            Money.from_dict(d["notional"]),
            SwaptionKind(d["kind"]),
        )
