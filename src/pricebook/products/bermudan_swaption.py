"""The Bermudan swaption (L2). JAECKEL-2002 §12.3.

The right to enter a coterminal interest-rate swap at a fixed strike on any of several exercise
dates — *"a Bermudan swaption contract denoted by 'X-non-call-Y' gives the holder the right to enter
into a swap"*. All exercise opportunities lead into a swap with **the same terminal maturity**, which
is what "coterminal" means and what makes the exercise decision one-dimensional in time: exercising
later gives up accrual periods, it does not change the end date.

Pure frozen data. The exercise policy is L3's (`models/exercise_boundary.py`), the pricing L4's.

**`span` + `first_exercise` is exactly Jäckel's X-non-call-Y.** The span fixes the swap (and so the
terminal maturity); `first_exercise` is the earliest index the holder may exercise at, and every
reset from there to the span's last period is an opportunity. A 6-period span first exercisable at
index 2 is a 6-non-call-2. Storing the opportunity *set* explicitly would be a second source of
truth that could disagree with the span, so it is derived (`exercise_indices`).

**The product cannot validate the exercise DATES, only the indices.** It does not know the tenor, so
it cannot tell that index 0 means today; an exercise opportunity that falls at `t = 0` has no
optionality and is refused by the engine, which does know the dates. The product refuses only what it
can see: an index outside its own span.

Provenance:
  stock:  parked/ng/.../options/bermudan_* — read for SHAPE only via docs/stock/mc_base_inventory.md;
          never imported (§5)
  source: JAECKEL-2002 (docs/references.md) §12.3 (the X-non-call-Y contract and the coterminal
          structure), §12.6 (the residual swap at each exercise time)
  oracle: tests/L4/test_bermudan_swaption_mc.py — the dict round-trip, the payload-kind refusal, the
          out-of-span exercise-index refusal, and the payer/receiver direction exercised by the
          single-exercise-date reduction to the tranche-17 European
  crosscheck: none — a frozen Bermudan description, no numerics
  slice:  tranche 18 — the Bermudan swaption under the LMM
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.money import Money
from pricebook.products.swaption import SwapSpan, SwaptionKind


@dataclass(frozen=True)
class BermudanSwaption:
    """A Bermudan swaption: the coterminal `span`, `strike`, `notional`, `kind` and `first_exercise`."""

    span: SwapSpan
    strike: float
    notional: Money
    kind: SwaptionKind
    first_exercise: int

    def __post_init__(self) -> None:
        if not self.span.first <= self.first_exercise < self.span.end:
            raise ValueError(
                f"the first exercise index must lie in [{self.span.first}, {self.span.end}) — "
                f"inside the swap it exercises into — got {self.first_exercise}"
            )

    @property
    def exercise_indices(self) -> tuple[int, ...]:
        """Every reset from `first_exercise` to the span's last period, each a chance to exercise.

        Derived rather than stored: a stored set could disagree with the span, and the swap the
        holder enters at index `i` is unambiguously the one spanning `i … span.end − 1`."""
        return tuple(range(self.first_exercise, self.span.end))

    def to_dict(self) -> dict:
        return {
            "kind_": "bermudan_swaption",
            "schema_version": SCHEMA_VERSION,
            "span": self.span.to_dict(),
            "strike": self.strike,
            "notional": self.notional.to_dict(),
            "kind": self.kind.value,
            "first_exercise": self.first_exercise,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BermudanSwaption:
        check_schema_version(d)
        if d.get("kind_") != "bermudan_swaption":
            raise ValueError(f"not a bermudan_swaption payload: {d.get('kind_')!r}")
        return cls(
            SwapSpan.from_dict(d["span"]),
            float(d["strike"]),
            Money.from_dict(d["notional"]),
            SwaptionKind(d["kind"]),
            int(d["first_exercise"]),
        )
