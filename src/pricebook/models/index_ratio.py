"""What a pricer needs of a model to value a year-on-year index ratio (L3).

The year-on-year swap and the inflation cap/floor both ask a model exactly two questions about the
period `[Tᵢ₋₁, Tᵢ]`:

* the **`Tᵢ`-forward expectation of the index ratio** `I(Tᵢ)/I(Tᵢ₋₁)` — BERHAN's `m` of eq 2.6.2;
* the **variance of that ratio's log** — BERHAN's `V²` of eq 2.6.4.

Everything else in those pricers is curve arithmetic and the schedule walk. So this protocol is the
whole of a model's contribution, and a second model is served by implementing it rather than by a
branch inside the pricer (§1's no-ladder law).

**It exists because a second model arrived, not in anticipation of one.** Jarrow–Yildirim was the
only model through tranches 4–11 and the pricers took it concretely; Market Model I is the second,
so rule-of-two fires now — the same timing as Black's extraction in tranche 9 and the Hull-White
core's in tranche 10.

**Both questions are quantities the SOURCE names, which is why one protocol fits two models rather
than bending one to the other.** Factoring BERHAN's two year-on-year leg formulas side by side makes
it plain — eq 2.5.12's later-period term and eq 3.1.11's are both

    ψᵢ · Pn(t,Tᵢ) · [ E − 1 ]

with `E` the forward ratio: for JY `E = (Pn(Tᵢ₋₁)/Pn(Tᵢ))·(Pr(Tᵢ)/Pr(Tᵢ₋₁))·e^C`, and for MM-I the
bivariate-lognormal integral of 3.1.11. Same shape, different `E`. The cap/floors agree too: JY's
2.6.5 and MM-I's §3.2 inner expectation are both Black on `(E, V²)`. So the seam is the *source's*
decomposition, not an abstraction invented to hold two things.

**Dates, not year fractions.** Tranche 7 lost a pointwise oracle to exactly that mismatch — "2.5
years" becomes 912 whole days, which is 2.4986 years on the curve's own clock. A date has one
spelling. The model reaches its own curves through `market`, so a pricer never passes market data
in alongside.

Provenance:
  stock:  — the quarry had one model per pricer and no such seam; built fresh from the two source
          formulas named above
  source: BERHAN-2012 (docs/references.md) eq 2.6.2 (`m`) and 2.6.4 (`V²`) as the two questions;
          eqs 2.5.12 and 3.1.11 factored side by side as the evidence that one shape holds both
  oracle: tests/L3/test_index_ratio_protocol.py — JarrowYildirim satisfies it at runtime; the two
          methods agree with the year-fraction forms they delegate to; and the tranche 4/5 oracles
          (unchanged) still pass through the refactored pricers
  crosscheck: none — a protocol declaration, no numerics
  slice:  tranche 12(a) — the model capability seam
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from pricebook.market.snapshot import MarketSnapshot


@runtime_checkable
class IndexRatioModel(Protocol):
    """A model that can price a year-on-year index ratio over a period.

    `runtime_checkable` so a test can assert a model satisfies it. Note what that does and does not
    buy: `isinstance` checks the *names* are present, never the signatures — tranche 5 learned that
    a runtime probe can report a protocol satisfied while pyright disagrees, and pyright was right.
    So the oracles check behaviour (the methods agree with the formulas they stand for), and the
    `isinstance` assertion is a cheap extra, not the evidence."""

    @property
    def market(self) -> MarketSnapshot:
        """The snapshot this model was calibrated to — the engine's only route to market data."""
        ...

    def expected_index_ratio(self, period_start: date, period_pay: date) -> float:
        """`m` — the `Tᵢ`-forward expectation of `I(Tᵢ)/I(Tᵢ₋₁)` (BERHAN eq 2.6.2)."""
        ...

    def index_ratio_variance(self, period_start: date, period_pay: date) -> float:
        """`V²` — the variance of `ln(I(Tᵢ)/I(Tᵢ₋₁))` under that measure (BERHAN eq 2.6.4)."""
        ...
