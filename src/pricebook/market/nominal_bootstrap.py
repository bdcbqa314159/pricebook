"""The nominal discount curve bootstrapped from par swap quotes (L1). BERHAN-2012 §5.1.

    S(Ti)   = (1 − P(0,Ti)) / Σ_{j=1..i} τ_j·P(0,Tj)                       (5.1.1, corrected)
    P(0,Ti) = (1 − S(Ti)·Σ_{j=1..i−1} τ_j·P(0,Tj)) / (1 + S(Ti)·τ_i)       (5.1.2, corrected)

**Both equations are misprinted in the thesis, and differently.** Read in `-raw` and `-layout`,
which *agree with each other* — so the dual-rendering rule (charter §4) does not catch these,
because the source itself is wrong rather than its rendering:

* **5.1.1 carries a spurious leading minus**, which would make every par rate negative;
* **5.1.2 sums `j = 0 .. i−1`** where inverting 5.1.1 requires `j = 1 .. i−1`. Since the thesis
  defines `P(0,T0) := 1` on the same line, the printed form adds a spurious unit to the annuity.

What caught both is the **requirement that the two be mutual inverses**: from a known flat 3% curve,
`j = 1..i−1` recovers it to the last digit (0.97044553) and the printed `j = 0..i−1` does not
(0.94089107). Transcription independence protects against a rendering; only a requirement oracle
protects against a typo.

**Closed form per pillar — no root-find, deliberately.** Given the pillars below `i`, 5.1.2 is an
exact rearrangement, so solving it numerically would be, in this repo's own words,
"solving a closed form" (`market/inflation_curve.py`). It is also the strongest possible version of
the negative-rate requirement: there is no bracket to fall outside, so a negative par rate yields
`df > 1` with nothing to configure. *(The spec proposed a sequential root-find on the
`foundation/solvers.py` seam; the closed form makes the seam unnecessary here.)*

**Why this is L1.** Quotes in, a `DiscountCurve` out: it imports L0 value types and L1's curve, and
touches no model and no engine — the same shape as `market/inflation_curve.py`, which bootstraps the
real curve from break-even quotes.

**Malformed input raises; it does not return a failure value.** Charter §3's "failure is a value"
governs `engine.price(product, model)`; a market-data constructor is a trust boundary, where the
house style is to fail fast and loud — and the established sibling, `real_curve_from_breakeven`,
raises `ValueError`. Two sibling constructors with opposite error conventions would be worse than
either choice. *(The spec asked for `PricingFailure`; raised as a finding rather than settled here.)*

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/ois.py — the sequential method and the par formula.
          Its solve is `brentq(objective, 1e-6, 3.0)` with the comment "wider bracket: handles
          negative rates", so the stock had already LEARNED this lesson rather than being a trap
          to avoid; the closed form supersedes the bracket entirely.
  source: BERHAN-2012 (docs/references.md) §5.1, eqs 5.1.1 and 5.1.2 — both corrected, above
  oracle: tests/L1/test_nominal_bootstrap.py — a known SLOPED curve round-trips through its own
          par rates pointwise to 1e-12 (non-telescoping: the test owns the forward map); both
          misprints are asserted to be wrong; later quotes must move later pillars and not earlier
          ones; negative par rates give df > 1; par-reprice kept only as a consistency guard
  crosscheck: quantlib PiecewiseYieldCurve / par-swap bootstrap — deferred to the optional QuantLib
          tier
  slice:  tranche 8 — the nominal curve bootstrap
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.curve import DiscountCurve


@dataclass(frozen=True)
class ParSwapQuote:
    """A par swap quote: the swap's `schedule` and the fixed rate that makes it worth zero.

    Two fields and no notional, because a par rate is notional-independent — carrying one would be
    a field the bootstrap never reads."""

    schedule: CouponSchedule
    par_rate: float


def swap_annuity(curve: DiscountCurve, schedule: CouponSchedule) -> float:
    """`Σ τ_j·P(0,T_j)` — the fixed leg's value per unit of rate.

    **Defined once so the bootstrap and any future swap pricer compose the same annuity** (§3d).
    `τ` comes from the schedule's own day count, never a hardcoded 1.0, so an annual-roll
    assumption cannot slip in silently."""
    return sum(psi * curve.df(pay) for _, pay, psi in schedule.periods())


def _validate(quotes: Sequence[ParSwapQuote], measure: TimeMeasure) -> None:
    """Every precondition the thesis leaves implicit, checked rather than assumed."""
    if not quotes:
        raise ValueError("a nominal curve needs at least one par swap quote")

    previous: tuple = ()
    last_maturity = None
    for quote in quotes:
        dates = quote.schedule.dates
        if dates[0] != measure.anchor:
            raise ValueError(
                f"a par swap schedule must start at the curve anchor {measure.anchor}, "
                f"got {dates[0]}"
            )
        maturity = dates[-1]
        if last_maturity is not None and maturity <= last_maturity:
            raise ValueError(
                f"par swap quotes must be in strictly ascending maturity order, "
                f"got {maturity} after {last_maturity}"
            )
        # The thesis's "swaps roll on a yearly basis" is a PRECONDITION: each quote's payment dates
        # must extend the previous quote's, or the annuity needs a pillar that was never solved.
        if previous and dates[: len(previous)] != previous:
            raise ValueError(
                f"each par swap schedule must extend the previous one; the quote maturing "
                f"{maturity} does not share the earlier payment dates"
            )
        previous = dates
        last_maturity = maturity


def nominal_curve_from_par_swaps(
    measure: TimeMeasure, quotes: Sequence[ParSwapQuote]
) -> DiscountCurve:
    """The discount curve implied by `quotes`, by 5.1.2 applied pillar by pillar.

    Returns an ordinary `DiscountCurve`, so every consumer reads it through `df` and there is no
    bootstrap-specific curve type and no second discounting path."""
    _validate(quotes, measure)

    times: list[float] = [0.0]
    rates: list[float] = []
    discount: dict = {}

    for quote in quotes:
        periods = list(quote.schedule.periods())
        *earlier, (_, maturity, psi_last) = periods
        # `Σ_{j=1..i−1}`: the already-solved pillars, which `_validate` guarantees are present.
        annuity = sum(psi * discount[pay] for _, pay, psi in earlier)
        df = (1.0 - quote.par_rate * annuity) / (1.0 + quote.par_rate * psi_last)
        if df <= 0.0:
            raise ValueError(
                f"the quote maturing {maturity} at {quote.par_rate} implies a non-positive "
                f"discount factor ({df}); the curve is unsolvable there"
            )
        discount[maturity] = df
        t = measure.year_fraction(maturity)
        times.append(t)
        rates.append(-math.log(df) / t)

    # t=0 is required of every curve, and the shortest pillar's zero rate extends back to the
    # anchor -- flat extrapolation at the short end, which a one-quote curve must do anyway and
    # which `real_curve_from_breakeven` already does for the same reason.
    return DiscountCurve(measure, tuple(times), (rates[0], *rates))
