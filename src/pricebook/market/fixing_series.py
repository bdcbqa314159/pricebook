"""The index fixing series and lag resolution (L1). BERHAN-2012 §1.1.2, §2.5.

An inflation index is **published monthly, in arrears, for a reference month**. Pricing needs a
level for a *date*, so something has to turn one into the other, and that something is a
convention with two parts — both properties of the index (L0):

1. the **publication lag**: the reference month is `date`'s month, less `publication_lag` months;
2. the **interpolation**: `FLAT` takes that month's print unchanged (UK index-linked gilts);
   `LINEAR` interpolates to the following month's print by `(day − 1) / days_in_month` (US TIPS and
   most markets).

**Resolution is defined once, here, and every pricer composes it.** Three engines need `I(t)` —
the ZCIIS, the linker and the year-on-year swap — and if each resolved its own, they could disagree
about which print `I(t)` is while every within-product oracle stayed green. That is the shape of a
cross-module convention bug, so it gets a shared atom and a register entry rather than three
implementations that happen to match today.

**Why this is L1 and not L0, where the tranche spec put it.** Resolution reads a `FixingSeries`,
and a series of published market observations is market data — the same kind of object as a
`DiscountCurve`, which lives here. Putting the atom at L0 while the series it reads is at L1 would
be an upward import and `acyclic` would reject it; putting *both* at L0 would make the foundation
hold market state. The precedent followed is `market/inflation_curve.py`: an L1 function over L0
value types.

**Refusal over extrapolation.** A date whose reference month has no print raises, and `LINEAR`
raises when the *following* month is missing rather than falling back to flat — a flat fallback
would return a plausible number computed under the wrong convention, which is indistinguishable
from a correct one at the call site.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/inflation_indices.py — mined for the CONVENTION
          FACTS only (the FLAT/LINEAR split and which markets use which; 3 months for most indices,
          8 for UK gilts). Its `InflationIndexDef` conflates publication and indexation lag in one
          field (the tranche 2 diagnosis) and it declares the interpolation enum without ever
          implementing resolution, so the series type and this atom are built fresh.
  source: BERHAN-2012 §1.1.2 (the index is a monthly print), §1.2 (`I(t)` as the FX-analogy spot),
          §2.5 eq 2.5.12 (where the published reset `I(T_{ι(t)−1})` is read)
  crosscheck: quantlib ZeroInflationIndex / CPI::laggedFixing — deferred to the optional QuantLib
          tier. The strongest target in this tranche: the FLAT/LINEAR split and the
          (day−1)/days_in_month weight are market conventions QuantLib implements directly, so a
          second engine would check the convention rather than only our arithmetic.
  oracle: tests/L1/test_fixing_series.py — the lag pointwise with the test computing its own
          reference month; LINEAR against a hand-computed weight; FLAT and LINEAR required to
          DISAGREE on a sloped series mid-month, and shown to agree on the 1st of a month and on a
          flat series (so the fixture's power is asserted, not assumed); refusal on a missing print
  slice:  tranche 7 — the index series and seasoned pricing
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from pricebook.foundation.inflation import IndexInterpolation, InflationIndex


@dataclass(frozen=True)
class FixingSeries:
    """Published index levels, keyed by **reference month** (the first of that month).

    Keyed by the month rather than the publication date because that is what the print *is* — "the
    index for March 2026" — and because a month has one canonical spelling here, so two callers
    cannot key the same print two ways."""

    prints: Mapping[date, float]

    def __post_init__(self) -> None:
        if not self.prints:
            raise ValueError("a fixing series needs at least one print")
        for month, level in self.prints.items():
            if month.day != 1:
                raise ValueError(
                    f"a reference month must be the first of the month, got {month}"
                )
            if level <= 0.0:
                raise ValueError(f"an index print must be positive, got {level} at {month}")

    def level(self, reference_month: date) -> float:
        """The print for `reference_month`, or `KeyError` — never an extrapolation."""
        try:
            return self.prints[reference_month]
        except KeyError:
            raise KeyError(
                f"no index print for reference month {reference_month}; "
                f"refusing to extrapolate the series"
            ) from None


def _shift_months(d: date, months: int) -> date:
    """The first of the month `months` before `d`'s month. Written out rather than inlined as
    `month - lag`, because month arithmetic across a year boundary is where an off-by-one lives
    and the oracle enumerates those cases against this."""
    total = (d.year * 12 + d.month - 1) - months
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)


def reference_level(series: FixingSeries, reference_date: date, index: InflationIndex) -> float:
    """`I(reference_date)` — the index level that applies on a date, by `index`'s own conventions.

    Takes the whole `index` rather than a loose `(lag, interpolation)` pair so a caller cannot
    pair one index's lag with another's convention."""
    month = _shift_months(reference_date, index.publication_lag)
    base = series.level(month)
    if index.interpolation is IndexInterpolation.FLAT:
        return base

    # LINEAR: weight towards the NEXT reference month by position within `reference_date`'s month.
    # At day 1 the weight is exactly 0 and this equals FLAT — asserted in the oracle, because a
    # fixture that only ever reads the 1st cannot tell the two conventions apart.
    days_in_month = calendar.monthrange(reference_date.year, reference_date.month)[1]
    weight = (reference_date.day - 1) / days_in_month
    nxt = series.level(_shift_months(reference_date, index.publication_lag - 1))
    return base + weight * (nxt - base)
