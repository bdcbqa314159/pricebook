"""A strictly ascending grid of year fractions (L0) - one piece of knowledge, one home.

**Why this module exists, measured rather than asserted.** Re-audit #7's duplication pass found the
same guard written **byte-identically in four modules across three layers**:

    market/curve.py                        (L1, tranche 1)
    market/survival_curve.py               (L1, tranche 38)
    risk/credit_valuation_adjustment.py    (L5, tranche 25)
    risk/mc_exposure.py                    (L5, tranche 26)

each spelling `if any(b <= a for a, b in zip(self.times, self.times[1:])): raise ValueError(...)`
with the identical message. §6's rule-of-two had fired **four times over** on a shared *shape* - an
ascending grid of year fractions - and six prior duplication passes did not name it, because an
identical two-line block in four files reads as idiom rather than as duplication.

**A guard, not a `TimeGrid` type - the tranche-62 precedent.** What the four consumers share is one
piece of *knowledge*: a time grid is strictly ascending, because every one of them either bisects it
or zips it pairwise, so a grid that doubles back is not a grid. They do **not** share a field group:
none of the four classes is near the §7 field limit, so no bundle is *compelled*, and §6's
"compelled bundling is vocabulary" does not apply to a bundle nothing compels. A value object would
also change `self.times` to `self.grid.times` at every site - a public-contract change bought for
nothing.

**L0, and finance-free.** An ascending tuple of floats carries no strikes, vols, payoffs or
discounting; `foundation/schedule.py` already holds the *dates* analogue, so the vocabulary lives
here already. Below all four consumers (two L1, two L5), so dependencies still point down.

**Deliberately NOT shared with the four domain-specific ascending guards** - the coupon dates in
`schedule.py`, the quote-maturity ordering in the nominal bootstrap and in the surface builder, and
the fixing dates of an average-rate option. Those check different objects, carry their own
vocabulary in their own messages, and sit inside different surrounding conditions: merging them
would be §6's shared-word-not-shared-shape error, which is the mistake the rule exists to prevent.
*(Ruled at re-audit #7, F-7.2; the exclusion is Code's identification, affirmed by Cowork.)*

Provenance:
  stock:  none - the predecessor repeated the check per class, as this tree did until tranche 65
  source: none - a structural invariant, not a published formula
  oracle: tests/L0/test_time_grid.py - the refusal on a repeat and on a decrease, acceptance of a
          strictly ascending grid and of a single point, and the message asserted to be ours (the
          T44b reachability shape: a guard whose removal still raises elsewhere needs its message
          pinned); the four consumers' own oracles are the regression, and the extraction is proved
          pure by bit-identity in-process (§4/T49 - committed at tolerance, never as bit patterns)
  slice:  tranche 65 - the re-audit #7 remediation
"""

from __future__ import annotations

from collections.abc import Sequence


def checked_ascending_times(times: Sequence[float]) -> None:
    """Refuse a grid of year fractions that is not **strictly** ascending.

    Strict rather than non-decreasing, and the strictness is the content: a repeated time means two
    pillars at one instant, which makes interpolation ambiguous and a pairwise difference zero. Every
    consumer either bisects the grid or zips it pairwise, so both failures are silent rather than
    loud - which is why this is a construction-time refusal and not a runtime check.

    Raises rather than returning a bool, because every caller's only sensible response is to refuse,
    and a predicate invites one of them to carry on.
    """
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("times must be strictly ascending")

def checked_positive_horizon(years: float, subject: str) -> None:
    """Refuse a non-positive year fraction — one home for a guard written **23 times**.

    **Measured at re-audit #9 (F-9.3), and the count is the point.** The shape
    `if years <= 0.0: raise ValueError(f"... needs a positive horizon, got {years}")` appeared in
    **23 places across 16 modules and two layers**, in 14 distinct message forms — four spelling
    "the inversion", four "the fair variance", three "the simulation", two "the characteristic
    function", and ten one-offs. §6's rule-of-two had fired **eleven times over**. The first
    measurement of this finding said "4x byte-identical", which was one cluster mistaken for the
    family; deriving the population mechanically rather than reading a grep is what found the rest
    (F-9.2's practice, adopted in the same re-audit that needed it three times).

    `subject` is kept because it is the one part that is genuinely per-site: *which* quantity
    refused tells a caller where they are, and collapsing it to a generic message would trade real
    diagnostic value for uniformity. The **knowledge** — a horizon must be positive, and the
    exception type and phrasing it refuses with — is what is single-homed. Same
    `ValueError`, same message text, so every existing `pytest.raises(..., match="positive
    horizon")` still holds.

    Sibling of `checked_ascending_times` above, and here for the same reason (re-audit #7's F-7.2).
    """
    if years <= 0.0:
        raise ValueError(f"{subject} needs a positive horizon, got {years}")
