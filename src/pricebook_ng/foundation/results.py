"""Engine I/O value types — PricingResult and PricingFailure.

Spine invariant 4: failure is a value, never a raised exception or a silent
NaN. The engine returns one of these; callers branch on the type.

Provenance:
  quarry: ad-hoc pricing returns across python/pricebook/pricing/
  source: redesign/03_vocabulary.md (Engine I/O); redesign/02_spine.md (inv. 4)
  oracle: N/A (result value types; exercised by the Slice 0 oracle)
  slice:  S00
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.foundation.money import Money


@dataclass(frozen=True)
class PricingResult:
    """A successful valuation."""

    pv: Money


@dataclass(frozen=True)
class PricingFailure:
    """A valuation that could not be produced — carried as a value."""

    reason: str
