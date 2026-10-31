"""Inflation index identity (L0).

§3c's corollary: a new asset class adds **one** L0 object — its identity — and nothing else. Spot
levels, curves and vols are market state keyed *by* this identity; the contract terms are L2.

**No lag field, and that is a decision.** The stock's `inflation_indices.py` carries a single
`publication_lag_months`: 3 for most indices and **8 for UK index-linked gilts**. Those are two
different quantities conflated under one name — UK RPI is *published* about a month after its
reference month, and the 8 months is the **indexation lag the gilt's contract specifies**. Two
linkers on the same index can specify different indexation lags, so that one is contract data (L2)
and only the publication lag could ever live here.

Even the publication lag is absent in this tranche, because nothing reads it: the snapshot carries
the already-resolved reference level `I(t)`, and resolution *through* a lag arrives with the
inflation-linked bond that first needs it. A field with no consumer is the speculative field §6b
forbids, and L0 is the worst place to put one — it is the hardest layer to change later.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/inflation_indices.py (purified to the bare
          identity; the 8-index registry, interpolation enum, deflation-floor notes and the
          conflated lag field are left behind — see above)
  source: BERHAN-2012 (docs/references.md) §1.2, where the index `I(t)` is the FX-analogy spot
  oracle: satisfies value-keying (two `InflationIndex("HICPxt")` are one snapshot key)
  slice:  tranche 2 — the inflation spine
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InflationIndex:
    """An inflation index, by `name` (e.g. `HICPxt`, `USCPI`).

    Frozen and hashable so it keys the snapshot by VALUE — two `InflationIndex("HICPxt")` are one
    key, which is what a curve lookup and a fixing lookup both need."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("InflationIndex needs a non-empty name")

    def __str__(self) -> str:
        return self.name
