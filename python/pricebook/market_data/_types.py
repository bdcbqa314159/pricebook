"""Market data types — the canonical L1 layer.

`QuoteKind` — enum of recognised quote types.
`QuoteId` — stable hashable identifier for "the same instrument across time".
`Quote` — a single observation: an `id`, a `value`, optional `bid_ask_bp`.
`MarketSnapshot` — a frozen bundle of quotes with a UUID and an `as_of` timestamp.
`FixingHistory` — frozen wrapper around historical fixings.

Zero dependencies on other pricebook subpackages — empirically L0 per
`tools/test_layer.py` (no pricebook imports landed yet). The design
target is L1 per DESIGN.md §5.1 A2; promotion happens when G1 P2
integration wires curves to read snapshots through this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Mapping
from uuid import UUID, uuid4


class QuoteKind(str, Enum):
    """Type of raw market observation.

    str-Enum so values serialise as plain strings and round-trip cleanly
    through JSON / dict / database columns.
    """
    DEPOSIT_RATE = "deposit_rate"
    FRA_RATE = "fra_rate"
    FUTURE_RATE = "future_rate"
    SWAP_RATE = "swap_rate"
    BASIS_SWAP_SPREAD = "basis_swap_spread"
    XCCY_BASIS = "xccy_basis"
    CDS_SPREAD = "cds_spread"
    BOND_PRICE = "bond_price"
    BOND_YIELD = "bond_yield"
    VOL_POINT = "vol_point"
    SWAPTION_VOL = "swaption_vol"
    CAPFLOOR_VOL = "capfloor_vol"
    FX_SPOT = "fx_spot"
    FX_FORWARD = "fx_forward"
    INFLATION_YOY = "inflation_yoy"
    INFLATION_ZC = "inflation_zc"
    OTHER = "other"


@dataclass(frozen=True)
class QuoteId:
    """Stable, hashable identifier for a market quote.

    Two quotes are "the same instrument" iff they have the same QuoteId —
    even if observed at different times with different values. The id is
    used to look up a quote in a snapshot and to link a calibration's
    `quotes_fitted` back to the underlying observations.

    The composite key `(kind, tenor, currency, label)` is intentionally
    coarse — for finer granularity (specific bond ISIN, specific exchange,
    etc.), use the `label` field.
    """
    kind: QuoteKind
    tenor: str               # e.g. "3M", "5Y", "10Y", "1M-3M" (FRA), "100x80" (smile)
    currency: str = "USD"
    label: str = ""          # optional sub-label (issuer / exchange / ISIN / smile-strike / ...)

    def __str__(self) -> str:
        s = f"{self.kind.value}:{self.tenor}:{self.currency}"
        if self.label:
            s += f":{self.label}"
        return s


@dataclass(frozen=True)
class Quote:
    """A single raw market observation at a snapshot time.

    Carries its identifier (`QuoteId`), its value, and optionally a
    bid-ask half-spread in basis points. The same `QuoteId` observed at
    different snapshot times yields different `Quote`s embedded in
    different `MarketSnapshot`s.
    """
    id: QuoteId
    value: float
    bid_ask_bp: float | None = None   # half-spread in basis points


@dataclass(frozen=True)
class MarketSnapshot:
    """A frozen, dated, identifiable bundle of market quotes.

    The canonical L1 type for raw market observations. Curves are *built*
    from snapshots; `CalibrationResult.market_snapshot_id` points to the
    snapshot used. Two snapshots with the same quotes but different `id`s
    are *different snapshots* — the id is the audit primitive.

    Construct via `MarketSnapshot.new(...)` to auto-generate the id and
    timestamp. Use `with_quote(q)` to derive a new snapshot with one
    quote replaced — the new snapshot has a fresh id.
    """
    id: UUID
    as_of: datetime
    quotes: tuple[Quote, ...]
    label: str = ""

    @classmethod
    def new(
        cls,
        quotes: list[Quote] | tuple[Quote, ...] = (),
        as_of: datetime | None = None,
        label: str = "",
    ) -> "MarketSnapshot":
        """Factory: auto-generate `id`; default `as_of` = now."""
        return cls(
            id=uuid4(),
            as_of=as_of if as_of is not None else datetime.now(),
            quotes=tuple(quotes),
            label=label,
        )

    @classmethod
    def empty(cls, label: str = "") -> "MarketSnapshot":
        """Convenience: empty snapshot. Useful as a starting point."""
        return cls.new(quotes=(), label=label)

    def get(self, qid: QuoteId) -> Quote | None:
        """Return the quote with this id, or None if absent."""
        for q in self.quotes:
            if q.id == qid:
                return q
        return None

    def filter(
        self,
        kind: QuoteKind | None = None,
        currency: str | None = None,
        label: str | None = None,
    ) -> tuple[Quote, ...]:
        """Return quotes matching all provided filter fields."""
        out = []
        for q in self.quotes:
            if kind is not None and q.id.kind != kind:
                continue
            if currency is not None and q.id.currency != currency:
                continue
            if label is not None and q.id.label != label:
                continue
            out.append(q)
        return tuple(out)

    def with_quote(self, q: Quote) -> "MarketSnapshot":
        """Return a new snapshot with `q` added (or replacing the existing
        quote with the same id). The returned snapshot has a fresh id —
        it is a different snapshot."""
        new_quotes = tuple(qq for qq in self.quotes if qq.id != q.id) + (q,)
        return MarketSnapshot.new(
            quotes=new_quotes,
            as_of=self.as_of,   # preserve as_of by default
            label=self.label,
        )

    def __len__(self) -> int:
        return len(self.quotes)

    def __iter__(self):
        return iter(self.quotes)

    def __contains__(self, qid: object) -> bool:
        if not isinstance(qid, QuoteId):
            return False
        return self.get(qid) is not None


@dataclass(frozen=True)
class FixingHistory:
    """Historical fixings for benchmarks (SOFR, EURIBOR, CPI, ...).

    Frozen wrapper around a `{(rate_name, date) -> value}` map. Distinct
    from `MarketSnapshot` because fixings are *backward-looking* (already
    fixed; no bid-ask, no uncertainty) while quotes are *forward-looking*
    market observations.
    """
    fixings: Mapping[tuple[str, date], float] = field(default_factory=dict)

    def get(self, rate_name: str, on: date) -> float | None:
        """Return the fixing for `rate_name` on `on`, or None if absent."""
        return self.fixings.get((rate_name, on))

    def for_rate(self, rate_name: str) -> dict[date, float]:
        """All fixings for one rate, keyed by date."""
        return {
            d: v for (rn, d), v in self.fixings.items() if rn == rate_name
        }

    def rate_names(self) -> set[str]:
        """All rate names with at least one fixing recorded."""
        return {rn for (rn, _d) in self.fixings.keys()}

    def with_fixing(self, rate_name: str, on: date, value: float) -> "FixingHistory":
        """Return a new history with this fixing added or replaced."""
        new = dict(self.fixings)
        new[(rate_name, on)] = value
        return FixingHistory(fixings=new)
