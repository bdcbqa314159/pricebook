"""The market snapshot (L1) — immutable market state, carrying its own valuation date.

**The valuation date lives here, not in a pricing call.** There is no clock read anywhere in the
library: "today" is `model.market.valuation_date`, which is what makes a reprice reproducible
months later and makes `price` a pure function of its arguments.

Provenance:
  stock:  _stock/src/pricebook/market/snapshot.py (purified: valuation_date + curves + fixing
          series. Its
          `scalars`/`surfaces`/`series` shapes and their key-unions grew one asset class at a
          time; each returns when a product pulls it)
  source: CLAUDE.md-era §2 invariant 2 (no ambient state) + doc 19 §2
  oracle: the snapshot is never mutated across a reprice (test_repricing_is_byte_identical)
  crosscheck: none — a frozen market-state value bag, no numerics
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from pricebook.foundation.inflation import InflationIndex
from pricebook.market.curve_set import CurveSet
from pricebook.market.commodity import CommodityAsset, CommodityUnderlying
from pricebook.market.equity import EquityAsset, EquityUnderlying
from pricebook.market.equity_skew import ImpliedVolSkew
from pricebook.market.fixing_series import FixingSeries


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the market said at `valuation_date`. Frozen; a bump produces a new snapshot."""

    # fields-exempt: the tree's one AGGREGATE — the gate's own named category ("a legitimately-wide
    # aggregate / output-record / config type"). It is a clock, the term structures, and one quote
    # store per KEYING DIMENSION: inflation prints by index, equity quotes and skews by ticker,
    # commodity quotes by symbol. They are not un-bundled vocabulary — nothing here is a loose
    # primitive that wants grouping — and the §6 precedent (`survivals`, `projections`) is that a
    # different keying dimension gets its own store rather than a `None` hung on an existing one.
    # Bundling them behind a `quotes` object is a real option (it would read `market.quotes.equities`
    # at 11 sites and change 26 construction sites across 12 test files), but it changes the DURABLE
    # PUBLIC CONTRACT §0 names, which is a design ruling and not tranche 60's to make. Raised to
    # Cowork at the T60 handoff; ledgered as OI-7 with that ruling as its removal trigger.

    valuation_date: date
    curves: CurveSet
    #: The published print history per index. **A series, not a resolved scalar, since tranche 7.**
    #: Tranche 2 carried `Mapping[InflationIndex, float]` — the already-resolved `I(t)` — which is
    #: enough for a zero-coupon swap at inception but cannot supply the *historical reset*
    #: `I(T_{ι(t)−1})` that a seasoned year-on-year period pays on (eq 2.5.12). The scalar is the
    #: degenerate one-point case of this, which is what makes the replacement provably faithful
    #: rather than a rewrite: see `test_a_one_point_series_reproduces_the_hand_fed_scalar`.
    #: *Which* print a date resolves to is `market.fixing_series.reference_level`, defined once and
    #: composed by every pricer — never re-derived per engine.
    fixings: Mapping[InflationIndex, FixingSeries] = field(default_factory=dict)
    #: The equity quote side per asset - spot, repo/borrow and dividends (tranche 30). Keyed like
    #: `fixings`: the snapshot is a mapping of what the market said, and an engine reaches it only
    #: through `model.market` (§3), so an equity option cannot be priced against another name's data.
    equities: Mapping[EquityAsset, EquityUnderlying] = field(default_factory=dict)
    #: The commodity quote side per asset - spot and convenience yield (tranche 60). Keyed like
    #: `equities`, and its own store rather than a widened one: a commodity underlying is a
    #: different *kind* of quote, not another equity. The precedent is `survivals` and
    #: `projections` - a different keying dimension gets its own store and its own accessor, never
    #: a `None` hung on the existing one (§6's carried-but-unread rule).
    commodities: Mapping[CommodityAsset, CommodityUnderlying] = field(default_factory=dict)
    #: The implied-volatility skew per asset (tranche 31) - the price source a model-free
    #: replication integrates across strikes. Market data, not a model: the variance swap's fair
    #: variance assumes no dynamics, only quoted prices.
    equity_skews: Mapping[EquityAsset, ImpliedVolSkew] = field(default_factory=dict)
