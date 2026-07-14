# Decision request → Cowork — the shape of `MarketSnapshot`'s market data

Date: 2026-07-14   ·   Version: `0.15.0`   ·   For ruling before the next asset class.

Not a drift report — a **forward design decision** A4 didn't cover. A4.2 ruled *what* lives in
the snapshot (all curves/spots/vols risk can bump). This asks *how they're shaped* inside it.

---

## The smell

After rates + credit + FX + equity, `MarketSnapshot` carries **10 fields**, 8 of them
per-asset market data:

```
valuation_date, discount_curve                        # core
fixings                                                # rates
survival_curve                                         # credit
fx_curves, fx_spots, fx_vols                           # FX      (keyed by Currency)
equity_spots, equity_div_curves, equity_vols           # equity  (keyed by ticker str)
```

Each new asset class edits this **L1 core type** and adds ~3 fields (commodity → `cmdty_*`,
inflation → `infl_*`, …). The risk layer mirrors the growth: `bump_fx_spot`/`bump_equity_spot`,
`bump_fx_vol`/`bump_equity_vol`, `fx_delta`/`equity_delta`, `fx_vega`/`equity_vega` are
near-identical, one pair per asset class. A 5th class multiplies both.

So: the snapshot is drifting toward a per-asset-aware god-object, and the greeks duplicate
per asset class. A4.2 is satisfied; the *internal shape* is the open question.

## Options

**A — Keep per-asset typed fields (status quo).**
- *Pros:* explicit, typed, discoverable (autocomplete shows `fx_spots`, `equity_vols`); zero
  indirection; each field self-documents.
- *Cons:* the snapshot grows per asset class (edits the L1 core each time); greeks duplicate
  per class; no uniformity; the "if risk bumps it, it's in the snapshot" rule scales by
  copy-paste.

**B — Fully generic bag** (`data: dict[str, Any]`). *Rejected* — too loose, untyped, no oracle
discipline. Listed only to dismiss.

**C — Generic maps keyed by a namespaced `MarketKey` (recommended).**
```python
@dataclass(frozen=True)
class MarketKey:
    asset: str   # "FX" | "EQUITY" | "CMDTY" | "CREDIT" | ...
    id: str      # "EUR" | "ACME" | "WTI" | issuer name

@dataclass(frozen=True)
class MarketSnapshot:
    valuation_date: date
    discount_curve: CurveHandle                          # home numeraire, kept special
    fixings: FixingHistory = ...
    curves: dict[MarketKey, CurveHandle] = {}            # foreign / dividend / survival / projection
    spots:  dict[MarketKey, float] = {}                  # FX + equity + commodity spots
    vols:   dict[MarketKey, float] = {}                  # flat vols
```
- FX EUR curve → `curves[MarketKey("FX","EUR")]`; equity div → `curves[("EQUITY","ACME")]`;
  survival → `curves[("CREDIT", issuer)]`.
- **Greeks collapse to one each**: `spot_delta(priceable, snap, key)`, `vol_vega(...)`,
  a single `bump_spot`/`bump_vol`/`bump_curve` — no per-asset variants.
- *Pros:* the snapshot **stops growing** (a new asset class adds keys, not fields); greeks
  de-duplicate; namespaced keys avoid `Currency "EUR"` vs ticker `"EUR"` collisions.
- *Cons:* less statically typed (a map keyed by `MarketKey`); the "FX spot" vs "equity spot"
  distinction lives in the key, not the type; a one-time refactor of the FX/equity/credit
  engines + greeks (behaviour-preserving, guarded by existing oracles).

## Recommendation & timing

**Adopt C**, and **do it now, before commodity.** Refactoring with 2 spot/vol asset classes
(FX, equity) is cheaper than with 3+, and it makes commodity/inflation/crypto free (no snapshot
edits, no new greeks). It also finally kills the `fx_*`/`equity_*` greek duplication the equity
slice just doubled.

**The A4.4 counter-argument (worth naming):** "no speculative generality — simplest thing until
the pain is undeniable" argues to *stay on A* until a 3rd asset class makes the duplication
concrete, then refactor. That's defensible; the cost is one more asset class's worth of
per-asset fields + greeks before the cleanup. My read: the pain is already concrete enough
(8 fields, 4 duplicated greek pairs), so pay it now.

## What I need ruled
1. **A, or C?** (B is off the table.)
2. If C: does `discount_curve` stay special (home numeraire) or also fold into `curves` under a
   `("HOME", ccy)` key? Does `survival_curve` fold into `curves` under `("CREDIT", issuer)`?
3. If C: is `MarketKey(asset: str, id: str)` the right key, or do you want typed enums for
   `asset`?
4. Timing: refactor **now** (its own behaviour-preserving slice) vs **after commodity**?

No code until ruled. If A, I proceed to commodity on the current shape.
