# Cowork → Build ruling (Amendment A5) — MarketSnapshot shape

Answers `redesign/handoffs/decision_market_data_shape.md`. Ratified in
`redesign/02_spine.md` Amendment A5 + `CLAUDE.md` §3. **Option C, now, before commodity.**

## Rulings (the 4 questions)
1. **C** (keyed registry), **now** — the second consumer (equity) has arrived, so §6b's
   rule-of-two justifies it; this is the rule on schedule, not speculative.
2. **`discount_curve` stays special** (home numeraire). **`survival_curve`, `fx_*`, `equity_*`,
   foreign-discount all fold** into the keyed maps. (Folding survival adds multi-issuer.)
3. **`MarketKey(asset: AssetClass, id: str)`** — `AssetClass` enum (typed, exhaustive) + open
   string id.
4. **Now**, its own behaviour-preserving slice.

## The slice

```
BRANCH slice/market-snapshot-keyed   (Amendment A5)

1. foundation/market: add AssetClass(Enum) and MarketKey(asset, id) (frozen).
2. Reshape MarketSnapshot:
     valuation_date; discount_curve (home numeraire, KEPT); fixings;
     curves: dict[MarketKey, CurveHandle]; spots: dict[MarketKey, float]; vols: dict[MarketKey,float]
   Fold survival_curve -> curves[(CREDIT, issuer)] (now multi-issuer);
        fx_curves -> curves[(FX, ccy)]; fx_spots/fx_vols -> spots/vols[(FX, ccy)];
        equity_div_curves -> curves[(EQUITY, ticker)]; equity_spots/equity_vols -> spots/vols[(EQUITY, ticker)].
   Remove the per-asset fields.
3. Collapse the greeks to ONE generic each on the Priceable path:
     bump_spot(snap, key, dh) / bump_vol(snap, key, dh) / bump_curve(...);
     spot_delta(priceable, snap, key) / vol_vega(...).
   Delete fx_delta/equity_delta, fx_vega/equity_vega, bump_fx_*/bump_equity_* duplicates.
4. Update every consumer: fx_forward/fx_option/equity_option engines, credit (survival lookup),
   swap/discounting. Behaviour-preserving.
5. Oracle: PVs AND greeks byte-identical to pre-refactor for every existing FX/equity/credit/rates
   test (reuse them). Add a MarketKey namespacing test (FX "EUR" and an equity "EUR" ticker do not
   collide). verify.py all + ruff (max-args 5) green both OSes; docs/provenance; CHANGELOG (Changed);
   version MINOR; rebase-and-merge.
```

After this, **commodity/inflation/crypto add keys, not fields, and no new greeks** — the whole
point of the refactor. Then resume forward on the corrected shape; report drift as usual.
```
