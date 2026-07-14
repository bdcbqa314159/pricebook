# Handoff Report — §5.1 promotions, keyed registry, and the asset-class run (v0.9.0 → v0.18.0)

Date: 2026-07-14   ·   Version: `0.18.0`   ·   Tests: 163 green (`verify.py tests --layer 6`)

Third build → design report (per `redesign/08_handoff_protocol.md`). Covers everything after the
v0.3–v0.8 report: the §5.1 promotions, Amendments **A4/A5**, and the breadth run across FX,
equity, commodity, and inflation. §5 carries the open items.

---

## 1. Slices landed

| Slice | Version | Layer(s) | Oracle |
|---|---|---|---|
| survival-in-snapshot | 0.9.0 | L1/L3/L5 | credit01 via `Priceable`; PV byte-identical |
| rename Pricable→Priceable | 0.9.1 | all | pure rename; oracles green |
| fx-forward | 0.10.0 | L2/L3/L4 | par forward → 0 (CIP) |
| fx-in-snapshot | 0.11.0 | L1/L5 | FX data in snapshot; `fx_delta` analytic |
| fx-option | 0.12.0 | L2/L4 | GK; put-call parity ties to fwd |
| fx-vega | 0.13.0 | L5 | `fx_vega` = analytic Black vega |
| equity-option | 0.14.0 | L2/L4 | BS; shared `black_76`; parity |
| equity-greeks | 0.15.0 | L5 | `equity_delta`/`vega` analytic |
| **market-snapshot-keyed (A5)** | 0.16.0 | L1/L4/L5 | keyed registry; PVs+greeks byte-identical; namespacing |
| commodity-option | 0.17.0 | L2/L4 | BS on carry; **greeks free**; shared spot-option engine |
| inflation-zcis | 0.18.0 | L2/L4 | par ZCIS → 0 (Fisher) |

Plus the ratified-docs commits for A4 and A5.

## 2. Ledger deltas (headline new-tree entries)
- `market/keys.py` (`AssetClass`, `MarketKey`) — the keyed-registry namespace (A5).
- `MarketSnapshot` reshaped to `curves`/`spots`/`vols` maps (A5); per-asset fields removed.
- `foundation/black.py` (`black_76`); `engine/spot_option.py` (shared BS-on-carry engine).
- Products: `fx_forward`, `fx_option`, `equity_option`, `commodity_option`,
  `inflation.ZeroCouponInflationSwap`; `cds` gained `issuer` (multi-issuer, A5).
- Risk collapsed to generic `spot_delta`/`vol_vega` + `bump_spot`/`bump_vol` (A5); the per-asset
  `fx_*`/`equity_*` greeks deleted.
- `SurvivalCurve` gained a `df` alias (a hazard curve is the credit-risky discount curve, A5).

## 3. Oracles
Closed-form self-consistency throughout: CIP / par-swap / par-CDS / par-ZCIS reprice to zero;
option engines vs put-call parity + independent Black; greeks vs analytic (delta/vega/dv01) or
independent FD (credit01). The A5 refactor reused every prior oracle unchanged (byte-identical).

## 4. Debt logged
- **None.** `verify.py debt` = 0. (Two would-be `# type: ignore`s were avoided with runtime
  asserts; no type checker runs in CI.)

## 5. Design drift / open items — for Cowork

1. **Curve greeks are still per-type — the next A5-style unification.** A5 made *spot/vol*
   greeks generic (`spot_delta`/`vol_vega` keyed by `MarketKey`). **Curve** greeks did not
   follow: `dv01` bumps only the home `discount_curve` (and assumes it's *flat*), `credit01`
   bumps a survival curve by key. Consequence — **no greek exists for**: the FX foreign curve,
   the equity dividend curve, the inflation **breakeven/real** curve, or the commodity carry
   curve; and there's no pillar-wise bump for a *bootstrapped* curve. **Proposal:** a generic
   `curve01(priceable, snapshot, key, numerics)` + `bump_curve(snapshot, key, shift)` that
   parallel-shifts the curve at `key` (home discount = special case; survival = the existing
   hazard bump). This is the curve analogue of what A5 did for spots/vols. **Ask: ratify a
   generic keyed curve-greek, and decide flat-only-shift now vs pillar-wise (bootstrapped)
   bumps.**

2. **`dv01` assumes a flat curve.** `bump_rate` shifts `FlatDiscountCurve.rate`; a bootstrapped
   `DiscountCurve` has no `.rate`, so `dv01` would fail on it. Real rate risk on the
   bootstrapped curve (pillar bumps / key-rate durations) is unbuilt. Ties into #1.

3. **Breadth vs depth — a direction check.** We've done a long *breadth* run: 5 asset classes,
   all pricing + (spot/vol) greeks, on the keyed spine. **Unbuilt depth:** unified calibration
   front (L3 — we hand-bootstrap curves), XVA/RWA (the `Priceable` is ready for snapshot
   simulation), curve greeks (#1), the data spine (persistence + fuller L6 lifecycle). **Ask:
   keep adding asset classes/products, or pivot to depth?** My lean: one depth slice next
   (generic curve greeks, #1) since it's cheap and closes a real risk gap, then reassess.

4. **Spot-option engine boundary (FYI, not a question).** Equity + commodity options were
   factored onto a shared `price_spot_option` (Black-Scholes on a carry curve, rule of two). FX
   options stayed separate (genuinely different: two currencies, keyed by `Currency`). The
   boundary reads right; flagging only for visibility.

5. **A4.4 defaults still holding, no action:** no engine/model registry yet (no mixed trade);
   `PricingResult` still `pv`+`accrued` (no consumer for the full breakdown); demand-migrate the
   minimum (stdlib `random`, no numerical toolkit pulled this arc).

## 6. Quarry status
Unchanged migration mode. The L0 numerical toolkit remains almost entirely un-migrated (we
pulled only `norm_cdf`, `bisect_root`, `black_76`, stdlib `random`); statistics: 0. This arc was
all new-tree construction on the keyed spine.

## 7. Ready for next?
- Healthy: 163 green, no debt, keyed spine proven across 5 asset classes.
- Recommended next: **generic curve greeks (§5.1)** — a cheap, high-value depth slice that
  closes the FX-foreign / equity-div / inflation-breakeven / bootstrapped-`dv01` risk gap and
  finishes the A5 unification for curves. Then reassess breadth vs depth (§5.3).
- Blockers: none. Questions: §5 items 1–3.

---

### One-line return message (paste into Cowork)

> Forward run v0.9→v0.18 landed (§5.1 promotions; A4/A5 keyed registry; FX/equity/commodity/
> inflation asset classes; option engines share black_76; a new asset class now adds keys not
> fields, greeks free); 163 green, no debt. Drift: §5.1 curve greeks are still per-type (dv01
> flat-only; no FX-foreign / equity-div / inflation-breakeven / bootstrapped-dv01 greek) —
> ratify a generic keyed curve01/bump_curve? Plus breadth-vs-depth direction check. See
> redesign/handoffs/forward_v0.9-v0.18_report.md.
