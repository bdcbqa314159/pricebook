# Cowork → Build ruling — CP-3 direction

Answers the CP-2c request. Grounded in `quarry_reconciliation.md` (drawdown 0/768).

## Ruling: CP-3 = cross-cutting-to-retire. Goal: drawdown moves off 0.

Breadth is **paused** while drawdown is stuck at 0 (CLAUDE.md §4). The map shows the
fixed-income vanillas (FRA, deposit, OIS, swap, bond) sit one-or-two residuals from deletable,
and **the residuals are shared** — so a few cross-cutting slices retire the whole cluster.

### The shared residuals to close (by retire-leverage)
1. **Serialisation** — `to_dict/from_dict` (the quarry `@serialisable` equivalent). Cross-cuts
   almost every module's deletability. Retire the simplest module with it first.
2. **Conventions + `RateIndex`** — deposit/FRA/OIS convention builders, SOFR/SONIA/ESTR
   (quarry `core/market_conventions` + `core/rate_index`). Closes a residual in *every* rate
   vanilla → retires **deposit** first.
3. **Multi-curve / projection curves** — `forward_rate(projection_curve)`, OIS-discount +
   projection. Closes the "single-curve only" residual → retires **FRA, swap, OIS**.
4. **Convenience surface** — `par_rate` / `annuity` / `dv01` / `pv_ctx`, patterned across the
   vanillas (a small shared mixin/protocol, not per-product copy-paste).

### The discipline (ratified, CLAUDE.md §4)
- **Every cross-cutting slice ends by ticking ≥1 quarry module to *deletable*** — no
  speculative infra ahead of the module that needs it. Deletable-bar rigor applies (read the
  quarry counterpart, empty residual).
- **CP-3 success criterion: drawdown > 0** — the first real quarry-module deletions. Report the
  new number in `quarry_reconciliation.md`.

### Suggested sequence (build confirms by "closest to deletable")
`serialisation` → `conventions/RateIndex` (retire **deposit**) → `multi-curve/projection`
(retire **FRA + swap + OIS**) → convenience surface (retire the rest of the vanilla cluster).
Target for the CP-3 checkpoint: **the fixed-income vanilla cluster deletable → drawdown ~5–8/768**.

Cadence unchanged (#11): ≤6 slices or cluster; each cross-cutting slice names the module(s) it
retires. Watch that cross-cutting doesn't sprawl — if a slice can't retire a module, it's too
big or premature; split it or defer it.
```
