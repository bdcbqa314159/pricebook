# The volatility-surface arc — project six (`GATHERAL-2006`)

**Proposed by Cowork, committed by Bernardo 2026-11-16 ("let's work with Gatheral and see how we can
go" → the full vol-surface capstone).** The sixth vertical: the implied-volatility surface as a
first-class object — its arbitrage-free parameterization, its asymptotics, the derivatives written on
it, and the model that generates it. Like the five closed arcs: build the oracle-able spine, ledger the
rest, close with a checkpoint.

## The thesis — a capstone of the vol complex, not a greenfield
The equity-and-hybrids arc (T30–T36) already built most of Gatheral's *model* layer: local volatility
(Dupire, T32), the Heston model by Fourier inversion (T33), the variance swap by static replication
(T31), SABR (`models/sabr_model.py`), and the implied-vol surface (T9/T32; `market/vol_surface.py` +
`models/implied_vol_surface.py`). Chapters 1–2 are a re-tread for us and are **not rebuilt**. So this arc
spends its new effort on what Gatheral adds *on top* of the models we hold, and the reuse is deep enough
that it is a **capstone of the vol complex**:

| Gatheral | we already hold | the arc adds |
|---|---|---|
| ch 1–2 local vol, Heston | T32, T33 | — (held; the base the new objects sit on) |
| ch 3 **SVI** parameterization (eq 3.20) | the implied-vol surface (T9/T32), total variance (T31) | the arbitrage-free surface fit |
| ch 7 **surface asymptotics** (Lee) | Heston (T33), SABR | the model-independent wing/short-time limits |
| ch 11 **volatility derivatives** | the variance swap (T31) | the vol swap's convexity, options on variance |
| ch 5 **SV + jumps / Bates** | the Heston characteristic function (T33) | the jump term, reducing to Heston at λ→0 |

Method-fit is strong: SVI's no-arbitrage conditions (calendar monotonicity, no butterfly, Lee's wing
bound) are **requirement oracles**; the asymptotics are **closed-form reductions** cross-checkable against
Heston and SABR; the vol swap is a **convexity (Jensen) oracle**; Bates is anchored by an **exact reduction**
to Heston. Two independent sources now cover Heston/local-vol/variance (Gatheral vs `BROCKHAUS-2016`), so a
cross-source check is available where they render the same object (§4's same-formula caveat applies).

## The spine (phased; each unit ruled at its own spec)

### Phase 1 — the surface
- **T46 — the SVI surface parameterization + no-arbitrage conditions (ch 3, eq 3.20).** The crown jewel:
  `σ²_BS(k) = a + b[ρ(k−m) + √((k−m)² + σ²)]` per expiration, fitted to a smile, with the arbitrage-free
  constraints. **Oracle:** the `b→0` reduction to a flat variance `a` (exact); the two linear **wings**
  `a + b(ρ±1)(k−m)` with **Lee's bound** (each wing slope ≤ 2) as a requirement oracle; **calendar**
  monotonicity (total variance non-decreasing in `T`); **no butterfly** (the implied density `g(k) ≥ 0`);
  and a **Heston↔SVI cross-check** (ch 3 fits SVI to a Heston surface — a second-source agreement where
  they render the same smile). **Layer ruling at build:** L3 model-free parameter calibration (SVI yields
  parameters evaluated directly, like `models/implied_vol_surface.py`) vs L1 surface-bootstrap
  (`market/vol_surface.py`) — decided by what it produces and reads (§2). **Source-boundary Ask:** the
  rigorous butterfly conditions are Gatheral–Jacquier (2014); rule at build whether the density-`g(k)≥0`
  requirement oracle suffices or that paper must be obtained (a `to_go_deeper` acquire-source note if so).
- **~~T47 — volatility surface asymptotics (ch 7)~~ — DROPPED by the §1 pull gate (ruled 2026-11-16, after
  T46 merged).** T46 took the one consumed part (Lee's wing bound `β*,α* ∈ [0,2]`, which pinned eq 3.20's
  radical). The quantitative remainder — `g(q*)`, the `β* ↔ q*` inversion, the Benaim–Friz refinement
  (eqs 7.13–7.14), the Drăgulescu–Yakovenko/Heston tail-genericity (p.99) — has **no consumer** in the
  committed spine: T48's vol-swap convexity is a Heston computation (ch 11) reusing T31+T33, and the Jensen
  bound is model-independent; neither needs it. So it is **`to_go_deeper`** (oracle-able, unpulled), pulled
  only if a tail-sensitive product (a far-wing extrapolation, a tail-sensitive replication) ever needs it —
  and then built *where consumed*, per Code's 0272 observation. This is the pull gate doing its job: T47 was
  flagged provisional precisely for this test, and it fails it cleanly.

### Phase 2 — the derivatives and the model extension
- **T48 — volatility derivatives (ch 11).** The **volatility swap** (`E[√(realized var)]`) and **options
  on variance**, beyond the variance swap T31 already prices. **Oracle:** the **Jensen convexity bound**
  (vol swap ≤ √(var swap), equality iff zero vol-of-vol — an exact reduction); the Heston **convexity
  adjustment** closed form (ch 11's figures); reuses the T31 static-replication variance whole (a
  cross-product anchor).
- **T49 — stochastic volatility + jumps / Bates (ch 5).** The jump term added to Heston's characteristic
  function. **Oracle:** the **exact reduction to Heston** at zero jump intensity (`λ→0`); put-call parity;
  the characteristic-function transcription (two renderings, §4); a companion exercising the non-degenerate
  jump (the reduction-anchor blind spot, T39 — the companion must move `λ`).
- **Checkpoint** `CP_gatheral_vol.md` closes the arc; its close fires **re-audit #5** (§9).

## The honest completion ratio (proposed)
Built as the spine: SVI + its no-arbitrage conditions, the surface asymptotics, the volatility
derivatives, and Bates. Deferred to `to_go_deeper` (oracle-able, unpulled): **Heston-Nandi** (ch 4, a
GARCH closed form), **default risk / Merton jump-to-ruin** (ch 6 — partly a re-tread of T35's defaultable
equity), **skew dynamics** (ch 8), **barrier options** (ch 9) and **exotic cliquets** (ch 10 — MC-heavy,
the equity exotic zoo already parked). **`out_of_method` candidate:** none expected — Gatheral is
closed-form and asymptotic country; flagged only if a surface-fitting objective turns out to have no
requirement-identity anchor, ruled at the §1 gate.

## Decision — the FULL vol-surface capstone is committed (Bernardo, 2026-11-16)
Bernardo chose the **full capstone** over a focused SVI-only start. **The queue:** T46 (SVI surface,
**merged v0.53.0**) → ~~T47~~ (dropped by the §1 pull gate → `to_go_deeper`) → T48 (volatility derivatives) →
T49 (Bates / SV+jumps) →
`CP_gatheral_vol.md` (closes the arc, fires re-audit #5). Cowork writes each per-tranche spec before its
build (§10); this is the queue, not the specs. **T46 is specced** (`docs/tranches/46_svi_surface.md`); the
rest follow in turn. `GATHERAL-2006` is registered `in-hand` in `docs/references.md` (flips to
`worked-from` at T46).
