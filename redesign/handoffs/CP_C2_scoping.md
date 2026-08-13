# C2 opening — scoping pass (models & calibration)

**Status:** scoping (pre-ratification). Build starts once Cowork rules the eight decisions below.
**Position:** v0.92.2 · cluster C1 CLOSED + audit-hardened (slice 6d) · drawdown 19/793 · full suite green.
**Reads:** `redesign/22_model_calibrator_foundation.md` (F2), `redesign/12_domain_build_order.md` (B4/B5),
`redesign/19` (snapshot shapes), CLAUDE.md §1 (`black.py`-at-L0 precedent), §2 (invariant 5).

---

## Where doc 22 (F2) leaves us — ratified foundation

Rev 3 ratified the *type contract*: `CalibratedModel` (frozen, carries `model.market`, opt-in capability
protocols), the **(A) "name-the-limitation" fork** (calibrate by repricing targets through the model's own
*closed-form* capabilities; defer `StateProcess`+`Payoff` until the first numerically-priced target), and
`calibrate(spec) → (model, result)`. C1 shipped only the degenerate case: `DiscountingModel` is the sole
`CalibratedModel`, `Discounting` the sole capability.

**C2's job is to put the foundation under load.** Everything in doc 22 was validated by exactly one
implementer. C2 introduces the **second capability and the second model** — the first real rule-of-two test
of the capability model itself.

## What the first C2 slice must establish

1. **A second capability protocol** beyond `Discounting` — a `Volatility`/Black capability. Per Q3′(a) it
   must ship *with its semantic contract* (vol law, forward, measure/numeraire), or it degrades to "shared
   signature + per-model semantics = `isinstance` with extra steps."
2. **A second `CalibratedModel`** — a `BlackModel` carrying a vol surface. Proves rule-of-two on `CalibratedModel`.
3. **The `surfaces` snapshot shape** — first consumer (doc 19 closed-shapes × open-keys; `scalars` was the
   first non-curve shape in 6a, `surfaces` is next).
4. **Black-76 analytic at L3** — the `black.py`-at-L0 precedent (CLAUDE.md §1) says a dynamics' closed-form
   block lives at L3, never L0.
5. **First non-linear product (L2)** — caplet or swaption.
6. **numerical-config (invariant 5)** — Black is closed-form, so the *engine* numerical-config still has no
   consumer; confirm it stays deferred, and reconcile naming (doc 22 sketch `numerics: SolverConfig` vs the
   shipped `solve: SolveConfig`).
7. **(A) fork** — a Black European caplet/swaption is analytically priced → fits fork (A); Bermudan stays the
   named re-open trigger.
8. **§3d continuity** — the C1 annuity `rpv01` is the swaption's numeraire; reuse, don't re-roll.

## Recommended first slice — Black caplet, vol read (not solved)

The leanest cut that proves the capability model. A `BlackModel` *built, not solved* (reads vol from the
surface, exactly as `DiscountingModel` reads curves) — defers vol **calibration** to a later slice, so slice 1
is pure closed-form.

Skeleton (new code):
- `market/snapshot.py` — add the `surfaces` shape + `SurfaceKey` (keyed by index/underlying; axes expiry × strike).
- `market/vol_surface.py` (L1) — a `VolSurface` handle + `BlackVol` protocol (`vol(expiry, strike) → float`), mirroring `CurveHandle`.
- `models/black_model.py` (L3) — `BlackModel(market)` satisfying `CalibratedModel` + the new `Volatility` capability, **with its semantic contract in the protocol**.
- `models/black.py` (L3) — the Black-76 closed form (**L3, not L0**).
- `products/option.py` (L2) — `Caplet` (≤5 fields).
- `engine/vanilla_option.py` (L4) — `price_caplet`, registered; composes `df` (Discounting) × Black(vol), no numerics.
- `tests_ng/L4/test_caplet.py`.

Oracle (closed-form, strong): Black-76 caplet vs hand-computed reference + **put-call parity**
(caplet − floorlet = `df·τ·(forward − strike)`) to ~1e-12.

Then: C2-2 swaption (Black, annuity numeraire); C2-3 vol calibration/stripping (first real `SolveConfig`
consumer beyond curves); C2-4+ Hull-White / SABR (more capabilities; the engine numerical-config returns
when MC/PDE lands, far later).

## Open decisions for Cowork

| # | Decision | Recommendation |
|---|---|---|
| D1 | First model | **Black-76** (build order B4) |
| D2 | First product | **Caplet** first (leanest), swaption next |
| D3 | Vol capability semantic contract (Q3′a) | **Lognormal** forward-rate vol under T-forward measure; normal/Bachelier deferred to its 2nd consumer |
| D4 | `surfaces` shape + `SurfaceKey` | keyed by index/underlying; axes **expiry × strike** |
| D5 | Model built-not-solved for slice 1 | **Yes** — read vol; defer calibration |
| D6 | numerical-config (inv. 5) | **Stays deferred**; reconcile `solve` vs `numerics` naming |
| D7 | (A) fork covers Black European | **Confirm**; Bermudan = re-open trigger |
| D8 | Black analytic layer | **L3 models/**, never L0 |

---

## Appendix — Cowork prompt (as sent)

> **# Task: Design the C2 opening (first dynamics model). C1 is closed and audit-hardened (v0.92.2).**
>
> **Context.** Cluster C1 (linear-rates curve world) is closed: single→dual curve, cash, global solve,
> Hagan–West, FX/collateral/xccy — plus slice 6d, which fixed all 8 findings from a third-party audit
> (negative-rate calibration, HW extrapolation, bad-input guards, config-tolerance, HW caching). Full suite
> green, all `verify.py` gates green, drawdown 19/793. Doc 22 (F2) ratified the capability *foundation*, but it
> has only ever been exercised by one implementer (`DiscountingModel`/`Discounting`). **C2 introduces the
> second capability + second model — the first rule-of-two test of the capability model.**
>
> **Request.** I recommend the first C2 slice be a **Black-76 European caplet, vol read-not-solved** (a
> `BlackModel` built like `DiscountingModel` — closed-form, no calibration, no numerics), then swaption
> (annuity numeraire), then vol calibration, then HW/SABR. Please rule these eight decisions (recommendations
> in brackets) and emit the ratified first-slice spec:
> - **D1** first model [Black-76, per build-order B4]
> - **D2** first product [caplet first, swaption next]
> - **D3** the vol capability's **semantic contract** per Q3′(a) — vol law + forward + measure/numeraire [lognormal forward-rate vol, T-forward measure; normal/Bachelier deferred to a 2nd consumer]
> - **D4** the `surfaces` snapshot shape + `SurfaceKey` design [keyed by index/underlying; axes expiry × strike]
> - **D5** model *built-not-solved* for slice 1, deferring vol calibration [yes]
> - **D6** whether the engine numerical-config (invariant 5) returns now or stays deferred, and reconcile the doc-22 `numerics: SolverConfig` name against the shipped `solve: SolveConfig` [stays deferred; unify naming]
> - **D7** confirm the (A) fork covers a Black European target (analytically priced), Bermudan = re-open trigger [confirm]
> - **D8** Black analytic lives at **L3 models/**, not L0 (the `black.py` precedent) [confirm]
>
> Emit the ratified C2-slice-1 "# Task:" spec: layers touched, new types with field counts (≤5), the
> capability protocol **with its semantic contract**, the oracle (closed-form Black + put-call parity), and
> the deferred list with named triggers. **Flag anything in doc 22 that the second capability *breaks* rather
> than *extends additively*** (§0 claims it extends — verify against a real second capability).
