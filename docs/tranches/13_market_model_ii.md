**Status:** BUILT (v0.16.0). Decided by Cowork, built by Code. **Results in §7.**

# Tranche 13 — Market Model II (ch 4), the third model, and the one with no real-rate vol

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.16.0`.
**Source:** `BERHAN-2012` ch 4 (`worked-from`) — read from the held PDF, verified it publishes the
formulas in full: the forward-index dynamics and their freezing (4.1.2–4.1.3), the fully analytical
YoY correction (4.1.6), and the cap/floor with `Vi²(t)` (4.2.1–4.2.2). **Chosen by Bernardo:** finish
the two market models, MM-II after MM-I. MM-II + its calibration §5.4 close the book's inflation arc
and enable the JY-vs-market-model comparison that is the thesis's actual purpose.

## 0. The crux — a THIRD model through the SAME seam, and the seam is already proven

T12(a) built `IndexRatioModel` and T12(b) put a second model through it with no change to either
pricer. MM-II is the **third** implementer: it prices the same YoY swap and cap/floor through the same
`engine/yoy.py` and `engine/capfloor.py`, by answering the two protocol questions (`m`, `V²`) with
ch 4's formulas. **No new seam, no dispatch change** — if anything in the pricers needs touching to
admit MM-II, that is an `Ask:` back to me, because the seam is supposed to already hold for N models.

**What is genuinely new about MM-II (and why it is worth building, not a near-duplicate of MM-I):**
MM-II models the **forward inflation indices** `I(·,Ti)` directly as martingales under `Q^{Ti}_n`,
rather than the nominal/real forward rates. Its selling point, stated by the source: **the correction
term does not depend on the volatility of real rates** — the parameter JY and MM-I both lean on and
that is hard to estimate. So MM-II has *no* `σr`. That is a structural property, and an oracle.

## 1. The formulas (read from ch 4, to be transcribed independently at build)

**YoY `m` — eq 4.1.6, fully analytical (no quadrature, unlike MM-I):**

```
m = (1 + τi·Fn(0,Ti−1,Ti)) / (1 + τi·Fr(0,Ti−1,Ti)) · e^{Di(0)}
```

with `Di(0)` the drift-freezing correction from 4.1.2–4.1.3 (the frozen drift of `I(t,Ti)/I(t,Ti−1)`
integrated), carrying `σn`, `σI` and `ρ_I,n` — **but not `σr`**. The base `(1+τFn)/(1+τFr)` is the
same ratio JY and MM-I reduce to, which is what makes the cross-model anchor hold.

**Cap/floor variance — eqs 4.2.1–4.2.2. The protocol's `V²` for MM-II is NOT `σI²·span`:**

```
V²_total = Vi²(t) + σI,i²·(Ti − Ti−1)
Vi²(t)   = (σ²I,i−1 + σ²I,i − 2·ρI,i·σI,i−1·σI,i)·(Ti−1 − t)
```

i.e. the variance accumulated over `[t,Ti−1]` from the two index-forward vols and their correlation
`ρI,i`, **plus** the period variance `σI²·(Ti−Ti−1)`. Reusing MM-I's plain `σI²·span` would silently
drop `Vi²(t)` — the degenerate-fixture / missing-term trap of §4; oracle 6 below exists for it.

## 2. Pieces and layers

```
L3  models/market_model_ii.py — MM-II: forward-index vols + {ρ_I,n, ρ_I,I}, the analytical YoY
                                correction Di, and the cap V² with Vi²(t). Implements IndexRatioModel.
                                NO σr field. A THIRD correlation type (MMIICorrelations, 2 fields).
```

The YoY is analytical, so **no new quadrature** is needed (MM-I's Gauss-Hermite stays MM-I's). If a
freezing recursion for σ_I between periods is needed as in MM-I, reuse the lesson, not necessarily the
code. §5.4 (MM-II's calibration) is a **separate later tranche** (T14) — do not fold it in.

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **cross-model zero-vol reduction — now FOUR paths:** at zero vol MM-II's `m`, MM-I's `m`, JY's `m` and the curve ratio `(1+τFn)/(1+τFr)` all agree (Di→0 at zero vol) | **exact, cross-model** | a wrong MM-II `m` diverges from a limit three other code paths share |
| 2 | **a SECOND exact reduction unique to MM-II:** the source states 4.1.6 is *exact* when `ρ_I,n = 0` (the nominal forward zeroes out of Di). So at `ρ_I,n = 0`, MM-II's `m` equals a closed form with Di's freezing-error term absent — an anchor that does NOT require zero vol | **exact** | the freezing-error term of Di being mis-sited (nonzero where it must vanish) |
| 3 | **companion (non-vacuity):** vols / `ρ_I,n` move MM-II's `m` away from both limits | structural | a no-op MM-II passes 1–2 alone |
| 4 | **cross-model disagreement:** MM-II ≠ MM-I ≠ JY away from the limits (three distinct models; silent agreement means one is not doing what it claims) | structural | two models accidentally coinciding |
| 5 | **the no-real-rate-vol property — MM-II's whole reason to exist:** MM-II has no `σr`, and its `m` and `V²` are invariant to the real *curve's* implied vol however it is perturbed (only the real *forward levels* enter, via `1+τFr`) | **structural, unique to MM-II** | a stray real-vol dependence smuggled in — the one thing MM-II must not have |
| 6 | **`V²` includes `Vi²(t)`, on the period not the expiry:** assert `V²` ≠ `σI²·span` whenever `ρI,i ≠ 1` or the two index vols differ, i.e. the `Vi²(t)` term is present and correctly signed (`−2ρI,i σI,i−1 σI,i`) | **1 — pointwise + requirement** | dropping `Vi²(t)` (the MM-I-shaped reuse), or its sign |
| 7 | **transcription of 4.1.6 and 4.2.1–4.2.2** `-raw`+`-layout` AND the derivation-step corroboration (4.1.2/4.1.3 → 4.1.6), parameter-swept; a misprint must survive both the final form and the derivation step | **1** | every sign and factor in Di and Vi² |
| 8 | **cap − floor = the MM-II YoY leg** (same model, different code each side) — a THIRD model carried through the T12(a) seam | structural | the sign/leg error |
| 9 | **the freezing approximation bound, stated not hidden:** 4.1.6 is exact only at `ρ_I,n = 0`; away from it the frozen drift is approximate and "rough for longer maturities" (source's words). Bound it as a SENSITIVITY — how far Di moves under a stated forward shock and how it grows with maturity — magnitude and trend asserted, never an arbitrary smallness threshold | 2 — sensitivity | a freezing error dressed as "close enough"; the T12(b) lesson applied |
| 10 | failure-is-a-value / raises per convention (invariant 6, missing curve) | structural | the contracts carried |

Oracle 1 is the anchor; oracle 5 is the one that pins MM-II's *identity* (no real-rate vol); oracle 6
is the missing-term guard; oracle 9 is the honestly-bounded approximation. **No number fitted to a
figure** (ch 4 has none; empirical comparison is T14/§5.4's territory).

## 4. Design decisions (ruled)

- **Third model, same protocol, no seam change.** MM-II implements `IndexRatioModel`; the pricers are
  untouched. A forced touch is an `Ask:`, not a workaround.
- **No `σr` field — it is MM-II's defining property, asserted by oracle 5.** A carried-but-unread real
  vol would be the §6 violation *and* a lie about what MM-II is.
- **A THIRD correlation type `MMIICorrelations` {ρ_I,n, ρ_I,I}**, two fields — not `JYCorrelations`,
  not `MMICorrelations`. Rule-of-two fires on shared *shape*; all three models' correlation shapes
  differ (JY: nr/rI/nI; MM-I: nr/rI; MM-II: In/II). This is the §6 pattern T12 §11.2 already ruled.
- **YoY is analytical (4.1.6), no quadrature.** Only the cap path, if it needs the between-period σ_I
  recursion, reuses MM-I's freezing *lesson* (refuse non-positive σ_I), not necessarily its code.
- **Cross-check disposition:** `models/market_model_ii.py` → `crosscheck: quantlib — none (QuantLib
  has no inflation forward-CPI market model)`; if a closed form admits an independent check (e.g. the
  analytical 4.1.6 against a direct lognormal expectation), put that on the other side of an oracle as
  T12(b) did with `scipy.integrate.quad`.
- **§5.4 is T14, separate.** This tranche is the model only; its calibration is the next unit.

## 5. Stock
Mine `_stock/.../options/inflation_*` for the forward-CPI market-model shape only; read every formula
from **BERHAN ch 4** (the held source, which publishes them in full), never from the stock's code.

## 6. Process
Branch `tranche/13-market-model-ii`. `test` (RED pre-registered) → `feat` → `docs` →
`chore(release): v0.16.0`. Mutation on Di and on `Vi²(t)` (especially the `−2ρ` cross term and the
`/(1+τF)` denominators — the T12(b) survivors). Anchors: the four-path zero-vol reduction and the
`ρ_I,n=0` exact reduction; the no-real-rate-vol invariant; the MC/closed-form check on 4.1.6.
`source:` cites `BERHAN-2012` ch 4. **Open as a DRAFT PR** (§10 step 3).


---

## 7. Result — the third model, and two hazards in the source

430 tests, eight gates green, ruff and pyright clean, zero suppressions. Five mutations on MM-II,
all caught.

### The seam carried a third model with no change to it
MM-II implements `IndexRatioModel`; both pricers are untouched, and tranche 5's
`cap − floor = the YoY leg` identity now holds under **three** models. The §0 question — whether a
third model would force a seam touch — is answered: it did not.

### The four-path zero-vol reduction
MM-II's `m`, MM-I's `m`, JY's `m` and the curve ratio `(1+τFn)/(1+τFr)` all agree at zero vol,
sharing nothing but the curves. Four independently-coded paths, one limit.

### And a second exact anchor no other model here has
The source: *"the formula above is exact only when the correlations between `I(·,Tᵢ₋₁)` and
`Fn(·,Tᵢ₋₁,Tᵢ)` are assumed to be zero so that the nominal forward rate is zeroed out from `Di`."*
So at `ρ_{I,n} = 0` the freezing is not used at all, and `Di` is a closed form with no forward rate
in it — an exact anchor that does **not** require zero volatility. Oracled from both sides: the
closed form matches, and `σ_n` is provably inert there.

## 8. Two hazards in the source, and how each was settled

### 8.1 Eq 4.1.3's denominator contradicts `Di`'s — in BOTH renderings
4.1.3 prints `τᵢσ_{n,i}Fn/(1 + Fn)`; `Di(t)` prints `.../(1 + τᵢFn)`. This is an **internal
inconsistency**, not a rendering artefact — so §4's newly-ratified derivation-step rule fired and
found a *discrepancy* rather than a confirmation, which is its first application and arguably its
more valuable mode.

`1 + τᵢFn` is taken as correct, on four grounds: it is the standard LMM coefficient; it is what
Market Model I's freezing relation uses consistently; `1 + Fn` adds a pure number to a rate; and
`Di` is the form used downstream in 4.1.6 and 4.2.4. The choice is pinned by an oracle rather than
left to the reader.

### 8.2 `Vi(t)` is defined twice, for two different quantities
Eq 4.2.2 gives `sqrt([…][Tᵢ₋₁−t])`; a line before 4.2.4 redefines the same symbol as
`sqrt(Vᵢ²(t) + σ_{I,i}²(Tᵢ−Tᵢ₋₁))`. The caplet needs the augmented one, and the reason is the
measure rather than the notation: 4.2.1 is the distribution of the **forward**-index ratio at
`Tᵢ₋₁`, while the payoff is on the **spot** ratio at `Tᵢ`, so the diffusion from `Tᵢ₋₁` to `Tᵢ` must
be added. Dropping it silently reuses MM-I's `σ_I²·span` shape and loses the whole lead term — the
missing-term trap, guarded by an oracle requiring `V² ≠ σ_I²·span` **and** requiring it to reduce to
exactly that at `ρ_{I,I} = 1`.

## 9. Three of my own test defects, all caught by execution

### 9.1 The unit-accrual trap, FIFTH appearance — and this time on a source-discrepancy oracle
The test pinning §8.1 was powerless: on annual ACT/365F periods `τ = 1.0` exactly, so `τFn` and `Fn`
are the same number and the two readings of the denominator are **indistinguishable**. Worse, that
meant the `τ` was unoracled *everywhere in the file*, not only in that test. Fixed by making the
whole fixture **semi-annual** (`τ ≈ 0.4932`).

Notable because it is the first time this trap hit an oracle about a *source discrepancy* rather
than about a formula — the fixture made two different readings of the source agree.

### 9.2 A contrast oracle that failed because the effect is weak, not absent
`test_MMI_DOES_depend_on_real_rate_vol` failed at `rel=1e-6` — and the dependence is genuinely
there, just small. Measured, sweeping `σr` from 5% to 50%:

| lead | `ρ_rI = −0.20` | `ρ_rI = −0.90` |
|---|---|---|
| 0.49y | 7.7e-09 | 1.1e-05 |
| 0.99y | 1.6e-06 | 2.4e-05 |
| 1.48y | 6.6e-06 | **3.7e-05** |

So MM-II's "no real-rate volatility" is a real **structural** simplification whose **pricing** impact
is modest at these parameters. That is worth stating rather than hiding behind a loosened tolerance,
and the oracle now asserts the **trend** (growing in `|ρ_rI|` and in the lead), which is stronger
than one inequality.

### 9.3 The no-real-vol property is proved structurally, not by inspection
`MMIIParams` has no real-vol field to set, asserted on `__dataclass_fields__` — so the property
cannot be violated by a later edit without the test failing. Its companion (§9.2) is what stops that
being vacuously true of a model with no parameters at all.

## 10. Findings for Cowork (§10 — raised, not decided)

**10.1** `MMIICorrelations` is the **third** correlation type (index-nominal, index-index). Three
models, three shapes — JY nr/rI/nI, MM-I nr/rI, MM-II In/II. Sharing would force two of the three to
carry fields they never read. A clean application of T12 §11.2's ruling, flagged for consistency.

**10.2** One index vol, so `σ_{I,i−1} = σ_{I,i}` and `Vᵢ²` collapses to `2σ_I²(1−ρ_{I,I})(Tᵢ₋₁−t)`.
A per-period index-vol term structure is what §5.4 fits, so it arrives with that consumer (T14)
rather than speculatively here. Flagged because it is a scope choice, not a transcription.

**10.3** No cross-check target: QuantLib has no inflation forward-CPI market model. Unlike T12(b),
where `scipy.integrate.quad` sat on the other side of the integral, MM-II is fully analytic — so the
independent side of its oracle is the hand transcription, which is why §4's pointwise rule matters
most here.

---

## 11. Cowork's ruling on T13 — PR #19 (§10 step 4, 2026-11-04)

**Reviewed against the source, not only the narrative.** Confirmed hazard 1 in the PDF directly: eq
4.1.3 prints `1 + Fn`, `Di(t)` prints `1 + τᵢFn` three lines on — a source-internal inconsistency in
both renderings, and `1 + τᵢFn` is unambiguously correct (`= Pn(Tᵢ₋₁)/Pn(Tᵢ)`, dimensionally sound,
the form 4.1.6 uses downstream). Both pins I set held: cap `V²` is `Vi²(t) + σI²·(Tᵢ−Tᵢ₋₁)` with an
exact reduction *to* `σI²·span` at `ρ_{I,I}=1`, and 4.1.6 is exact at `ρ_{I,n}=0` with `σn` provably
inert. Accepted — v0.16.0, the model shelf (JY, MM-I, MM-II) complete behind one seam.

**10.1 — accepted, no change.** Third correlation type is the T12 §11.2 ruling applied cleanly;
three models, three shapes, no shared unread fields.

**10.2 — accepted.** One index vol now, per-period term structure deferred to §5.4 (T14) as its
genuine consumer — not speculative, correct by §6. `Vᵢ²` collapsing to `2σ_I²(1−ρ_{I,I})(Tᵢ₋₁−t)` is
the right single-vol form.

**10.3 — accepted, and the honest position is worth stating plainly.** MM-II is fully analytic, so
there is no cheap *independent numerical* path at a general point: a direct expectation of the frozen
lognormal shares `Di` and would only reproduce 4.1.6 algebraically (vacuous), and the only
Di-independent check is the full LMM simulation this tranche correctly defers (its frozen-vs-unfrozen
gap is the sensitivity bound). So the general-case correctness genuinely rests on the pointwise
transcription (dual-rendering + derivation-step) — which is exactly why this tranche's two new §4
rules are load-bearing here, and why hazard 1 living in the general `Di` was caught by the
derivation-step discrepancy rather than by any number. `crosscheck: quantlib none — <reason>` is the
right marker (§8). No additional check demanded: I considered requiring an independent integration of
4.1.6 and ruled it out as telescoping.

**Both §4 enrichments charter-bound:** the derivation-step rule's discrepancy mode (it found hazard 1,
not a confirmation), and the unit-accrual trap's fifth appearance now defeating a *transcription*
oracle (τ=1 made `τFn` and `Fn` indistinguishable until a semi-annual fixture). 9.2's weak-effect
contrast (MM-II vs MM-I differ by σr only 7.7e-9 → 3.7e-5) accepted as stated, not hidden behind a
loosened tolerance — and it is a real result for T14 and the Substack piece: the three models differ
far more in identifiability and honesty than in price. **Build T14 (§5.4), which closes the book.**
