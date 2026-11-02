# Tranche 10 — the Hull-White nominal cap engine and the (κn, σn) fit (§5.3.1)

**Status:** **(a) BUILT** (v0.11.0) — the HW nominal cap engine. **(b) pending** — the `(κn, σn)` fit.
Decided by Cowork (§10 step 1), built by Code. **Results in §7.**
**Source:** the HW cap formula from **`MCINERNEY-ZASTAWNIAK`** (`in-hand`) §3.5 (Hull-White) + the
caplet-as-put-on-a-zero-coupon-bond decomposition and bond-option put-call parity it gives; the
**calibration target** (ATM nominal cap vols) from `BERHAN-2012` §5.3.1. **BERHAN performs §5.3.1
with a HW cap formula it declines to publish** ("beyond the scope of this thesis") — so §5.3.1 is
followed for the *fit*, and McInerney supplies the *forward model* BERHAN omits. (Code's hard-stop
caught this; Bernardo acquired the source 2026-11-02.)
**Why now:** the finish-the-book ladder's next rung, and it **closes the two-of-six gap** — σn has
been a hand-supplied input since T6; §5.3.1 is where the thesis sources it, by fitting `(κn, σn)` to
ATM nominal cap vols. It is the first **non-inflation** instrument (a vanilla rates cap), flagged
plainly, but it is the thesis's own calibration step, not a detour.

## 0. Split — two rungs, build (a) first

- **(a) the Hull-White nominal cap ENGINE** — a priceable rates cap under HW, closed form.
- **(b) the CALIBRATION** `(κn, σn)` to ATM nominal cap vols — inverts (a).

They land as separate PRs; (b) composes (a). If (a)'s own oracle will not check in one pass, it may
split again, but I expect it whole.

## 1. Pieces and layers

```
L3  models/hull_white.py  — EXTRACT HullWhiteParams + hw_beta from JY (§3d, rule-of-two: JY's
                            nominal factor + this cap are the two consumers), ADD the HW caplet
                            closed form (a Jamshidian option on a zero-coupon bond)
L2  products/nominal_cap.py — NominalCap: schedule + strike + notional + kind (CAP/FLOOR), frozen
L4  engine/nominal_cap.py   — price_nominal_cap: a strip of HW caplets off the nominal curve        (a)
L4  engine/hw_cap_calibration.py — fit (κn, σn) to ATM cap vols, via the solver seam                (b)
```

**The HW caplet is L3** (a model analytic — a bond-option closed form, finance, like Black and the
JY convexity). **The cap pricer is L4** (an engine: `price(NominalCap, HullWhiteModel)`). **The
calibration is L4** — it **inverts the cap engine** (reprices to the ATM market, solving `(κn,σn)`),
so by §2 it is engine-inverting and lives in `engine/`, like `jy_calibration`.

`HullWhiteParams` and `hw_beta` already exist inside `models/jarrow_yildirim.py` (T4). T10 is their
**second** consumer, so they move to `models/hull_white.py` and JY composes them — a pure refactor
guarded by T4's green oracles, numbers unmoved. The nominal HW model is JY's nominal factor
standing alone.

## 2. Oracle — part (a), the cap engine

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **the HW caplet closed form vs an independent transcription** of the caplet-as-put-on-a-ZCB formula from **`MCINERNEY-ZASTAWNIAK` §3.5**, read in `-raw`+`-layout`, parameter-swept | **1** | every `κ`, factor-of-2 and `B(S,T)` error lives here; the independent transcription must not share the implementation's rendering (§4), and after §5.1 we assume even a held source can misprint |
| 2 | **anchor reduction: `σn → 0` ⇒ the caplet is its intrinsic value** on the deterministic forward, exactly — with the mandatory companion that **σn MOVES the price** (not a no-op) | exact + companion | `σn` is the vol; zero vol is the degenerate case with a known answer, and the companion stops a reduction that an empty implementation also passes |
| 3 | **put-call parity: cap − floor = the forward-rate swap value** (the fixed-vs-float identity), composed from the same caplet atom, different code each side | structural cross-product | the sign/leg error no magnitude check catches |
| 4 | **a requirement that survives a source misprint:** the caplet is positive, monotone in `σn`, and → 0 as the strike → ∞; and the HW caplet-vol term structure has the right `κn` sign (vol of the `T`-caplet rises then saturates in maturity) | structural | the §4 "formula must satisfy a property" guard — a transcription or source typo that oracle 1 might share |
| 5 | failure-is-a-value / raises per convention (an unpriceable cap, a bad schedule) | structural | the engine returns `PricingFailure`; a malformed product RAISES at construction (T8 convention) |

## 3. Oracle — part (b), the calibration

| # | oracle | rung | why it can go red |
|---|---|---|---|
| B1 | **parameter recovery:** generate ATM cap vols from a known `(κn*, σn*)` through (a), fit back, recover both to tolerance | 2 — round trip | the inverse of (a) |
| B2 | **identifiability (B1's non-vacuity companion, the T6 pattern):** `(κn, σn)` are a 2-parameter fit, so **multi-start recovery to one point** AND/OR a bounded Jacobian condition number — not single-parameter sensitivity | 1 — requirement | `κn` shapes the caplet-vol *term structure* while `σn` sets its *level*, so they SHOULD separate across a maturity strip — but that must be proven, not assumed (the T6 identifiability valley) |
| B3 | a deliberately-wrong start strictly improves; residual at truth ≈ 0 | 1 | the objective is not flat |
| B4 | failure-is-a-value: a non-converging fit returns `PricingFailure` | structural | the solver seam's `converged` flag (T6) |

The thesis reports the ATM nominal-cap fit is reasonable ("not too bad in the long end", §5.3.1) —
unlike the 6-parameter YoY fit it feeds. That is context, not an oracle; B1–B2 are self-contained.

## 4. Design decisions (ruled)

- **HW core extracted to L3, shared — rule-of-two fires (JY + this).** `models/hull_white.py` holds
  `HullWhiteParams`, `hw_beta`, and the caplet; JY and the nominal cap both compose it, so a HW
  convention cannot diverge between the inflation and rates uses. JY's refactor is pure, T4's
  oracles re-run green, numbers unmoved.
- **The NominalCap is a real L2 product priced by an L4 engine**, not a calibration-internal formula
  — a nominal cap is a first-class instrument (and §5.3.1's forward model), priced through
  `price()` like every other product.
- **This closes T6's σn gap but does NOT re-open §5.3.2.** The full six-parameter YoY fit stays as
  T6 left it; wiring σn from here into a §5.3.2 re-run is a *later* unit, not T10.
- **Cross-check dispositions (§8 gate) for the new modules:**
  - `models/hull_white.py` → `crosscheck: quantlib HullWhite / HullWhiteModel caplet`
  - `engine/nominal_cap.py` → `crosscheck: quantlib BlackCapFloorEngine / AnalyticCapFloorEngine`
  - `engine/hw_cap_calibration.py` → `crosscheck: quantlib HullWhite calibration (CapHelper)`
  - `products/nominal_cap.py` → `crosscheck: none — a frozen cap description, no numerics`

## 5. Stock
**Mine for content:** `_stock/.../jarrow_yildirim.py` already has the HW `β` and parameter bundle;
the HW caplet formula read from the **source** (standard HW/Jamshidian, §5.3.1), not from a stock
rates module, which may use a different parameterisation. Build the caplet fresh on the extracted
`models/hull_white.py`.

## 6. Process
Branch `tranche/10a-hw-nominal-cap` then `tranche/10b-hw-cap-calibration`. `test` (RED) → `feat` →
`docs` → `chore(release)`. the HW caplet's `source:` is **`MCINERNEY-ZASTAWNIAK` §3.5**; the calibration's is `BERHAN-2012` §5.3.1. The HW
extraction's provenance notes it supersedes JY's inline `HullWhiteParams`/`hw_beta`, with T4's
oracles re-run green to prove the refactor moved nothing.


---

## 7. Result — (a) built, and the structural finding is that a HW bond option IS Black

332 tests, eight gates green, ruff and pyright clean, zero suppressions.

### No new Jamshidian formula was needed
§5 said "build the caplet fresh". It turned out not to need building: **a Hull-White bond option is
Black on the forward bond price** `B(0,T)/B(0,S)` with total variance `ν`, discounted by `B(0,S)`.
Writing out eqs 3.16/3.17 shows it immediately:

```
B(0,S)·ω·[F·Φ(ω d₊) − K·Φ(ω d₋)]  =  B(0,T)Φ(d₊) − K·B(0,S)Φ(d₋)     for ω = +1
```

So `zero_bond_option` **composes `black_price`** — tranche 9's extraction, already oracled — and the
only genuinely new formula in the tranche is `ν(0,S)` (eq 3.20). Transcribing Black a second time
would have duplicated the convention across two modules, which §3d exists to prevent. The oracle
still transcribes 3.16–3.18 longhand, so there is independent code on each side.

`ν` is also written through `hw_beta` rather than as the printed closed form, because eq 3.19's
`D(S,T)` **is** `β` — verified by reducing 3.19 to the printed 3.20, not assumed.

### The source's `d₋` drops its radical in BOTH renderings
Each prints `d₋ = d₊ − ν(0,S)` while `ν` is defined as a *variance*. The §5.1 shape again — a held
source misprinting identically in `-raw` and `-layout`, so transcription independence cannot settle
it.

**And the obvious discriminator does not work.** I reached for put-call parity first; it is
satisfied **identically** for any `d₊`/`d₋`, because `Φ(d) + Φ(−d) = 1` collapses both terms. A
check that cannot fail, on the one question that mattered. The `ν → 0` limit fails too — both
readings converge, since `√ν` and `ν` both vanish. What settles it: **the literal reading prices a
call at −0.0049**, which no option can be. That now has its own test rather than living in a
comment, and the vacuity of parity here is stated in both the test and the module.

## 8. Process failures of mine, both recorded

### 8.1 The L4 oracles were NOT pre-registered — charter §4 violated
The L3 caplet analytics were committed RED before the code (`test(models): HW caplet oracles RED`).
The **engine** tests were not: I wrote `products/nominal_cap.py` and `engine/nominal_cap.py`, then
the tests, which passed first time. That is implement-then-check, which §4 forbids precisely because
it invites bending the test to the code.

Un-ringable, so the substitute evidence is **mutation**, which is the only thing that can show a
not-pre-registered test has power:

| mutation | caught? |
|---|---|
| reverse the caplet↔bond-put direction | **yes** — 3 tests |
| variance on `(pay, pay)` instead of `(start, pay)` | **yes** — 4 tests |
| factor-of-2 in the variance denominator | **yes** — 36 tests (the L3 sweep) |
| **drop the accrual from `(1 + τK)`** | **NO — survived all 312** |

### 8.2 The unit-accrual trap, FOURTH appearance, and it left a real hole
Every fixture used annual ACT/365F periods, where `τ = 1.0` exactly — so `1 + τK` and `1 + K` are
the same number and the accrual's role in eq 3.21 was **completely unoracled**. Not review, not
reasoning: mutation found it.

Fixed on both sides — the L3 caplet test now sweeps `τ ∈ {0.25, 0.5, 1.0, 2.0}` and the L4 fixture
is **semi-annual** (`τ ≈ 0.4932`), with the non-unit accrual asserted as a premise. The mutation now
fails 23 tests.

This is the same trap as tranche 8's annuity (`Σ τ·df` vs `Σ df`) and tranche 9's variance horizon.
Three of four instances were in fixtures I wrote in the same tranche that had just learned it.

### 8.3 A wrong requirement, caught by its own test
The distinctness test asserted every caplet is `> 0`. The first one is exactly **zero** — its period
starts at the anchor, so `ν(0,0) = 0`, the rate has already fixed, and on this curve it is out of
the money. Correct behaviour, wrong assertion. It became two oracles: the first caplet **is** its
intrinsic at zero variance (a reduction worth having), and the *live* caplets are distinct.

## 9. Findings for Cowork (§10 — raised, not decided)

**9.1** `zero_bond_option` composes `black_price` instead of transcribing 3.16/3.17, against §5's
"build the caplet fresh". Flagged because it changes what the module contains, though the oracle is
unaffected — the test still transcribes longhand.

**9.2** `CapletPeriod` is a new L3 bundle (`df_start`, `df_pay`, `accrual`), created because
`hw_caplet` otherwise takes six arguments. §6's compelled-bundling clause, not a new abstraction —
concrete, frozen, no polymorphism.

**9.3** `NominalCap` deliberately does **not** share a type with `InflationCapFloor` despite
identical fields. One strikes a year-on-year index ratio, the other a nominal rate: the shared
*word* "cap", not a shared shape. Stated so a future reader does not read the duplication as an
oversight.
