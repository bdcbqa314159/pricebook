# Tranche 4 — the year-on-year inflation swap and Jarrow–Yildirim

**Proposed:** 2026-10-31 · **By:** Code · **Status:** BUILT, green — awaiting Cowork's pre-merge pass · **Target:**
`v0.4.0` · **Source:** `BERHAN-2012` §1.2.3 and **ch 2** (§§2.4–2.5), formulas **read in the PDF**,
not inferred.

## 0. STOP-AND-READ: the anchor oracle in the handoff is wrong, and I found it by reading ch 2

Thread 0022 specifies the T4 anchor as *"JY → deterministic limit as **vol/ρ → 0** (the stock test
already shows convexity = 0 at ρ = 0)"*. **The ρ half is wrong**, and had I built it the oracle
would have failed against the correct formula — or, worse, invited me to bend the formula until it
passed.

BERHAN eq **2.5.11**, read in the source:

```
C(t,Tᵢ₋₁,Tᵢ) = σr · βr(Tᵢ₋₁,Tᵢ) · [ βr(t,Tᵢ₋₁)·(ρ_rI·σI − ½·βr(t,Tᵢ₋₁))
                                   + (ρ_nr·σn/(κn+κr))·(1 + κr·βn(t,Tᵢ₋₁))
                                   − (ρ_nr·σn/(κn+κr))·βn(t,Tᵢ₋₁) ]
```

**`σr` is an overall factor; `ρ` is not.** The thesis says so in the next sentence — *"it vanishes
for σr = 0"* — and the `−½βr(t,Tᵢ₋₁)²` term is a **pure real-rate convexity with no correlation in
it at all**. Computed (κn=.05, σn=.01, κr=.04, σr=.008, σI=.012, ρ_nr=.3, ρ_rI=−.2, t=0, T=2→3):

| | C |
|---|---|
| all parameters | −1.4738e-02 |
| **ρ_nr = ρ_rI = 0** | **−1.4486e-02** — barely moves, does **not** vanish |
| **σr = 0** | **0** exactly |

**So the anchor is `σr → 0`, not `ρ → 0`.**

### Why the stock's test said otherwise — and a second finding
`_stock/.../inflation_advanced.py::yoy_convexity_adjustment` computes

```python
adj = inflation_vol * nominal_vol * nominal_real_corr * tenor * year
```

a **first-order approximation** in which ρ *is* a factor, so ρ=0 kills it. It captures only the
`σI·σn·ρ` cross term and **drops the real-rate convexity entirely**. And third:
**it takes `real_vol` as a parameter and never uses it** — a dead parameter, which is the
carried-but-unread smell we have caught three times now, and it is the very parameter JY makes the
term's overall factor.

So the stock's function is not JY's convexity; it is a different, weaker object. **This tranche
implements 2.5.11 and does not purify that function.** Its test is kept only as a regression anchor
for the *approximation's* own behaviour, if at all.

## 1. Target and pieces

Per-period inflation leg, eq **2.5.10**:
`YYIIS(t,Tᵢ₋₁,Tᵢ,ψᵢ,N) = N·ψᵢ·[ Pn(t,Tᵢ₋₁)·(Pr(t,Tᵢ)/Pr(t,Tᵢ₋₁))·e^{C} − Pn(t,Tᵢ) ]`,
summed over periods; fixed leg `N·Σψᵢ·C_fix` (1.2.18). With `βn(t,T)=(1−e^{−κn(T−t)})/κn` and
`βr` likewise (2.4.3, 2.4.16).

```
L0  (nothing new)
L1  (nothing new — both curves already keyed)
L3  JarrowYildirim(kappa_n, sigma_n, kappa_r, sigma_r, index_vol_and_corr)   — see §3.1
    carrying its MarketSnapshot, like every model here
L2  YearOnYearInflationSwap(index, leg, notional, ...)   — a RealLeg second consumer (§3.2)
L4  price_yoy_inflation_swap — composes both curves + the model's convexity
```

**This is the first model-dependent instrument**, so it is also the first tranche where a model
carries parameters rather than only a snapshot.

## 2. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **`σr → 0` ⇒ the deterministic-real-rate limit** `N·ψᵢ·[Pn(t,Tᵢ₋₁)·Pr(t,Tᵢ)/Pr(t,Tᵢ₋₁) − Pn(t,Tᵢ)]`, **exactly** | **exact reduction** | the house pattern, third time. `σr` is 2.5.11's overall factor, so the reduction is structural, not a limit taken numerically |
| 2 | **`C` computed against 2.5.11 written independently in the test**, over a parameter sweep | **1** | the correction term is where every sign and `κn+κr` denominator error lives |
| 3 | **ρ → 0 does NOT remove the convexity** — asserted explicitly, with the surviving `−½βr²` quantified | **structural** | it pins the finding above into the test suite so nobody "simplifies" the formula back to the stock's approximation |
| 4 | **YoY → ZCIIS when the first fixing is known**: the first period's leg, with `I(Tᵢ₋₁)` observed, is T2's ZCIIS leg | **structural, cross-product** | ties T4 to T2's verified atom — the T3 reconciliation pattern again |
| 5 | **`κ → 0` limits**: `β(t,T) → (T−t)`, so `C` degenerates to its small-`κ` form | **exact** | catches a `κ` in the wrong denominator; `β` is where Hull-White implementations go wrong |
| 6 | `C` grows with `σr` and with the period length; sign follows `ρ_rI` for the `σI` term | structural | monotonicity a scalar match misses |
| 7 | historical periods excluded; matured → PV 0; missing curve/fixing/params → failure; round-trip | structural | the contracts carried from T1–T3 |

**What I am NOT claiming.** No external engine, and no reproduction of the thesis's Figure 5.x
calibration — its quotes are plots. Oracles 1, 4 and 5 are *reductions to independently-verified
things*; 2 is the formula written twice. That is the strength the QuantLib deferral rests on, and
thread 0022 reasoned the same way.

## 3. Design decisions, flagged

### 3.1 JY has nine parameters; the five-field limit bites
`κn, σn, κr, σr, σI, ρ_nr, ρ_nI, ρ_rI` plus the snapshot = nine. The limit says bundle, so I
propose `JarrowYildirim(market, nominal, real, index)` with `nominal = HullWhiteParams(κ, σ)`,
`real = HullWhiteParams(κ, σ)`, and `index = IndexParams(vol, rho_nominal, rho_real)`. Four fields,
and the grouping is the model's own structure rather than an arbitrary split — the thesis derives it
as two HJM/HW factors plus the index. **`ρ_nI` appears in 2.5.11 nowhere**: only `ρ_rI` and `ρ_nr`
do, so I will carry `rho_nominal` **only if** the cap/floor formulas (T5) need it — otherwise it is a
speculative field and I will leave it out and say so. *Flagging that I have not yet read ch 2's
cap/floor section to check.*

### 3.2 `RealLeg`'s second consumer — rule-of-two fires here
T3 kept `RealLeg` concrete with the generalisation decision deferred to "the second leg type", which
is this tranche. **My reading: it still should not generalise.** A YoY leg pays an index *ratio*, not
a rate on a notional; sharing a base class would put two different payoff shapes behind one name. I
propose a separate `YoYLeg` (schedule + fixed rate for the paying side) and `RealLeg` left alone —
i.e. rule-of-two fired and the answer was *no*. Worth your challenge.

### 3.3 Stock usage
**Purify:** `jarrow_yildirim.py` (510 lines) for `JYParams` shape, `βn/βr` helpers and the ZC/YoY
structure. **Do not purify:** `yoy_convexity_adjustment` (§0 — it is a different, weaker object with
a dead parameter). Mine the 22 inflation test files for regression values *after* oracles 1–5 pass,
never before.

## 4. Process
Branch `tranche/04-yoy-jarrow-yildirim`. `test` (RED) → `feat` → `docs` → `chore(release)`.
`source:` cites `BERHAN-2012` with equation numbers (§2.5.10, 2.5.11, 2.4.3, 2.4.16).
`# crosscheck: quantlib YoYInflationSwap / YoYInflationCapFloor`.

---

## 5. Result

**64 oracles green**, all eight gates, ruff + pyright clean, **zero suppressions**. Cowork owned the
`ρ→0` error without qualification and ruled in favour of all four points (§0's correction, not
purifying the stock's convexity, deferring `ρ_nI` to T5, and keeping `YoYLeg` separate).

### What landed

`yoy_convexity` (eq 2.5.11) with `hw_beta` (2.4.3/2.4.16), the `JarrowYildirim` model carrying
`JYParams` and its snapshot, `YoYLeg`, `YearOnYearInflationSwap`, and the pricer composing eq
2.5.13 — whose **first period is tranche 2's ZCIIS leg**, because the thesis prices that cash flow
by 1.2.13 when its previous fixing is known.

### The anchor, and the test that stops it being vacuous

`σr → 0` ⇒ `C = 0` exactly ⇒ the whole swap collapses to the deterministic-real-rate limit. Exact
by construction, since `σr` is 2.5.11's overall factor.

**Paired with `test_the_convexity_MOVES_the_leg_so_the_anchor_is_not_vacuous`** — without it, the
anchor would pass on an implementation that ignored `C` entirely. That is the powerless-check lens
applied to a reduction oracle: a reduction proves the general case *collapses* correctly, and says
nothing about whether the general case does anything at all.

And `test_the_convexity_does_NOT_vanish_at_zero_correlation` pins §0's finding into the suite, with
the surviving `−½βr²` term quantified, so nobody simplifies 2.5.11 back toward the stock's
approximation.

### Three gate catches, all mine, all fixed at the cause

1. **`PLR0913`** — `yoy_convexity` took seven arguments. Bundled into **`JYParams`** (two HW
   factors, index vol, correlations), which is the model's own structure rather than a split to
   satisfy a count — and `JarrowYildirim` then carries just `market` + `params`.
2. **`layers`** — the word *"payoff"* in `YoYLeg`'s docstring tripped the L0 finance-free gate,
   which greps prose as well as code. **Reworded rather than loosening the gate.** A gate that
   over-reports on comments is still doing its job, and relaxing it so my own docstring passes is
   the wrong direction; the weakness is noted in the docstring itself rather than acted on.
3. **`debt`** — four `# noqa: E731` on test lambdas. Replaced with real local functions; the gate
   is right that a suppression needs justifying, and here there was none to give.

### What is deferred, named
The **seasoned** case (eq 2.5.12, `ι(t) > 1`) needs the *historical* fixing `I(T_{ι−1})` — an index
**series** in the snapshot, a new market-data shape, deferred with the indexation-lag resolution it
shares a consumer with. `ρ_nI` lands with the cap/floor at T5, per Cowork's reading of §2.6.
