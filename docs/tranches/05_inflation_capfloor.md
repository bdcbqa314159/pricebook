# Tranche 5 — the inflation cap/floor under Jarrow–Yildirim

**Proposed:** 2026-10-31 · **By:** Code · **Status:** BUILT, green — awaiting Cowork's pre-merge pass · **Target:**
`v0.5.0` · **Source:** `BERHAN-2012` §1.2.4 and **ch 2.6**, eqs 2.6.1–2.6.5 — **read in the PDF
first-hand**, as T4 taught.

## 0. The formulas, transcribed

Caplet/floorlet (`ω = +1` cap, `−1` floor), eq **2.6.5**:

```
ILCFLT(t,Tᵢ₋₁,Tᵢ,ψᵢ,K,N,ω) = ω·N·ψᵢ·Pn(t,Tᵢ)·[ m·Φ(ω·d₁) − K·Φ(ω·d₂) ]

m  = (Pn(t,Tᵢ₋₁)/Pn(t,Tᵢ))·(Pr(t,Tᵢ)/Pr(t,Tᵢ₋₁))·e^{C}          (2.6.2)
d₁ = [ ln(m/K) + ½V² ] / V ,   d₂ = d₁ − V                       (2.6.5)
K  = 1 + κ   (κ the quoted strike rate, 1.2.22–1.2.23)
```

`m` is the `Tᵢ`-forward expectation of the index ratio and **`C` is T4's convexity** — the same
2.5.11 object, composed, not restated. `V²` is eq **2.6.4**, eight terms, with
`Δ = Tᵢ − Tᵢ₋₁`, `τ = Tᵢ₋₁ − t`, `βn = βn(Tᵢ₋₁,Tᵢ)`, `βr = βr(Tᵢ₋₁,Tᵢ)`:

```
V² = (σn²/2κn)·βn²·[1−e^{−2κn·τ}]  +  σI²·Δ  +  (σr²/2κr)·βr²·[1−e^{−2κr·τ}]
   − 2ρnr·(σn·σr/(κn+κr))·βn·βr·[1−e^{−(κn+κr)·τ}]
   + (σn²/κn²)·[ Δ + (2/κn)e^{−κn·Δ} − (1/2κn)e^{−2κn·Δ} − 3/(2κn) ]
   + (σr²/κr²)·[ Δ + (2/κr)e^{−κr·Δ} − (1/2κr)e^{−2κr·Δ} − 3/(2κr) ]
   − 2ρnr·(σn·σr/(κn·κr))·[ Δ − βn − βr + (1−e^{−(κn+κr)·Δ})/(κn+κr) ]
   + 2ρnI·(σn·σI/κn)·[Δ − βn]  −  2ρrI·(σr·σI/κr)·[Δ − βr]
```

**`ρnI` appears here and only here** — the eighth term — exactly as Cowork read §2.6. This is its
first consumer, so it lands now rather than in T4, which is the pull discipline working as intended
rather than a gap being filled.

## 1. Pieces

```
L3  JYCorrelations gains `nominal_index`      — ρnI's first consumer
    yoy_variance(t, Ti-1, Ti, params)         — eq 2.6.4
    JarrowYildirim.variance(...)              — the capability, beside .convexity(...)
L2  InflationCapFloor(index, leg, notional, kind)   — reuses YoYLeg (see §3.1)
L4  price_inflation_capfloor + caplet_value   — eq 2.6.5 per period, summed
```

**Nothing new at L0 or L1.** Five tranches in, the market-data layer has not changed since T2 —
which is the keys-not-fields design's whole claim, now tested by a model with options on it.

## 2. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **eq 2.6.5 and 2.6.4 against independent transcriptions** in the test, parameter-swept | **1** | `V²`'s eight terms are where every sign, `κ` and factor-of-2 error lives; the transcription is from the PDF, not the implementation |
| 2 | **the anchor — `σn = σr = 0` ⇒ `V² = σI²·Δ` and `C = 0`**, so the caplet is plain Black on the deterministic forward ratio with vol `σI√Δ`, **exactly** | **exact reduction** | six of eight `V²` terms carry `σn` or `σr`; this collapses them. A sign error in any of the six survives |
| 3 | **companion: the rate vols MOVE the price** (per the charter's reduction clause) | structural | without it, oracle 2 passes an implementation that computes only `σI²Δ` and ignores `V²`'s other seven terms |
| 4 | **put-call parity `cap − floor = N·ψᵢ·Pn(t,Tᵢ)·(m − K)`**, and at `K = 1` that is **exactly T4's YoY period value** | **structural, cross-product** | parity is invisible to magnitude checks, and the `K=1` case ties T5 to T4's verified leg through different code |
| 5 | **`V² ≥ 0`** over a wide parameter sweep including adverse correlations | structural | a variance that can go negative is the classic sign error in an eight-term expression, and it would surface as a domain error far from here |
| 6 | **deep in/out limits**: `K → 0` ⇒ caplet → `N·ψᵢ·Pn(t,Tᵢ)·m`; `K → ∞` ⇒ 0; floor mirrors | **1** | catches `ω` applied to the wrong term |
| 7 | **`σI, σn, σr → 0` ⇒ intrinsic** on the deterministic forward | exact | the no-vol degenerate case |
| 8 | failures, matured → PV 0, round-trip + version | structural | the contracts carried from T1–T4 |

**Oracle 2 is the anchor and oracle 3 is its companion**, per the charter clause T4 produced. I am
stating that pairing explicitly because this is the first tranche written under the rule.

## 3. Design decisions, flagged

### 3.1 Reusing `YoYLeg` — rule-of-two fires, and this time the answer is YES
T4 kept `YoYLeg` separate from `RealLeg` because they are different cash-flow shapes. A cap's
strike is a *rate applied per period on a schedule* — the **same shape** as a YoY swap's fixed rate,
differing only in interpretation (`K = 1 + κ`). So `InflationCapFloor` reuses `YoYLeg`, giving four
fields, and `YoYLeg` now has its two consumers honestly.

That is the ruling test applied in the opposite direction from T4: *consumers sharing a shape, not a
word.* Worth your challenge — the counter-argument is that a strike and a coupon are different
*concepts* even at the same shape, and if you read it that way I will add a `CapFloorLeg` instead.

### 3.2 `ρnI` lands in `JYCorrelations`, which becomes three fields
No bundling pressure (3 ≤ 5). Its absence in T4 was correct and its arrival here is the pull
discipline, so the field has a consumer the day it appears — which is the test the charter sets.

### 3.3 Cap vs floor: `ω` as an enum, not a bool
`kind: CapFloorKind` (CAP/FLOOR) rather than `is_cap: bool`. A boolean at a call site reads
`InflationCapFloor(..., True)`, which is unreadable and silently reversible; the enum cannot be
passed backwards.

### 3.4 Not building: the cap/floor **surface** or calibration
T5 prices a cap/floor at a given parameter set. Calibrating JY *to* cap or YoY-floor quotes (ch 5)
is the next tranche, and its oracle is the synthetic parameter-recovery round-trip Cowork named —
not the thesis's Figure 5.x, which is plots.

## 4. Stock
**Purify:** `jy_yoy_caplet` (`jarrow_yildirim.py:293`) for structure. **Mine for values only after
oracles 1–7 pass:** the ZC/YoY cap tests in `inflation_vol` — regression, never anchor, per the
stock-is-not-authority rule T4 earned the hard way.

## 5. Process
Branch `tranche/05-inflation-capfloor`. `test` (RED) → `feat` → `docs` → `chore(release)`.
`source:` cites `BERHAN-2012` with equation numbers. `# crosscheck: quantlib YoYInflationCapFloor`.

---

## 6. Result — and a bug found in TRANCHE 4, which is the headline

**87 oracles green**, all eight gates, ruff + pyright clean, zero suppressions.
Cowork's §3.1 ruling applied: the cap carries `strike: float` and reuses **`CouponSchedule`** (its
honest second consumer); no `YoYLeg`, no `CapFloorLeg`.

### The T4 convexity formula was wrong, and T5's cross-product oracle found it

**Eq 2.5.11's bracket nesting.** The `ρ_nr` cross term belongs **inside** the `βr(t,Tᵢ₋₁)` factor:

```
C = σr·βr(Tᵢ₋₁,Tᵢ)·{ βr(t,Tᵢ₋₁)·[ ρrI·σI − ½βr(t,Tᵢ₋₁) + (ρnr·σn/(κn+κr))(1+κr·βn(t,Tᵢ₋₁)) ]
                      − (ρnr·σn/(κn+κr))·βn(t,Tᵢ₋₁) }
```

T4 placed it *beside* that factor. Error size: **1.8% of the convexity** at a 2y–3y period, 1.5% at
4y–5y — and **infinite** at `t = Tᵢ₋₁`, where the correct value is exactly zero and T4's was 2.6e-4.

### Why nothing in T4 caught it — the part worth keeping

1. **The sweep oracle compared two copies of the same mistake.** T4's "independent transcription"
   was made from the same `pdftotext -layout` output as the implementation, and that rendering
   *merges 2.5.11's parentheses*. Two transcriptions of one corrupted source are **not
   independent** — the 108-point sweep passed because both sides were wrong identically. This is
   the telescoping failure mode in a new disguise: we guarded against reading the formula from the
   *stock*, and then both readings came from one mangled rendering of the *source*.
2. **The `σr → 0` anchor could not see it either.** `σr` remains an overall factor under both
   nestings, so the reduction passed.
3. **What found it:** T5's `cap − floor = T4's YoY leg` oracle disagreed by 0.65%. That is the
   cross-product check doing exactly its job — different code on each side, so a shared
   transcription error has nowhere to hide.
4. **What identified the cause:** a **structural requirement**, not a number. `C` corrects
   `E^{Tᵢ₋₁}[Pr(Tᵢ₋₁,Tᵢ)|Ft]`, so at `t = Tᵢ₋₁` the quantity is already known and `C` must be
   **exactly zero** — a property of the object, true for every parameter set. The wrong nesting
   violates it. That oracle now exists and should have existed in T4.

**Two practices follow, and they are charter-grade:**
- **Re-read a load-bearing formula in the `-raw` extraction, not only `-layout`.** Layout mode
  reflows fractions and merges parentheses; raw mode preserves the token order. For anything whose
  nesting carries meaning, read both.
- **Ask what the object must satisfy regardless of parameters**, and test that. `C(t=Tᵢ₋₁) = 0`,
  `V² ≥ 0`, parity — none of these are numbers from a source, and all three outlive a transcription
  error.

### The rest of the tranche
`V²`'s eight terms (2.6.4) with `ρ_nI` landing as its first consumer and a test pinning that T4's
convexity is **invariant** to it; eq 2.6.5 swept over strikes and both kinds as the primary oracle;
the `σn = σr = 0` reduction to plain Black with its non-vacuity companion; parity, deep-ITM/OTM
limits, no-vol intrinsic; and `foundation/distributions.py`, a thin scipy adapter whose oracles test
*the adapter* (`Φ(x)+Φ(−x)=1` is what the parity oracle leans on) rather than scipy's numerics.
