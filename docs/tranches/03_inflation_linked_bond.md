# Tranche 3 — the inflation-linked bond (linker), model-free

**Proposed:** 2026-10-31 · **By:** Code · **Status:** BUILT, green — awaiting Cowork's pre-merge pass · **Target:** `v0.3.0`
**Source:** `BERHAN-2012` (`worked-from`) — §1.2.1, eqs 1.2.4–1.2.5. Extraction:
`docs/papers/inflation-derivatives-berhan-2012.md` §3.2.

## 0. Target, and why it is the right next step rather than the interesting one

`ILB(t,TM) = I(t)/I0 · N[ C·Σᵢ Pr(t,Tᵢ) + Pr(t,TM) ]` (1.2.5), with the unit real value
`P_IL(t,T) = I(t)·Pr(t,T)` (1.2.4).

**It is a consolidation step and I am proposing it as one.** No new mathematics: the linker is a
real bond scaled by the index ratio, and — worth noticing — **1.2.5 does not involve the nominal
curve at all.** What it does buy is three things T2 deliberately left out:

1. **A schedule.** T2's single exchange needed no coupon dates. A linker does, so this is where
   the schedule enters — a real piece of L0 that every later coupon product needs.
2. **The indexation lag, as L2 contract data** — the finding T2 recorded and deferred.
3. **A cross-product reconciliation oracle.** The linker's payments are each a zero-coupon
   inflation bond, so it must reconcile with T2's `inflation_leg_value` rather than being an
   independent formula. T2's guards were all *within* one product; this is the first that holds
   two products to one atom.

The alternative was skipping to the YoY swap (the first model-dependent instrument, where convexity
and the `vol → 0` reduction appear). I am following the extraction doc's own build order instead:
the YoY leg's deterministic-real-rate limit is `Pn(t,Tᵢ₋₁)·Pr(t,Tᵢ)/Pr(t,Tᵢ₋₁)`, which composes the
same `Pr` strip this tranche builds and tests — so T3 is the thing that makes T4's reduction oracle
trustworthy rather than a detour before it.

## 1. The pieces

```
L0  CouponSchedule(dates, day_count)      — coupon dates + the basis accruals are measured on
    RealLeg(schedule, coupon_rate)        — the §3b bundling (see §4.1)
L2  InflationLinkedBond(index, base_index, leg, notional, indexation_lag)   — 5 fields
L4  price_inflation_linked_bond — composes Pr.df over the schedule and the index ratio
```

**Nothing new at L1 and nothing new at L3.** The curve key, the fixings and the model all came in
T2 — which is the keys-not-fields design paying off: a second inflation product costs no market-data
change at all.

## 2. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **1.2.5 closed form** off curves and a schedule the test specifies | **1** | the reference is a sum the test computes; nothing read back from the engine |
| 2 | **strip reconciliation with T2** — the linker equals `Σ C·ZCIB(Tᵢ) + ZCIB(TM)` where each `ZCIB(T) = inflation_leg_value + N·Pn(T)` | **structural, cross-product** | the §3d guard across products: if the linker and the ZCIIS hold two different `Pr` strips, or disagree about the index ratio, this breaks while each product's own oracle still passes |
| 3 | **zero-coupon limit:** a linker with `C = 0` and one payment IS a zero-coupon inflation bond — equals `I(t)/I0 · N · Pr(t,TM)` exactly | **exact reduction** | the reduction shape the extraction doc says to reach for; a mis-built schedule or a double-counted principal shows here |
| 4 | **real-rate invariance:** with `I(t)=I0` and `Pr = Pn`, the linker prices exactly as a nominal fixed-coupon bond on the same schedule | **1** | catches the index ratio applied to the wrong leg, or `Pr`/`Pn` transposed |
| 5 | index ratio scaling: double `I(t)`, the whole price doubles (1.2.5 is linear in it) | structural | catches the ratio applied to coupons but not principal, or vice versa |
| 6 | historical coupons excluded; a matured linker is PV 0 (not a failure) | structural | invariant 6, and T2's must-fix 2 precedent now across *many* flows rather than one |
| 7 | missing real curve / missing fixing → `PricingFailure`; round-trip + version enforced | structural | the contracts carried from T1/T2 |

**Note on oracle 2.** This is the one I most want reviewed. It is the first oracle that spans two
products, and it is deliberately built from T2's **exposed** `inflation_leg_value` rather than from a
re-derivation — so if someone later changes how the index ratio is applied in one product and not
the other, it fails. That is the `rpv01` lesson at product scale.

## 3. Design decisions, flagged

### 3.1 `RealLeg` and `CouponSchedule` — compelled bundling, not new abstractions
A linker needs index, base index, coupon rate, schedule, day count, notional **and** indexation lag
— seven loose fields. The rule's answer is never to raise the limit but to group the cohesive ones,
so: `CouponSchedule(dates, day_count)` and `RealLeg(schedule, coupon_rate)`, giving the bond five
fields. Both have one consumer today, which §6b would normally question — but §3b's bundling
requirement is what *compels* them, and the same argument carried `Term` in T2.

### 3.2 The indexation lag is on the contract, and the stock now confirms it
T2 argued the lag the stock conflates is really two quantities, and that the **indexation** lag is
contract data. **The stock's own linker agrees:** `InflationLinkedBond.__init__` takes
`cpi_lag_months: int = 3` with the comment *"3 for US TIPS, 8 for UK ILG"* — i.e. the stock puts it
**on the instrument** here, while putting a conflated `publication_lag_months` on the index in
`inflation_indices.py`. Two homes for two different facts, one of them mislabelled. So:
`indexation_lag` is a field on the bond, and the index identity stays bare.

### 3.3 The lag is CARRIED but not yet RESOLVED — and I want this ruled
Here is the honest limit of this tranche. Eq 1.2.5 uses `I(t)`, the reference index level, which the
snapshot supplies **already resolved**. The indexation lag's job is to decide *which monthly print*
is `I(t)` for a given date — and that resolution needs a **monthly index history**, not a single
fixing, which is a new market-data shape.

Three options; I propose (b):
- **(a)** carry the lag and resolve it now → needs an index *series* in the snapshot, a new L1
  shape, and interpolation conventions (TIPS interpolate daily, UK ILGs do not). That is a tranche
  of its own.
- **(b) carry the lag as contract data, resolve nothing; the snapshot's fixing stays the resolved
  `I(t)`.** The lag is then *recorded where it belongs* and is **round-tripped in the
  representation** (so a wire payload is complete and a future reader loses nothing), but no code
  reads it yet. **I am flagging that it is an unread field** — exactly what I argued against for the
  publication lag in T2 — and the distinction I would defend is that this one is **contract data
  the representation must carry to be faithful**, whereas the other was market metadata with no
  consumer at all. If you do not accept that distinction, (c).
- **(c)** omit it entirely until the resolution tranche, and accept that a serialised linker is
  incomplete until then.

### 3.4 Only annual ACT/365F schedules, with explicit dates
`CouponSchedule` takes an **explicit date list**, not a frequency plus a generator. Date generation
is calendars, roll conventions and business-day adjustment — a genuine tranche, and the stock's
`generate_schedule` pulls all of it. Deferred; the test states its dates, which is also what makes
oracle 1 independent.

## 4. Stock → purify or build fresh

| piece | stock source | call |
|---|---|---|
| `InflationLinkedBond` | `fixed_income/inflation.py` (565, class at 339) | **mine for content, not shape.** 8 mutable ctor params, a bound schedule generator, `cpi_lag_months` on the instrument (keep that insight) |
| the pricer | same, its PV method | **build fresh** from 1.2.5 |
| `CouponSchedule`, `RealLeg` | `foundation/schedule.py` in the built core | **purify, hard** — an explicit date list and a day count; the frequency/roll/stub machinery waits for the schedule tranche |
| advanced linker analytics | `inflation_bond_advanced.py` (284) | **not carried** — real yield, breakeven analytics, deflation floors each want their own oracle |

## 5. Process
Branch `tranche/03-inflation-linked-bond`. `test` (RED) → `feat` → `docs` → `chore(release)`.
`source:` names `BERHAN-2012`. `# crosscheck: quantlib CPIBond` at the pricer. Cowork's pass first.

---

## 6. Cowork's design pass (thread 0019) and the result

**All three calls ratified; its new question answered a third way, which I think is better than
either option it offered.**

### The `day_count` question — resolved by making the field genuinely read

Cowork spotted that BERHAN 1.2.5 is annual with no accrual factor, so pricing the literal formula
would leave `CouponSchedule.day_count` carried-but-unread — the exact smell this tranche is careful
about. It offered two resolutions: read it (and register the convention), or drop to a dates-only
schedule.

**I took a third: price the general form `C·ψᵢ`, which a real linker pays, and make 1.2.5 its exact
reduction at `ψᵢ = 1`.** That is better than either option on three counts — `day_count` is read, so
nothing is smuggled; the instrument is the real one rather than the paper's simplification; and it
buys a **reduction oracle** (the same shape as T2's 1.2.13→1.2.14 and the predecessor's best oracle),
which neither offered option provides. The schedule's periods are exactly 365 days, so every `ψ` is
exactly 1.0 and the reduction is an identity, not an approximation — asserted in the test alongside
the premise itself.

Per its instruction, the sibling convention is registered: **C8** — a schedule's accrual day count
and its curve's measure are one convention, OWED, trigger shared with C7.

### The lag: carried and round-tripped, on Cowork's sharper test

I had argued the distinction was "contract data vs market metadata" and asked to be ruled rather
than deciding in my own favour. Cowork's test is better than mine: **constitutive of the
instrument's identity, not "does the current pricer read it."** Two linkers on one index with
different indexation lags are *different contracts*, so the lag is constitutive and the
**round-trip is its present consumer** — which is precisely what T2's publication-lag field lacked.
So it is a round-tripped field, not an unread one, and the oracle asserts both that the lag survives
serialisation and that `lag=3 ≠ lag=8` as contracts.

### Oracle 2 confirmed non-circular

Cowork traced the arithmetic: `ZCIB(T) = inflation_leg_value + N·Pn(T)` undoes the swap leg's `−Pn`
to recover `I(t)/I0·Pr(t,T)`, and the strip is exactly 1.2.5. Non-circular because the two sides
compose *different code* — the pricer sums `Pr.df` over the schedule, the oracle routes through
T2's independently-verified leg function. Its note is kept explicit in the test docstring: **oracle
1 is the correctness anchor; oracle 2 is a consistency guard**, and a reconciliation alone only
proves two things agree.

### Bundles kept concrete
`CouponSchedule` and `RealLeg` are frozen bundles with no base class and no polymorphism, per the
§6b guard. Whether `RealLeg` generalises is decided at the second leg type — the year-on-year swap.

### Result
**50 oracles green** (first run, no failures), all eight gates, ruff + pyright clean, zero
suppressions. Nothing new at L1 or L3 — a second inflation product cost no market-data change at
all, which is the keys-not-fields design paying off.
