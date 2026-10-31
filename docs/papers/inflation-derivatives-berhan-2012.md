# Paper reference — Berhan (2012), *Pricing Inflation Derivatives*

**Citation.** Damir Tewolde Berhan, *Pricing Inflation Derivatives: A survey of short-rate and
market models.* MSc thesis, KTH Royal Institute of Technology, Stockholm, 2012. TRITA-MAT-E
2012:13 / ISRN-KTH/MAT/E--12/13--SE. Supervisor/examiner: Boualem Djehiche.
**Source file:** `_stock/NewDocuments/pricing-inflation-derivatives.pdf` (60 pp).
**Scope decision (Bernardo, 2026-10-31):** build the **full arc** — linear benchmarks →
year-on-year → cap/floor → Jarrow–Yildirim and the market model, calibrated to YoY floors (the
thesis's own endpoint).

---

## 1. The oracle honesty note — READ FIRST

**The thesis's empirical results are PLOTS, not tables.** Figures 1.1 (break-even/nominal swap
quotes), 1.2 (YoY cap/floor quote surface), and 5.1–5.10 (calibrated curves, implied-vol surfaces,
relative-error surfaces) are the only "numbers", and a plot cannot be transcribed to a 1e-12
oracle. So for this vertical the oracle does **not** come from transcribed paper numbers. It comes,
in descending strength, from:

1. **The paper's closed-form formulas, no-arbitrage identities, and EXACT REDUCTIONS** (below) —
   rung-1/structural, independent of any implementation. The reductions (JY → the
   deterministic-real-rate limit as vol/correlation → 0; `cap − floor = YoY swap` by parity) are
   independent *by construction*, not reproductions — they are the correctness anchor at the
   model-dependent instruments, where a self-consistent closed form could be self-consistently
   wrong. These are the backbone of every tranche oracle.
2. **The stock's inflation test files** (`_stock/.../*inflation*`, `jarrow_yildirim` tests) — a
   **regression** anchor: did we faithfully reproduce the known-good migration implementation.
   Strong, but note it is reproduction, not independent correctness, so it rides behind the
   reductions in (1).
3. **An external cross-check against QuantLib — deferred, and deliberately not gated to a tranche**
   (Bernardo, 2026-10-31). QuantLib implements these exact instruments, so it is a genuine second
   engine — but it is belt-and-suspenders on top of (1), not the thing correctness rests on. It
   lands **much later as its own optional cross-check test tier**: a dev-only optional dependency,
   its own skippable test folder, **never in the core reproducibility contract or a merge gate**,
   so it validates without complicating our pinned-dependency story. Near-term focus is our own
   library covered by our own oracles.

The figures are a **visual sanity check only** (shape, sign, order of magnitude). No code is ever
fitted to a figure.

---

## 2. Notation

`Pn(t,T)` nominal zero-coupon bond (discount factor); `Pr(t,T)` real zero-coupon bond; `I(t)` the
inflation index level; `I0` the base index at issue; `b(0,T)` the market-quoted break-even
inflation rate; `N` notional; `ψi` the contract year fraction on `[Ti−1,Ti]`. Foreign-currency
analogy (the thesis's engine of derivation): **real ↔ foreign, nominal ↔ domestic, the index ↔ the
FX spot** (eqs 1.2.8–1.2.12).

---

## 3. The instruments and their pricing (the strong oracles)

### 3.1 Zero-coupon inflation swap (ZCIIS) — §1.2.2 — MODEL-FREE
Inflation leg payout `N[I(TM)/I0 − 1]` (1.2.6); fixed leg `N[(1+b(0,TM))^TM − 1]` (1.2.7).
Value: `ZCIIS(t,TM) = N[ I(t)/I0 · Pr(t,TM) − Pn(t,TM) ]` (1.2.13); at `t=0`,
`ZCIIS(0,TM) = N[Pr(0,TM) − Pn(0,TM)]` (1.2.14). **Depends on no dynamics — pure no-arbitrage.**
Par/break-even relation (this is also how the real curve is bootstrapped from quotes):
`Pr(0,TM) = Pn(0,TM)·(1+b(0,TM))^TM` (1.2.16).

**Oracles:** (a) closed form `N[Pr−Pn]` with `Pr`,`Pn` from independently-specified curves in the
test (not read back from the engine); (b) **par identity** — a swap struck at the break-even rate
reprices to zero (1.2.16); (c) QuantLib `ZeroCouponInflationSwap`; (d) stock test values.

### 3.2 Inflation-linked bond — §1.2.1 — MODEL-FREE
Unit real-value `P_IL(t,T) = I(t)·Pr(t,T)` (1.2.4); coupon bond
`ILB(t,TM) = I(t)/I0 · N[ C·Σ Pr(t,Ti) + Pr(t,TM) ]` (1.2.5).
**Oracles:** closed form; replication consistency — the linked leg equals a strip of ZC inflation
bonds, so it must reconcile with §3.1; QuantLib; stock.

### 3.3 Year-on-year inflation swap (YoY) — §1.2.3 — MODEL-DEPENDENT
Inflation leg `N Σ [I(Ti)/I(Ti−1) − 1] ψi` (1.2.17); fixed `N Σ ψi C` (1.2.18). Per-period value
(1.2.21) reduces to a nominal PV of the `Ti−1`-forward price of the `Ti` real bond, which is
**model-dependent through the convexity** of `E^Qn[e^{−∫rn} Pr(Ti−1,Ti)]`. The
**deterministic-real-rate limit** `Pn(t,Ti−1)·Pr(t,Ti)/Pr(t,Ti−1)` is the no-convexity baseline.
**Oracles:** JY/market-model closed form **vs the deterministic limit as vols/correlation → 0**
(an exact-by-construction reduction — reach for this pattern, it is the strongest the predecessor
produced); YoY→ZCIIS when the first fixing is already known; QuantLib `YoYInflationSwap`; stock.

### 3.4 Inflation cap/floor — §1.2.4 — MODEL-DEPENDENT
Caplet/floorlet payout `Nψi[ω(I(Ti)/I(Ti−1) − 1 − κ)]^+`, `ω=+1` cap / `−1` floor (1.2.22);
with `K=1+κ`, value (1.2.23) under the nominal `Ti`-forward measure. Model-dependent (Black-type
under JY / the market model). **Oracles:** closed form; **put-call parity** `cap − floor = YoY
swap` (structural, invisible to magnitude-only checks); QuantLib; stock.

---

## 4. The models (chapters 2–4)
- **Jarrow–Yildirim (ch 2)** — HJM for nominal and real forward curves related by the FX analogy,
  in a **Hull-White specification**; yields analytic formulas for ZCIIS, YoY and cap/floor. Stock:
  `_stock/.../fixed_income/jarrow_yildirim.py` (510 lines) — primary purify target.
- **Market Model I (ch 3)** — an LMM for nominal and real forward rates.
- **Market Model II (ch 4)** — models the **forward inflation indices** directly (Mercurio;
  Belgrade–Benhamou–Koehler); analytic for YoY and cap/floor under approximations. Note the
  thesis's own finding: it calibrates only to **ATM** (no smile).
- **Calibration (ch 5)** — nominal+real curves from nominal swaps + ZCIIS; JY nominal vols to ATM
  caps, then fit to YoY floor quotes; Market Model II likewise. Results are the Figure 5.x plots —
  visual check only (§1).

**Placement against the canon** (for Code's "where do these sit"): JY = Jarrow & Yildirim (2003),
the foreign-currency-analogy HJM; Market Model II = Mercurio (2005) and Belgrade, Benhamou & Koehler
(2004), the forward-CPI line that parallels the LIBOR Market Model. This thesis is a *survey that
implements both*, so it is a good Rosetta stone but not the primary citation — when a formula is
load-bearing, pin it to the original author too.

---

## 5. Build order (the tranche sequence this implies)
- **T2 — inflation spine (model-free):** inflation-index identity (L0, carrying its publication
  lag), nominal+real discount curves in the snapshot (a new curve *key*, not a new field), the
  ZCIIS product (L2, serialised), the pricer composing `df` from both curves. Oracle §3.1.
- **T3 — inflation-linked bond:** same curve machinery, second consumer. Oracle §3.2.
- **T4 — YoY swap + first model (JY):** dynamics and convexity enter. Oracle §3.3 (the →0 limit).
- **T5+ — cap/floor + JY/market-model calibration to YoY floors:** the thesis endpoint. Oracle §3.4.

**Design question ruled for T2 (Cowork, contract/vocabulary → §6 immediate-stop lane):** the
**publication lag lives on the inflation-index identity (L0)** — it is intrinsic to the index
(HICPxt is 3-month-lagged by definition), not the contract, and risk never bumps it; the
*reference-value resolution* that uses the lag is a snapshot lookup (L1). Open to Code's challenge
at the tranche proposal, but this is the default.
