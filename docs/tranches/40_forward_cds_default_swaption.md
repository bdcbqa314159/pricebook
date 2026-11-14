# Tranche 40 — forward CDS + the default swaption (Black on the forward spread)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.47.0`.
**Source:** `OKANE-2008` ch 9.2 (forward starting CDS) and ch 9.3 (the default swaption, knockout).
**Pre-registered RED.** Third unit of the credit arc; closes the single-name CDS vertical. Reuses
`credit_legs` (T39), `black_price` (T5/T30), and the `SurvivalCurve`.

## 0. The product worth having
**The forward-starting CDS and the option to enter one — the CDS swaption, in closed form.** Two
products, the second priced on the first:

```
Forward spread (9.2):   S(t,tF,T) = [S(t,T)·RPV01(t,T) − S(t,tF)·RPV01(t,tF)] / [RPV01(t,T) − RPV01(t,tF)]
Forward annuity:        RPV01(t,tF,T) = RPV01(t,T) − RPV01(t,tF)
Forward CDS value:      V(t) = (S(t,tF,T) − S(0,tF,T)) · RPV01(t,tF,T)          (long protection)

Default swaption (9.3, knockout), under the risky-annuity measure P where S(tE,T) is a MARTINGALE:
  VPay(0) = RPV01(0,tE,T) · E^P[ max(S(tE,T) − K, 0) ]   = Black payer  on S(0,tE,T), strike K, × RPV01
  VRec(0) = RPV01(0,tE,T) · E^P[ max(K − S(tE,T), 0) ]   = Black receiver
  E^P[S(tE,T)] = S(0,tE,T)                                                       (eq 9.6 — the martingale)
```

The knockout `1{τ>tE}` is already in `RPV01(0,tE,T)` (the forward risky annuity weights by survival to
`tE`), so the standard default swaption is exactly Black on the forward spread, scaled by the forward
risky PV01 as numéraire — the same numéraire-martingale structure as the LMM swaption (T17/T19).

## 1. The pieces and their layers
- **L1 `market/credit_legs.py` (extend) — the forward annuity.** `RPV01(t,tF,T) = RPV01(t,T) −
  RPV01(t,tF)` and the forward spread `S(t,tF,T)` are curve arithmetic over the survival + discount
  curves (the T39 shared functional), so they extend `credit_legs`, not a new module. Pure refactor of
  nothing (new functions), but **the forward spread must reduce to the spot par spread when `tF→t`** —
  assert it against T38's par spread (cross-product, different code).
- **L2 `products/` — `ForwardCDS` and `CreditDefaultSwaption`.** Frozen data. `ForwardCDS`: forward
  start `tF`, maturity `T`, contractual forward spread, notional, direction. `CreditDefaultSwaption`:
  option expiry `tE`, underlying CDS maturity `T`, strike spread `K`, payer/receiver (its own enum,
  §6), and the **knockout** flag (knockout is the standard and the scope here; see §4). No pricing.
- **L3 `models/` — the credit-swaption model (thin).** Carries the `MarketSnapshot` (Q, Z) + the
  **forward-spread volatility `σ`** (the Black vol, a model parameter like `EquityBlackModel`'s vol,
  T30) + recovery `R`. The forward CDS itself needs no vol (it is curve arithmetic); only the swaption
  does — so the forward CDS prices off the `CreditModel` (T38), and the swaption off this one.
- **L4 `engine/` — two pricers, `@register`ed.** The forward CDS value `(S(t,tF,T)−S₀)·RPV01(t,tF,T)`
  under `(ForwardCDS, CreditModel)`; the default swaption `RPV01(0,tE,T)·black(S(0,tE,T), K, σ²tE, ω)`
  under `(CreditDefaultSwaption, CreditSwaptionModel)` — **reusing `black_price`** on the forward
  spread, scaled by the forward risky PV01.

## 2. The oracle (RULED)
The forward-spread and zero-vol reductions are the exact anchors; the **annuity-measure martingale**
and **put-call parity** are the transcription-proof requirements; Black is reused.

- **Forward-spread reductions — the preferred anchors (§4), both exact.** (a) `tF → t`: the forward
  spread `S(t,tF,T)` → the **spot par spread** `S(t,T)` (T38) exactly (the `RPV01(t,tF)→0` limit), and
  the forward CDS value → the spot CDS value — a cross-product reduction to T38, different code. (b) A
  **flat spread curve** `S(t,T)=S(t,tF)`: the forward spread equals the spot (O'Kane 9.2.1, exact). A
  no-op satisfies (b), so **companion:** a **term-structured** spread curve where `S(t,tF,T)` differs
  materially from both spot spreads — the per-the-reduction-anchor-blind-spot rule (§4, T39): the flat
  curve is exactly where the forward-annuity *difference* degenerates, so the companion must be
  non-flat, and the `RPV01(t,T) − RPV01(t,tF)` difference must be a real subtraction (read beyond one
  segment; semi-annual accruals — the τ=1 history baked in).
- **The annuity-measure martingale `E^P[S(tE,T)] = S(0,tE,T)` (eq 9.6) — the drift pin.** The forward
  spread is a martingale under the risky-annuity (numéraire RPV01) measure; this is *why* Black applies
  and it pins the swaption's drift independently of `σ`. Assert it (it is the credit analogue of the
  LMM numéraire-relative martingale, T19 — a free, exactly-centred check, no model result involved).
- **Put-call parity — the headline requirement oracle (survives a Black transcription error).**
  `VPay(0) − VRec(0) = (S(0,tE,T) − K)·RPV01(0,tE,T)` = the forward CDS value (derived from 9.6). It
  holds for every `σ, K` and is blind to a slip in the Black `d±` — the §4 requirement oracle, and a
  **cross-product** tie to the forward CDS (part 1) with different code each side.
- **The zero-vol reduction — exact.** At `σ → 0` the payer swaption → the intrinsic
  `max(S(0,tE,T) − K, 0)·RPV01(0,tE,T)` (and receiver symmetric) — Black's `σ→0` limit, exact.
  **Companion `σ > 0`:** genuine time value, asserted material, and the swaption value monotone
  increasing in `σ`.
- **Black reused, and transcribed pointwise once.** The swaption is `black_price` on the forward
  spread — the reuse is the oracle for the Black machinery (verified since T5), but the **composition**
  (forward spread in, `σ²tE` total variance, `× RPV01(0,tE,T)` out, ω for payer/receiver) is new and
  gets its own pointwise transcription against O'Kane 9.3 (both renderings) plus the parity + zero-vol
  requirements (§4: a reuse still needs the new composition pinned).
- **Mutation (prove power).** Each must die: drop one `RPV01` term in the forward-spread difference
  (→ the `tF→t` reduction and flat-curve both still pass? no — dies on term structure; assert); swap
  `S(t,T)`/`S(t,tF)` (→ forward-spread sign/parity die); flip payer/receiver ω (→ parity dies, zero-vol
  intrinsic dies); drop the `RPV01(0,tE,T)` scaling on the swaption (→ parity dies, units wrong); use
  the **spot** spread `S(0,T)` not the **forward** `S(0,tE,T)` in Black (→ dies on a term-structured
  curve, not flat — the degenerate-fixture premise). **Premises:** term-structured spread curve,
  semi-annual accruals, `σ>0`, `K ≠ S(0,tE,T)` (so parity is not trivially `0=0`).

## 3. Process
Branch `tranche/40-forward-cds-default-swaption`. `test` (RED: the `tF→t` + flat-curve reductions +
the martingale 9.6 + put-call parity + the zero-vol reduction + the Black composition transcription +
the mutation table) → `feat` → `docs` → `chore(release): v0.47.0`. **Open the draft PR early;
pre-register the martingale, parity and zero-vol requirements first** (no implementation needed).
**Transcribe the forward spread (9.2) and the knockout swaption (9.3) in both renderings**; O'Kane
**derives** the forward spread from the two-spot-protection-leg difference (9.2.1) and the swaption
from the measure change (9.3.3) — use the derivation steps as the independent check (§4 stronger rung).
`source:` `OKANE-2008`; `crosscheck: quantlib` — QuantLib has a `CdsOption` / Black CDS-swaption engine
(a real target). **Raise an `Ask:`** on (a) whether the **non-knockout** swaption is in scope — it is
`knockout + front-end protection` (O'Kane's own decomposition; the front-end protection is a short
protection leg over `[t,tE]`, which `credit_legs` already computes, so it is cheap — rule fold-in vs
defer), and (b) where the forward-spread volatility `σ` is sourced (a flat model input now vs a quoted
CDS-swaption vol surface later — the latter is `to_go_deeper`).

## 4. What this tranche is NOT (scope fence)
**Knockout default swaption only** — the standard contract; **non-knockout** (= knockout + front-end
protection) is the `Ask:` above, folded only if cheap, else `to_go_deeper`. No **CMDS** (constant
maturity default swaps, ch 9.4 — `to_go_deeper`, as the arc plan said). No **CDS-swaption vol surface**
(a flat `σ` input; a quoted surface is `to_go_deeper`). No **multi-name** anything (phase 2, T41+). No
**CS01 / spread risk** sensitivities (ch 8 — a later risk tranche if pulled). T40 is the forward CDS and
the knockout default swaption, closing the single-name vertical; T41 opens the correlation half.
