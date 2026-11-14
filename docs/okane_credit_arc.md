# The credit-derivatives arc — project five (`OKANE-2008`)

**Proposed by Cowork for Bernardo, 2026-11-14 ("a good and solid spine of O'Kane").** The fifth
vertical: the canonical **credit derivatives** library — the CDS and its curve, then the multi-name
correlation products. Like the four closed arcs: build the oracle-able spine, ledger the rest, close
with a checkpoint.

## The thesis — why this is the natural next arc, and a second capstone
The XVA arc (T25–T29) built the survival curve and the exposure/CVA machinery but **deliberately
deferred the CDS itself** — T25's note: *"inverting market CDS quotes for λ is a curve bootstrap,
arriving with the product that needs it."* O'Kane is that product's book. And the reuse is deep enough
that this is a **second capstone**, not a greenfield:

| O'Kane | we already hold | reused by |
|---|---|---|
| ch 2 Libor curve | discount curves (T1/T8) | everything |
| ch 3 hazard / Cox / survival `Q(t)` | `market/survival_curve.py` (T25, from CZ ch 2) | every credit product |
| ch 6 protection leg `(1−R)∫Z·λQ dt` | **the CVA integral** `LGD·∫DF·λS dt` (T25) | the CDS — same shape, cross-product oracle |
| ch 7 CDS bootstrap | the **T8 nominal par-swap bootstrap** round-trip pattern | the CDS curve |
| ch 9 CDS swaption | `models/black` (T5/T30) on the forward spread | the default swaption |
| ch 13/16 Gaussian copula, LHP tranche | **Φ / Φ₂** (T36b), `norm_cdf` | the correlation products |
| ch 15/18 baskets, loss distribution | the **Jäckel MC engine** (T15–T22) | multi-name simulation |

So the arc spends its new effort on the genuinely new credit objects (the CDS legs, the bootstrap, the
copula, the tranche) and *connects* the survival curve, the CVA integral, Black, and the Gaussian
machinery that four arcs already built. Method-fit is strong: O'Kane is a practitioner text full of
closed forms (the credit triangle, RPV01, the LHP tranche) — reduction and requirement oracles
everywhere.

## The two phases of the spine

### Phase 1 — single-name: the CDS vertical (the clear, oracle-rich core)
- **T38 — the CDS: premium + protection legs, par spread, RPV01 (ch 5–6).** The premium leg `S·RPV01`
  (risky annuity + accrual-on-default) and the protection leg `(1−R)∫Z·λQ dt`; the mark-to-market
  `V = (S(t,T) − S(0,T))·RPV01`. **Oracle:** the **credit triangle** `S = λ(1−R)` (exact in the flat-
  hazard continuous limit — a closed-form reduction anchor); the protection leg reconciled against the
  **T25 CVA integral** (different code each side — the cross-product oracle); par-spread-reprices-to-zero
  as the self-consistency guard (telescoping, so the triangle + the CVA cross-check are the correctness
  anchors). Reuses the `SurvivalCurve` and the discount curve whole.
- **T39 — the CDS survival-curve bootstrap (ch 7).** Invert a strip of par CDS quotes for the
  piecewise hazard curve — the L1 bootstrap T25 deferred (it *produces* market data, so L1 in
  `market/`, per §2's calibration split). **Oracle:** the **known-curve round trip** (bootstrap from
  quotes, then reprice every CDS to par = 0) — exactly T8's nominal-bootstrap anchor, and the one that
  caught two misprints in BERHAN §5.1; plus arbitrage detection (survival monotone, hazards ≥ 0).
- **T40 — forward CDS + the default swaption (ch 9).** The forward-starting CDS spread and the CDS
  option under Black's model, with RPV01 as the annuity numéraire. **Oracle:** reduction to the spot
  CDS as the forward start → 0; put-call parity on the payer/receiver swaption; Black reused (T5).
  *(CS01 / analytic sensitivities, ch 8, fold in here or defer — ruled at the spec.)*

### Phase 2 — multi-name: correlation products (the copula half)
- **T41 — the one-factor Gaussian copula / latent-variable model (ch 13).** Conditional independence:
  each name defaults when a Gaussian latent variable crosses its threshold `Φ⁻¹(1−Q_i)`, correlated
  through one market factor. **Oracle:** the conditional default probability closed form; the
  `ρ→0` reduction to independent defaults; reused `Φ`/`norm_cdf`.
- **T42 — default baskets: first-to-default (ch 15).** The FtD basket spread via the copula.
  **Oracle:** reduction to a single-name CDS as the basket size → 1; the independent-case closed form
  at `ρ=0`; an independent MC (the Jäckel engine) within SE.
- **T43 — the LHP tranche closed form (ch 16).** The large-homogeneous-pool tranche spread — a
  beautiful closed form in `Φ`/`Φ₂`. **Oracle:** the LHP formula transcribed pointwise + the
  reductions (equity tranche → expected loss as detachment → 1; `ρ→0`); the tranche survival curve.
- **T44 — the recursive loss distribution + base correlation (ch 18, 20).** The exact conditional
  loss recursion (vs the LHP approximation — a measured gap, the §4 two-method companion), and the
  base-correlation curve the market quotes tranches on. **Oracle:** the recursion vs LHP vs MC (three
  methods); base-correlation round-trip (strip → reprice tranches).
- **Checkpoint** `CP_okane_credit.md` closes the arc; its close fires re-audit #4 (§9).

## The honest completion ratio (proposed)
Built as the spine (phase 1 and/or 2 per Bernardo's scope): the CDS + curve + swaption, and — if the
full spine — the Gaussian copula, baskets, LHP tranches, the loss distribution and base correlation.
Deferred to `to_go_deeper` (oracle-able, unpulled): **bonds & asset swaps** (ch 4), **CDS indices +
index options** (ch 10–11 — a clean later pull, Black revisited), **CMDS** (ch 9.4), **implied &
compound correlation** (ch 19), the **copula-skew zoo** (ch 21 — student-t, double-t, Marshall–Olkin,
random-factor-loading, implied copula — a correlation sub-arc), **CPPI / CPDO / CDO² / tranchelets**
(ch 22). **`out_of_method` candidate:** the **dynamic credit models** (ch 23 — intensity-gamma, affine
jump-diffusion) if their market calibration has no closed form or requirement-identity anchor, only a
twin-MC — ruled at the §1 gate if interest ever pulls there.

## Decision — the FULL credit spine is committed (Bernardo, 2026-11-14)
Bernardo chose the **full credit spine (T38–T44)** over the compact single-name or middle options — the
canonical credit-derivatives library, single-name *and* the multi-name correlation core, the second
capstone. **The queue:** T38 (the CDS) → T39 (the CDS survival bootstrap) → T40 (forward CDS + default
swaption) → T41 (the one-factor Gaussian copula) → **T41b (survival re-keyed by ReferenceEntity — §3 prep for multi-name, ruled at T41's review)** → T42 (first-to-default basket) → T43 (the LHP tranche)
→ T44 (recursive loss distribution + base correlation) → `CP_okane_credit.md` (closes the arc, fires
re-audit #4). Cowork writes each per-tranche spec before its build (§10); this is the queue, not the
specs. **T38 is specced** (`docs/tranches/38_credit_default_swap.md`); the rest follow in turn.
