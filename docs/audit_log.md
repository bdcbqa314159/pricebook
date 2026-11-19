# Audit log — the periodic whole-tree re-audit

Charter §9: a full re-audit **every 11 tranches, and at each arc close**, run with independence (the
`analysis/` read-only copy + `AUDIT_FINDINGS.md` pointed at current `main`). It targets what the
per-push gates cannot — oracle power (a full-tree mutation sweep), cross-tranche consistency and
duplication, register accuracy (`to_go_deeper`/`OPEN`/`references`/arc plans), and charter staleness.
Findings are ruled by Cowork and become tranches or ledger entries.

This file is the **counter**, so the trigger is mechanical, not remembered.

| re-audit | at version | tranche | findings | notes |
|---|---|---|---|---|
| — | — | — | — | mechanism established 2026-11-07 (cadence 11, Bernardo). |
| #1 (**complete**) | v0.27.0 | T22 | **2 HOLES + 1 MED + 3 LOW** | first re-audit, Jäckel arc close, whole tree T1–T22; run independently of the builder from two fresh contexts (Code built all 22), findings → `docs/audit/reaudit_01_findings.md`, ruled by Cowork. See below. |
| #2 (**complete**) | v0.35.0 | T29 | **CLEAN — 0 holes + 1 LOW (fixed)** | **XVA arc close** (arc-boundary trigger, not the cadence counter — eight tranches since #1). Whole tree **T1–T29**, independent. No oracle holes, no silent weakening from the XVA refactors, charter clean; one LOW (stale conventions-register summary) fixed in place. Findings + rulings → `docs/audit/reaudit_02_findings.md`. |
| #3 (**complete**) | v0.44.1 | T37 | **CLEAN src — 0 holes; 2 blind oracles (fixed T37)** | **equity & hybrids arc close.** Whole tree T1–T36b; oldest-first. No `src/` defect; the τ=1 degeneracy bit a second and third instrument (F-3.1 bootstrap, F-3.2 vol surface — blind oracles, test-only fix at T37). Mechanical τ=1 guard assessed and declined (dynamic, 7-of-14). One charter edit (§9 gateability boundary). `docs/audit/reaudit_03_findings.md`. |
| #4 (**complete**) | v0.52.0 | T44b | **5 blind oracles (fixed T45); 0 src defects** | **credit-derivatives arc close.** Register + charter passes clean (OKANE + CREPEY stale headers fixed; CREPEY stale since #2, missed twice). The proactive invariant-6 sweep's **first run found 5 holes** (F-4.1…F-4.5 — expired-option bypasses + YoY comprehensions), all blind oracles, remediated test-only at **T45** (v0.52.1). `docs/audit/reaudit_04_findings.md`. |
| #5 (**complete**) | v0.55.0 | T49 | **3 blind oracles + 1 shipped defect (OI-4); all remediated T50–T53** | **volatility-surface arc close.** Sweep 71/68/3: the proactive invariant-6 run clean (23/23), three blind oracles F-5.1/5.2/5.3 (put arms + a Dupire-band discounting hole), and **OI-4** — a negative Heston/SVJ price on `main` since T33, mapped to a quoted strike. Remediated: **T50** (OI-4 adaptive truncation, closed), **T51** (the three hardenings, test-only), **T52** (the mutation sweep committed as an instrument, which found a 4th instrument clause on its first run), **T53** (φ-across-strikes, closing OI-5). `OPEN.md` empty on T53 merge. MM-I single killer → re-audit #6. `docs/audit/reaudit_05_findings.md`. |
| #6 (**open**) | v0.60.0 | T58 | **(3)+(4) DONE: T29 oracle narrowed, 1 stale header; (1)+(2) PENDING Code** | **collateralised-discounting arc close.** Cowork register + charter passes done (T29's "collateral never increases" narrowed to a plausibility check — §4 floor rule on an oracle's own description; LICHTERS header closed). Sweep scope set oldest-first, proactive invariant-6 first, **the single-curve design invariant's reachability a NEW named sweep** (its failure is silent, T33b), **MM-I's owed second killer the highest-value single-killer target**, OI-6 a candidate to close in the remediation. `docs/audit/reaudit_06_findings.md`. |

## Re-audit #2 — result (2026-11-10, v0.35.0, T1–T29)

A **clean bill**, cleaner than #1. Fired by the XVA arc close. Findings/rulings in
`docs/audit/reaudit_02_findings.md`.

- **Oracle power: CLEAN.** ~48 mutations, 0 holes; one proven-equivalent survivor. The signature-defect
  checks — did generalizing T25 into `credit_adjustment` (T27), extending T26's engine with collateral
  (T29), or the `is_historical` atom across five engines weaken an older oracle? — all held. F-1.2
  confirmed closed; the new `references` gate itself powered.
- **Duplication: CLEAN** (F-2.1 still one consumer short, correctly open).
- **Register accuracy: 1 LOW.** F-3.1 — the conventions-register "Immediate reading" summary was T24
  prose not updated under T26 (called C10 unguarded, counted five conventions). Rows were correct;
  summary **fixed in place**, no fix-tranche.
- **Charter staleness: CLEAN.**

## Re-audit #1 — result (2026-11-07, v0.27.0, T1–T22)

Two independent mutation passes (fresh contexts; mutants regenerated, not reused — Code's T15–T22
sweeps did not survive its session) plus the three tree-wide judgment passes. Findings and rulings in
`docs/audit/reaudit_01_findings.md`.

- **Oracle power — Jäckel arc (T15–T22): CLEAN.** 26 mutations, 0 holes; 2 proven-equivalent survivors.
- **Oracle power — inflation arc (T2–T14): 1 HOLE.** F-1.2 [MED] — the linker coupon `ψ` accrual
  survives a drop under all 818 tests (every fixture `ψ=1`, the §4 `τ=1` trap); production code
  correct, oracle degenerate; ~8.4% latent mispricing on a semi-annual linker. → **fix-tranche T23**
  (oracle hardening). *Found only because the sweep was extended past the first pass's domain — the
  §4 rule applied to the audit itself.*
- **Duplication & consistency:** 2 LOW (F-2.1 driftless-forward terminal inlined twice → rule-of-two,
  `to_go_deeper`; F-2.2 discount-from-forwards ×3 but different shapes → no action).
- **Register accuracy:** 1 MED + 1 LOW. F-3.1 — `conventions_register.md` C1–C6 describe the archived
  predecessor tree → **fix-tranche T24** (rebuild). F-3.2 — `to_go_deeper` stale lines → folded into T24.
- **Charter staleness: CLEAN.**

**Remediation: COMPLETE.** **T23** (v0.28.0) closed F-1.2 — the linker accrual oracle, test-only, the
drop-`ψ` mutant reproduced then killed. **T24** (v0.29.0) rebuilt the conventions register, corrected
`to_go_deeper`'s two stale lines and recorded F-2.1.

**T24 found a SECOND hole the audit had not, which is why the count above reads 2.** Re-deriving the
invariant-6 row meant mutating the valuation-date boundary in all five engines rather than reading
them: `>` → `>=` **survives** in three. Two of those three (`nominal_cap`, `yoy`) are **proved
equivalent mutants** — bit-identical prices across four schedules including the boundary, because a
Hull-White caplet whose expiry has passed is worth exactly zero. The third (`capfloor`) was a **real
41% mispricing** (5,929 → 8,364 on a schedule paying exactly on the valuation date), closed by
`test_INVARIANT_6_excludes_a_period_PAYING_EXACTLY_on_the_valuation_date`.

Worth recording for re-audit #2: the finding came from **re-deriving a register row by mutation
instead of by reading the code**. The register rebuild was scoped as a docs tranche; the hole was a
by-product of doing it honestly. Target 3 (register accuracy) and Target 1 (oracle power) are less
separable than the §9 list suggests.

**Next due (superseded — re-audit #2 ran at the T29 arc close; see its result above):** re-arms at
whichever of {next arc close, 11 tranches since **re-audit #2** at T29 ≈ **T40**} comes first.

---

## Re-audit #3 — equity & hybrids arc close (v0.44.0, T36b, 2026-11-13)

Fired by the **arc-close trigger** (the eleven-tranche cadence counter is at nine since re-audit #2 at
T29: T30, 31, 32, 33, 33b, 34, 35, 36a, 36b — so arc-close fires first, as at #2). Findings:
`docs/audit/reaudit_03_findings.md`.

- **Register-accuracy + charter-staleness passes (Cowork): clean.** One stale `to_go_deeper` header
  fixed (BROCKHAUS arc marked closed; the 13.1.2-for-the-stock route added); one overdue code-hygiene
  route re-flagged (the dividend-filter consolidation, `dividends_before(T)`, never done across the
  arc). `references.md`, `OPEN.md` (zero suppressions), the arc plan all current. No charter clause
  stale or in tension — the arc's lessons were ruled instances of existing clauses, so the charter
  covered new cases rather than growing.
- **Oracle-power mutation sweep (Code): DONE — 31 mutations, 30 killed, ONE survivor.** **F-3.1**: the
  T8 nominal bootstrap drops the accrual `ψ` and survives all 1430 tests — the `τ=1` degeneracy (every
  fixture ψ=1.0 on annual ACT/365F), **F-1.2's exact shape in a second, pre-Jäckel instrument**. The
  `src/` code is correct; the oracle is blind. **Ruled a real oracle-power hole → T37** (test-only:
  semi-annual fixture + `ψ≠1` premise, the T23 remedy; **plus a proactive τ=1 sweep of all
  accrual-bearing fixtures**, since the degeneracy has now bitten twice; plus a second killer for the
  thinnest single-point-of-failure oracle, the Sobol Gray code). The **oldest-first scoping was
  vindicated** — all eight T30–T36b mutations died; the only hole was pre-Jäckel. Re-audit #3 **logged
  complete**; the arc boundary opens once T37 lands.
- ~~**Oracle-power mutation sweep (Code): PENDING**, scope set~~ — full-tree, **oldest-first** (§9's
  binding emphasis; re-audit #1's only hole was in the oldest instrument), single-point-of-failure
  oracles flagged, the new L0 `Φ₂` and the equity closed forms swept for the first time. Survivors
  ruled by Cowork, appended to the findings doc.

**Addendum (T37 build, same day):** the proactive τ=1 sweep ordered above immediately found the
**third** instance — **F-3.2**, the T9 implied-vol surface dropping `ψ` (annual fixture, measured 9.00%→
5.80% vol understatement semi-annually). τ=1 has now bitten three pre-Jäckel instruments (F-1.2 linker,
F-3.1 bootstrap, F-3.2 vol surface) from one fixture convention. Ruled: fold F-3.2 into T37 (same class,
same remedy), make the sweep exhaustive (fix same-class survivors in-tranche, raise only different-class),
and — the T26b "gate a thrice-drifted discipline" precedent — **assess a mechanical guard** for the
all-unit-accrual fixture convention, building it in T37 if a clean mechanism exists. Four single-killer
oracles recorded (Gray code, MM-I denominator, YoY ψ, T26 Leg B); second killers added to the thinnest.
See `reaudit_03_findings.md`.

**CLOSED (T37, v0.44.1, 2026-11-14).** Remediation landed, test-only: F-3.1 and F-3.2 fixed (semi-annual
fixtures + `ψ≠1` premise, mutants die); the τ=1 sweep was exhaustive (every accrual multiplication in
`src/` mutated — no third same-class and no different-class survivor); the Gray-code second killer added
(a different method); MM-I/YoY recorded for re-audit #4. The **mechanical guard was assessed with two
built prototypes and declined** — τ=1 load-bearing-ness is a dynamic property no static gate can see
(7-of-14 false positives), so it stays the mutation sweep's job; charter §9 gained the gateability
boundary, guard → `to_go_deeper`. Re-audit #3 is complete; no `src/` defect was ever present (all holes
were blind oracles), no new debt, one charter edit (the §9 gateability boundary).

**Next due (superseded — re-audit #4 ran at the credit arc close; see below):** re-arms at whichever of
{next arc close, 11 tranches since **re-audit #4**} comes first.

---

## Re-audit #4 — the credit-derivatives arc close (v0.52.0, T44b, 2026-11-16) — COMPLETE (remediated T45: 5 blind oracles)

Fired by the **arc-close trigger** (the cadence counter sits at ten since re-audit #3 at T37; arc-close
fires first, as every time). Full scope and the two completed Cowork passes:
`docs/audit/reaudit_04_findings.md`.

- **(3) Register accuracy — DONE (Cowork).** Two stale `to_go_deeper` arc headers fixed in place: the
  `OKANE-2008` header ("T38–, in progress" → "T38–T44b CLOSED v0.52.0") and the **`CREPEY-2026` header**,
  stale ("T25–, in progress") since the XVA arc closed at re-audit #2 (T29) and **read past by re-audits
  #2 and #3**. `references.md`/`OPEN.md`/`conventions_register.md`/arc plan all clean (and the register's
  own id/citation integrity is now gated by `verify.py conventions`, T41c). A mechanical register-header
  gate is flagged as a candidate (statically decidable, thrice-adjacent) — not built, for a ruling.
- **(4) Charter staleness — DONE (Cowork): clean.** The arc's §4/§9/§3.5 additions describe reality;
  the deliberate non-additions (three dissolved questions, the vacuous-oracle restatement) left the
  charter leaner-relative-to-coverage, as at the equity capstone.
- **(1) Oracle power — PENDING (Code), scope set oldest-first.** FIRST named item: the proactive
  invariant-6 / historical-flow sweep (the tenth-shape escalation's first run — drop the filter on every
  module that has one, confirm a straddling fixture kills it; **verify, do not trust, the T43/T44a/T44b
  straddle claims**). Then the single-killer re-run table (below), the new L0/credit closed forms (first
  sweep — probe for the T43 scale-invariance and T39 reduction-anchor blind spots), then T1–T37.
- **(2) Duplication — PARTIAL.** The T44a tranche-composition duplication (flagged, likely resolved by
  `models/tranche_loss.py`'s one protocol + free functions — **Code verifies**, content until shown);
  `dividends_before(T)` still owed (no equity touch this arc); C2 still the widest OWED convention.

**Conduct rule for #4 (this arc's own lesson):** a surface's report that it did something is content, not
the artifact — Code re-derives each premise it reports; a sweep "no survivors" is evidence only of what
was swept.

### Accumulating for re-audit #4

Single-killer oracles recorded as they are found, so the sweep has its re-run targets ready rather
than rediscovering them:

| from | mutant killed by exactly one test | note |
|---|---|---|
| re-audit #3 | MM-I's `τ/(1+τ)` denominator | load-bearing; second killer still owed |
| re-audit #3 | the YoY leg's first `ψ` | already-shipped hole (F-1.2); given a second killer at T37 |
| re-audit #3 | tranche 26's Leg B numéraire divisor | single **by design** — one leg, one identity |
| T38 | `CreditModel`'s recovery-range refusal | a guard with one behaviour; single killer is inherent |
| T38 | `CreditModel`'s shared-clock refusal | same shape |
| T39 | the leg's non-positive-horizon guard and the bootstrap's five refusals (arbitrage, its `>` boundary, maturity order, empty strip, recovery range) and the leg's horizon guard | six guards, one condition each — inherent |
| T40 | seven refusals across the forward CDS, the swaption and its model (forward start on/before the schedule, expiry on the schedule, positive strike, positive forward annuity, par-spread annuity, finite non-negative vol) | guards, one condition each — inherent. The three *load-bearing* single-killers (the direction factor, and `σ²` vs `σ²t` twice) were hardened in-tranche |
| T41 | three copula refusals (`|β| ≤ 1`, non-empty portfolio, shared clock) | guards, one condition each — inherent. The three load-bearing single-killers (the unit-β indicator twice, the name index) were hardened in-tranche |
| T42 | four basket refusals (two-name minimum, duplicate entities, mixed unit/sub-unit loadings, recovery range) and the missing-entity check | guards, one condition each — inherent. The three *load-bearing* survivors (notional scaling, the contract's entity list, the invariant-6 filter) were all **fixture gaps**, closed in-tranche |
| T43 | the LHP's recovery-range refusal, the tranche bounds' unit-interval refusal, the invariant-6 filter, the notional scaling, and the pool-entity mismatch | guards and scalings, one condition each. The three survivors shared a cause — **every oracle was relative**, so the tranche's *level* was unpinned while its shape was over-determined; closed with absolute-level oracles in-tranche |
| T44a | the pool-size minimum | one guard, one condition. Both survivors were about where an oracle **lives**: the width division was already fixed at T43 but only against the LHP model, and the step count survived because its own test **telescopes** (it reads the model's committed count). The tranche composition being duplicated across the two models is the reason the first was reachable — flagged for re-audit #4 |
| T44b | the empty-quote-set, positive-spread and unquoted-strike refusals | guards, one condition each — inherent. **No survivors.** The reachability refusal (the tranche's non-vacuous arbitrage oracle) was hardened 1→2: dropping it still *raises*, so only a test asserting the message is ours and names the reachable range discriminates |

T38's two load-bearing single killers were **hardened in-tranche** rather than recorded: the
protection-leg quadrature rule (1 → 3 killers, via a closed-form anchor) and the direction factor
(1 → 2, via a zero-spread sign asymmetry). The two that remain are refusal guards, where one
assertion is the whole behaviour — distinct from a numerical coefficient, which can be wrong by a
factor no single test pins.
