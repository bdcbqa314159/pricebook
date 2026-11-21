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
| #8 (**open**) | v0.74.0 | T72 | **PENDING Code's sweep — (3)+(4) done** | **single-curve short-rate arc close** (T66–T72, `MCINERNEY-ZASTAWNIAK` revisit). Cowork register + charter passes done: MZ `worked-from`, **`BROCKHAUS-2016` now cites the CEV caplet** (MZ §8.3 is figure-only — recorded so not re-sought), the **Brockhaus §6.1 misprint logged** (first source *correction* in the project), arc plan closed, charter **clean and converged** (two advances in T66/T67, then T68–T72 grew it not at all). Sweep scope set oldest-first: **proactive invariant-6 FIRST and load-bearing — the tenth shape bit a FOURTH time at T72** (a premise written-but-wrong with `abs()` on an OTM fixture); the arc's reduction anchors (T66/T67 rate-in-a-regime per model); **the CEV reconstructed-formula anchor set** (requirement oracles + reductions + committed MC, swept hardest — misprint-corrected); the HJM drift requirement; MM-I re-run; full tree. **(2) duplication has a named target: C16's strip walk** across the four caplet engines (rule-of-two overdue; the extraction gives C16 its missing structural guard). Findings → `docs/audit/reaudit_08_findings.md`. |
| #7 (**complete**) | v0.66.0 | T64 | **1 blind oracle (fixed T65); 0 src defects** | **commodity-options arc close.** Cowork register + charter passes clean (`CLARK-2014` worked-from, arc plan closed, to_go_deeper routes recorded, charter clean — one arc advance, §4's obligation-not-form clause). Sweep scope set oldest-first (proactive invariant-6; the arc's new closed forms; MM-I; full tree). **Sweep 77 applied / 76 killed / 1 survived:** the invariant-6 run clean (31/31 across 23 modules), the arc's closed forms all pinned — **zero single-killers**, 10–38 killers each, the T43 magnitude mutation `ρ→ρ²` dying to 18 (both correlation instruments bite). The one hole **F-7.1** (a blind oracle fifteen tranches old: `reference_level`'s hard-coded `days_in_month=30`, invisible because the one pointwise fixture read a 30-day month *and* its reference wrote `/30` too — §4 independence satisfied formally, blind to the coincidence). Plus **F-7.2** four byte-identical ascending-time-grid guards (duplication), **F-7.3** the single-killer count hand-rolled beside the committed instrument (a timeout corrupted a baseline mid-mutation), **F-7.4** a non-probe (no action). **Remediated at T65** (v0.67.0, one test/tooling-only tranche): F-7.1 closed with a bracketed Feb/31-day fixture whose reference *derives* the month length; F-7.2 extracted to `foundation/time_grid.py` (guard-only, all four consumers tested); F-7.3 folded into the sweep instrument as a separate counting mode (its own known-answer tests then caught a *second* defect — counting from the `-x` verdict run would call every kill a single-killer, the false-positive mirror). Two charter clauses (§4 eleventh degenerate-fixture shape, §9 instrument single-killer ownership). `docs/audit/reaudit_07_findings.md`. |
| #6 (**complete**) | v0.60.0 | T58 | **6 findings, 2 real holes (both blind oracles), 0 src defects; all remediated T59** | **collateralised-discounting arc close.** Cowork register + charter passes done (T29's "collateral never increases" narrowed to a plausibility check — §4 floor rule on an oracle's own description; LICHTERS header closed). Sweep scope set oldest-first, proactive invariant-6 first, **the single-curve design invariant's reachability a NEW named sweep** (its failure is silent, T33b), **MM-I's owed second killer the highest-value single-killer target**, OI-6 a candidate to close in the remediation. **Sweep 76 applied / 74 killed / 2 survived:** items 1, 2 and 5 clean (24/24, 6/6, 24/24), MM-I's second killer added (1 -> 2 per denominator), **F-6.1** the LGM numeraire's half (inside a band measured against a *different* defect, and eq 11.1 never transcribed pointwise) and **F-6.2** T4's YoY `psi` (the FIXED leg's survives all 2323 tests; the two inflation-leg sites die only to another product's cross-check) - F-1.2's sibling site, the `tau=1` fixture again. Plus **F-6.3** Pyflakes never selected, **F-6.4**/**F-6.5** register, **F-6.6** the era selection written twice. **Remediated at T59** (v0.61.0, one tranche, the reaudit/05 precedent): F-6.1 0->21 killers, F-6.2's three sites 0->2 / 1->3 / 1->3 off ONE non-unit-`psi` fixture, MM-I 1->2 per denominator, `curves_for_era` single-homed at L1 (bit-identical), the T29 names narrowed, the table brought current, `F841` gated and 12 dead items removed, and **OI-6 closed - the ledger is empty again**. `docs/audit/reaudit_06_findings.md`. |

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

### The single-killer re-run table (standing)

Headed "Accumulating for re-audit #4" until re-audit #6 found it two re-audits stale (F-6.4) - re-audit #5's record went to `docs/audit/reaudit_05_findings.md` instead, so the table §9 names as the standing re-run target had stopped holding the targets. Renamed to say what it is; the rows after T44b are owed and are the remediation's to add.

Single-killer oracles recorded as they are found, so the sweep has its re-run targets ready rather
than rediscovering them:

| from | mutant killed by exactly one test | note |
|---|---|---|
| re-audit #3 | MM-I's `τ/(1+τ)` denominators (`:154` real, `:155` nominal) | load-bearing; **second killer added at T59** (re-audit #6) — a per-leg *saturation requirement*, since the existing companion dropped **both** denominators and a one-leg drop landed on a third value. Each now dies to 2 |
| re-audit #3 | **T3's LINKER** first `ψ` | already-shipped hole (F-1.2); given a second killer at T37 — **re-measured at 2 at re-audit #6, T37 holds**. *(This row read "the YoY leg's" until re-audit #6: two different products' `ψ` were conflated, and the hardening credited here belongs to the linker. The YoY module's own sites are the row below.)* |
| re-audit #6 | **T4's YoY `ψ` — three sites, the fixed leg's at ZERO killers** | F-6.2, and not a single-killer note but a **surviving mutant**: `yoy_fixed_leg_value`'s `ψ` died to nothing in 2322 tests (+100.5% semi-annual, +301% quarterly, +1117% monthly), while both `yoy_inflation_leg_value` sites died only to T13's cross-product oracle **in another product's file**. One `τ=1` fixture (`VAL + 365·k` on ACT/365F) caused all three. Closed at T59 by one non-unit-`ψ` schedule: 0→2, 1→3, 1→3 |
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

| T45–T49 | the Gatheral arc's refusals — SVI's five parameter guards and the `b ≥ 0`/`σ > 0` bounds, the volatility swap's non-negative-variance and positive-maturity refusals, Bates' jump-intensity and grid guards | guards, one condition each — inherent. No load-bearing single killer survived re-audit #5's sweep |
| T50–T53 | the adaptive grid's decay threshold and `checked_call_price`'s three branches; the sweep instrument's own guards | the truncation branches are one condition each; `checked_call_price`'s **raise** arm was hardened 1→2 in-tranche (dropping it still raises, so the message discriminates) |
| T54–T55 | the bootstrap's one-equation-one-unknown refusal, the anchor and non-positive-factor guards; the swap's `annuity == 0` refusal | guards, one condition each — inherent. Re-audit #6 swept the load-bearing closed forms (20 mutations) with one survivor, F-6.1, which is not a single-killer but a **zero**-killer |
| T57–T58 | the LGM's non-negative-horizon and maturity-order refusals; the CSA threshold's finite/non-negative guards; the capstone's straddling-date and too-short refusals | guards, one condition each — inherent. **F-6.1** (eq 11.1's `½`) had **no** killer at all until T59's pointwise transcription took it to 21 |
| T60–T64 | **the whole commodity arc's closed forms** — the carry forward, both Asian double sums, the TW moment ratio, the spread cross term (including `rho -> rho^2`, the magnitude-only mutation), the Kirk weight, the Schwartz variance factor and forward decay | **zero single-killers**: measured at re-audit #7, every load-bearing target dies to **10-38** tests (lowest: T64's forward decay at 10; highest: T64's `1/(2k)` at 38). The two correlation instruments both bite independently -- `rho -> rho^2`, which preserves presence AND sign and changes only magnitude (T43's trap), dies to 18. Counts measured, not guessed |
| T59 | `curves_for_era`'s two branches | 42 killers — the era lookup is read by every swap and capstone oracle, so single-sourcing it (F-6.6) made it the most heavily covered line in the arc |

*(The five rows above were owed since re-audit #4: the table was headed "Accumulating for re-audit #4"
and stopped at T44b while #5's record went to its own findings doc — F-6.4, added at T59.)*

T38's two load-bearing single killers were **hardened in-tranche** rather than recorded: the
protection-leg quadrature rule (1 → 3 killers, via a closed-form anchor) and the direction factor
(1 → 2, via a zero-spread sign asymmetry). The two that remain are refusal guards, where one
assertion is the whole behaviour — distinct from a numerical coefficient, which can be wrong by a
factor no single test pins.
