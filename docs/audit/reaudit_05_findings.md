# Re-audit #5 — the volatility-surface arc close (v0.55.0, 2026-11-17)

Fifth whole-tree re-audit (§9), fired by the **arc-close trigger** (three builds since re-audit #4 at the
O'Kane close — T46, T48, T49, with T47 dropped — so arc-close fires, not the eleven-cadence). Run with
independence: Cowork sets scope and rules; the mutation sweep is Code's execution; the register-accuracy
and charter-staleness passes (no execution) are done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree sweep — Code, PENDING), **(2) duplication**, **(3)
register accuracy**, **(4) charter staleness**. (1) and (3) run together.

**Conduct rule (the arc's own lessons):** a surface's report that it did something is content, not the
artifact (§4) — Code re-derives each premise it reports; and a sweep "zero survivors" is read only against
a full applied count **and** a confirmation the mutated code is the code that ran (the T46 bytecode clause).

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9), in order:

1. **FIRST NAMED ITEM — the proactive invariant-6 / historical-flow sweep** (the tenth-shape escalation's
   standing run, now at every arc close). Drop the filter on every module that has one; the vol arc added
   `engine/vanilla_option.py` and the Heston/SVJ pricers to the population. The first-shape twice this arc
   (T46 OTM intrinsic 0, T48 zero rate) means **also vary the rate off zero and strikes into the money** in
   the swept fixtures — the same guard's two blind parameters.
2. **OI-4 — characterise, do not just confirm.** The Heston/SVJ Fourier grid's negative call price in the
   far short-dated high-vol wing is OPEN debt (found at T49). Re-audit's job here is the §8 severity sweep:
   **map the defect's domain** (the (σ_v, T, K) region where the committed grid gives a wrong — not merely
   imprecise — number), so the remediation tranche is scoped to a measured region, not one sample. This is
   the "a reference is code too" hole (T20a/T25); the sweep confirms the oracle's reference grid is itself
   the weak side there.
3. **The single-point-of-failure re-run table** (`docs/audit_log.md`'s "Accumulating" list) — every
   load-bearing mutant recorded as killed by exactly one test; a deleted killer reopens a hole green.
4. **The new vol-arc closed forms, first sweep** — the SVI fit and its arbitrage conditions (T46), the
   volatility-swap convexity (T48), the SVJ double reduction and `engine/fourier_inversion.py` (T49). Probe
   the T43 scale-invariance and T39 reduction blind spots this arc has repeatedly paid; confirm the double
   reduction's asserted mutual blindness holds (a jump mutation leaves A green and kills B, vice versa).
5. **The rest of the tree (T1–T37)** as the standing full-tree sweep.

Survivors ruled by Cowork, appended here. OI-4's remediation is a **separate tranche** out of this
re-audit (the T45-from-#4 precedent), its **fix shape ruled then** — adaptive/regime-aware truncation vs a
wider constant is a design decision, not pre-committed in the ledger (Code's 0281 flag, agreed).

## (2) Duplication — PARTIAL (Cowork read; Code sweep informs)

- **`engine/fourier_inversion.py` — the Gil-Pelaez extraction (T49) is the rule-of-two done right** (second
  consumer, §6; the alternative broke §7). Confirm no third copy lingers and the two consumers (Heston
  probabilities, SVJ) both call it.
- **C2 (vol lives on one clock) — still OWED, still the widest-unguarded**, and the vol arc *added* readers
  (the SVI slice reads `T` to form total variance — C14 sits directly on C2). Carried; the standing item.
- **`dividends_before(T)` consolidation — still owed** (no equity touch this arc). Carried.

## (3) Register-accuracy pass — DONE (Cowork)

- **`to_go_deeper.md` — the `GATHERAL-2006` header fixed in place** ("project six, in progress" →
  "T46–T49, CLOSED v0.55.0"), the standing register-pass fix (the BROCKHAUS/OKANE/CREPEY precedent). The
  three arc routes (ch 7 asymptotics, variance options, G–J 2014) are current.
- **`references.md` — clean.** `GATHERAL-2006` is `worked-from` with `src/` citers (flipped at T46); the
  `verify.py references` gate passes by construction.
- **`OPEN.md` — clean and correct.** OI-1/2/3 CLOSED, **OI-4 OPEN** (the T33 truncation defect) with its
  measurement, interim exclude-with-reason (itself guarded by a test asserting the grid *does* disagree, so
  it cannot become a silent hole), and re-open trigger = this re-audit. The `debt` gate counts only `- [NG-`
  prefixes, so the `### OI-` heading does not unbalance it (Code confirmed).
- **`conventions_register.md` — clean and gated.** C14 (total-variance dimension, T46) is `guarded`
  (the Heston↔SVI-through-`black.py` end-to-end path); `verify.py conventions` holds the id run.
- **Arc plan (`gatheral_vol_surface_arc.md`) — current.** T47 struck, queue closed, completion ratio
  matches the checkpoint.

## (4) Charter-staleness pass — DONE (Cowork)

- **No clause stale or in tension.** The two arc additions (§4 bit-identity-is-a-measurement; §9
  mutated-code-is-the-code-that-ran) describe reality and each cite the earning tranche. The deliberate
  non-additions (the double reduction = T39; the blind variance-option set = T39+T43; the analytic butterfly
  = requirement oracle) left the charter leaner-relative-to-coverage, as at the last two capstones.
- **Two method-watch items logged (not yet charter), tracked for the next arc:**
  1. **The lost-glyph recovery pattern — two instances, classified by authority**, not just counted: T46
     from a *later chapter of the same source*, T48 from *a prior tranche's oracled code*. A **third** of
     the T48 kind earns a §4 line distinguishing oracled prior-tranche code (admissible) from a parked
     `docs/` number (not). Tracked here so the third classifies rather than merely counts.
  2. **The catalogue's first shape ("a zero makes a factor invisible") twice this arc** (T46 OTM intrinsic
     0, T48 zero rate). A **third** in a single arc opens the question of a proactive first-shape sweep (the
     tenth-shape precedent); the counter is what is kept. Folded into item 1 of the oracle-power scope above
     as the rate/moneyness fixture variation.

---

## (1) Findings — Code's execution, in progress

**Conduct:** every sweep below cleared `__pycache__` per mutant (the T46 clause — a mutation written
inside one mtime tick leaves the `.pyc` valid and the interpreter runs the *original* while printing a
verdict for the mutant), and every count is reported as **applied/killed/survived** so a zero is read
against a full applied count.

### Item 1a — the proactive invariant-6 / historical-flow sweep: **CLEAN**

**applied 23/23, killed 23, survived 0.** Every `is_historical` call site in `src/` dropped in turn
(the early-return form → `if False:`, the comprehension-filter form → `if True`, `all(...)` → `False`),
full suite per mutant. The population is 23 sites across 18 modules, including the vol arc's additions
(`engine/bates_fourier.py`, `engine/volatility_swap.py`, `engine/heston_fourier.py`,
`engine/variance_replication.py`, `engine/equity_vanilla.py`).

Two sites (`engine/yoy.py:136` and `:152`) are **textually identical lines** in different functions, so
a string-replacement sweep refuses them as non-unique — they were re-run by line index rather than
counted as killed (§9: a mutant that did not apply is not a mutant that died). Both die. This is a
small methodological note for the standing sweep: identical guard lines need positional mutation.

**Reading:** T45's remediation of F-4.1…F-4.5 holds, and the arc's new engines were built with the
premise already asserted (T48 at a 3% rate, T48/T49 with in-the-money expired fixtures). The
escalation's second run is clean.

### Item 1b — the FIRST-shape sweep (a zero makes a factor invisible): **3 HOLES**

Cowork's emphasis ("vary the rate off zero") is the first shape, and the fixtures already carry
non-zero rates (0.04 Heston, 0.025 SVJ, 0.06 local vol) — so instead of varying a fixture the sweep
**drops the discount factor itself** at every terminal discounting in `engine/`, which is the exact
mutation that found T48's hole. **applied 15/15, killed 12, survived 3.** Not a zero-rate problem in
the end: all three are *unexercised code paths*.

| # | site | why it survived | severity |
|---|---|---|---|
| **F-5.1** | `engine/heston_fourier.py:136` — the **put** branch's parity discounting | **No test prices a PUT through the registered Heston pricer.** The branch *condition* is covered (a mutation reversing it kills, because a call then takes the put arm), but the put **arm** is never taken, so its `df` is unguarded. From **T33 — sixteen tranches on `main`**. | a put mispriced by `1/df` |
| **F-5.2** | `engine/bates_fourier.py:117` — the same parity discounting | Same cause, inherited by building T49 in T33's shape. **And `test_bates_svj.py` *does* assert put-call parity** — but it computes the put **in the test** from `bates_call_price`, so it never enters the engine's put arm. *Asserting parity in a test does not exercise the engine's parity code.* | same |
| **F-5.3** | `engine/local_vol_mc.py:151` — `surface_call_price`'s discounting | **The MC repricing band absorbs it.** Measured: the band is `4·SE` = **11.8% / 18.5% / 33.1%** of the price at `K/F = 0.85 / 1.00 / 1.15`, while a dropped discounting is **6.18%** (`1/DF − 1` at a 6% rate over one year) — inside the band at every strike. From **T32**. | a 6.2% pricing error invisible |

**F-5.3 is the sharpest of the three, and it is a resolution claim that was true and insufficient.**
The same file carries `test_the_repricing_band_is_tight_enough_to_RESOLVE_a_dupire_defect`, which
measures that dropping Dupire's `(r−q)K ∂C/∂K` term moves the worst repricing by **8.9 SE** — a
correct, measured, §4-compliant resolution statement. It is simply about a *different defect*: the
band resolves an 8.9-SE Dupire error and cannot resolve a 1.4-SE discounting error. **A band's
resolution is a property of the defect it was measured against, not of the band** — stated as a
candidate for §4 beside "assert the band's resolution", because the existing clause reads as
satisfied here and the hole is real. Ruling is Cowork's.

### Item 2 — OI-4 characterised: the defect region is MAPPED, and it reaches a liquid strike

The §8 severity-by-domain sweep, over `σ_v ∈ {0.3, 0.6, 0.9, 1.2} × T ∈ {7, 30, 90, 365}d ×
K/F ∈ {1.0, 1.15, 1.3, 1.6}`, committed grid (`upper=200`) against `upper=1600, count=32001`:
**12 of 64 points are wrong** — a negative price, or >1% relative.

- **Negative call prices** (arbitrage violations) at `σ_v ≥ 0.6`, `T ≤ 30d`, `K/F ≥ 1.15` — five points,
  e.g. `σ_v=0.9, 30d, K/F=1.3`: committed **−2.478e-04** against reference **+1.651e-05**.
- **Catastrophic relative error** further out: `σ_v=0.9, 30d, K/F=1.6` reads **1.928e-04** where the
  truth is **8.352e-10** — five orders of magnitude, a price invented from truncation residue.
- **And it is not confined to an exotic corner.** At `σ_v ∈ {0.9, 1.2}`, **30 days, K/F = 1.15** the
  error is **4.0% / 4.4%** on prices of 4.5e-03 / 8.6e-03 — a one-month 115% strike is a *quoted*
  instrument, not a tail curiosity. Even `σ_v = 0.3` (a modest vol-of-vol) is **53.9%** wrong at
  30d/1.3.
- **The boundary:** trouble begins where `T ≤ 90d` meets `K/F ≥ 1.15`, worsening monotonically in
  `σ_v`; **nothing fails at `T = 365d`** at any tested `σ_v` or strike.

That is the measured region the remediation tranche should be scoped to. It also confirms Cowork's
"a reference is code too" reading: the region where the production grid is wrong is exactly the region
T33's oracles never priced against an independent high-precision reference.

### Item 3 — the single-killer re-run table: one target unchanged

`docs/audit_log.md`'s accumulating list is mostly single-condition **guards**, where one killer is
inherent. The one load-bearing entry carrying "second killer still owed" since re-audit #3 was
re-run:

- **MM-I's `τ/(1+τ)` denominators** (`models/market_model_i.py:154` and `:155`) — both still die, and
  both die to **exactly one test, the same one**:
  `tests/L3/test_market_model_i.py::test_the_index_vol_step_matches_an_independent_transcription`.
  Deleting that single test would reopen **two** load-bearing holes with the suite green. This is the
  charter's own §4 example (the dropped `/(1+τF)` denominator worth ~3% that survived all 404 tests),
  so the thinness is known; it is **recorded again, unhardened**, and is the re-audit's standing
  highest-value hardening target.

### Items 4 and 5 — NOT YET RUN

Stated plainly rather than implied: the first sweep of the new vol-arc closed forms against the
T43/T39 blind spots (item 4) and the standing T1–T37 full-tree sweep (item 5) have **not** been
executed in this pass. The per-tranche sweeps covered T46 (27 mutants), T48 (20) and T49 (22) at build
time, but that is not the re-audit's independent re-run and must not be counted as it.

---

## Status

- (3) and (4): **DONE** (Cowork). One stale header fixed in place; no fix-tranche.
- (1) and (2): **PARTIAL** — items 1a (invariant-6, clean), 1b (first shape, **3 holes**), 2 (OI-4
  mapped) and 3 (single-killer re-run) done; **items 4 and 5 not yet run.**
- **Three holes for Cowork to rule: F-5.1, F-5.2, F-5.3** (two in merged code, T33 and T32), plus a
  §4 charter-line candidate from F-5.3 and an unhardened single-killer target (MM-I).
- OI-4's remediation (the Fourier-grid fix) is a **separate tranche** out of this re-audit, fix shape ruled
  then.

Re-audit #5 **logged open**; the arc boundary is settled once the sweep lands and OI-4's remediation is
ruled.
