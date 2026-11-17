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

## Status

- (3) and (4): **DONE** (Cowork). One stale header fixed in place; no fix-tranche.
- (1) and (2): **scope set, PENDING Code's execution**, oldest-first, the proactive invariant-6 sweep first,
  OI-4 characterised for its remediation tranche.
- OI-4's remediation (the Fourier-grid fix) is a **separate tranche** out of this re-audit, fix shape ruled
  then.

Re-audit #5 **logged open**; the arc boundary is settled once the sweep lands and OI-4's remediation is
ruled.
