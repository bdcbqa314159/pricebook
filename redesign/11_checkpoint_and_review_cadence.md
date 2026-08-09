# Artifact #11 — Checkpoint & Review Cadence (Ratified)

**Status:** Ratified. Extends the handoff protocol (#08). Makes the build → Cowork
review a **planned rhythm**, not an ad-hoc "I felt it was necessary." Governs when Claude Code
stops and what the review must interrogate.

**Stance: full migration, corrected.** Every quarry module crosses — but *realigned* to the
better design and clarity as it does (copy-ADAPT, shed debt). Nothing is archived-instead-of-
migrated; "done = quarry empty = v1.0" is taken literally. Time is not the constraint; quality
and legibility are.

---

## 1. When the build stops (checkpoint triggers)

Claude Code stops and writes a checkpoint report at the **first** of:

- **Cadence ceiling:** every **≤ 6 slices** (versions) since the last checkpoint. Never run
  more than 6 unreviewed. (The L5 run went ~18 and let L6 drift — that is the anti-pattern.)
- **Capability-cluster boundary:** a coherent unit completes — an asset class, a risk/greek
  family, a layer vertical, a calibration family. Stop even if < 6 slices.

**Plus mandatory *immediate* stops (any one, mid-cluster):**
- Any **design drift** beyond the ratified guardrails / a contract or measure question.
- Any **new cross-cutting abstraction or vocabulary type** introduced (rule-of-two moment).
- Any **debt logged** (`OPEN.md` gains an entry) or **suppression** added.
- An **oracle weaker than closed-form/cross-check** is the best available for a slice.
- A **quarry entry resists clean realignment** (can't be adapted to the spine without a smell).

The build **names its next checkpoint** at the end of every report (predicted boundary +
trigger), so the next stop is planned one hop ahead.

## 2. What the checkpoint report must carry (extends #08 §format)

Beyond the #08 sections (slices landed, ledger deltas, oracles, ready-for-next), every report
adds the four mandatory review inputs:

1. **Oracle-quality audit** — for each slice, state the oracle and classify it: closed-form >
   external cross-check (QuantLib/ISDA) > self-consistency (reprice-to-par) > trusted mark.
   Flag any slice whose oracle is *only* self-consistency — that is the "no real oracle" risk.
2. **Quarry-drawdown reconciliation** — modules crossed this checkpoint + cumulative:
   `<N> / 768 migrated (<x>%)`, and which quarry subpackages advanced. Keeps v1.0 countable.
3. **Design choices to challenge** — an explicit *"challenge me"* list: the non-obvious calls
   made since last checkpoint, each with its rationale, surfaced for Cowork to push back on
   *before* they harden. (This is where the build answers for what it chose.)
4. **Smell + debt scan** — new smells (oversized signatures caught by `PLR0913`, primitive
   obsession, god-objects, duplication), suppressions, shims, TODOs — each with the build's
   proposed disposition.
5. **Spine-conformance audit** *(added after the `foundation/black.py` drift)* — every module
   **created or moved** since the last checkpoint, justified against its **layer's definition**
   (not merely against the acyclic check, which cannot see semantic misplacement). Explicitly
   confirm: **L0 holds no finance** (strikes/vols/payoffs/discounting), analytic blocks of a
   dynamics sit at **L3** (A4.3), products are pure data at L2, pricing composition at L4, state
   only at L6. `verify.py layers` mechanises the L0 rule; the rest is reviewed.
6. **Deferred-ledger sweep** *(added after the foundation-audit closure, V3)* — re-read **every**
   `AC-*` / `NG-*` entry in `OPEN.md`'s audit-closure ledger and ask of each: *has its trigger
   fired?* If so, it is now scoped work, not a deferral — promote it. **No automated gate watches
   these** (they are deliberately outside the `verify.py debt` balance — a deferred capability
   suppresses nothing), so this human re-read is the only thing that catches them. Six triggers are
   **condition-driven, not topic-driven** (AC-T4.7/8/9/11/17/18 — a future calendar/consumer, an
   external weekend-rule change, error-message quality, a fourth consumer): no roadmap progress will
   ever fire them, so they need this calendar-based sweep, not a trigger. A ledgered item a
   checkpoint cannot account for is a forgotten item with paperwork (CLAUDE.md §5).

## 3. The Cowork review ritual (what happens here on receipt)

Per checkpoint, Cowork runs the same pass, in order:
1. **Reconcile** — apply ledger deltas; update the drawdown number.
2. **Challenge** — interrogate the §3 choices list; push back, ratify, or send back.
3. **Smell/debt hunt** — rule each item: fix-now / debt-ledger / accept-with-rationale.
4. **Oracle audit** — any self-consistency-only oracle gets a real reference value demanded
   or a re-open trigger logged.
5. **Reshape** — if a pattern is emerging (a recurring value object, a boundary), amend the
   spine/vocabulary now, before more slices build on the un-named pattern.
6. **Retune & release** — issue the rulings as a `rulings_<id>.md` hand-back; the build resumes
   to the named next checkpoint.

No forward slice begins until the checkpoint is ruled.

---

## 4. Forward checkpoint map (planned ahead)

Indicative, not binding — each report re-confirms the next. Ordered by the current design
priorities (A6: L6 before more depth), then the systematic quarry drawdown.

| # | Cluster | Trigger | Focus of the Cowork review |
|---|---|---|---|
| **CP-1** | measure-consistency oracle + first L6 vertical (Trade/Book/benefit table) + `xva_report` | cluster (A6) | A3 realized-vs-mark proven end-to-end; the binding measure oracle |
| **CP-2** | general-curve (bootstrapped) HW; lift the flat-curve ceiling | ≤6 or cluster | modelling honesty under the XVA stack |
| **CP-3** | breadth: commodity (+ inflation) on the A5 keyed snapshot | cluster | confirm "keys not fields" pays off; no new greeks |
| **CP-4** | quarry drawdown — `fixed_income` (130) systematic crossing | ≤6 rolling | copy-ADAPT realignment quality; oracle coverage |
| **CP-5** | quarry drawdown — `credit` (93) + `models` (90) | ≤6 rolling | de-duplication vs quarry sprawl |
| **CP-n** | … `options`, `structured`, `desks`, `regulatory`, `crypto`, `viz`, `ts` … | ≤6 rolling | until 793/793 crossed → **v1.0** |

The long tail (CP-4 onward) is the real bulk: full migration means each quarry subpackage is
mined to empty, realigned, oracle-gated. Drawdown % is the honest progress bar.

---

## 5. Relationship to the other artifacts
- Extends **#08** (report format) and **#06** (a checkpoint is not a release; releases are
  per-slice).
- Feeds **#05/#09** — smells/debt found here land in `OPEN.md`; oracle audit enforces the
  green-oracle gate retroactively.
- The **spine amendments** (A1–A6…) are the output of the "reshape" step — this cadence is how
  they keep getting made deliberately instead of discovered late.
