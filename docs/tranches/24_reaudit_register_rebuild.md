# Tranche 24 — rebuild the conventions register to this tree (re-audit #1 F-3.1/F-3.2/F-2.1): docs

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.29.0`.
**Source:** none external — a docs-accuracy correction of our own registers, pulled by re-audit #1.
Second of two re-audit-remediation tranches (T23 → T24). **Docs-only**, unless the honest re-derivation
of a convention row warrants a cheap cross-module guard (see §2), which is in scope.

## 0. The defect
Re-audit #1 (`docs/audit/reaudit_01_findings.md` F-3.1) found `docs/conventions_register.md` rows
**C1–C6 describe a tree that does not exist** — XVA/equity/SABR/vanilla modules carried wholesale from
the archived predecessor at the v0.0.0 scaffold (`236c184`). Only C7/C8 (inflation day-count, OWED)
and C9 (index resolution, guarded) belong to this tree. Worse, the file's **"Immediate reading"**
section calls C2 "the widest-spread convention in the tree … the one to worry about" — a worry about
modules that aren't here. §4's cross-module-convention clause relies on this register as the
enumeration of which conventions exist; today it points a reader at a phantom tree while this tree's
real conventions are only partly covered.

## 1. The change — `conventions_register.md`
Rebuild the file so it describes **this** tree:
- **Partition.** Separate "predecessor lessons (archived tree)" — the C1–C6 material kept for its
  *lessons* per the reset ruling (`handoffs/rulings_reset_to_fresh_repo.md`), clearly marked as the
  archived tree's inventory — from "this tree's live register." No row may read as live inventory of a
  module that does not exist.
- **Re-derive the live rows** from the actual modules, each with its honest three-state Guard column
  (§7: guarded / per-module / owed):
  - C7/C8 inflation day-count — verified real (only `ACT/365F` exists; OWED + "unrepresentable, one
    day count"). Keep, confirm wording against code.
  - C9 index resolution — real, structurally guarded. Keep.
  - **invariant-6** (historical flows → PV 0) across the **5 real engines** — `cash`, `inflation`,
    `nominal_cap`, `capfloor`, `yoy` — which currently has **no row**. Add it; its honest Guard state
    is almost certainly *per-module* (each engine tests its own historical→PV0), so record that, not a
    guarantee it doesn't have.
  - the **`model.market`/snapshot discounting** convention (the engine reaches market data only through
    `model.market`, §3) — add a row with its Guard state.
- **Rewrite "Immediate reading"** to name this tree's actual widest-spread / highest-risk convention,
  not C2.

## 2. A cheap guard, only if the re-derivation earns it (in scope, not required)
If writing the invariant-6 row makes it obvious that a *per-module* convention wants a cheap
cross-module oracle (one test comparing the 5 engines' historical-flow handling), add it and mark the
row **guarded**. Do not manufacture a guard to tidy the column — an honest *per-module* or *owed* is
the correct entry when no structural guarantee exists (§7). This is the one place T24 may touch `src`
tests; everything else is docs.

## 3. Also in this tranche — `to_go_deeper.md` (F-3.2) and F-2.1 record
- **F-3.2:** `BERHAN` item 6 "Six `crosscheck:` markers" → **23** (the real non-`none` targets across
  the BERHAN + Jäckel arcs). `BERHAN` item 1 "L5 is empty — no deltas, no scenario engine" → reword:
  L5 now holds Black/forward Greeks (T22, `risk/sensitivities.py`); the *route* it describes (an
  inflation risk layer over the existing engines via a `Priceable` protocol, bump-and-reprice) is
  genuinely still unbuilt, so keep the route and correct only the false blanket sentence.
- **F-2.1 record:** add a `JAECKEL` / cross-cutting `to_go_deeper` note — the driftless-forward
  terminal `F·exp(-½V²+Vz)` is inlined in `models/black_mc.py:118` and `risk/sensitivities.py:130`
  (`_terminal`); a rule-of-two second consumer sharing the shape. Extract a shared helper at the next
  touch of either file (both are identical one-liners today; the risk is the convention drifting
  between them). Not a tranche of its own.

## 4. Process
Branch `tranche/24-register-rebuild`. `test` (RED only if §2's guard is added — otherwise the gates are
the check: `verify.py provenance`/`layers` stay green, and the rebuilt register must not name a
non-existent module) → `docs` → `chore(release): v0.29.0`. `source:` none. **Open as a DRAFT PR**
(§10 step 3). Closes the re-audit #1 remediation; after it lands, `docs/audit_log.md` entry #1 is
complete and the next-source direction is settled with Bernardo.


---

# BUILT — results (appended at build; §0–4 above are Cowork's spec, unedited)

`v0.29.0`. 821 tests, all eight gates green, ruff and pyright clean. The register is rebuilt,
`to_go_deeper`'s stale lines are corrected, F-2.1 is recorded — **and §2's optional guard turned out
to be earned, for a reason the audit had not found.**

## 5. §2's guard was earned — by mutation, not by reading

The spec allowed a cheap cross-module guard *"only if the re-derivation earns it"*, and warned against
manufacturing one to tidy the Guard column. My carry-in said I would re-derive first and judge after.
Re-deriving the invariant-6 row meant establishing what the five engines actually agree on, and the
honest way to establish that is to **flip the boundary and see who notices**:

| engine | `>` → `>=` | verdict |
|---|---|---|
| `cash` | KILLED | tested |
| `inflation` | KILLED | tested |
| `nominal_cap` | SURVIVED | **proved equivalent** |
| `capfloor` | SURVIVED | **a real hole — 41%** |
| `yoy` | SURVIVED | **proved equivalent** |

So three survivors, and the first instinct — "three engines have an untested boundary" — was **wrong
for two of them.** §4 requires proving an equivalent mutant rather than assuming it, and the proof is
that prices are **bit-identical** across four schedules including the boundary case: a Hull-White
caplet whose expiry has already passed is worth exactly zero, so including or dropping such a period
changes nothing. I nearly wrote a guard on that false premise; measuring stopped me.

`capfloor` is the one that matters. An inflation cap/floor period that has already paid has **both
index levels known** — the payoff is *determined*, not an expired option — so it carries real value,
and including it moved the price from **5,929.24 to 8,363.52, about +41%**. That is a genuine
mispricing a wrong comparison would have caused, with nothing to catch it. Closed.

## 6. Why the row is `per-module` and not `guarded`

A cross-module test comparing all five engines would catch **nothing** that the per-engine boundary
tests do, while duplicating three market fixtures and claiming a guarantee that does not exist. What
would make C10 structurally guarded is what makes C9 guarded: **an atom.** Five engines inline
`pay > valuation_date`, so a divergence is expressible; extract one `is_historical(pay, valuation_date)`
and the row joins C9 and C11 at *guarded — structurally*. That is recorded as the row's trigger rather
than done, because it is a `src` change and this was ruled a docs tranche.

This is the spec's own warning honoured rather than quoted: an honest `per-module` with a measured
exposure and a named upgrade path beats a `guarded` that rests on a test comparing fixtures instead of
modules.

## 7. The register, rebuilt — and one of my own rows was overstated

C1–C6 described `risk/exposure.py`, `risk/xva.py`, `engine/linear.py`, `engine/xccy.py`,
`engine/equity.py`, `models/sabr_model.py` and seven more modules that **do not exist here**. They are
now a clearly-labelled *archived* section (lessons only, not inventory), with the IDs **kept stable**
because `docs/handoffs/`, `docs/audit/` and CLAUDE.md cite them and renumbering would silently change
what those citations mean. The live register is C7–C11, and a check confirms **every module it names
exists in `src/`**.

**C11 as I first wrote it was wrong in two ways**, which is worth recording because it is the same
failure mode as the rows I was replacing — a convention row asserting more than the code supports:

* *"every L3 model carries a `MarketSnapshot`"* — false. Six do; `black`, `black_mc`,
  `libor_market_model`, `swap_rate_vol`, `exercise_boundary` and `index_ratio` do not, because they
  are closed forms, parameter objects and numerical helpers that never read market data. That is the
  convention's **scope**, not a list of exceptions.
* *"there is no second place to pass a snapshot from"* — false inside an engine. `inflation_leg_value`,
  `resolved_index_level` and the three calibration objectives all take a snapshot explicitly, quite
  legitimately, since they are called *from* a pricer that got it from `model.market`. The guarantee
  is at the **entry point**: a caller cannot supply a mismatched snapshot, because every registered
  pricer's signature is `(product, model)`.

Both corrected before commit. The lesson is the one F-3.1 is about: a register row is only worth the
grep behind it.

## 8. Findings for Cowork (§10 — raised, not decided)

1. **Target 3 and Target 1 are less separable than §9's list suggests.** The `capfloor` hole was found
   by a *docs* tranche, because re-deriving a register row honestly means mutating the convention, not
   reading the code. Worth folding into §9's re-audit description for #2: *re-deriving a register row
   is an oracle-power probe*, so the register pass and the mutation pass should inform each other
   rather than run as separate targets.
2. **The `is_historical` atom (C10's trigger)** — a four-line extraction that would upgrade the
   widest-spread live convention from `per-module` to `guarded — structurally`. Out of scope here by
   your docs-only ruling; worth a small tranche of its own, or folding into whatever next touches an
   engine. Not urgent: the exposure is now measured and two of the five survivors are proved harmless.
3. **No charter change proposed.** §4 already carries the rules that caught both holes (mutation over
   reading; prove an equivalent mutant rather than assume it). Third tranche running where the
   machinery was right and only its application needed widening.

---

## 9. Cowork's ruling on T24 — PR #33 (§10 step 4, 2026-11-08)

Accepted — v0.29.0, 821 green, the register rebuilt to this tree. Re-audit #1 remediation complete;
entry #1 closes when #33 merges.

**The docs tranche earned its keep by finding a correctness-material hole, which is the headline.**
Re-deriving the invariant-6 row honestly (mutating the boundary, not reading it) surfaced a **41%
`capfloor` mispricing** the whole suite could not catch. I read the result as: production `capfloor`
is **correct** (`pay > valuation_date` excludes the already-paid, fully-determined period per invariant
6); the `>=` mutant would have added that determined value, +41%; nothing tested the boundary; you
closed it with a per-engine boundary test and changed no pricing code. Same shape as T23's F-1.2 —
code right, oracle blind. (If you meant production was itself wrong, flag it — but "a wrong comparison
*would have* caused" reads as the mutant, not the ship.)

**Two things done exactly right:**
- **You proved the equivalent mutants instead of assuming them.** "Three untested boundaries" was the
  easy read; §4 made you prove, and `nominal_cap`/`yoy` are bit-identical across the boundary because
  an expired HW caplet is 0. You began a guard on the false premise and the measurement stopped you —
  that is the rule working.
- **You caught your own overstated C11 twice before commit** — the F-3.1 lesson pointed back at the
  person applying it ("a register row is only worth the grep behind it"). Keeping C1–C6's IDs stable
  while archiving them, because `docs/` and CLAUDE.md cite them, is the right call too.

**Your three §8 findings:**
1. **Target 1 and Target 3 inform each other — ruled into §9, not §4.** This is the real lesson and
   you are right: the `capfloor` hole came out of the *register* pass because honestly re-deriving a
   row is a mutation probe. §9 now says the register and oracle-power passes run together for re-audit
   #2, with the `capfloor` cite. Edited.
2. **The `is_historical` atom (C10's trigger) — left as the recorded trigger, pulled by product-pull;
   not a speculative tranche now.** Extracting a four-line atom with no present consumer is exactly the
   speculative abstraction §6 forbids, and C10 at `per-module` with a measured exposure and a named
   upgrade path is a charter-legitimate state (§7), not debt. The CVA integral's lower limit *is* the
   historical boundary, so an XVA tranche (likely T26, the MC exposure engine over the real engines)
   will **pull** the atom when it needs it — and that pull upgrades C10 to `guarded — structurally`
   for free, closing your trigger by product-pull rather than by a tranche built for tidiness. If the
   XVA arc reaches its checkpoint without touching the boundary, C10 stays `per-module` (legitimate)
   or earns its own tiny tranche then. Recorded in C10's trigger and noted in the T25 spec.
3. **No §4 change — agreed.** §4 already carried both rules that caught the holes (mutation over
   reading; prove an equivalent mutant). The machinery was right; only its application widened, which
   is the §9 edit above.

**Next: the XVA spine arc — already settled and specced (thread 0119, which crossed your 0120).** The
next source is *not* open — Bernardo brought Crépey and I scoped the elementary XVA spine. After #33
merges, read the queue: **T25** (`docs/tranches/25_unilateral_cva.md`), the arc plan in
`docs/xva_spine_arc.md`.
