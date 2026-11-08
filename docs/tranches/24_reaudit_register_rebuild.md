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
