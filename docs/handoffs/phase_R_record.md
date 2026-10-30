# Phase R — archive & recreate: what was done

**Date:** 2026-10-30 · **By:** Code · **Executes:** `rulings_reset_to_fresh_repo.md` §4 +
thread 0079. **End state reached:** empty, CI-green `pricebook` at `v0.0.0`, stock shelved and
gitignored, no library code.

## What happened

| step | result |
|---|---|
| preserve | final commit `52faf86` pushed to the old `main` (Cowork's two ruling docs — the record of this decision, which was uncommitted) |
| tag | `migration-archive` → `32cac55` (v0.144.0 · 600 oracles green · drawdown 32/793) |
| rename | `pricebook` → **`pricebook-migration`**, public, **archived read-only** |
| local | `~/work/pricebook` → `~/work/pricebook-migration`, remote URL updated |
| create | **`pricebook`** — public, MIT, described — cloned to `~/work/pricebook` |
| shelve | `_stock/` = the whole tracked old tree, **45M**, incl. all **77 oracle test files** |
| ignore | `_stock/` gitignored (confirmed: 0 entries staged) |
| scaffold | 16 files, one commit `chore: clean project scaffold (v0.0.0)` |
| verify | all 8 gates green · ruff clean · pyright clean · **CI green on ubuntu AND windows** |

## Three questions put to Bernardo before executing

Cowork's step list left these open and said to ask rather than guess:

1. **Local directories** — the new repo is also named `pricebook`, so the clone needed a home.
   **Ruled:** move the old tree to `~/work/pricebook-migration` (matching its archived repo name)
   and clone the new repo into `~/work/pricebook`, so the live project keeps the path everything
   already assumes.
2. **Stock contents** — step 3 named `src/pricebook`, `quarry/` and the old `docs/`; the **77 test
   files were not listed**, yet they encode every oracle (closed forms, reference values, the
   identities). **Ruled:** everything tracked, minus `.git`/`.venv` — mining is by hand and
   reference-only, so more material costs nothing and the tests are the richest source of *what is
   true*.
3. **`conventions_register.md`** — the ruling called it a keep-artifact *and* said it travels into
   the gitignored stock, which would have meant it was not committed anywhere. **Ruled:** tracked
   in the new `docs/`, as a living doc.

## Decisions taken under delegated mechanics — flagged for ratification

Cowork delegated "how `verify.py`'s layer/provenance gates read against an empty tree". Four calls:

1. **Provenance `quarry:` → `stock:`.** Same role (where the *content* came from) against the new
   reference tree. But `_stock/` is **gitignored and therefore absent in CI**, so a resolve-check
   would have had nothing to dereference there. A silent skip would have made it *a gate that
   cannot fail* — this project's named defect class — so the skip is **printed**: locally paths are
   dereferenced, in CI the gate says out loud that it checked headers only. `source:` and `oracle:`
   are verified in both, and under the new model those are the ones that matter.
2. **The population floor no longer fires on an empty library.** It counted zero resolved paths as
   a violation; with no landed modules that is the scaffold's honest state, so it now prints
   "nothing to check". The floor is **retained with teeth** for the real case: once modules exist,
   if none declares a stock ancestor and none resolves, that is still an error — all-fresh-built
   modules are *counted*, so the floor can see them.
3. **`verify.py tests` falls back to a flat `tests/`.** The old gate required `tests/L0..LN` tiers;
   those return as the layers do. Failing until then would have made the gate a thing to remember
   later, which is exactly how the 177-of-366 incident happened.
4. **Deleted:** the `quarry` import-floor gate and the drawdown reconciliation. Both measured
   migration progress. The header of `verify.py` says what was removed and why, so the next reader
   does not go looking.

## One thing I changed on my own judgement

**`.cowork/README.md` was rewritten, not carried.** The copied version instructed Cowork in the
retired frame — a read-only `quarry/`, the drawdown bar, `done = quarry empty`. A stale protocol
file is worse than none precisely because it *will* be followed. The new one states the project
model, points at the ruling as the governing document until the charter lands, warns that the
stock's own `CLAUDE.md` is the most dangerous file in it, and adds one limit learned the hard way:
Cowork's `git status` can only confirm Code's prior commits, never the artifact it wrote this turn.

## What was deliberately NOT done

- **No library code.** Nothing is priced; the package is a version string and a docstring.
- **No charter.** `CLAUDE.md` is a placeholder pointing at the ruling. Writing it now would be
  guessing at conventions the foundation tranches have not yet forced.
- **The vol-clock oracle, and the branch cleanup** — both stood down per thread 0079. The vol-clock
  finding survives in `docs/conventions_register.md` (C2, **OWED**) and is re-laid *with* its
  cross-module oracle when the rebuild reaches that convention.
