# Ruling — reset to a fresh repo (the migration frame is retired)

**Date:** 2026-10-30 · **By:** Cowork, ruling Bernardo's decision · **Supersedes:** the scope
deliberation recorded in `rulings_CP_xva_reporting.md` §4 (the "full migration stands / Path 1"
line) and the fork in `CP_xva_reporting.md` §5. **Status of this tree:** to be archived.

## 1. The decision
The migration frame is retired. Not Path 1 (full migration), not a narrowed v1.0 — a **clean
reset**. The driver was honest and is recorded as the reason: three consecutive checkpoints showed
complete, oracled capability against a nearly-static drawdown bar (32/793 = 4.0%), and the
completionist contract (`done = quarry empty = v1.0`, drawdown as scoreboard) had drained the
project's meaning while the quant work itself stayed good. The fix is to keep everything that had
value — the architecture, the oracle discipline, the hardening/review passes, the Code↔Cowork flow,
the conventions — and drop the one thing that didn't: the obligation to cross every old module.

## 2. The new project model (for context; full charter lands as the new repo's CLAUDE.md)
Same **aim**: a pricing analytics library, every number **verified against the book or paper that
defines it**. What changes is the direction of pull.

- **Unit of work = a product you want.** "I'd like to go for this product" → plan what it needs
  (this method, this model, these core pieces) → **tag against the stock**: do we already have each
  piece? → **purify** (review the stock version, adapt to our conventions) where we do, build fresh
  where we don't → **oracle** against the reference → **reflect** into the library. Top-down,
  demand-pulled — the product drags in exactly the core it needs and nothing it doesn't.
- **Interest picks the target; the discipline and conventions are the constant; the books/papers
  are the oracle.**
- **Scope line — analytics only, pluggable not implemented.** In: products, models, calibration,
  engines, risk/XVA, the market-data value types. The durable public contract is **how a product
  and a pricing request are represented** (a serialization/wire format over the frozen product
  data). A **pricing server** (e.g. a socket that prices a request) is the first consumer that
  proves the representation. **Out of this repo:** GUI, database, broker/feed — they are downstream
  projects that *consume* the representation, never built here.

## 3. The stock
The entire current tree — the migrated core (`src/pricebook`) **and** the non-migrated `quarry/`,
**and** this `docs/` — becomes read-only reference material ("the stock"), mined by hand for
*content* (what is true: conventions, formulas, edge cases), never for *structure*. It lives in a
gitignored folder in the new repo; the archived old repo is its pushed backup.

**Carry the lessons, not just the code.** `conventions_register.md` is a keep-artifact: it describes
the built core, not the migration. Its one OWED item — the **vol clock** (`vol_time_measure`, read
by ten modules, unguarded, the 5.14%-bug shape at twice the spread) — is a known hole in the stock's
core; when the rebuild re-lays that convention, it is re-laid **with** its cross-module oracle, not
without. (No point writing that oracle into this tree now — it is being archived.)

## 4. Phase R — the clean-slate process handed to Code (thread 0079)
Settled parameters: new repo **`pricebook`**, **public**, **MIT** (pyproject corrected to match the
LICENSE file, which already reads MIT; the old `pyproject` said Proprietary — resolved to MIT for a
public repo). Package **`pricebook`**. Old repo renamed **`pricebook-migration`** (distinct from the
existing `pricebook-legacy`) and archived read-only, tagged `migration-archive` first. Stock folder
**`_stock/`**.

End state: an empty, CI-green `pricebook` at `v0.0.0`, the stock shelved and gitignored, nothing
priced. Then Bernardo and Cowork plan the foundation tranches before any library code is written.

Full step list is in thread 0079.
