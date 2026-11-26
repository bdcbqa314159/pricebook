# T81 — the live status instrument (the structural track opens here)

**Cowork spec, 2026-11-26.** The first unit of the structural track, and infrastructure rather than
representation: a committed tool that **generates `docs/STATUS.md` from the tree**, so the library's coverage
is a live, drift-proof scoreboard that the structural track (R1→R5) fills column by column. Ruled a
tooling-emitted instrument, not a hand-maintained page (0425) — a hand-maintained STATUS.md drifted within
hours and duplicated four other docs; this makes it mechanical.

## Target
A `tests/tooling/status.py` instrument (the `mutation_sweep.py` committed-instrument pattern) that walks the
tree and **emits the mechanical rows of `docs/STATUS.md`**, plus a CI check that the committed `STATUS.md`
equals the freshly-generated one — so it **cannot go stale** (the whole point; §7 guarded-not-folklore).

## Pieces
- **`tests/tooling/status.py`** — walks `src/pricebook/` and emits the mechanical rows: model / engine /
  product counts; the registry's `(product, model)` pairs; a per-asset-class engine inventory (rates /
  credit / vol / equity / inflation / commodity / FX / XVA — classified by a small explicit map, not guessed);
  **which products have greeks** (grep the risk layer — today: Black MC only); test count (from the floor),
  gate count, re-audit count (from `audit_log.md`), the `OPEN.md` ledger. **Structure and status only — never
  a formula or source line** (IP-safe, so the page can later render publicly).
- **The curated rows stay curated** — the asset-class readiness assessment (`●●●…`), the bibliography table,
  and the **1.0 scorecard** are a template the tool preserves verbatim between `<!-- curated -->` markers and
  regenerates the mechanical rows around; those are refreshed by hand at each re-audit (the §1 judgement half).
- **`docs/STATUS.md`** — becomes the generated file (the current hand-drafted one is the seed for the curated
  template).

## Oracles (RED-first, §4)
1. **Known-answer test** (the instrument tests itself, the `mutation_sweep.py` precedent): on a small fixture
   tree the generator emits the exact expected rows (counts, a planted engine in a known asset class, a
   product with/without greeks) — so a miscount in the generator is caught, the F-9.2 discipline made
   mechanical (a count the tool derives, tested).
2. **The drift gate** — a test (and `ci.yml` step) asserting the committed `docs/STATUS.md`'s mechanical rows
   equal `status.py`'s fresh output on the real tree. This is the anti-folklore guarantee: a stale status
   page **fails CI**, so it can never lie. The curated rows between the markers are exempt (judgement, not
   mechanical).
3. **Idempotence** — running the generator twice is byte-identical (no timestamp churn in the mechanical
   rows; the "last generated" line, if any, is outside the compared region).

## Degeneracy premises (§4 / §9)
- The known-answer fixture tree has **≥2 asset classes and a greeks/no-greeks pair**, so the classifier and
  the greeks-detection do real work (not a one-class tree where every row is trivially right — the τ=1 shape).
- The drift gate is proven to **fail** on a planted divergence (add a model to the fixture, confirm the gate
  reddens) — a gate asserted to work, not assumed (the F-9.1 "verify the gate fails" discipline).

## Provenance / gates
- `tests/tooling/status.py` is tooling (like `mutation_sweep.py`) — not `src/`, no layer/provenance header,
  but its own known-answer test. The **drift gate joins `ci.yml`** (and the charter §9 gate list if it
  becomes a named `verify.py` subcommand — Code's call on placement).
- Version → **v0.83.0**; one floor bump (the known-answer + drift + idempotence tests); one `CHANGELOG` entry.
- `docs/STATUS.md` becomes **tracked** here (it was untracked, 0425) — now that it is generated and
  drift-gated, the 0425 objection (a hand-maintained duplicate) is resolved.

## Out of scope
- Any capability itself (greeks, risk) — the page *reports* coverage; R2–R5 *add* it.
- A rendered GitHub Pages site — markdown first; public rendering is a later option, and the IP-safety
  (structure-only) is already honoured so it stays available.

## After T81
**T82 = R1.1 — the `Trade` type** (`docs/arcs/R1_representation_mtm.md`): a product + notional + direction,
oracled `Trade value == ±notional × price(product, model)`. The scoreboard then shows the representation
column beginning to fill.
