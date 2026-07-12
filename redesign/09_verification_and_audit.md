# Artifact #09 — Verification & Audit, Consolidated (DRAFT)

**Status:** Draft for reaction. Merges the quarry's scattered audit apparatus into **one
small setup** for the new tree, and delivers the staged-test theme. It obeys its own
rule: the verification setup is itself simple, not another framework.

---

## What the quarry accumulated (the sprawl to collapse)

| Thing | Size | Fate for the new tree |
|---|---|---|
| `tools/layer_deps.py` | 22 KB | **fold** the acyclic check + layer computation into `verify.py` |
| `tools/test_layer.py` | 9 KB | **fold** the layer-scoped test selection into `verify.py` |
| `tools/instrument_greeks_coverage.py` | 5 KB | quarry-specific; not carried (coverage is per-slice oracle) |
| `L0_DEPS.md … L7_DEPS.md` | ~95 KB (8 files) | **retire** — AST reading-orders for the *old* audit; the new order is the ledger + slice plan |
| `ARCHITECTURE.md` (regenerated) | 22 KB | **replace** with `verify.py --layers` on demand (no maintained 22KB doc) |
| `AUDIT_PLAN.md` | 13 KB | **superseded** by migration policy (#05) + this artifact |
| `MODULE_HEALTH.md` | — | superseded by per-slice oracles + `OPEN.md` |

All of these stay in the quarry (read-only history). The **new tree** gets one tool and a
short check list — nothing more.

---

## The new setup: one tool, `verify.py`

A single entry point with a few subcommands. No plugin system, no config framework —
just the checks the design actually needs.

```
verify.py layers            # print the layer of each new-tree module (replaces ARCHITECTURE.md)
verify.py acyclic           # FAIL if any import points upward (the one load-bearing invariant)
verify.py tests --layer N   # run only the tests at layer ≤ N  (staged tests / merge gate)
verify.py debt              # FAIL if suppressions − OPEN.md entries ≠ 0
verify.py provenance        # FAIL if a landed module lacks its provenance header
verify.py version           # FAIL if __version__ ≠ top CHANGELOG.md entry
verify.py all               # everything above (the CI gate)
```

Each subcommand is a small function; `verify.py all` is what CI and the merge gate run.

---

## Staged tests (the theme, delivered)

Correctness is checked **incrementally, by layer** — never the whole universe per slice.

| Tier | When it runs | Scope |
|---|---|---|
| **Slice tier** | every commit on a slice branch | `verify.py tests --layer <slice layer>` — only tests at/below that layer |
| **Merge gate** | before rebase-and-merge to `main` | the slice's layer tier, green, + `acyclic` + `debt` + `version` |
| **Full sweep** | nightly / at a layer checkpoint | the entire new-tree suite + the quarry regression suite |

So a Slice-0/L0 change runs the small L0 tier (fast), not the full suite. The tier grows
as migration climbs. This is exactly the quarry's own insight ("the test universe above L
is irrelevant during a slice") — kept, and made the default.

---

## Provenance header (what `verify.py provenance` checks)

Every landed module carries a short, machine-checkable header — the educational +
migration record in one place:

```python
"""Discount curve — flat and bootstrapped.

Provenance:
  quarry: python/pricebook/core/discount_curve.py
  source: Hull, Options Futures & Other Derivatives, ch.4  (docs/…)
  oracle: DF = exp(-r t) closed form; QuantLib cross-check
  slice:  S02
"""
```

`verify.py provenance` fails if a module under `src/pricebook_ng/` lacks these four
lines. Cheap to check, and it keeps the "legible, traceable" promise honest.

---

## Lint rules that carry design intent (CI ruff, new tree only)

Beyond formatting, CI's ruff pass on `src/pricebook_ng/**` enforces a few rules that are
really design guardrails (the quarry is excluded — it would explode):

- **`PLR0913` `max-args = 5`** — signature discipline (CLAUDE.md §3b). Over the limit ⇒
  bundle into a value object, never suppress.
- (add further design-carrying lint rules here only when a real, present need appears —
  not speculatively.)

These run in the same CI ruff step, not a bespoke checker.

## Pre-commit stays surgical

Keep the existing philosophy: pre-commit does **only** fast, safe auto-fixes (the ruff
E3 blank-line hook already there). The heavier checks (`acyclic`, `debt`, tests) run in
CI / the merge gate, not on every keystroke. Do not grow pre-commit into a second CI.

---

## Why this honours the simplicity rule

The temptation is to build an "audit framework." We don't. `verify.py` is a flat script
of small functions, each solving a check that the design *actually requires today*:
dependency direction, staged tests, the debt invariant, provenance, version drift. No
abstraction layer, no extensibility hooks, no speculative generality. If a sixth check is
ever needed, it's one more function then — not a plugin API now.

---

## Ratified decisions (2026-07, Bernardo)
1. **One `verify.py`, sprawl retired.** ✅ The new tree's entire audit setup is a single
   `verify.py` (`acyclic`, `tests --layer N`, `debt`, `provenance`, `version`, `all`).
   The 8 `L*_DEPS.md`, the regenerated `ARCHITECTURE.md`, and `AUDIT_PLAN.md` are **not
   carried forward** — they stay as frozen quarry history. A human layer view comes from
   `verify.py layers` on demand, not a maintained doc.
2. **Provenance enforced.** ✅ `verify.py provenance` fails CI if any `src/pricebook_ng/`
   module lacks the four-line header (quarry / source / oracle / slice). Four lines per
   module keeps the legibility-and-traceability promise honest.
3. **Staged tests are the default.** ✅ Slice tier per commit, layer tier at the merge
   gate, full sweep nightly / at layer checkpoints.
4. **Pre-commit stays surgical** (ruff E3 auto-fix only); heavy checks run in CI, never
   grown into a second CI on every keystroke.
