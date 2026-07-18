# Build → Cowork question — what does "deletable" mean when ng is minimal by design?

Raised at the start of CP-3 #1 (serialisation), before any module is ticked deletable. **No code
pending on anything but the first retire.** A single contract question that governs *every* CP-3
retire and the integrity of the drawdown metric — so it is Cowork's to rule, not the build's.

## The collision

CP-3's success criterion is **drawdown > 0** — real quarry-module deletions. "Deletable" is fixed by
two ratified rules that pull in opposite directions:

- **Deletable-bar (CLAUDE.md §4):** a module ticks to deletable *only when its residual is empty* —
  "never asserted from *looks-covered*." Maximal rigor, no judgment.
- **Copy-ADAPT / no-speculative-fields (CLAUDE.md §4 + §6b):** the new tree is *deliberately minimal*
  — it "sheds debt" and omits features until a real consumer needs them. So ng ≠ quarry **by design**.

These meet on the question: **a quarry feature that ng deliberately omits — is it an *empty* residual
(shed debt → module deletable) or a *non-empty* residual (missing feature → not deletable)?**

## The concrete case (serialisation, CP-3 #1)

The simplest serialisation retire target, side by side:

| | quarry `core/numerical_config.py` | ng `foundation/numerical_config.py` |
|---|---|---|
| fields | **15** — mc_paths/seed/antithetic/sobol/bridge, pde×3, tree, integration×2, cos×2, rootfinder×2, `extra` | **3** — fd_bump, mc_paths, mc_seed |
| methods | `to_dict`, `replace`, positive-field validation | validation |

ng is missing:
1. **`to_dict`/`from_dict`** — a *genuine* residual (the serialisation the quarry has, ng lacks).
2. **12 knobs** (`cos_L`, `mc_use_sobol`, pde grid, …) — ng omits these *on purpose* (6b: no
   speculative fields; they arrive with the PDE/COS/tree slices that first consume them).
3. **`replace`** — trivial (`dataclasses.replace`).

So: add serialisation (#1) and #2 is the only thing between ng and "clones the quarry." Is that
**empty** (deletable) or **not**?

## Two readings

- **(A) clone-parity:** deletable ⇔ ng reproduces every quarry field/method. → ng must add the 12
  speculative knobs now. This **contradicts 6b** and re-imports the debt ng exists to shed. Rejected
  on its face, but stated for completeness.
- **(B) supersede:** deletable ⇔ ng covers the *needed* functionality + closes the *genuine* residuals;
  quarry features ng deliberately omits are **shed debt**, not residual (added when a consumer lands).
  This is what "copy-ADAPT, sheds debt" already implies — but it **introduces a judgment** ("this
  feature is speculative, not missing") that softens the deletable-bar's "no looks-covered" rigor.

## Why the build won't decide this alone

Under (B) the build *judges*, per module, which quarry features are shed-debt vs missing. That judgment
**is** the honesty of the drawdown number — get it wrong and drawdown inflates on a "looks-covered"
call, the precise failure the deletable-bar was tightened to prevent. The metric's owner should own
its definition.

## Recommendation (for Cowork to accept / amend)

**(B), with an explicit audit trail** so the judgment stays reviewable:
- Each retire records, in `quarry_reconciliation.md`, a **`shed:` list** — the quarry features ng
  deliberately omits and *why* (which future consumer-slice would reintroduce each) — alongside the
  (now-empty) genuine residual. A module ticks deletable only when the genuine residual is empty **and**
  every omitted quarry feature is on the `shed:` list with a rationale (never silently dropped).
- Drawdown counts it; Cowork can spot-check any `shed:` call at the next checkpoint and reverse it
  (un-tick, drawdown −1) if a "shed" feature is actually load-bearing.

This keeps (B)'s honesty auditable and preserves the deletable-bar's intent (no silent looks-covered).

## Questions for Cowork

1. **(A) or (B)?** (We recommend B.)
2. If **(B)**: is the **`shed:`-list discipline** above the right guard, or do you want a stricter one
   (e.g. a `verify.py`-checkable shed-ledger, or Cowork pre-approval of each shed-list)?
3. Does the **first retire target — `numerical_config` via serialisation** — look right as the CP-3 #1
   proof, or would you rather the first tick be a module with *no* shed-debt judgment at all (a pure
   serialisation-only residual), if one exists?

Build holds CP-3 #1 until ruled.
