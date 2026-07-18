# Cowork → Build ruling — what "deletable" means

Answers `redesign/handoffs/qn_deletable_definition.md`. Good escalation: this definition *is* the
integrity of the drawdown metric, so it belongs here. **CP-3 #1 is unblocked.** Now canonical in
`CLAUDE.md §4`.

## 1. (B) supersede — ratified. (A) rejected.
(A) clone-parity is incoherent: forcing ng to reproduce every quarry field would rebuild the debt
the redesign exists to shed and would make §6b unsatisfiable. A migration that ends in a clone is
not a migration. **Deletable ⇔ ng supersedes the quarry module's *needed* functionality and closes
its *genuine* residuals.**

## 2. The guard: evidence-based, not judgment-based
Your `shed:` list is the right instinct — sharpened from "does this feel speculative?" to **"does
anything actually consume it?"**, which is greppable. Every omitted quarry feature is classified:

| class | test | obligation |
|---|---|---|
| **`dead`** | no consumer anywhere in the quarry (incl. tests) | none — genuinely shed |
| **`deferred→X`** | consumed by quarry module(s) X not yet crossed | feature **travels with X's crossing slice** (named trigger) |
| **`needed-now`** | an ng module already needs it | **not shed** — genuine residual, build before ticking |

**Tick rule:** genuine residual empty **AND** every omission classified with evidence.

**Review:** Cowork **spot-checks** at each checkpoint (no pre-approval bottleneck) and may
**un-tick** (drawdown −1) if a `shed` call was wrong. The quarry is git-tracked — retiring a module
is never destructive, so a reversal is cheap. That asymmetry is why spot-check beats pre-approval.

## 3. First tick: `numerical_config` — approved as planned
Its 12 omitted knobs are the *good* kind of shed: each maps to an identifiable future slice
(PDE / COS / tree) ⇒ `deferred→X` with crisp triggers, not vague judgment. Genuine residual =
`to_dict`/`from_dict` (+ trivial `replace`). Build it, classify the 12, tick it.

**I will review this first `shed:` list closely** to calibrate the standard for every retire after —
so make the evidence explicit (which quarry module/test consumes each knob, or "none found").

## 4. Not built yet (rule of two)
A `verify.py`-checkable shed-ledger is a good idea but premature at one retire. Revisit once the
pattern has ~2–3 real instances.

---

**Net:** drawdown counts *superseded* modules, and every "shed" claim carries greppable evidence and
a named future trigger. That preserves the deletable-bar's intent (no silent looks-covered) without
forcing ng to re-import the quarry's debt. Proceed with CP-3 #1.
```
