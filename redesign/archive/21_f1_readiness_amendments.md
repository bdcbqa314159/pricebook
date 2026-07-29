# Artifact #21 — F1 readiness: what the audit changed in the design

**Purpose:** before market data (F1) opens, reconcile the ratified design with the post-audit tree
(v0.84.0). The audit deleted four types and added three; the docs still describe the old world.

**Verdict:** `19_market_data_design.md` needs **three amendments**, none structural. The urgent work
is elsewhere — **`CLAUDE.md` references four types that no longer exist and describes a snapshot
shape that `19` superseded.** It is read as law every session, so stale law is the highest-value fix.

---

# Part A — `CLAUDE.md` drift (fix FIRST; law referencing absent types)

Verified against `src/pricebook_ng/`: `NumericalConfig` exists only as a stale `.pyc`;
`DiscountBasis` and `ReferenceEntity` are absent.

| # | line | says | reality | fix |
|---|---|---|---|---|
| **A-1** | §3 ~107 | snapshot is `curves`/`spots`/`vols` maps keyed by `MarketKey(asset, id)` (A5) | **superseded by `19` §2** — closed shapes × open keys: `curves`/`surfaces`/`scalars`/`series`/`schedules` | replace the A5 paragraph with `19` §2's shape; note A5 superseded |
| **A-2** | §2 inv. 5 · §3 ~120 · §3b ~140 · §3b ~157 | reproducibility knobs arrive in `NumericalConfig`; the decomposition worked example | **deleted** (Phase 4 — guessed knob names, no engine consumer) | keep the *principle* (config explicit, ≤5 fields, decomposed by method family); state the type **lands with the first engine that reads it** |
| **A-3** | §3 ~121 | `PricingResult` carries `sensitivities`, `diagnostics`, cashflow breakdown | **deleted** (Phase 4) | `PricingResult` = `pv` · `accrued` · `clean` · `basis: Currency \| None`. Risk output returns with L5, shaped by it |
| **A-4** | §3c ~179 | worked example: `ReferenceEntity` → **L0** | **deleted** (Phase 4) | the *reasoning* stands (multi-layer identity ⇒ L0); mark it as the shape credit will follow, not a type that exists |
| **A-5** | §3 ~122 | "the economy a model is built on = curves + `FixingHistory`" | `FixingSource` protocol now exists; `FixingHistory` is its trivial impl | depend on the protocol |
| **A-6** | §3 Time | `Calendar` | `CalendarProtocol` (+ `calendar.py` → `calendars.py`) | type slots take the protocol |

**Why this is first.** §4 warns that inheriting the quarry's shape is how the old design re-enters
through the back door. Stale *law* is the same failure with a shorter path: a builder reading
`CLAUDE.md` will faithfully implement a deleted type.

---

# Part B — `19_market_data_design.md`: three amendments

### B-1. FX quote order and triangulation — `AC-3.6b` is now F1 scope (**the real gap**)

`19` §2 files FX spot as a **scalar keyed by a pair**, and §9 repeats it. That is under-specified in
exactly the way §6 warns about.

A scalar keyed by `(EUR, USD)` does not say **which quote convention the number is in.** EURUSD 1.08
and USDEUR 0.926 are the same market; stored as a bare float they are indistinguishable, and an
inverted spot builds an xccy curve that **calibrates perfectly and prices everything wrong** — the
§6 mis-resolution blind spot, in the one asset class where inversion is a coin-flip.

**Amendment:** FX spot is not a bare scalar. It carries its **quote convention** (base/quote order),
and the snapshot exposes a **rate accessor that takes the direction you want** and inverts internally:

```python
def fx_rate(self, base: Currency, quote: Currency) -> float: ...   # asserts, never guesses
```
Plus the **pair-conventions registry** promoted out of the ledger (`AC-3.6b`): market-standard quote
order per pair, spot lag (already moved off `CurrencyPair` identity), and **triangulation** through a
declared vehicle currency for crosses. Triangulation is a *stated* path, never an inferred one — §6.1
(fail loud on ambiguity) applies unchanged.

*Oracle:* `fx_rate(A,B) · fx_rate(B,A) == 1` to tolerance; a cross triangulated through USD equals the
directly-quoted cross where both exist.

### B-2. Snapshot serialisation — the convention now exists, so rule it

Audit 3.4 landed **identities by name, atoms by value**. `19` predates it and is silent. A
`MarketSnapshot` holds both (`RateIndex`, `Currency`, `Calendar` = identities; curves, spots, fixings
= values), so the rule applies directly and unambiguously.

**Amendment:** state that `QuoteSet` and `MarketSnapshot` serialise under the 3.4 convention, and that
**a serialised `QuoteSet` + a `CalibrationSpec` is sufficient to reproduce a snapshot exactly.** That
is what makes engine invariant 1 (identical inputs ⇒ identical output) checkable across processes
rather than only within one — and it follows from §1's "`QuoteSet` is authoritative" without adding
anything new.

### B-3. `TimeMeasure` is the sanctioned anchor (A1 ruling)

`19` §7 says "a curve carries its own anchor date." The A1 ruling names
`TimeMeasure(anchor, day_count)` as the **only** sanctioned `date → t` map.

**Amendment:** §7's anchor **is** a `TimeMeasure` — anchor *and* day count together. An anchor date
without its day count is half a specification and lets two curves measure the same interval
differently. Per A1 it lands as an **L0 module in Topic 1's first slice**; `19` records the dependency
so F1 doesn't invent a second one.

---

# Part C — carried unchanged (verified, no amendment)

- **§1 dual-role rule** — `QuoteSet` authoritative, par risk bumps quotes and rebuilds, zero/pillar
  bumps the built curve. Untouched by the audit.
- **§2 closed shapes × open keys** — survives Phase 4 intact. `AssetClass` was **kept** (the deleted
  siblings were the sibling *dataclasses*, not the enum), so keys still have their asset dimension.
- **§3 `CurveSet` typed accessors** · **§4 adapter boundary** · **§5 provenance beside the curve** ·
  **§6 resolution safety** — all unaffected.
- **§9 asset-class slot-in test** — still the right test; B-1 strengthens the FX row.

---

# Part D — `18` and `20`: dangling references

| doc | line | issue | fix |
|---|---|---|---|
| `18` §4 | ~56 | "**`DiscountBasis`** on `PricingResult` … (L0, done)" | **deleted**; the concept is now `basis: Currency \| None`. Statement is wrong as written |
| `20` Part A | ~20 | Engine row cites "explicit `NumericalConfig`" | principle stands, type deferred (A-2) |
| `20` B1 | ~55 | `CalibrationSpec` carries "its `NumericalConfig` slice" | the spec carries **its own** solver settings until the type returns |
| `20` B5 | ~128 | "`DiscountBasis` already does this for a PV" | same as `18` §4 |

---

# Order of work

1. **Part A** — `CLAUDE.md`. Six edits, no code. Do before anything else touches the tree.
2. **Part D** — four one-line doc corrections.
3. **Part B** — amend `19` in place (rev 3), recording the amendments as `19` already records its
   adversarial findings in §8.
4. Then **F1 builds**, carrying `AC-3.6b` as scope rather than as a deferral.

**Note on B-1:** it is the only amendment that adds design surface, and it arrived from the audit
rather than from the adversarial pass — the pair-conventions registry was invisible while FX spot
looked like "a number keyed by a pair." Worth noting as evidence that `19` §9's slot-in test is
necessary but not sufficient: FX *did* slot in by adding a key, and was still under-specified.
