# Cowork → Build — Topic 0 gate spot-check (5 items)

Checked against the code at v0.70.0. **Two of the five are corrected; three confirmed.**

---

## 1. `data_registry` / `notional` — reassigned, not parked

**`data_registry` → re-classify as `dead` (superseded by design), NOT reassigned.**
Its whole purpose is loading convention registries from JSON at import. We **ruled that capability
away** — S5: *"explicit construction, no import-time I/O"*, after the quarry's registry-replacement
bug (one bad JSON entry silently dropped the other 27). ng declares calendars and indices **in code**.
So nobody will ever build this; "reassigned" implies a future owner that does not exist. Mark it
`dead` with that rationale — a design decision, not an omission.

**`notional` → reassignment is fine, but sharpen why.** The *value* concept is **absorbed**: ng `Leg`
holds cashflows each carrying their own `Money`, so amortising/accreting is expressed directly and
needs no normalisation helper. What remains — expanding `scalar | list → per-period list` — is
**product-construction convenience → L2**. Not an L0 gap either way.

## 2. `interpolation` / `solvers` — **SUPERSEDED by S17: complete them NOW, tick at the Topic 0 gate**

> **This item is revised.** The hold below assumed hand-rolling the missing schemes was expensive.
> With **S17** ratified (Python may use numpy/scipy), completing them is *adapter* work, not
> algorithm work — so there is no reason to carry them into Topic 1 as partials.
>
> **Do now, in the Topic 0 numerics slice:**
> - `foundation/interpolation.py` → the five **point** schemes: `LINEAR`/`LOG_LINEAR` ours;
>   `CUBIC_SPLINE`/`MONOTONE_CUBIC`/`AKIMA` as scipy adapters; **extrapolation policy stated per
>   end** (`FLAT | CONTINUE_SLOPE | RAISE`).
> - `foundation/solvers.py` → Brent · Newton · secant · least-squares (LM), **replacing**
>   `bisect_root`/`nelder_mead`. `distributions.py` → `scipy.stats.norm`.
> - ⇒ **`core/interpolation.py` and `core/solvers.py` TICK COVERED at the Topic 0 gate.**
>
> **Stays in Topic 1:** **Hagan–West monotone-convex** — it is *not* point interpolation (it
> reconstructs a function from **interval averages**) and exists to build forward curves. The quarry
> agrees: it sits in `core/forward_interpolation.py`, which is forward-space curve construction.
> That module is Topic 1 work and ticks at Topic 1 close.

*Original reasoning, retained for the record:*

**Corrected.** ng has **2 of 5** interpolation schemes (LINEAR, LOG_LINEAR — missing CUBIC_SPLINE,
MONOTONE_CUBIC, AKIMA) and **2 of 5** solvers (missing Brent).

The omissions are **`needed-now`, not `deferred`** — and `needed-now` blocks a tick:
- **Topic 1 requires monotone-cubic / Hyman-filtered interpolation** for curve construction (it is
  the shape-preserving scheme the whole curve discussion in C4 turns on).
- **Topic 1's bootstrap requires Brent** — the quarry's sequential bootstrap solves each pillar with
  `brentq`; bisection is materially weaker for that job.

Ticking these "covered" would be the *looks-covered* assertion the deletable bar forbids, and would
mean un-parking within weeks. **Hold both as partial crosses; they tick at Topic 1 close**, once the
curve work has confirmed nothing further is owed.

## 3. Serialisation ticked covered-by-pattern — ACCEPTED, with one condition

**The framework is genuinely superseded**: we ruled *no framework, per-class `to_dict`/`from_dict` +
`schema_version`*. That is a design decision, not a coverage gap, and the un-implemented per-class
methods are `deferred→persistence` (CP-3 §4.5), which does not block a tick.

**But one class is not a demonstrated pattern.** Only `NumericalConfig` implements it. Before the
tick counts, exercise the pattern on a **genuinely hard case** — one carrying a *nested value object*,
an *enum*, and a *collection*. `Schedule` (tuples of dates) or `FixingHistory` (nested mappings) will
do. Otherwise "the pattern works" is an assertion about the easy case.

## 4. `fixings` split L0 / later — CORRECT, and better than my hand-off

**Your judgement overrides the hand-off.** The hand-off listed `core/fixings.py` in Topic 0's set;
that was wrong. The quarry module is a **mutable store with file persistence**
(`set`/`bulk_set`/`save`/`series`) — state and I/O, which do not belong in L0. ng's `FixingHistory`
is an **immutable read model**, which is exactly the L0 value type. Clean layering split.

One correction: reassign it to the **market-data / persistence topic**, not Topic 1 — that is where
the store and its I/O belong (the same topic that owns the ECB/FRED plumbing we deferred).
The *fixing-lag* logic is correctly already in `accrued_rate`, not the store.

## 5. `PricingResult` `fields-exempt` — LEGITIMATE, not config-smuggling

Confirmed. Its fields (`pv` · `accrued` · `basis` · `cashflow_breakdown` · `sensitivities` ·
diagnostics) are **independent facets of one valuation**; forcing them into sub-objects would give
`result.valuation.pv` and destroy legibility for no gain. This is the `XvaReport` case exactly.

**The distinguishing test, so this never has to be re-argued:**
> A **config** is *input* and groups naturally **by method family** → it decomposes; exemption banned.
> An **output record** is *result* and its fields are **independent facets** → decomposition adds a
> layer without meaning; exemption legitimate.

`NumericalConfig` failed that test (it grouped cleanly into MC/lattice/integration/solver).
`PricingResult` passes it.

---

## Gate status
Blocking: **item 2** (hold interpolation/solvers as partial) and **item 3's condition** (hard-case
serialisation demonstration). Items 1, 4, 5 are re-classifications/confirmations only.
Once those clear — park the Topic 0 set, refresh the roll-up, and Topic 1 opens.
