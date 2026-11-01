# Future projects (parked ideas)

Downstream projects that *consume* this library — out of scope for the library itself (which stays
analytics-only), recorded so the idea is not lost. None of these is a tranche.

## FP-1 — the narrative / explainer project (Bernardo, 2026-10-31)

**Idea.** A separate project built *on* the library that retells a source's story — for the
inflation work, Berhan's: nominal vs real curves, break-even inflation, the JY-vs-market-model
comparison, how calibration behaves — rendered from the library's **own** priced numbers.

**The honesty principle it is built around, which generalises beyond this one source.** When the
source material is a PDF whose results are *plots, not tables* (Berhan's Figs 1.1, 1.2, 5.1–5.10),
we cannot and should not claim to reproduce its exact figures. So the narrative:

- reproduces the **shapes and relationships** the source demonstrates (break-even below nominal at
  the short end; convexity growing with correlation; a model that fits ATM but not the smile), using
  numbers the library computes and that are internally self-consistent;
- states plainly that it is **not** the source's exact dataset — the source did not publish one —
  and attributes every number to the library + its oracle, never to the paper's figure;
- leans on the same thing the library's oracles do: the **structure is the truth that transfers**
  (a reduction, a sign, a monotonicity), not a specific basis-point value read off a plot.

This is the presentation-layer analogue of the library's own "figures are a visual check only,
numbers come from closed forms" rule — the right way to honour a source you cannot fully see into.

**Why it is a separate project.** It is a narrative/presentation surface (a consumer of the
representation contract), not pricing analytics. It belongs with the GUI / dashboard / server family
the scope line keeps out of this repo.

**Activated 2026-11-01 (Bernardo): a Substack series.** A quant-finance publication whose through-
line is *working good books and papers by building them* — quant the subject, programming the
vehicle, each resource-backed implementation doubling as teaching. Berhan's inflation thesis is
**project one**; finishing the book (the T7→ tranche ladder) is what generates the material.
- **Authored by Cowork, kept OUT of the library's code** — drafts live in the workspace / a sibling,
  not `src/` or this repo's tracked tree, so §0's analytics-only scope is untouched. Bernardo
  publishes (a Substack connector or a paste — sorted when the first piece is ready).
- **Cadence: backlog-first.** The six built tranches (the whole inflation arc + the bugs the oracle
  discipline caught) are a strong opening; then one piece as each new tranche lands.
- **Governed by the honesty principle above** — structure, not a source's pixels; every number
  attributed to the library and its oracle.
