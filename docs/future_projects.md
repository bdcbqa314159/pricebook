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
the scope line keeps out of this repo. Revisit once the inflation arc is priced and the
representation is stable.
