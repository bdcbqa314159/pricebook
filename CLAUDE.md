# CLAUDE.md — pricebook charter

The law for this repo. When anything you would do conflicts with this file, this file wins. It is
deliberately lean: the detail lives in `docs/`, this is the enforceable and ruled subset. It is
also a teaching text — legible beats clever, everywhere.

---

## 0. What this is

A **pricing analytics library** where **every number is verified against the book or paper that
defines it**. Functional core, imperative shell: pricing is a pure function
`price(product, model) → result`; everything stateful is a thin shell around it.

**Analytics only.** In scope: products, models, calibration, engines, risk, the market-data value
types, and the **representation** of a product and a pricing request (the durable public contract).
Out of scope, built in *other* projects that consume the representation: GUI, database, broker/data
feeds. The library is pluggable, not plugged in. (Downstream ideas are parked in
`docs/future_projects.md`.)

---

## 1. The working loop — product-pull, driven by interest

The unit of work is **a product worth having**:

```
name a product  →  plan what it needs (method, model, core pieces)
                →  check the stock for each piece
                →  purify it (adapt to our conventions) or build it fresh
                →  ORACLE it against the held source
                →  reflect it into the library
```

Interest picks the target; the discipline is the constant; the books and papers are the oracle.
Top-down: the product drags in exactly the core it needs and nothing it does not. There is no
module-count finish line — the library grows by interest. `0.x` holds through the build; **`1.0.0`
is when Bernardo declares the public API and the representation contract stable**, not a count.

---

## 2. The spine (layers — dependencies point DOWN only)

```
L6  LIFECYCLE / SHELL        stateful; remembers, never computes
L5  RISK & CAPITAL           on the engine + a Priceable protocol
L4  ENGINES                  the stateless heart: price(product, model) → result
L3  MODELS + CALIBRATION     carry the MarketSnapshot they were calibrated to
L2  PRODUCTS                 pure data — describe, never price themselves
L1  MARKET DATA              immutable MarketSnapshot · curves · fixings · vols
L0  FOUNDATION               time, money, value types — FINANCE-FREE
     _stock/ (side)          read-only reference; never imported by the tree
```

A lower layer never imports a higher one (`verify.py acyclic`). Layer membership is **semantic,
not just acyclic** (`verify.py layers`): L0 is finance-free (no strikes, vols, payoffs, call/put,
discounting *in code* — a docstring explaining finance context is fine). Products are frozen data;
they do not price themselves — pricing lives in L4.

---

## 3. The stateless engine contract (L4)

`engine.price(product, model) → PricingResult | PricingFailure`. The **model carries the
`MarketSnapshot` it was calibrated to** (`model.market`); the engine reaches market data only
through it, never as a second argument, so a model/market mismatch cannot be expressed.

1. **Referential transparency** — identical inputs ⇒ identical output.
2. **No ambient state** — no clock reads; "today" is `model.market.valuation_date`.
3. **No mutation** — product, model and snapshot are frozen.
4. **Failure is a value** — return `PricingFailure`, never raise-and-hope or a silent `NaN`.
5. **Config is explicit, and deferred to its first consumer** — a numerical-config value arrives
   with the first engine that reads it; never threaded speculatively (the dropped `numerics`).
6. **Valuation-date-aware (invariant 6)** — a flow on or before `valuation_date` is historical:
   excluded from the mark, **PV 0** (the shell's business), never discounted with non-positive `t`.
   `PricingFailure` is for inputs that *cannot* be priced, never for a settled trade.

---

## 4. Oracles — the heart of the method

Nothing lands without a **named, green oracle** proving it against a known value. Preference order:
**closed form > external cross-check > self-consistency > trusted mark.** "Runs and looks right" is
not an oracle. Ruled principles:

- **Pre-register the oracle.** The failing reference-value test (RED) is committed *before* the
  code that satisfies it. Implement-then-check invites bending the code to a number.
- **Independence.** The test computes its own reference (e.g. `t = days/365`), never reads a value
  back from the code under test. The predecessor's 5.14% bug passed because its oracle restated the
  builder's own composition.
- **Reduction is the preferred anchor, and it needs a companion.** A capability that generalises
  another is anchored by the **exact reduction** to the old case at the degenerate parameter
  (`σr → 0` ⇒ the deterministic limit; `I(t)=I0` ⇒ the inception formula) — exact by construction,
  better than a numerical limit. It **requires a companion oracle exercising the non-degenerate
  general case**: a reduction a no-op implementation also satisfies is vacuous.
- **Self-consistency telescopes.** A par-reprice-to-zero is satisfied by construction when
  construction and pricing share a convention — *even a wrong one*. Keep it as a consistency guard;
  the correctness anchor is a non-telescoping check of the structure itself (a pointwise identity,
  a count, an independent transcription).
- **No number is fitted to a figure.** When a source's results are plots, not tables, reproduce
  its structure, not its pixels; numbers come from closed forms, an external engine, or the stock's
  regression values — never read off a chart.

---

## 5. The stock — content, never authority

`_stock/` holds the whole previous tree, gitignored, **mined for CONTENT (what is true —
conventions, formulas, edge cases, reference values), never for STRUCTURE**, and never imported. A
stock file's organisation carries no authority here.

**The stock is content and regression; the held source is authority** — for *every* reader,
including whoever writes the oracle plan. Oracle anchors and formulas are read from the source, not
inferred from the stock's code, which may be a weaker or different object. *(Precedent: the T4
anchor was mistakenly taken as `ρ → 0` from the stock's first-order `yoy_convexity_adjustment`; the
source's eq 2.5.11 makes it `σr → 0`. Caught at the handoff by pre-registration.)* Each purified
piece earns its own oracle; an imported-but-unoracled convention is worse than an absent one.

---

## 6. Simplicity — solve the real problem

- **Nothing speculative.** No hooks, config knobs or extension points for hypothetical futures.
- **No abstraction without two present consumers sharing a SHAPE, not a word.** A real leg and a
  year-on-year leg share the word "leg" and not the payoff — so they stay separate types, no base
  class. Rule-of-two triggers on a shared shape.
- **Compelled bundling is vocabulary, not abstraction.** Grouping cohesive fields to meet the limit
  (`Term`, `HullWhiteParams`, `RealLeg`) is required, not speculative — kept **concrete and frozen**,
  no polymorphism. Whether it generalises is decided at the second real consumer.
- **A carried-but-unread field is allowed only when constitutive of the instrument's identity /
  required for faithful representation** (and then it is exercised by the round-trip) — never
  "the pricer might read it later." The test is constitutive-of-identity, not does-the-pricer-read
  it. *(An indexation lag distinguishing two otherwise-identical linkers: carried. A publication
  lag the index name already implies: not.)*

---

## 7. Signatures, fields, conventions, references

- **Max 5 arguments per function and 5 fields per dataclass** (`verify.py fields`, ruff `PLR0913`).
  Over the limit is **un-bundled vocabulary**, never a reason to raise it: group into a frozen
  value object (`Money`, `Term`, `HullWhiteParams`, …). **Never suppress** (`# noqa` / `# type:
  ignore`) — fix the cause.
- **Cross-module conventions** (shared day counts, the time measure, discounting dimension) live in
  `docs/conventions_register.md` with a three-state Guard column — **guarded** (a test compares the
  modules), **per-module** (each tested, nothing asserts they agree), **owed** (no oracle). A shared
  convention with no structural guarantee needs a cross-module oracle; until it has one it is
  recorded, never folklore.
- **References.** Every provenance `source:` names a key in `docs/references.md`, and may cite only
  an `in-hand`/`worked-from` key. A work merely *referenced-by* a held source is not citable until
  obtained — no reference by inference.

---

## 8. Provenance, debt, cross-checks

- **Provenance header on every `src/` module:** `stock:` · `source:` · `oracle:` · `slice:`
  (`verify.py provenance`).
- **No debt is silenced silently.** Every suppression, shim, skip or load-bearing TODO goes in
  `OPEN.md` with a rationale and a re-open trigger, or it does not exist. CI invariant: suppressions
  − ledger entries = 0. Debt may never buy a green oracle.
- **Deferred external validation is tagged, not ignored.** A `# crosscheck: <library> <thing> —
  deferred …` marker sits at every site that wants a known-library cross-check, so the optional
  QuantLib tier (a later, dev-only, skippable tier — never in the reproducibility contract or a
  merge gate) is complete by construction, not from memory.

---

## 9. Gates, branches, flow

- **`verify.py`**: `acyclic` · `layers` · `fields` · `debt` · `version` · `provenance` · `slow` ·
  `tests` (floored). CI runs ubuntu + windows, Python 3.12. Tolerance-based oracles are a hard rule
  (transcendental results differ 1–2 ULP across runtimes); the byte-identical statelessness check
  is same-process only.
- **One branch per tranche** (`tranche/<NN>-<slug>`), commit sequence `test (RED) → feat (GREEN) →
  docs → chore(release)`. One version bump + one `CHANGELOG.md` entry per tranche; CI asserts
  `__version__` matches the top entry.
- **The merge method is not legislated here.** The draft said *"land by squash-and-merge, the only
  GitHub merge that leaves `main` both linear and Verified"* — but **every PR to date (#1–#5) landed
  as a merge commit**, so the clause described a practice the repo does not follow, and a charter
  stating a false practice is worse than one that is silent. The signed-and-linear argument is a
  migration-era concern Bernardo has explicitly set aside as not worth the attention, and nothing
  about it is broken: the RED→GREEN sequence is preserved either way, on `main` under a merge commit
  and on `refs/pull/<N>/head` under a squash. If it ever matters, it becomes law then, from a
  ruling — not from a doc describing an aspiration.
- **Review before merge.** Nothing merges until the pre-merge pass is in.

---

## 10. Code ↔ Cowork

**Code leads the build**: proposes each tranche (target, pieces, stock-to-purify, oracle) and builds
it; owns code, tests, git, and **verification by execution** (Cowork cannot run the suite and never
marks anything passing). **Cowork reviews before merge** — an independent read of the design and the
oracle choice, fresh eyes — rules design and contract questions (a layer, a vocabulary type, a
measure), and has the external reach (sources, documents). The prompt channel is `.cowork/thread.md`
(append-only, one `Ask:` per entry, pointers not pastes); durable state goes to `docs/`, never only
the thread.
