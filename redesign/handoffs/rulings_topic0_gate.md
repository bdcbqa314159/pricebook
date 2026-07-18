# Cowork → Build rulings — Topic 0 gate review

Two items raised on the "L0 is done" claim. One is a required fix before the gate; the other is a
standing rule now in `CLAUDE.md`.

---

## 1. REQUIRED BEFORE THE GATE — decompose `NumericalConfig`

The `# fields-exempt: config aggregate` marker I authorised was wrong. **A config aggregate is
exactly the case that must decompose** — configs group naturally by method family, and a flat
15-field bag fails §3b in substance while passing it by exemption.

```python
MonteCarloConfig(paths, seed, antithetic, sobol, brownian_bridge)   # 5
LatticeConfig(time_steps, space_steps, n_std_devs, tree_steps)      # 4
IntegrationConfig(quad_tol, quad_max_iter, cos_n, cos_L)            # 4
SolverConfig(root_tol, root_max_iter, fd_bump)                      # 3

NumericalConfig(monte_carlo, lattice, integration, solver)          # 4  ✓ no exemption
```

- **Remove the `fields-exempt` marker**; `verify.py fields` must pass on merit.
- Reads better too: `numerics.monte_carlo.paths` states what it is.
- *Judgement call:* tree folded into `LatticeConfig` (both are discretisation grids). A separate
  `TreeConfig(steps)` is acceptable — it puts `NumericalConfig` at 5, still legal, slightly less
  coherent. Builder's choice; state which and why.
- **Rule tightened (CLAUDE.md §3b):** `fields-exempt` is for genuinely irreducible **aggregates and
  output records** (`MarketSnapshot`, `XvaReport`), **never for configs**. Reaching for the marker on
  a config means it is not decomposed yet.

## 2. STANDING RULE — onboarding a new asset class (CLAUDE.md §3c)

The recurring question — *"is this fundamental, and is it at L0 or above it?"* — now has a written
procedure, because the L0 membership test alone does not settle it:

```
1. VALUATION?                                   → L3/L4. Never L0.
2. Market state that RISK BUMPS?                → L1 snapshot (A4.2)
3. CONTRACT DESCRIPTION?                        → L2 product data
4. STATE / LIFECYCLE?                           → L6 shell
5. IDENTITY or CONVENTION spoken by ≥2 layers?  → L0
6. Otherwise                                    → its natural layer, NOT L0
```

**Corollary: L0 grows only by identity/convention siblings, never by asset-specific machinery.**
A new asset should add roughly **one** L0 object — its identity, as a sibling under the general
index/underlying concept. Wanting to add more is a smell: stop and route it to Cowork.

**Worked example — credit (next up), so the shape is concrete:**
| candidate | layer | why |
|---|---|---|
| `ReferenceEntity` | **L0** | survival curves keyed by it (L1) · CDS references it (L2) · credit risk bumps by it (L5) — multi-layer identity |
| hazard / survival curve | **L1** | risk bumps it (A4.2) |
| recovery rate | **L1** | market data, bumped |
| credit-event definitions · restructuring clause · seniority | **L2** | contract description; not spoken below it |
| CDS IMM rolls | **L0** | already built (S3) |

Credit therefore adds **one** L0 object. That is the expected shape for every asset class.

---

## Gate
Land the `NumericalConfig` decomposition, then the Topic 0 gate: park the set to
`parked/topic-00-foundation/`, refresh the roll-up, and report. **Topic 1 (yield curves) does not
begin until the gate is green.**
