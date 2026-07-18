# Cowork → Build ruling — spot-check of retire #1 (`core/numerical_config.py`)

First deletable tick audited per `rulings_deletable_definition.md` ("Cowork reviews this first
`shed:` list closely to calibrate the standard"). Version `0.47.0`, drawdown claimed 1/768.

## Verdict: TICK STANDS. Drawdown = 1/768 confirmed.
`core/numerical_config.py` is genuinely superseded — the real residual (`to_dict`/`from_dict`,
`replace`) is closed and no ng module needs the omitted knobs. **First honest non-zero drawdown.**

But the *evidence method* has a false-negative hole, and the *labels* need correcting. Both are
fixed below and are binding on every future retire.

---

## 1. Evidence method — the grep was too narrow (MANDATORY FIX)

The list used `grep '\.<knob>' python/` — **attribute access only**. That cannot see constructor
kwargs, dict keys, `getattr`, `**kwargs` forwarding, or serialisation round-trips.

Re-running on bare names found a real miss:

| knob | claim | audit finding |
|---|---|---|
| `cos_n` | dead | **WRONG — consumer exists.** `tests/test_pricing_context_round_trip.py:125` constructs `NumericalConfig(…, cos_n=2048)` — a *serialisation round-trip* consumer, invisible to `\.cos_n` |
| `tree_steps` | dead (same-named local kwarg) | **correct** — the 4 hits are `compare_engines`/`engine_comparison.py`'s own parameter, not a config read. Well caught. |
| `extra` | dead | **correct** — the 123 hits are a *different* `extra` on calibration types (`spec.extra`, `diagnostics.extra`). `NumericalConfig.extra` is orphaned. |
| other 9 | dead | **correct** — zero bare-name hits. |

**Binding protocol for every future retire.** A `dead` claim requires all of:
1. **bare-name** grep across `python/` (source **and** tests), not `\.name`;
2. explicit check of **constructor kwargs** (`Type(name=…)`), **dict/string keys** (`"name"`),
   `getattr`, and `**kwargs` forwarding;
3. explicit check of **serialisation paths** (`to_dict`/`from_dict` round-trips that carry the field);
4. the shed-list records **how** it was verified (patterns used) and the hit count, not just the verdict.
Anything reachable dynamically is **not** `dead`.

## 2. Labels — `dead` vs `deferred→X` (CORRECTION)

`dead` means *no obligation, ever*. `deferred→X` carries a **real obligation**. The list filed all
12 as `dead`, but its own prose reasoned correctly ("a future ng PDE/COS/tree engine adds its own
knob when it lands") — that is `deferred`, not `dead`. Re-classify:

- **`dead`** (no consumer, no identifiable future one): `mc_antithetic`, `mc_use_sobol`,
  `mc_brownian_bridge`, `extra`.
- **`deferred→X`** (identifiable future consumer in the un-crossed `numerical/` toolkit):
  - `cos_n`, `cos_L` → `numerical/_fourier` (COS)
  - `pde_time_steps`, `pde_space_steps`, `pde_n_std_devs` → `numerical/_pde`
  - `tree_steps` → `numerical/_trees`
  - `integration_tol`, `integration_max_iter` → `numerical/_integrate`
  - `rootfinder_tol`, `rootfinder_max_iter` → `numerical/_rootfinding`

Mislabelling `deferred` as `dead` silently drops the obligation — the exact failure the taxonomy exists
to prevent.

## 3. Forward-linking (NEW RULE)

A `deferred→X` obligation must be written **on X's row** in `quarry_reconciliation.md`, not only on the
retired module's entry. The retired entry is never re-read; X's row is read the moment X is picked up.

> e.g. `numerical/_fourier` row gains: *"on crossing: add `cos_n`/`cos_L` to `NumericalConfig`
> (deferred from `core/numerical_config` retire, v0.47.0)."*

Same for `_pde`, `_trees`, `_integrate`, `_rootfinding`.

## 4. The retire flow (canonical, per module)

1. **Migrate** — build/realign the ng counterpart to cover what's needed.
2. **Read the old module end-to-end** — enumerate everything it has.
3. **Assess each omitted feature's status *in the quarry*** (a fact, not a taste judgment):
   nothing used it ⇒ `dead` · something un-crossed used it ⇒ `deferred→X` (forward-link it) ·
   ng needs it ⇒ `needed-now` (build it; it blocks the tick).
4. **Then tick** deletable. The assessment is the evidence for the tick, so it completes *before* it.

Per-module and just-in-time — never a big upfront pass over all 768.

## 5. Doc hygiene (fix now)
`quarry_reconciliation.md` contradicts itself: the new section reads `Drawdown = 1/768` while the stale
Headline still reads `Drawdown = 0/768 … Nothing is deletable yet`. That file is the metric's source of
truth and cannot disagree with itself. Update the Headline (and keep it updated at every retire).

---

**Actions:** (a) re-label the 12 per §2, (b) forward-link the 5 deferred groups per §3, (c) record the
verification method per §1, (d) fix the Headline per §5. No code change required — the tick stands.
Proceed with CP-3 #2.
```
