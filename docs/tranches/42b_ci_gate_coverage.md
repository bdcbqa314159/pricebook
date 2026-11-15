# Tranche 42b — the gates-gate covers all CI gates, not just `verify.py`

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.49.1` (tooling only). **Source:**
none external — a gate-coverage gap T42 surfaced. **Runs before T43.** The T38 (gates-gate) precedent, one
category wider.

## 0. Why this is a tranche
T38 built `verify.py gates` so a gate **named in §9 but not invoked in `ci.yml`** (or the reverse) fails
CI — closing the "a gate silently stops running" defect. But it covers **`verify.py` subcommands only**.
`ci.yml` also runs **pyright, ruff, and pytest** — real, load-bearing gates (pyright caught a T42 type
defect a local run missed) that are **outside the gates-gate's domain**: any of them can be deleted from
`ci.yml` and nothing notices, because neither the charter nor the gates-gate knows they should be there.
That is the exact T38 defect, one category wider, and it is **statically decidable** (list the named
gates, check each appears in `ci.yml`), so gateable by §9's boundary.

## 1. The change
- **Name the load-bearing CI gates in §9.** The gate list gains `pyright`, `ruff`, `pytest` (alongside the
  `verify.py` subcommands) — the checks whose silent removal would let a defect ship. This edit lands **in
  this tranche, with the gate that enforces it** (the T38 lesson: the charter must not name a practice its
  gate does not yet enforce).
- **Extend `verify.py gates`** to assert each **named** gate is invoked in `ci.yml` — whether it is a
  `verify.py` subcommand or a standalone step (`pyright`, `ruff`, `pytest`). The existing checks stay: the
  reverse direction (*every `ci.yml` `verify.py` invocation is named*) remains scoped to `verify.py`
  subcommands, because classifying an arbitrary `ci.yml` step as gate-vs-scaffolding (checkout, setup-python)
  needs judgement — a check needing an exemption list is the τ=1 over-reach (§9), so it is **not** built.
  Named-forward only for the non-`verify.py` gates.

## 2. The oracle (seeded drift — the T26b/T38 pattern)
In `tests/tooling/test_gates_gate.py` (extend): on a copy of the real `ci.yml`, **delete the `pyright`
step** → the gate rejects (it is named in §9, not invoked); **delete `pyright` from the §9 list too** →
the gate passes (named-and-invoked agree on its absence — the gate checks agreement, not presence of a
fixed set, so removing a gate *consistently* is allowed, exactly as for a `verify.py` subcommand). Same
for `ruff`, `pytest`. The **real repo passes**. Mutate the gate (make it ignore non-`verify.py` names) →
the pyright-deleted test fails. This is "a gate that can't fail is empty" (§9) applied to the extension.

## 3. Process
Branch `tranche/42b-ci-gate-coverage`. `test` (RED: the seeded-drift tests failing on the un-extended
gate) → `feat` (extend `verify.py gates` + add the three names to the §9 list) → `docs` →
`chore(release): v0.49.1`. The `gates` gate must stay green. `source:` none; `crosscheck: none — a
CI-config integrity check`. **Raise an `Ask:`** only if a CI gate's invocation in `ci.yml` is not a clean
grep (e.g. pytest invoked indirectly through `verify.py tests` rather than as its own step) — then rule
what "invoked" means for that gate rather than guess.

## 4. What this is NOT (scope fence)
No numerics, no `src/` change. Not a classifier of every `ci.yml` step (scaffolding vs gate — needs an
exemption list, declined). Not a new CI gate — it protects the ones already running. T42b is the one
coverage extension: the named load-bearing CI gates (pyright, ruff, pytest) can no longer be silently
dropped.

---

## 5. Cowork's ruling on T42b — (reserved for §10 step 4, after the build)
