# Ponytail Debt Ledger

**Date:** 2026-07-19
**Scope:** `/Users/bernardocohen/work/analysis/foundation/`
**Method:** `grep -rn 'ponytail:'` over the tree. Reads and reports only.

Each deliberate ponytail shortcut names its ceiling (the known limit) and its upgrade path (the trigger to revisit). Collected here so a deferral can't quietly become permanent.

---

## interpolation.py

`interpolation.py:75` — scipy spline (cubic / PCHIP / Akima) is rebuilt on every `interpolate()` call instead of being constructed once.
- **ceiling:** O(N) rebuild per evaluation → O(N·M) for M queries.
- **upgrade:** Topic 1 (curve construction) — a hot curve caches the interpolator itself.

> Cross-ref: same hot spot the numerical audit flagged — the per-call rebuild also drives a 2× cost in `_boundary_slope` during extrapolation (`interpolation.py:88`).

---

**1 marker, 0 with no trigger.**

---

## Closure disposition (2026-07-19)

The single marker is **LEDGERED** for tracking (it is a real, correctly-flagged Topic-1 deferral, not a
bug): `OPEN.md` → **AC-PD.1** (= **AC-T4.13**), trigger *Topic 1 — a hot curve caches the interpolator*.
The `ponytail:` marker stays in `interpolation.py` as the in-code record; its ceiling and upgrade path
are intact. Nothing to fix at L0.
