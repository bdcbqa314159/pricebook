# HAND-OFF — L0 final cleanup (last work before Topic 1)

From the full L0 codebase audit at v0.74.0. **The audit passed on everything substantive** — all
dataclasses frozen · no mutable registries rebound at import · no import-time I/O · zero debt markers
· provenance complete on all 16 modules · internal import graph a clean DAG · `day_count → calendar`
correctly `TYPE_CHECKING`-only · `Currency.minor_units` a single source of truth · 125 tests.

Three cleanups remain. One slice, then Topic 1.

---

## A1 — `Accrual` is mis-homed; the import graph proves it

`Accrual(start, end, day_count)` sits in `cashflow.py`, but its consumers are `cashflow`,
`rate_basis` **and** `rate_index`. That forces the edge:

```
rate_index -> cashflow
```

which is semantically wrong: **an interest-rate index has no business knowing what a cashflow is.**
`Accrual` is a *time-period* value object (a period that knows its day count), not a cashflow concept.

**Fix: move `Accrual` → `day_count.py`.** It is cohesive there (`Accrual.year_fraction()` delegates
straight to `year_fraction`; the type *is* an applied day count), and it adds **no new edges** —
`cashflow` and `rate_index` both already import `day_count`. Net effect:

```
rate_index: 8 imports -> 7,  and the cashflow edge disappears
```

*(A dedicated `accrual.py` is an acceptable alternative if `day_count.py` gets crowded; the point is
that it leaves `cashflow.py`.)*

**Oracle:** pure move — every existing `Accrual`/`year_fraction`/`accrued_rate` test stays green
(§8 pure-refactor carve-out: no new RED required, but run them). `verify.py acyclic` green.

## A2 — registries are plain mutable dicts

`_CALENDARS`, `_REGISTRY` (rate indices) and the currency registry are module-level `dict`s, private
by convention only. Given "frozen everywhere" is this tree's ethos — and given the **quarry's bug was
precisely registry mutation** — wrap them in `types.MappingProxyType` after population. Near-free
hardening; the `get_*` accessors are unchanged.

**Oracle:** attempting to mutate a registry raises.

## A3 — record the `Currency` trade-off, so nobody "fixes" it back

`setattr(Currency, _code, ...)` makes `Currency.USD` work but **invisible to type checkers and
autocomplete**. That is the deliberate price of S1's open registry — and without a written rationale
someone will reasonably "improve" it back into an enum in six months, which **silently removes BRL and
the whole LatAm scope**.

**Fix:** an explicit note in `money.py`'s module docstring — *why* `Currency` is an open registry, what
reverting to an enum would cost, and the pointer to the ratified scope contract. No code change.

---

## Then Topic 0 is closed for good
```
[ ] A1 Accrual moved; rate_index no longer imports cashflow; acyclic green
[ ] A2 registries MappingProxyType; mutation raises
[ ] A3 rationale recorded in money.py docstring
[ ] full L0 suite green; verify.py all green (layers · fields · acyclic · debt · version · provenance)
[ ] version PATCH; no parking change (the 13 are already parked)
```
**Then Topic 1 — multicurve + linear rates.** L0 covers its base.
