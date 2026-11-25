# Re-audit #9 — the vol-derivatives arc close (v0.81.0, T79, 2026-11-26)

Arc-close trigger (7 tranches since #8 at T72). Run with independence (the `analysis/` copy pointed at
`main`). Code's sweep (1)+(2) in thread 0431; Cowork rulings in 0432; (3)+(4) passes below. **Zero `src`
defects** — the ninth re-audit running, every hole a blind oracle or hygiene, as every time.

## (1) Oracle power — sweep: 108 applied / 106 killed / 2 survived / 0 errored / 0 single-killers
| item | applied | killed | survived |
|---|---|---|---|
| proactive invariant-6 | 76/76 | 76 | 0 |
| the arc's closed forms | 16/16 | 14 | 2 → **F-9.5** |
| single-killer table (counting mode) | 7/7 | 7 | 0 |
| oldest tree T1–T37 | 9/9 | 9 | 0 |

Highlights: the invariant-6 run **doubled** #8's 32 sites (tree-generated discovery found guard + negated +
positive comprehension shapes); **OI-3 proved NOT re-introducible** (restoring `norm_cdf`'s cancelling form
dies — T20c holds, F-8.6's floor never resurrected the real defect); every standing single-killer at/above
its last count (MM-I 2/2, F-1.2's shipped-hole second killer 2, T4's YoY 3/3/2). Tree byte-identical to HEAD
afterwards (F-8.2 check), suite green at 3286.

## Findings and rulings (Cowork, 0432)

**F-9.5 [the one real hole] — a REGIME blind oracle in `vix.py`'s quadrature span; production correct,
remediate test-only.** Both survivors are in `_survival_integral:138-139`. The span centre
`mean = scale·(dof+nc)` is correct, but every fixture sits at `nc = 2…34`, where the 12-σ span absorbs a
mis-centred grid; at `nc ≈ 438` (`v0 ≫ θ`, low `κ`, short `T`) a mutant truncates the integral and the price
is materially wrong, unreached. The T66/T67 "measure the sound regime per model" applied to a quadrature's
*placement*, sibling to the T74 per-weight-grid lens. **Ruling:** add a **targeted high-`nc` fixture** pinning
the centre formula; **do NOT tighten `_SPREAD_SPANS`** (shrinking a production correctness margin to make a
test fire is §9's trade-the-oracle-for-the-gate, inverted); the `spread` factor-2 (absorbed everywhere
reachable) → a per-weight span-adequacy assertion + a margin-watch (T79 shape). OI-8 absorbs the ~1.4s — the
§4 "contain the failure" cost that is not cut. → remediation tranche.

**F-9.1 [hygiene] — `F401` gateable now; close it.** `ruff.toml`'s stated precondition ("once modules carry
`__all__`") is met (`engine/__init__.py` has `__all__`, 67 names). **Ruling:** add the 11 missing names,
**delete the 5 dead imports** (esp. the two dead `is_historical` imports that misrepresent where the
invariant-6 guard lives), gate `F401` zero-exemption; resolve `_interest_rate_swap` to public-with-`__all__`
or not-package-imported by its actual use. → remediation.

**C17 + F-9.3 [duplication] — consolidate.** C17 `implied_carry = ln(F/S)/T` is **7 sites / 5 modules**
(`bates`×2, `heston_fourier`×2, `heston_mc`, `time_changed_levy`, `timer_option`) → one atom (L0/L1), the
C16 pattern. F-9.3 the positive-horizon `ValueError` is **4× byte-identical** → one guard helper, the F-7.2
`time_grid` pattern. **Ruling:** both into the remediation tranche. The registered-pricer skeleton (13×/8×)
stays boilerplate-by-design (Code's non-finding, C16's ruling — recorded).

**F-9.2 [method] — adopt "a count that enters a finding or register row is parser-derived, not eyeballed".**
Three wrong counts this session (invariant-6 24-vs-38, F-9.1's 11, C17's row), each a grep read not an `ast`
derivation; the C17 row's error propagated into the 0429 scope document (§5 reaching the audit's own brief).
The F-7.1 two-parsers / F-8.6 mechanical-audit family applied to counts. **Ruling:** adopt the practice; it
is a dynamic property (not statically gateable — §9's boundary), so the sweep/`ast` is the backstop, like
τ=1. **No new charter clause** (lean-charter; the family exists). Fix the propagated damage — the C17 row's
count — in the remediation.

**F-9.4 [non-finding] — the `SmileLike` T33b probe premise was wrong** (no production `isinstance` guard,
only a pyright annotation), so the hazard can't apply and the protocol is sound; Code's logged prediction
held. My spec over-specified the probe. Recorded, no action.

**F-9.6 [register] — pin the T26 single-killer row by PATH.** The standing table names it in prose (the
cause of the T3/YoY conflation at #6); Code measured 11 at the plausible site. **Ruling:** confirm it is the
T26 numéraire divisor and rewrite the row by path (if 11, it is no longer single-by-design). → remediation.

## (3) Register accuracy — CLEAN
`references.md` (the two new defect-log entries — eq 3.30 argument-order, Prop 4.8 — plus
`ZHAO-CUI-FLORESCU-2018` `worked-from` and `PAGES-2018` `in-hand`) current; no stale "in progress" headers.
`to_go_deeper.md` carries the §6.2.2 numerical-track pull, C17, gamma/corridor MGF. `OPEN.md`: OI-8 and OI-9
both open with live removal triggers. Arc plan marked closed. `audit_log.md` #9 row updated to the result.
CHANGELOG top = `__version__` = 0.81.0. One propagated error (C17's count) folded into the remediation per
F-9.2.

## (4) Charter staleness — CLEAN, plus the direction evolution → Bernardo
No clause stale or in tension: the arc's two method events filed as instances of existing clauses (T34, T43),
zero new clauses. One standing watch: **§4's size** (the degenerate-fixture catalogue is now large) — a
condensation pass recorded as a candidate tooling/docs tranche, not done here. **And the arc-boundary
ruling §1 reserves:** the two-track + 1.0-purpose amendment is drafted at `docs/proposals/
two_track_and_1_0_purpose.md` with the first structural arc at `docs/arcs/R1_representation_mtm.md`, for
**Bernardo's §1/§9 ruling**. Re-audit #9 is the "ripe" moment §1 names.

## Remediation
One **test/tooling/hygiene tranche** (the T45/T59/T65/T73 precedent): F-9.5's high-`nc` fixture + the
`implied_carry`/guard consolidations (C17, F-9.3) + F-9.1's `F401` close + F-9.6's row-pin + the
F-9.2/F-9.4 records. Zero `src` defects. After it merges, re-audit #9 closes and — on Bernardo's ruling —
the structural track opens with R1.
