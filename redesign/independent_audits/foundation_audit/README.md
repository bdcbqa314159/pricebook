# Foundation audit — closed. What's left, and when it comes back.

**Read this file, not `OPEN.md`.** `OPEN.md` is the machine-side ledger with ids, triggers and
disposition mechanics. This is the human-side answer to the two questions you'll actually have:
*is the foundation done?* and *what did we agree to postpone, and when does it return?*

**Status:** Topic 0 / L0 closed after three full passes. **15 items deferred, every one owned by a
named downstream topic.** Nothing is condition-driven, nothing is unowned, nothing can rot silently.

**Branch state (the one thing to action):** all of this lives on `fix/post-closure-seams`, pushed but
**not merged**. `main` sits at **v0.82.0** via PR #114. The v0.83.0 + v0.84.0 work lands when you
merge.

---

## The deferral register — grouped by the topic that will surface it

The rule these obey: *no deferred item unless a downstream topic owns it and will shape it.*
"Waits on an event" was disallowed and those items were closed instead (v0.84.0).

**When you open a topic below, read its rows first.** That is the entire point of this table.

### ► Market data (F1) — next up
| id | item | note |
|---|---|---|
| **AC-3.6b** | FX-spot completion — pair-conventions registry (quote order, triangulation) + the asymmetric ACI intermediate-day rule | **Promoted out of deferral into F1 scope.** Not postponed work anymore. The ACI rule is gated on a citable oracle — we kept joint counting because no primary source with a worked example could be cited, and `fx_spot_date` says so in its docstring. |

### ► Topic 1 — multicurve / curves
| id | item | note |
|---|---|---|
| **AC-T4.5** | `TimeMeasure(anchor, day_count)` | **Ruled (A1).** The only sanctioned `date → t` map — no curve pairs anchor and day-count ad hoc. Built as an **L0 module** in T1's first slice, not a T1 module. |
| **AC-T4.15** | `Frequency` ↔ ICMA `frequency: int` bridge | **Ruled (A2).** `per_year()` raises for tenors with no integer periods-per-year (28D, daily, bullet); BUS-period products (TIIE) don't meet ICMA at all — that's the answer to the undecidable mapping. |
| **AC-T4.10** | guard `log(y ≤ 0)` in interpolation | Fires when a curve first stores negative-carrying series. |
| **AC-T4.13** / **AC-PD.1** | interpolator rebuilt per call | Same item, two rows (one audit, one ponytail marker). Performance only — a hot curve caches the interpolator. |

### ► Models / calibration (Topic 2+)
| id | item |
|---|---|
| **AC-T4.3** | `distributions.py` thin — bivariate-normal CDF, non-central χ² |
| **AC-T4.4** | `least_squares` can't bound — needs `trf` for Feller, \|ρ\|<1 |

### ► Credit
| id | item |
|---|---|
| **AC-T4.6** | CDS-2015 maturity roll (pre/post-2015 conventions) |

### ► EM / per-index rates
| id | item |
|---|---|
| **AC-T4.1** | remaining index declarations — EFFR, BBSW, AONIA, CORRA, TIIE 28D, SELIC, WIBOR/PRIBOR/BUBOR/JIBAR |
| **AC-T4.12** | CDI rate-application trap — returns an annualised exponential rate; an `r·yf` consumer is silently wrong by convexity |

### ► Equity / FX asset classes
| id | item |
|---|---|
| **AC-2.2b** | separate NYSE + Fed-bank/EFFR calendars (different Good-Friday / half-day / observance rules) |
| **AC-2.4b** | Tokyo Silver Week (sandwiched holiday) + 2020/21 Olympic one-off shifts |
| **AC-T4.16** | time-of-day / timezone story — expiry cuts, equity closes |

### ► L6 booking / settlement
| id | item |
|---|---|
| **AC-T4.14** | `Money` is unrounded float, `minor_units` decorative — document so no ledger assumes rounding |

### ► First fixing-cutoff consumer
| id | item |
|---|---|
| **AC-C1** | half-day / early-close concept — the *concept*, after the unused table was deleted |

### ► Not a deferral
**AC-T4.18** — month-arithmetic triplication (3 near-duplicate add-months/EOM helpers). A **recorded
considered exception** under rule-of-three, not postponed work. Reopens only if a 4th consumer
appears. Consolidating now would add indirection with no present consumer.

---

## What the three passes did

**1 · Independent audit closure** — v0.75.0 → v0.82.0
Three adversarial reports, 48 + 8 + 1 findings. **26 fixed with tests.** All five Tier-1 computational
bugs, all four convention bugs, all seven structural gaps. Included `AC-T4.2` (`Tenor` accepting zero
and negative) and `EURIBOR_6M`, both promoted out of the ledger on the reclassification rule: *does
this give a wrong answer or accept invalid input today?* Yes ⇒ it's a bug, and bugs aren't deferrable.

**2 · Post-closure seam pass** — v0.83.0
An independent re-verification executed every audit counterexample cold — 17/17 pass. It then found
what a findings census structurally can't: **the seams between two correctly-FIXED findings.** `A3`
(JointCalendar can't round-trip) sits between 3.2 FIXED and 3.4 FIXED and holds no id at all. Fixed
`B1` (lockout underflow silently freezing rates to early dates), `B2` (`clean` mixing currencies),
`A3`; ruled `A1`/`A2`; scoped `B3`; deleted ~41 lines of residue.

**3 · Standing-rule ledger tightening** — v0.84.0
Applied your rule. Closed the five condition-driven items (`AC-T4.7/8/9/11/17`) that waited on events
rather than topics, promoted `AC-3.6b` into F1 scope, reclassified `T4.18` as an exception.

**Signal worth keeping:** the closure added ~370 lines of audit-mandated machinery and a fresh
over-engineering sweep afterwards found only ~41 lines of residue, most of it pre-existing. Growth
under audit pressure without new speculation is the durable result — more than any individual fix.

---

## Two things that keep this from decaying

**The checkpoint re-reads the whole ledger.** `redesign/11` gained a 6th standing review input:
at every checkpoint, re-read all `AC-*` items and ask whether each trigger has fired. Nothing
automated watches them — `verify.py`'s gates are all silent on deferred capability, correctly, since
a deferral suppresses nothing and counting it would corrupt the debt balance. Human review is the
only mechanism, so it's scheduled rather than hoped for.

**`closed_` is reversible.** A report carries the prefix only when every finding is fixed-with-a-test,
ledgered-with-a-trigger, or rejected-with-reason. If a disposition is later found false, the file gets
renamed **back**. The prefix has to be able to be wrong, or it's decoration.

---

## The files here

| file | what it is | when you'd open it |
|---|---|---|
| `closed_AUDIT.md` | the original 4-way adversarial audit, 48 findings, + closure disposition block | "what exactly was finding 2.2?" |
| `closed_PONYTAIL_AUDIT.md` | over-engineering audit, 8 findings | "why did we delete X?" |
| `closed_PONYTAIL-DEBT.md` | performance-debt markers | "what's the interpolator ceiling?" |
| `closed_POST_CLOSURE.md` | independent cold re-verification, all counterexamples executed | "did the fixes actually work?" |
| `closed_POST_CLOSURE_FINDINGS.md` | the seam findings, actionable list | "what did the second pass find?" |
| `CLOSURE_VERIFICATION.md` | the control document — how to re-check every claim without trusting anyone | "how do I audit the audit?" |

`CLOSURE_VERIFICATION.md` Part 5 is a cold-start runbook: structural checks, the executable check
(revert a fix, confirm its numbered test fails), the four numerical spot-checks, and the ledger
integrity pass.

---

## Bottom line

The foundation is load-bearing. Value semantics with no ambient state, errors that stop at layer 0
instead of falling through, identity flowing coherently from calendar → `RollRule` → `RateIndex`,
and consumers depending on protocols rather than concretions.

**Next:** merge `fix/post-closure-seams`, then **F1 — market data**, carrying `AC-3.6b` as scope.
