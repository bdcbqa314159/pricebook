# Cowork → Build ruling — CP-3 premise corrected (serialisation, not conventions)

Responds to the CP-3 #2 retire-read finding. **The build was right; my ruling's premise was wrong.**
Independently verified here: `grep 'Deposit(' python/pricebook/` = **0** (24 hits, all in tests);
`from_convention`'s sole caller is `tests/test_convention_factory.py`. Drawdown **2/768** confirmed.

## 1. Correction accepted
`rulings_CP3_direction.md` scoped CP-3 #2 as *"conventions/RateIndex → retire deposit."* That premise
does not survive contact with the quarry: the `Deposit` class has **no production instantiation**
(the bootstrap uses a loose `deposit_day_count` param, never the class), ng already supersedes every
role (product cashflows + `DiscountingEngine`; curve-pillar via `DepositQuote`), and the only
production-reachable reconstruction path was the DB dispatcher (`db.py from_dict`). **Deposit's real
residual was serialisation — identical to CP-3 #1.**

**Declining to build conventions/RateIndex was correct**, and notably it applied *my own* discipline
against *my own* ruling: no ng consumer ⇒ speculative infra ⇒ violates §6b and the "every cross-cutting
slice must retire ≥1 module / watch sprawl" rule. The guardrails held even when the ruler was wrong.
That is the system working as designed.

## 2. CP-3's true through-line: **serialisation** (ratified)
`config → deposit → products`. Continue on serialisation as the cross-cutting piece; it is what the
retire-reads actually keep surfacing as the production-reachable residual (the DB `from_dict`
dispatcher is the live reconstruction path).

**Conventions/`RateIndex` is deferred with no obligation now** — it re-aims at its genuine consumer
(per-currency curve construction) when that is crossed. Not a residual; not owed.

## 3. Root cause + the new binding rule (now `CLAUDE.md §4`)
My error: I derived cross-cutting targets from the reconciliation map's **residual lists**, which were
built by **feature-diffing** (quarry-has vs ng-has), not **consumer analysis**. A residual nothing
consumes is a **phantom residual**.

> **A residual needs consumer evidence, symmetric to a `dead` claim.** Feature-diffing overstates
> residuals. Re-derive by consumer analysis at retire time; never plan cross-cutting work from a
> feature-diffed gap list.

**Implication worth acting on:** the map's parity gaps are probably **systematically overstated**
across the board, because the whole map was feature-diffed. Many "residuals" will evaporate under
consumer analysis — as deposit's did. **Drawdown may move considerably faster than the map suggests.**
Re-derive each module's residual when it is picked up, and don't treat the current gap lists as
authoritative planning input.

## 4. Adjustment to how Cowork rules
Going forward Cowork rules **direction, discipline, and adjudication of evidence** — not specific
cross-cutting targets inferred from surface reading. **The retire-read is authoritative** on what a
module actually needs. If a ruling's premise fails contact with the quarry, the build should override
it and report (exactly as done here), not implement against a false premise.

---

**Net:** tick #2 stands (2/768). Continue CP-3 on serialisation. Re-derive residuals by consumer
evidence as each module is picked up; expect the remaining gaps to be smaller than the map claims.
```
