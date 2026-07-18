# Cowork → Build rulings — CP-3 checkpoint (serialisation cluster, 0 → 5/768)

Reviews `CP3_checkpoint.md` (v0.51.0, 276 green, gates all green). **Cluster accepted.** The
cross-cutting-to-retire pivot delivered its goal: **first honest drawdown, 0 → 5/768**, with zero
debt and the §6b discipline held (declining conventions against a false premise).

## §3 — ratified
Parity gaps are **feature-diffed and systematically overstated**. Keep retiring by **consumer
analysis at pickup**; do not plan from the map's gap lists. Expect drawdown to keep outpacing them.

## §4 rulings

1. **ZCB retired against `FixedCashflow`, no named type — upheld.** A named wrapper with no distinct
   behaviour is the speculative-abstraction smell (§6b). *Escape hatch if legibility bites:* a thin
   named **factory** `zero_coupon_bond(...) -> FixedCashflow` — a name + provenance anchor, no new
   type, no indirection. Add only when a real consumer (docs / example / desk API) wants the name
   (rule-of-two applied to naming).
2. **Self-consistency oracle — accepted** for this capability: round-trip fidelity + version safety
   *is* serialisation's contract, and declining a cross-format oracle against a compatibility we do
   not owe was right. **Cheap strengthening (do it):** make the round-trip **property-based over
   generated instances** rather than one representative — "round-trips *any* valid instance" instead
   of "round-trips this example."
3. **Per-class `schema_version`, no central registry — upheld.** Local and legible; no consumer for a
   global envelope. **Forward-link the trigger:** *when a persisted portfolio/book format needs one
   envelope version.*
4. **Leg encoding inlined in OIS — accepted** (one copy is not duplication). **Detach the trigger from
   swap:** it lifts at **the 2nd serialising leg consumer**, whoever arrives first (inflation, bond,
   swap) — otherwise a delayed swap silently strands it.
5. **Serialisation: `deferred→persistence` (classification) + build-as-you-go (policy).** ✅ See below.

## §4.5 — the ruling that changes practice

**Classification (honest):** serialisation's only consumer is the quarry's `db.py from_dict`
dispatcher — itself un-crossed — and **ng has no serialisation consumer** (persistence was split out
at CP-1 and has not landed). By our own §4 phantom-residual rule it is **`deferred→persistence`, NOT
a genuine residual.** Applied consistently, that means the five retires did not *require* it.

**Policy (deliberate):** we nonetheless **build it per-product while already inside the module**,
because retrofitting serialisation across dozens of products after persistence lands costs far more
than adding it in passing. This is an explicit engineering choice, not a residual claim — the
taxonomy stays honest.

**The binding consequence:** **serialisation must never *block* a tick.** If a module is otherwise
deletable, tick it and forward-link the serialisation to `→persistence`. Do not hold drawdown hostage
to a deferred capability. Where serialisation is cheap and you are already in the module, add it;
where it is expensive or awkward, retire without it and forward-link.

Retro-note the five CP-3 retires in `quarry_reconciliation.md` as *serialisation = deferred→persistence,
built early by policy* — they stand; nothing to rework.

## §5 — smells/debt
Clean. Zero ng suppressions/ignores/skips/TODOs; `fields`, `provenance`, `debt` green; rule-of-two
lifts (`Money`/`Accrual`/`Cashflow`) correctly taken. The headline discipline win — refusing
speculative conventions infra — is exactly the behaviour to keep.

## §6 — next
**CP-3 tail** (`fixed_rate_bond`, `leg`, `inflation`) by consumer-analysis retire-read — and now,
under §4.5, **check deletability *without* serialisation first**; several may tick immediately.

**Then the swap** — genuinely load-bearing (29 production instantiations); its real residual is the
**multi-curve / curve-pillar role**, not serialisation. Analyse whether ng's `ParSwapQuote` +
bootstrap already supersedes the pillar use before assuming a multi-curve build is owed (the deposit
lesson: check consumers before assuming infra).

**CP-4 confirmed** at the first of: (a) fixed-income vanilla cluster retired + a swap decision,
(b) 6 slices since CP-3, (c) multi-curve introduced (immediate-stop trigger).
```
