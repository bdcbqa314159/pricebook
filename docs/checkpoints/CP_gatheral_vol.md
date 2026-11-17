# Checkpoint — the volatility-surface arc (project six, `GATHERAL-2006`)

**Closed 2026-11-17 at `v0.55.0`, 2089 tests, eleven gates green.** The sixth vertical and a capstone of
the volatility complex: the implied-volatility surface as a first-class object (SVI), the derivatives
written on realized volatility, and the model that generates the surface's short-dated skew (SVJ). The
full capstone was committed by Bernardo on 2026-11-16; it built in **three** tranches, not four — T47 was
dropped by the §1 pull gate mid-arc.

## What was built — and what the pull gate dropped
- **T46 — the SVI surface parameterization (ch 3).** `w(k) = a + b[ρ(k−m) + √((k−m)²+σ²)]` in **total
  variance**, fitted to a smile, with the no-arbitrage conditions (Lee's wing bound, calendar monotonicity,
  the Breeden–Litzenberger butterfly density). The arc's crown jewel. `v0.53.0`.
- **~~T47 — surface asymptotics (ch 7)~~ — DROPPED (§1 pull gate).** T46 took the one consumed part (Lee's
  wing bound, which pinned eq 3.20's radical); the quantitative remainder had no consumer in T48/T49, so it
  is `to_go_deeper` route 2, not a build. The pull gate working exactly as flagged.
- **T48 — volatility derivatives (ch 11).** The **volatility swap** `K_vol = E[√V_T]` and its **convexity
  adjustment** `√(K_var) − K_vol`, a Heston computation (because `√V_T` has no static replication). `v0.54.0`.
- **T49 — the SVJ / Bates model (ch 5).** Heston plus Merton lognormal jumps, priced by T33's Fourier
  inversion reused whole, the SVJ characteristic function being `φ_Heston · φ_jump`. The arc's last build.
  `v0.55.0`.

## What the arc reused — the capstone map
The arc rebuilt none of Gatheral ch 1–2 (local vol, Heston — already held, T32/T33). It spent its new
effort on the surface/derivatives/jump objects *on top*: T46 reused the implied-vol surface (T9/T32),
`models/black.py`'s total-variance parameterisation, and **ch 7's Lee bound** to pin its own radical; T48
reused T31 (variance replication) and T33 (Heston) whole; T49 reused T33's Fourier inversion (extracted at
its second consumer to `engine/fourier_inversion.py`, §6) and the T33b `(product, model)` registry. Two
independent sources now cover Heston/local-vol/variance (Gatheral vs `BROCKHAUS-2016`).

## Method advances this arc produced (charter accretions — two, both earned)
- **§4 — bit-identity is a property of a MEASUREMENT, not a committed TEST (T49).** Even where the refactor
  genuinely is bit-identical, that holds only in-process; committing the bit patterns as an equality (or an
  ULP-distance) silently promotes it to a cross-platform claim that §9 forbids. The committed regression
  gets a *tolerance*, and for an accumulated transcendental sum that tolerance is *relative*, not ULP
  (25,475 ULP vs 3.5e-12 relative). A different failure from a *false* bit-identity claim, which is why the
  existing op-order clause did not stop it.
- **§9 — the mutation sweep must ensure the mutated code is the code that RAN (T46).** A mutation written
  within one mtime tick leaves the `.pyc` valid, so CPython runs the *original* and prints a confident
  verdict for the mutant; the applied-count guard is blind to it (the mutation did reach the source). A
  false *"killed"* hides a hole like a false green. Invalidate the bytecode cache per mutant.

## The lessons that did NOT grow the charter — the healthy outcome
- **The double reduction (T49)** — Reduction A (→Heston at `m=0`) and Reduction B (→Merton at `η=0`) *are*
  each other's companion, each blind to what the other tests. It is the T39 discipline made symmetric, the
  arc's best oracle design, and an *instance* of T39, not a new clause.
- **The blind variance-option oracle set (T48)** — the two oracles the spec proposed (zero-strike reduction,
  put-call parity) are both structurally blind (the reduction is carried by an exact term where the
  inversion vanishes — T39; parity is by-construction — T43 at the set level). A Cowork-oracle error caught
  by measurement; ruled `to_go_deeper` route 3 with the lesson recorded, no charter edit.
- **The analytic butterfly = Lee's bound (T46)** — the no-butterfly density's tails are *exactly*
  `b(1±ρ) ≤ 2`, unifying two rulings; an instance of the requirement-oracle discipline.

## Method-watch items carried to re-audit #5 (not yet charter)
- **The lost-glyph recovery pattern — two instances, two sub-kinds, classified not just counted.** When
  both renderings drop a glyph (extraction loss, not a misprint), the recovery is a requirement oracle
  against independent code — but the authority differs: **T46** recovered the radical from a *later chapter
  of the same source* (ch 7's Lee bound — still `GATHERAL-2006`); **T48** recovered the overbar from *our
  own prior tranche's oracled code* (T33's `integrated_variance` — not the source at all). The second is
  stronger and nearer the §5 cliff; the distinction that keeps it legitimate is *oracled code*, not a
  parked `docs/` number. A **third** instance of the T48 kind earns a §4 line roughly: "oracled code from a
  prior tranche is an admissible independent reference where a rendering fails; a parked formula is not."
- **The catalogue's first shape ("a zero makes a factor invisible") twice in one arc** — T46's OTM expired
  option (intrinsic 0) and T48's zero *rate* (`df=1`), both in parameters nobody would flag by reading. The
  sweep caught both. A **third** in an arc opens the question of a proactive first-shape sweep, the
  tenth-shape precedent.

## Honest completion ratio — the deferred ledger
Built as the spine: SVI + its arbitrage conditions, the vol swap + convexity, and SVJ. Deferred to
`to_go_deeper` (oracle-able, unpulled): **ch 7's quantitative asymptotics** (route 2, the dropped T47),
**options on variance** (route 3 — needs an interior-strike absolute-level anchor, most of a tranche),
**Gatheral–Jacquier 2014's closed-form butterfly conditions** (route 1, an acquire-source note — the one
place T46's certificate has a named gap), **SVJJ** (jumps in vol too), **Heston-Nandi** (ch 4),
**default/jump-to-ruin** (ch 6, partly T35's territory), **skew dynamics** (ch 8), **barriers** (ch 9),
**exotic cliquets** (ch 10 — the MC exotic zoo), and SVI/jump **calibration to a live surface**. **Open
debt:** `OPEN.md` **OI-4** — the Heston/SVJ Fourier grid returns a negative call price in the far
short-dated high-vol wing (a real defect on `main` since T33, found at T49), a named re-audit-#5
remediation target; its fix *shape* (adaptive truncation vs a wider constant) is left to that tranche, not
pre-committed, since a wider constant would tax every pricing call to repair a corner.

## Governance this arc
- **The pull gate dropped a queued tranche (T47)** — the first time an arc shortened itself mid-flight on
  the §1 gate rather than by Bernardo's scope call. The discipline, not a scope change.
- **A Cowork-oracle error caught by Code's measurement (T48 variance options)** and **a Cowork spec error
  caught by Code's derivation (T48 convexity order, quadratic not linear; a `T→∞` not small-`T` statement)**
  — both the two-surface split working, both owned, neither grew the charter.
- **The draft-PR mechanism held** — every tranche opened draft, marked ready only at step 6.

## Close
The volatility-surface arc is complete on its builds. The close triggers **re-audit #5** (§9, arc-close
trigger — three builds since re-audit #4 at the O'Kane close, so arc-close fires). Its scope and findings
are `docs/audit/reaudit_05_findings.md`, with the **proactive invariant-6 sweep** as its standing first
item, **OI-4** as a named remediation target, and the two method-watch items above in the register/charter
pass. After it, the library is at a fresh **arc boundary**: the next direction is Bernardo's interest plus
a Cowork spec.
