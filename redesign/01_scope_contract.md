# Artifact #1 — Scope Contract (Ratified)

**Status:** Ratified. Breadth decisions ratified 2026-07 (Bernardo): full
cross-asset library stays in the disciplined core; crypto stays live. Remaining open
items flagged inline.
**Target language:** Python. A C++ port is real ambition but an explicit *later* — see NEVER.
**Governing purpose:** A **pro + educational** cross-asset pricing library. Its
*disciplined heart* is rates and credit in EUR/USD/GBP and LatAm/BRL, but **all present
asset classes are in scope and held to the same bar.** "Educational" is a first-class
constraint: code must map legibly back to the paper/book/model it implements — which is
precisely why breadth is kept rather than cut.

---

## Reframe (why this contract, and not a rebuild)

The codebase is **not** structurally broken. Layering is acyclic and machine-verified
every commit; ~12,800 tests pass with ~232 closed-form/reference oracles; debt is
tracked in a live ledger (`OPEN.md`) with a single production TODO and no swallowed
exceptions. The "lost its spine / no oracle / silenced debt" framing is largely
**already addressed** in the current tree.

So this redesign is not a rescue. It is a **discipline pass**: take a broad, working,
well-tested library and make its structure and correctness bar *uniform and explicit*,
so future growth can't erode it. What is actually loose:

- **Three localized coupling defects** — `risk` sits at L3 and switches on instrument
  type; calibration logic is scattered across packages; `registry.py` fans out from one
  top-level module. (→ spine artifact #2.)
- **Schema versioning** — `@serialisable` records type but not version; stored trades
  can't migrate. (→ migration/debt policy #5.)
- **Uneven oracle coverage** — oracles are strong in numerical/rates, thinner in the
  outer asset classes. Discipline means *every* asset class reaches the same oracle bar.

---

## THE BAR (what "in the disciplined core" means)

Every in-scope capability — every asset class below — is held to all of:

1. **One-way dependencies.** Lower layers never import higher ones. Enforced by the
   per-commit acyclic check that already exists.
2. **A red/green oracle per slice.** Correctness is a comparison to a reference value
   (closed form, QuantLib, published number, trusted mark) — never eyeballed.
3. **Legible provenance.** Each model traceable to its source paper/book (the
   educational constraint).
4. **Tracked debt.** Every suppression, shim, or skip is logged in the ledger, never
   silent.

---

## MUST — in scope, full discipline

Currencies for the rates/credit heart: **EUR, USD, GBP + LatAm/BRL.** Other asset
classes carry their natural currency/underlying scope.

**Foundations**
- Time & conventions — dates, day counts (incl. ACT/ACT ICMA, 252/exponential),
  calendars, schedules, compounding. (`core`)
- Core value types & infra — `PricingContext`, quotes, cashflows, serialisation, `db`
  spine. (`core`, `db`)
- Numerical toolkit (finance-free) — MC, PDE, Fourier/COS, optimize, AAD, distributions,
  approximation. (`numerical`, `statistics`)
- Market data & curves — discount/projection/hazard curves, bootstrap, NSS,
  Smith-Wilson; `MarketSnapshot`/`Quote`/fixings. (`market_data`, `curves`, `data`)
- Models — Black-76, Hull-White, G2++, LGM, LMM, SABR, Heston, jump/rough, plus the
  cross-asset dynamics the outer classes need. (`models`)

**Rates & credit heart**
- Fixed income — bonds, swaps, FRA/FRN, repos, inflation, money market, futures,
  sovereign curves. (`fixed_income`)
- Credit — CDS, CLN, CLO, loans, recovery, hazard bootstrap. (`credit`)

**Full cross-asset surface (ratified in scope)**
- Options — swaptions, caps/floors, vanillas, exotics, Bermudan (tree/LSM/LMM),
  barriers, Asians, TARFs, convertibles, variance/dispersion. (`options`)
- FX — forwards, swaps, NDFs, options, barriers, PRDC, smile cubes, EM-FX. (`fx`)
- Equity — forwards, dividends, TRS, variance/vol swaps, autocallables, ELN, index
  futures. (`equity`)
- Commodity — forwards, basis, swing, storage, Schwartz family, carbon, power, freight,
  spread options. (`commodity`)
- Structured — CMS/CMO, notes, hybrids, ABS/MBS/CMBS, OAS, CAT bonds, longevity, real
  estate. (`structured`)
- Crypto — perpetuals, funding curves, crypto options, 24/7 vol, AMM/DeFi, staking.
  (`crypto`) — *live and in scope; needs at least one consumer + oracles to earn its
  place (currently zero production consumers).*

**Cross-cutting**
- Risk — Greeks, XVA, VaR, stress — *relocated above instruments* (see artifact #2).
  (`risk`)
- Regulatory — Basel III/IV, FRTB, SA-CCR, SIMM. (`regulatory`)
- Calibration — unified front over per-family calibrators (see artifact #2). (`calibration`)
- Desks / portfolio / reporting — books, P&L, PE analytics. (`desks`, `pe`, `ts`, `viz`)

## NOT-NOW — real, but deferred within this cycle

Deferred *work items*, not deferred scope. Named so they don't leak into slices early:

- **Schema-v2 migration bundle** — the deferred shim-removal set in `OPEN.md` (LD.1,
  LD.6–LD.10). Coordinated single cut at a release boundary, not piecemeal.
- **Crypto oracles + first consumer** — crypto is in scope but must earn its keep;
  building its oracle set and wiring a desk consumer is scheduled, not immediate.
- **3D interpolation primitive** — vol-cube third-axis handling is currently per-cube;
  central primitive deferred to the L1 numerical pass.

## NEVER — out of scope for this redesign cycle

- **A C++ port / C++ interop.** Genuine future ambition; nothing this cycle is designed
  around it. Revisit once the Python spine is settled and stable.
- **New asset classes** beyond those already present in the tree.
- **Feature parity as a goal.** The redesign disciplines and migrates what exists; it
  does not chase completeness for its own sake. New features arrive as constraint-checked
  slices, never as a parity backlog.
- **Blank-slate rewrite.** The old tree is a read-only quarry; nothing is deleted or
  edited in place.

---

## Open questions before ratification

1. **LatAm beyond BRL** — is "LatAm/BRL" shorthand for BRL only, or does it commit to
   MXN/CLP/COP curves too? Affects the market-data scope.
2. **Crypto's earn-its-keep deadline** — in scope, but do we gate it (must have N oracles
   + 1 consumer by slice X) or leave it open-ended?
3. **`pe` placement** — stays in scope; confirm it becomes a top reporting layer (L7/L8)
   rather than its current L0 artifact placement. (Structural — lands in artifact #2.)
