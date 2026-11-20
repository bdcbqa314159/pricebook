# Checkpoint — the commodity-options arc (project eight, `CLARK-2014`)

**Closed 2026-11-20 at `v0.66.0`, 2625 tests, eleven gates green.** The eighth vertical: the
**commodity** layer — the forward built from cost-of-carry and **convenience yield**, Black-76 on futures,
the closed-form exotics a commodity desk trades (geometric and arithmetic Asians, Margrabe/Kirk spreads),
and a **mean-reverting (Schwartz) spot model** giving the Samuelson term structure of volatility. Committed
by Bernardo on 2026-11-19 as "a big spine, balanced with to-go-deep"; it built in **five** tranches, every
one closed-form or closed-form-plus-MC-companion, and it is the arc with the **most reuse** the tree has
seen — the genuinely-new code each tranche was a formula or two, the engine almost always `black_price`.

## What was built
- **T60 — the commodity forward + Black-76** (ch 2.1–2.2). `F = S·e^{(r−δ)T}` with convenience yield; the
  option engine **reuses `black_price`** (Black-76 already existed — the step-1 grep caught it). The source
  gave a free stronger oracle: Clark prints the derivation in steps, so the pre-martingale form checked by
  **Gauss-Hermite quadrature** anchors the closed form. `v0.62.0`.
- **T61 — geometric Asian, exact closed form** (ch 2.7.3, Kemna–Vorst). The geometric average is lognormal,
  so it is Black-76 on an adjusted forward + variance (the `Σ min(tᵢ,tⱼ)` double sum). The `n=1` reduction
  is **exact in ℝ but not bit-exact** (a `log`/`exp` round trip) — banded, with the gap's **invariance in
  volatility** proving it a pure round trip. `v0.63.0`.
- **T62 — arithmetic Asian, Turnbull–Wakeman** (ch 2.7.4). No exact closed form (a sum of lognormals is not
  lognormal); TW moment-matches. The MC companion **measures** the approximation error (0.41%→4.81% across
  σ√T, crossing 0.5% at σ√T≈0.12 — the honest negative, in the CHANGELOG). `v0.64.0`.
- **T63 — spread options: Margrabe + Kirk** (ch 2.9). They are **one function** — Black-76 with the second
  forward as strike, Kirk weighting the short vol, K=0 recovering Margrabe exactly. The finding:
  **Kirk degrades in correlation, not strike**, and a crack spread is a highly correlated pair, so the
  approximation is least trustworthy where the desk most uses it (+8.37% at ρ=0.9). `v0.65.0`.
- **T64 — the Schwartz one-factor mean-reverting model** (ch 2.10.1), the capstone. OU log-spot, closed-form
  futures and Samuelson option vol; the MC companion **reuses `simulate_lgm_states`** because the Schwartz
  state **is** the LGM state scaled by `e^{−κT}` (an exact identity, eq 2.150b). `v0.66.0`.

## What the arc reused — why "big" was affordable
The whole arc rode on three existing pieces: **`black_price`** (T60's Black-76, and every option engine
after it — the Asians and both spread legs project onto it), **`mc_estimator`** (the Asian and Kirk MC
companions), and the **LGM/OU simulator** (`simulate_lgm_states` — T64's Schwartz MC, the same object that
anchored T57's short rate). The genuinely-new code each tranche was a formula or two: the convenience-yield
forward, the two Asian double sums, the combined spread vol, the Schwartz futures and Samuelson variance.
The step-1 reuse grep (the §10 edit) paid again at T60 — Black-76 "existed but was misunderstood," the T54
shape caught at spec time.

## The method advance this arc produced — one charter clause, earned
- **§4 — a spec names the oracle's OBLIGATION, not the FORM of its result.** Three instances
  (T61 `==`, T62 "resolution per strike", T63 "monotone Kirk error in strike"), each a spec pre-committing
  the *shape* of a measured answer the mathematics does not supply, each caught by Code verifying rather
  than ratifying. Ruled clause-worthy at the third because it is a **bending-invitation** pattern (every
  instance a live chance to bend the build toward the stated answer), distinct from the
  derivation-structure watch which is a *missed-opportunity* pattern (clause-worthy only when it
  discriminates). The distinction — "a temptation rather than a gap" — is the operating lens for when a
  watch item crosses to a clause. It binds the reviewer's own instruction, and it bound mine immediately:
  T64's oracle 4 measures the vol term structure rather than asserting the Samuelson monotonicity.

## The lessons that did NOT grow the charter — the healthy outcome (fifth arc running)
- **Cowork's own spec errors, caught by the method.** T64's κ→0 reduction claim was **wrong** — I asserted
  it reduces to the cost-of-carry forward without transcribing that the Schwartz drift has no `r−δ`; the
  limit is driftless-log GBM (88.51, not T60's 83.98). A **§5 miss** (a spec asserting a model fact the
  source contradicts), caught by Code re-transcribing and the reduction oracle going red — **not** a fourth
  instance of the §4 form-clause, kept off that count. The corrected anchor (agree at `δ=r−σ²/2`, >20% apart
  at genuine κ) is strictly stronger. Carried as a method-watch: a spec naming a reduction target/model fact
  is a transcription claim to re-derive at spec-time (the §10 reuse-grep's mathematical sibling) — but
  self-catching (a measured reduction cannot be bent to pass), so watched, not claused.
- **`SchwartzParams` its own type, not `HullWhiteParams` + a level** — a shared SDE is not a shared idea
  (contrast LGM, whose numbers *are* Hull-White's reparameterised). §6, no accretion.
- **Carry the composite `α̂_X`, document the decomposition** — T11 identifiability arriving at *design time*:
  the physical level and the market price of risk reach every futures price only as one composite, so the
  type carries what is observable and names the unidentifiable parts in docs. §6 + T11 applied, no clause.
- **The two Asian products share a guard, not a value object** — rule-of-two fired on the family, but the
  bundle was *not compelled* (both at the five-field limit), so guard-only extraction (the shared
  ascending-schedule knowledge, zero contract change) over a 42-site value object. §6/§7.

## Honest completion ratio — the deferred ledger
Built as the spine: the convenience-yield forward + Black-76, geometric and arithmetic Asians, Margrabe/Kirk
spreads, the Schwartz mean-reverting model. Deferred to `to_go_deeper` (oracle-able, unpulled):
**partly-fixed (seasoned) Asians** (needs a commodity fixing series — recorded at T61), **American options /
the lattice** (ch 2.6), **two-factor / stochastic-convenience-yield models** (ch 2.10.2–2.10.3),
**calendar and Asian spread options** (ch 2.9.3–2.9.4), **Schwartz market-forward calibration** (`α̂_X` fit,
surfaced at T64). Out-of-method / market-specific (ch 3–9): vol surfaces from quotes, **seasonality /
deseasonalised forward curves**, electricity spike models, weather/freight — quotes-and-plots, no
closed-form oracle. **Open debt:** `OPEN.md` **OI-7** alone (the `MarketSnapshot` `# fields-exempt` marker,
re-open at the next market-data arc that adds a keying dimension).

## Governance this arc
- **Cowork stayed one spec ahead** the whole arc (T61–T64 written before each merge), so Code never waited.
- **The journal capture discipline held** — a raw beat appended at every step-4, live, the first arc since
  the ~15-tranche slip and the new raw-bar + checkpoint-backstop rule. The journal is current to T64.
- **The draft-PR mechanism held**; every tranche opened draft, marked ready only at step 6.

## Close
The commodity arc is complete on its builds. The close triggers **re-audit #7** (§9 — arc-close trigger, and
the cadence lands here, ~eight tranches since #6). Scope and findings in `docs/audit/reaudit_07_findings.md`:
the proactive invariant-6 sweep, the arc's new closed forms first (the T39/T43/T17 concentrations — the
Schwartz κ→0 companion, the spread correlation pins, the Asian double sums), and the standing full-tree
sweep. After it, the library is at a fresh **arc boundary** — Bernardo's interest (a new reference is
likely) plus a Cowork spec.
