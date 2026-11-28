# pricebook — library status

> **Live status page.** The **inventory block is generated** by `tests/tooling/status.py` from the tree and
> **drift-gated in CI** — a stale page fails the suite, which is the only reason the numbers here can be
> trusted (§7's guarded-not-folklore line, applied to a status page). Everything between the `curated`
> markers is **judgement** — asset-class readiness, the bibliography, the 1.0 scorecard — refreshed by hand
> at each re-audit (§9's (4) pass). Regenerate with `python tests/tooling/status.py --write`.
>
> **IP note:** structure and status only — never a transcribed formula or source line, asserted by the
> instrument's own test — so the page stays safe to render publicly.

<!-- generated:start -->
### Inventory — derived from the tree, regenerated and drift-gated

- **Models:** 39
- **Engines:** 49, dispatching **36** `(product, model)` pairs
- **Products:** 25

**Greeks and book-level risk are DETECTED, not asserted** — the columns below are derived from the risk layer, so they cannot drift. A hand-curated greeks cell said "Black MC only" and was wrong on this instrument's first run.

| Asset class | Engines | Greek fns | Book-level risk |
|---|---|---|---|
| commodity | 4 | — | — |
| credit | 8 | — | `credit_valuation_adjustment`, `funding_valuation_adjustment`, `wrong_way_mc`, `wrong_way_risk` |
| equity | 7 | 4 | `equity_cva`, `sensitivities` |
| inflation | 5 | — | — |
| rates | 12 | — | `collateral`, `collateralised_capstone`, `mc_exposure` |
| volatility | 7 | — | — |

**Representation (R1) — 3 of 5 built.** Detected by presence, so this column fills itself as the structural track proceeds.

| Piece | Built | What it is |
|---|---|---|
| `Trade` | **yes** | a holding: product + units + direction (R1.1) |
| `Position` | **yes** | trades in one product, netted (R1.2) |
| `Portfolio` | — | positions across products (R1.2) |
| `PricingRequest` | **yes** | what to value, as of when, against which market (R1.3) |
| `MarkToMarket` | — | the valuation procedure over a book (R1.4) |

**Unclassified engines (2):** `position`, `pricing_request` — reported rather than dropped, so the table cannot look complete when it is not.
<!-- generated:end -->

<!-- curated:start -->
## What this is
A **pricing-and-risk analytics library** where every number is verified against the book or paper that defines
it. Functional core, imperative shell: pricing is a pure `price(product, model) → result`. The goal (proposed
1.0 purpose): **price *and* risk a position, for the asset classes a desk trades, ingesting any market
provider through the representation contract — to retire the expensive third-party black box.**

## At a glance
- **Version:** see `CHANGELOG.md` (the `version` gate keeps them equal)
- **Spine:** L0 foundation · L1 market · L2 products · L3 models + calibration · L4 engines · L5 risk · L6 shell
- **Verification:** ~3,250 tests (floored in CI), 11 structural gates, mutation-swept; **8 whole-tree re-audits**
- **Projects built (literature arcs):** inflation · Monte-Carlo infrastructure · XVA · equity & hybrids ·
  single/multi-name credit · volatility surface · collateralised discounting · commodities · stochastic rates
  · volatility derivatives *(closing)*

## Bibliography — where we stand with the literature
Every built capability is verified against a held source. `worked-from` = held and built/oracled against;
`in-hand` = held, not yet built from. (Source PDFs are gitignored for IP; only this status is tracked.)

| Source | Domain | Status | Delivered / deferred |
|---|---|---|---|
| BERHAN-2012 | Inflation derivatives | worked-from | ch 1–5 built; later ch → `to_go_deeper` |
| JAECKEL-2002 | Monte-Carlo methods | worked-from | 9 of 14 ch (Sobol, bridge, variance reduction, MC Greeks); rest deferred |
| CREPEY-2026 | XVA | worked-from | elementary XVA spine built; ML/BSDE engine → `out_of_method` |
| BROCKHAUS-2016 | Equity & hybrids | worked-from | equity/hybrid/CEV built |
| OKANE-2008 | Single/multi-name credit | worked-from | CDS, baskets, CDO, base correlation built |
| GATHERAL-2006 | Volatility surface | worked-from | SVI surface built |
| LICHTERS-2015 | Collateralised discounting | worked-from | OIS/multi-curve, LGM, CSA exposure built |
| CLARK-2014 | Commodities | worked-from | forward, Asian, spread, Schwartz built |
| MCINERNEY-ZASTAWNIAK | Stochastic rates | worked-from | Vasicek→G2++→HJM→Bachelier→CEV; §8.5 SABR → `out_of_method` |
| KWOK-ZHENG | Volatility derivatives | worked-from *(current arc)* | variance swaps, options-on-variance, VIX, 3/2, Lévy; timer (T79) to come |
| ZHAO-CUI-FLORESCU-2018 | 3/2 transform (paper) | worked-from | the 3/2 Laplace transform (resolved a book defect) |
| PAGES-2018 | Numerical probability (MC/American/Greeks) | in-hand | shelved; the structural/numerical track's reference |
| KROESE-TAIMRE-2011 · MILLER-EDELMAN-APPLEBY-2007 · OVERHAUS-2002 | MC / numerical / equity | in-hand | shelved; mined when a topic pulls them |
| *candidates (not yet held)* | Seydel · Savine · C. Alexander | — | trees+PDE, AAD, market risk — for the structural track |

## Coverage — asset class × depth (the JUDGEMENT half)
`●●● deep · ●● solid · ● partial · — absent`

**Greeks and book-level risk are NOT here** — they are *detectable*, so they live in the generated
inventory above and cannot drift. This table carries only what needs a judgement: how deep the pricing
and calibration go for each class. A hand-curated greeks cell read "Black MC only" and was wrong within
the same re-audit that wrote it; the generator found the fourth function on its first run.

| Asset class | Price | Calibrate |
|---|---|---|
| Rates | ●●● | ●● |
| Credit | ●●● | ●● |
| Volatility / variance | ●●● | ●● |
| Equity | ●● | ●● |
| Inflation | ●● | ●● |
| Commodity | ●● | ● |
| **FX** | **—** | **—** |
| Counterparty / XVA | ●● | — |

**Reading:** pricing is broad and deep; calibration is broader than it looks (curves, SVI, Heston, HW,
CEV, copula, JY). **FX is the one untouched asset class** — and it is the literature arc after R1,
deliberately, as the test that the representation is currency-general. For greeks and risk, read the
generated table: the one annotation worth adding, which the generator cannot know, is that equity's greek
functions are **Monte-Carlo only — there are no analytic greeks anywhere yet**, which is R2's subject.

## The representation layer — the known gap (why it can feel like a catalog)
At the pricer level this is a genuine library: one dispatch spine, one `price` contract, a shared,
actively-de-duplicated foundation. **Above the pricers, the representation layer §0 promised was deferred:**

| Representation piece | State |
|---|---|
| Product (the economic contract) | ✅ built (frozen dataclasses per asset class) |
| `MarketSnapshot` (market data the model carries) | ✅ built |
| **Trade / Position / Portfolio** | ❌ absent — cannot represent *a book* |
| **Pricing / Valuation request** (§0's other half) | ❌ absent |
| **Calendar / holiday / business-day engine** | ❌ absent |
| Schedule structure | ◐ half-unified (one foundation + ~15 per-product builders) |
| **Mark-to-market procedure** (value a position vs market) | ❌ absent |

This layer sits *above* the 45 engines and reuses them — building it does not rewrite them, and the verified
closed-form catalogue becomes its oracle set.

## Direction — a two-track rhythm, switched at the re-audit
- **Literature arcs** (book-grounded): add pricing capability for a model/asset class; oracle = the source.
- **Structural arcs** (library-level, pluggable with any asset/model): representation, greeks, scenario,
  P&L, lifecycle, XVA-on-position, time-series; oracle = the catalogue + decomposition/convergence identities.

The whole-tree re-audit (every ~arc close) is the switch point: assess the tree, pick the next track.

**Next (proposed):** vol arc → re-audit #9 → **structural arc: representation & risk** (R1 Trade/Position/
PricingRequest → R2 mark-to-market → R3 schedule+calendar rework → R4 position greeks → scenario/P&L →
lifecycle → XVA-on-position → time-series/estimation) → re-audit #10 → **FX literature arc** (the test that
the representation is currency-general).

## The 1.0 scorecard (proposed finish line — purpose, not module count)
- [ ] **Represent** a position (trade / portfolio / pricing request)
- [ ] **Value** a position against any provider's market data (mark-to-market)
- [ ] **Risk** a position (greeks, scenario/stress, P&L attribution)
- [x] **Provider-agnostic by design** (§0 — the library owns the representation; providers adapt into it)
- [ ] **Desk-complete** (price + calibrate + greeks + risk) for the chosen asset classes
- [ ] Statistical-oracle tier ruled (needed for VaR backtesting / time-series) — §4 extension, open

## How this page stays honest
Regenerated at each re-audit; mechanical rows tooling-emitted (proposed, so they cannot drift); judgement
rows curated. A stale status page is worse than none — the §7 *guarded-not-folklore* rule applies to it too.
<!-- curated:end -->
