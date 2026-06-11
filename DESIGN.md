# Pricebook — Theoretical Design

A first-principles design document for a financial analytics engine, written
independently of what we currently have, and then used as the lens through
which to read pricebook today.

The first three sections (Principles, Reference Architecture, Patterns vs
Anti-patterns) describe *what a quant analytics engine should look like*,
without reference to pricebook. The remaining sections (Pricebook through
this lens, Delta list, Roadmap) overlay pricebook on the reference and
produce a prioritised list of changes.

This is a living document. Pushback is expected on sections 1-3; the layer
cut and the calibration placement (S1.3) are the load-bearing decisions —
everything downstream is downstream of those.

---

## Table of contents

1. [Principles](#section-1)
   - 1.1 [What a financial analytics engine is for](#section-1-1)
   - 1.2 [Non-functional requirements](#section-1-2)
   - 1.3 [The domain shape](#section-1-3)
2. [Reference architecture](#section-2)
   - 2.1 [Layering rationale](#section-2-1)
   - 2.2 [Cross-cutting concerns](#section-2-2)
   - 2.3 [The model abstraction](#section-2-3)
   - 2.4 [The trade and portfolio abstractions](#section-2-4)
   - 2.5 [Math kernels vs financial logic](#section-2-5)
   - 2.6 [Python vs C++ — the right cut](#section-2-6)
3. [Patterns vs anti-patterns](#section-3)
   - 3.1 [Industry references](#section-3-1)
   - 3.2 [Common pitfalls](#section-3-2)
   - 3.3 [Concrete shapes](#section-3-3)
4. [Pricebook through this lens](#section-4)
   - 4.1 [Where we match — keep as is](#section-4-1)
   - 4.2 [Where the shape is right but coverage incomplete](#section-4-2)
   - 4.3 [Where the shape is wrong](#section-4-3)
   - 4.4 [Anti-patterns visible in the codebase](#section-4-4)
5. [Delta list](#section-5)
   - 5.1 [High-value adds](#section-5-1)
   - 5.2 [Wrong-shape refactors](#section-5-2)
   - 5.3 [Nice-to-haves](#section-5-3)
   - 5.4 [Won't-fix](#section-5-4)
6. [Roadmap](#section-6) — five gates (G1-G5), ten phases (P1-P10), 59-79 slices total

---

## Executive summary

**The principled design** (sections 1-3) lands on a 9-layer architecture with calibration as its own layer (L6) parallel to risk, both depending on pricing (L5). Models are `Protocol`s, not inheritance trees. Trades are frozen dataclasses; the `PricingContext` is the only place market state lives, passed in by every call. Risk is *orthogonal* to instruments — it depends on the `Pricable` protocol, never on concrete instrument classes. Calibration produces transferable `CalibrationResult` artefacts with provenance; pricing consumes parameters, never market quotes. Scenarios are first-class composable functions on contexts. AAD is supported at the kernel level by making L0 generic over the scalar type. The Python/C++ cut is empirical, not ideological: hot loops migrate, scaffolding stays Python forever.

**Pricebook today** (section 4) matches the reference design *better than expected at the boundaries* — the layer count is comparable (7 actual vs 8 in the reference), models already use Protocol-style composition in many places, `PricingContext` exists as a real type, and the dependency graph is verified acyclic. But it diverges in three structurally important places:

1. **Risk lives at L3 alongside instruments**, not at L6 above pricing. This is the largest structural mismatch and the most consequential to fix — the current placement forces every risk module to know about specific instrument classes, exactly the anti-pattern Section 3 argues against.
2. **Calibration is distributed**, not a layer. Calibrations live next to the models, curves, or instruments they fit; there is no `CalibrationResult` type, no provenance trail, no audit story.
3. **Market data is not a layer.** Quotes, fixings, and the interpreted curves they feed into share the same layer, which makes it impossible to say crisply "this number came from a quote vs this number came from a fit."

**The delta list** (section 5) sorts the changes by leverage. The top three are: introduce a `CalibrationResult` type and a calibration layer; move risk out of L3 into a meta-layer that depends only on `Pricable`; split market data from curves into its own L1. Each is one or two slices of work, with broad downstream impact.

**The roadmap** (section 6) is structured as **five gates** (each a user-visible promise) decomposed into **10 phases** (each one architectural focus). G1 Audit-ready (right types in place) → G2 Production-grade (failure handling, scenarios, persistence) → G3 Architecturally clean (risk relocated, structural cleanups) → G4 Capability-complete (AAD protocol, payoff algebra) → G5 Performant at scale (C++ port). Total: 59-79 slices, **7-19 weeks** at the pricebook slice rate. G1 (~14-16 slices, 1-3 weeks) is the prerequisite for the bottom-up module audit, which is the next major task after the design is accepted.

---

<a id="section-1"></a>
## 1. Principles

<a id="section-1-1"></a>
### 1.1 What a financial analytics engine is for

A financial analytics engine exists to answer one question, repeatedly, under every imaginable perturbation: **what is this position worth, and how does that value move when the world moves?** Everything else — calibration, scenarios, capital, P&L attribution — is a corollary of that single question asked in different ways. State this baldly because it has design consequences: any abstraction that does not serve the value-and-sensitivity question is overhead, and the cost of overhead in this domain compounds badly (you pay for it in every bump, every MC path, every overnight batch).

The load-bearing capabilities, in strict order of precedence:

1. **Pricing.** A function from `(trade, market_state) -> value`. If this is wrong, nothing downstream matters. Pricing is *the* contract; everything else negotiates with it.
2. **Sensitivities (risk).** First and second derivatives of value with respect to market inputs. The whole reason quants exist as a job function is that traders need to hedge, which means they need derivatives, fast.
3. **Calibration.** A function from `(market_quotes, model_class) -> model_parameters` such that the model reprices the quotes. Calibration is *upstream* of pricing in the data flow, but *downstream* of pricing in the design — you cannot design calibration before you know what consumes its output.
4. **Scenario analysis and stress.** Deterministic perturbations of market state and re-pricing. A special case of risk, but with named perturbations rather than infinitesimal ones.
5. **P&L attribution.** Decomposition of `Δvalue` over a time step into contributions from market moves, time decay, carry, and unexplained residual. The residual is the diagnostic — large residual means the risk model is incomplete.
6. **Regulatory capital.** A specific, prescriptive set of pricing-and-sensitivity computations dictated by Basel/FRTB/SA-CCR/SIMM. Treat it as a downstream consumer with a frozen calling convention, not a first-class capability — let it pull from the same primitives the front office uses.
7. **Reporting, persistence, automation.** Everything that turns numbers into decisions. Important, but not load-bearing for the engine's correctness.

When these conflict — and they will — the order above is the tie-breaker. Concrete example: if a faster calibration would compromise pricing reproducibility, do not take it. If a more elegant risk API would slow pricing by 10%, the pricing wins. Performance lives below correctness; ergonomics lives below performance for the inner loop and above performance for the outer loop.

A note on what is **not** the engine's job: portfolio construction, alpha generation, execution, settlement. These consume the engine; they are not it. Conflating them produces god-systems that no one can refactor.

<a id="section-1-2"></a>
### 1.2 Non-functional requirements

Pick four properties to be load-bearing. Everything else is desirable but negotiable.

**Correctness — operationally defined.** "Correct" cannot mean "the author believes it is correct." It must be falsifiable. Three concentric tests:

- *Closed-form anchors.* For every model with a known analytical case (Black-Scholes against MC under GBM, Hull-White swaption against Jamshidian, Heston against Lewis), the numerical implementation must match the closed form within a published tolerance. These are the golden references.
- *Cross-engine cross-check.* Any instrument priceable by two independent numerical methods (MC vs PDE, COS vs FFT, tree vs MC) must agree within tolerance. A surprising amount of subtle wrongness is caught by forcing two methods to converge to each other.
- *Round-trip invariants.* Calibrate to quotes, reprice the quotes, recover the inputs. If the round-trip fidelity is below the calibration tolerance, the model is mis-specified or the optimiser is wrong — either way, fail loud.

Correctness is not a test suite property; it is an architectural property. The architecture must make it cheap to add a new closed-form anchor and a new cross-engine check whenever a new model lands. If adding those is expensive, the architecture is wrong.

**Reproducibility.** Given the same trade, the same market data snapshot, the same code version, the same machine — the same number, to the bit. This implies:

- All randomness routed through explicit, named RNG streams. No implicit `np.random.default_rng()` calls anywhere. The MC engine takes a seed; if it is not supplied, it raises.
- All "current date" parameters explicit. No `date.today()` inside a pricing function, ever. The valuation date lives on the `PricingContext` and nowhere else.
- All numerical tolerances explicit and versioned. `solver_tol=1e-10` belongs in a config object that is serialised with the result, not in a default argument that someone might change.
- Build determinism. Pinned dependencies; no floating versions for numerics-critical libraries (NumPy, SciPy in particular). A reproducibility break on a SciPy minor version is a real outage, not a theoretical one.

Reproducibility is not just nice-to-have. It is the precondition for *every* downstream diagnostic: if you cannot reproduce yesterday's number, you cannot do P&L attribution, you cannot debug a calibration regression, you cannot defend a regulatory submission.

**Observability.** When a calibration fails, "calibration did not converge" is a useless error. The system must answer: *which* parameters were being fit, *to what* quotes, *under what* tolerance, *with what* initial guess, *what residual at termination*, *which quote had the largest contribution to the residual*. Concretely: every numerical routine returns a structured result, not a bare float — `(value, diagnostics)`, where `diagnostics` is a typed object the caller may ignore but never a routine may omit. The diagnostics carry the *story* of how the number was produced, in a form a quant can read at 2am.

**Evolvability at the boundaries that matter.** Two boundaries should be cheap to extend; everything else is allowed to be rigid.

- Adding a new *instrument* should require: define the cash flows, declare which model interface it dispatches to, register a serialisation tag. No edits to risk, calibration, or reporting code.
- Adding a new *model* should require: implement the model protocol, register it under a name. No edits to instruments.

Anything that violates these two evolvability rules — a switch statement in the risk engine listing every instrument type, a model that knows about specific trade classes — is technical debt by construction, not by accident.

I am deliberately *not* listing performance as a top-tier NFR. Performance is per-layer and per-call-site; making it a global property leads to premature optimisation everywhere and the wrong optimisations in the inner loop. State it locally: the MC inner loop, the curve evaluator, the bumping kernel are hot; everything else is cold and may be implemented for clarity first.

Likewise, auditability is implied by reproducibility + observability + serialisation. It does not need to be a separate axis.

<a id="section-1-3"></a>
### 1.3 The domain shape

The standard data flow — market data → curves → models → instruments → portfolios → risk → reporting — is right in its rough shape and wrong in two specific places.

**Where the layers are right.** Market data is genuinely upstream of curves (curves are *fit* to data). Curves are upstream of models (a short-rate model is calibrated to a discount curve; a vol surface is built on a discount curve to discount expected payoffs). Instruments need both curves and models to price. Portfolios are aggregations of instruments. So far, uncontroversial.

**Where risk goes.** Risk is downstream of instruments in the data flow but it must not depend on specific instruments in the code. The right model is: risk is *a transformation* of `(instrument, market_state) -> sensitivities`, and it does its work by manipulating `market_state` and re-invoking `instrument.pv(...)`. So risk depends on the *protocol* `Pricable`, not on `Bond` or `Swap`. This is the only way to keep the risk layer from becoming a 5000-line dispatch table. The phrase "risk downstream of instruments" is true at the data-flow level and false at the code level — risk is *orthogonal* to instruments and depends only on their protocol.

**Where calibration goes — this is the interesting question.** The naive answers are all wrong:

- "Calibration belongs with the model." Wrong because calibration takes *market quotes* (which the model layer does not know about), produces *parameters* (which the model needs), and chooses *which quotes to fit* (a business decision, not a model decision).
- "Calibration belongs with the curve." Wrong for the same reason: curves are *bootstrapped* (which is a degenerate calibration), but a SABR fit to swaption vols is not a curve operation.
- "Calibration belongs with the instrument." Wrong because calibration fits to *many* instruments at once.

The correct answer: **calibration is its own layer**, parallel to risk, depending on models and on a market-data abstraction, producing typed `CalibrationResult` objects that models consume. A calibration is a *contract*: "these quotes, this model class, this objective, this tolerance, produced these parameters, with this residual, in this many iterations, with this convergence story." The result is a first-class citizen — versioned, serialisable, stored. Calibration *runs* are auditable artefacts, not transient inputs.

This has a direct architectural consequence: a model does *not* call a calibration routine in its constructor (eager calibration). A model is constructed with parameters; if those parameters came from a calibration, the `CalibrationResult` is carried alongside as provenance. Calibration and pricing are *decoupled in time*; you can calibrate at 9am and price all day with the result.

Putting it together — the layer shape:

```
L0  foundations:    math kernels, conventions, types
L1  market data:    quotes, fixings, indices (raw, observed)
L2  curves:         discount, survival, vol, fitted from quotes
L3  models:         pricing models, parameterised
L4  instruments:    trades, contracts, payoffs
L5  pricing:        instrument.pv(context) — the central operation
L6  derived:        risk, calibration, scenarios — orthogonal transforms
L7  portfolio:      aggregation, attribution
L8  delivery:       reporting, persistence, regulatory, automation
```

Calibration sits at L6, not on the data-flow path between curves and models. It produces artefacts that *feed* L3 (model parameters) and L2 (curve fits), but the dependency edge is data flow, not code import. In code, calibration imports from L3; L3 does not import from calibration.

---

<a id="section-2"></a>
## 2. Reference architecture

<a id="section-2-1"></a>
### 2.1 Layering rationale

The nine layers above are not arbitrary. Each cut is defensible:

- **L0 → L1.** Math kernels (PDE solvers, MC engines, root finders, AD) must not know about money. A Crank-Nicolson scheme is the same whether it discretises Black-Scholes or the heat equation. Putting financial vocabulary into `numerical/` couples the math to the finance and prevents both from evolving. Test: would a physicist use this code unchanged? If yes, it belongs in L0.
- **L1 → L2.** Market data is what you *observe*; curves are what you *infer*. The split forces you to be honest about which numbers came from quotes and which came from fitting. The day a curve construction bug shows up in production, you need to be able to point at the inputs and the fit separately.
- **L2 → L3.** Curves are objects you evaluate (`df(t)`, `vol(K, T)`); models are objects that have *dynamics* (an SDE, a tree, a PDE). The distinction is that a model can simulate forward; a curve cannot. This matters because risk against curves and risk against models is different (key-rate bumps vs model-parameter bumps), and the type system should reflect that.
- **L3 → L4.** Models are pricing engines that take *parameters*; instruments are objects that have *cash flows* and *terms*. An instrument knows it pays a fixed coupon on these dates; it does not know the discount factor. Keeping cash flows and engines separate is what lets you reprice the same swap under three models without rewriting the swap. This is the single most important boundary in the system.
- **L4 → L5.** Instruments declare their structure; pricing applies a model to the structure. The reason this is a separate layer rather than a method on the instrument is the next sub-section.
- **L5 → L6.** Pricing is the primitive; risk and calibration are *meta*-operations on pricing. They must depend on pricing; pricing must not depend on them. Otherwise the simplest case (price one swap) has to drag in the entire risk and calibration machinery.
- **L6 → L7.** Portfolios aggregate; they need risk + pricing, not vice versa. A single trade does not need to know about its portfolio.
- **L7 → L8.** Reporting and regulatory pull from everything below; nothing depends on them. If your regulatory module is imported by anything other than the entry point, the dependency is backwards.

The single most common architectural mistake in quant libraries is collapsing L4 and L5 — making pricing a method on the instrument that knows about specific models. This produces the QuantLib `Instrument`/`PricingEngine` pair, which is closer to right than wrong but suffers from inheritance overuse; and it produces the worse pattern of `bond.price_with_hull_white(...)` proliferating across the codebase.

<a id="section-2-2"></a>
### 2.2 Cross-cutting concerns

Cross-cutting concerns are the test of an architecture. They must traverse the layers without creating cycles.

**Serialisation.** Lives at L0 as a *protocol* (the `Serialisable` interface, a `to_dict`/`from_dict` contract, a global registry of tags), and is *implemented* by every L4 instrument and L3 model. Serialisation must not introduce a backwards dependency from L0 to higher layers — the registry is populated by decorators that run when L4/L3 modules import, not by L0 reaching upward. This is the only correct way to do it in Python; any attempt to centralise the registry in L0 ends up importing from L4 and creates a cycle.

The serialisation contract is opinionated: every serialised object carries a *type tag*, a *schema version*, and the *data*. No magic, no pickle, no automatic reflection of `__dict__`. The schema version is what lets you migrate; without it, you are committed to never changing a field name.

**Calibration.** Lives at L6. Takes a `MarketQuoteSet`, a model class identifier, a calibration spec (objective, weights, tolerance, optimiser), produces a `CalibrationResult` (parameters, residuals, diagnostics, provenance). The crucial design move: calibration *does not return a model*; it returns *parameters*. The caller assembles the model from the parameters. This decouples "I have parameters" from "I have a model object" and makes the calibration result a transferable artefact.

**Scenarios.** A `Scenario` is a transformation `market_state -> market_state`. Lives at L6 alongside risk. Scenarios compose (parallel shift then key-rate bump then vol shock). The composition algebra is what makes a stress framework usable; without it, every named scenario is a one-off and the system rots.

**Automatic differentiation.** The hardest cross-cutter. AD must work at L0 (the math kernels need to support dual numbers / tapes), but the *application* of AD to compute Greeks is an L6 concern. The clean split: L0 provides AD-aware primitives (a dual number, a tape, an AD-compatible solver); L3 models can be written generically over the scalar type; L6 risk *applies* forward or reverse AD by passing AD scalars into the pricer. The model writer does not opt in to AD — the type discipline of L0 makes it free. This requires that L0's primitives are written against a numeric protocol, not against `float` directly, which is the single biggest constraint on the L0 design.

**Persistence.** L8 concern. Persistence depends on serialisation but not vice versa. The repository pattern: `TradeRepository`, `MarketSnapshotRepository`, `CalibrationResultRepository`. Each is a thin wrapper over the serialisation layer plus a database. The database is an implementation detail; the repository is the contract.

**Testing.** L0 through L8, but the architecture imposes structure on tests: every model gets a closed-form anchor test, every instrument gets a cross-engine test, every calibration gets a round-trip test. These are not optional; they are the operational definition of correctness. Test code lives in `tests/<layer>/` and follows the same layering as production code (test code for L3 may depend on L0–L2 fixtures but not on L4 instruments).

**Validation.** A pricer may produce a value that is structurally valid (a finite float) but financially nonsense (negative volatility, negative survival probability). Validation is *inside* the model — never at the boundary. The boundary trusts the type; the type was built with invariants enforced in its constructor. `Volatility(0.20)` raises if you pass `-0.20`; `SurvivalCurve` raises if monotonicity is violated at construction.

<a id="section-2-3"></a>
### 2.3 The model abstraction

Pricers are not classes inheriting from a god-base. Pricers are **structural types** (Python `Protocol`) implementing a small set of operations, registered into a dispatch table.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EuropeanIROptionPricer(Protocol):
    def price(self, payoff: IRPayoff, market: PricingContext) -> PricingResult: ...
    def implied_vol(self, payoff: IRPayoff, market: PricingContext, target: float) -> float: ...
```

Why protocol, not inheritance:

- Inheritance trees in QuantLib became a tax: `Instrument <- Bond <- FixedRateBond <- AmortisingFixedRateBond`, with overrides at every level and no clear answer to "where does the price method live?" Protocols flatten this. There is one contract; everything that implements it is a pricer, regardless of internal structure.
- Protocols are nominal-light. A new model written by a team that has never read the codebase but implements `.price(...)` correctly is automatically a pricer. No subclass declaration, no `super().__init__()` archaeology.
- Protocols enable mocks and fakes trivially. The test suite is a primary consumer.

**Stateful vs stateless pricers.** A `Black76Model` is stateless: vol comes in, price goes out. A calibrated `HestonModel` is stateful: it holds `(v0, kappa, theta, xi, rho)`. The Python idiom: stateless pricers are *callables* or *modules*; stateful pricers are *frozen dataclasses* with a `price` method. Either way, **state is immutable after construction**. There is no `model.calibrate(...)` that mutates the model in place. Calibration returns a new model (or, preferably, returns parameters and the caller builds the new model). This makes models hashable, cacheable, and parallel-safe by construction.

**Composition with risk.** Bumping is mechanical: take a `PricingContext`, perturb one field, re-call `pricer.price(...)`. Because models do not own market data — the context is passed in — bumping is a context transformation and a re-call. No model needs to know about risk.

**Composition with calibration.** Calibration is `optimise(parameters -> objective(parameters, quotes, model_class))`. The model class is a constructor; the optimiser proposes parameters; the closure builds a candidate model and prices the quotes. Because models are cheap to construct (just frozen dataclasses), constructing thousands of them in an optimiser inner loop is free.

**Dispatch.** A registry maps `(instrument_type, model_class) -> pricer`. The registry is populated at import time by decorators. The user-facing call `price(instrument, market, model=None)` looks up `(type(instrument), model or default_for(instrument))`. When the lookup fails, the error message lists which models *are* registered for that instrument — observability at the dispatch layer.

<a id="section-2-4"></a>
### 2.4 The trade and portfolio abstractions

A `Trade` is **a frozen dataclass plus a thin interface**. It owns:

- Its economic terms: notional, currency, dates, schedule, coupon, payoff specification.
- Its identity: trade id, counterparty, book.
- Its serialisation contract.

A `Trade` *delegates*:

- Pricing — to a model, via the dispatch registry.
- Cash-flow projection — to a cash-flow generator, which is a pure function of the trade's terms.
- Risk computation — to the risk engine, which only needs the trade and a context.

It is critical that the trade does *not* own its valuation date, its discount curve, or its model. These all live on the `PricingContext`. Why: a trade is a *static contract*. The market state varies; the trade does not. Conflating them produces trades that are bound to a snapshot, which means you cannot reprice yesterday's trade against today's market without surgery.

```python
@frozen
class FixedRateBond:
    issue_date: date
    maturity: date
    coupon_rate: float
    coupon_freq: Frequency
    notional: Money
    day_count: DayCount
    business_day: BusinessDayConvention
    calendar: Calendar
    # No curves, no market data, no dates other than economic.

    def cash_flows(self) -> CashFlowSchedule: ...  # pure function of self
```

**PricingContext shape — pass-in, not attached, not thread-local.** The three options for where market state lives, and why pass-in wins:

| Approach | What it looks like | Why not |
|---|---|---|
| Attached to trade | `trade.pv()` reads `trade._curves` | Trade is no longer pure data; reprice-against-another-market is hard |
| Thread-local global | `set_market(...); trade.pv()` | Magic; tests interfere; parallelism breaks; debugging is hell |
| Passed-in (chosen) | `trade.pv(context)` | Explicit; reproducible; parallel-safe; testable |

The argument against pass-in is verbosity: every call site has to thread the context. The mitigation is a `PricingSession` that bundles a context and offers convenience methods (`session.price(trade)`, `session.risk(trade)`) without becoming a global. The session is constructed once per evaluation, scoped lexically, and never lives longer than the evaluation.

```python
@frozen
class PricingContext:
    valuation_date: date
    discount_curves: Mapping[Currency, DiscountCurve]
    fx_spots: Mapping[CurrencyPair, float]
    vol_surfaces: Mapping[VolKey, VolSurface]
    survival_curves: Mapping[Entity, SurvivalCurve]
    fixings: FixingHistory
    config: NumericalConfig  # tolerances, MC paths, PDE grid, seeds
```

The context is *frozen*. Bumping produces a *new* context (`context.with_curve_bump(...)`). Immutability + structural sharing means bumping is cheap.

**Portfolio.** Trivially, an iterable of trades plus a name. The portfolio is *not* the right place to put pricing or risk methods — those are functions over portfolios, not methods on them. Reason: you want to pickle a portfolio, store it, ship it across processes, without dragging the entire pricing engine. The portfolio is data; pricing is a function applied to it.

```python
def portfolio_pv(p: Portfolio, ctx: PricingContext) -> PortfolioValue: ...
def portfolio_risk(p: Portfolio, ctx: PricingContext, spec: RiskSpec) -> RiskRun: ...
```

<a id="section-2-5"></a>
### 2.5 Math kernels vs financial logic

The right cut: **L0 knows nothing about finance.** Not "almost nothing"; nothing. A function in `numerical/` taking a `Volatility` type is wrong; it should take a `float` (or a generic scalar). A PDE solver that has a `coupon` parameter is wrong; it has `boundary_conditions` and `source_terms`.

The reason is reuse and clarity. A Crank-Nicolson PDE solver is a numerical method. The day someone uses it for the Fokker-Planck equation on an FX rate, then for a hazard-rate PDE, then for an inflation density evolution — you do not want to discover that the function signature has `option_strike: float` baked in. Worse: financial vocabulary in numerical code attracts financial logic. Two months later, the PDE solver has a special case for barriers, and now it is no longer a PDE solver — it is a barrier pricer pretending.

The discipline test: every function in L0 should have a docstring that does not mention any financial concept. If the docstring needs to mention "option" or "swap" or "discount", the function belongs in L3 or higher.

**Conversely, models do not know about specific instruments.** A `HullWhiteModel` knows how to price a *generic* IR payoff — a function of the short rate at exercise. It does not know it is being used to price a swaption versus a callable bond. The instrument constructs the payoff; the model evaluates it. This is the classical separation between model and payoff that QuantLib gets nearly right and that most in-house systems get badly wrong.

The contract between instrument and model:

```python
class IRPayoff(Protocol):
    def evaluate(self, short_rate_path: np.ndarray, times: np.ndarray) -> np.ndarray: ...

# Instrument-side:
def swaption_payoff(swaption: Swaption, ctx: PricingContext) -> IRPayoff: ...

# Model-side:
class HullWhiteModel:
    def price(self, payoff: IRPayoff, ctx: PricingContext) -> PricingResult: ...
```

The payoff is the lingua franca. Instruments speak it; models consume it. Neither imports the other.

<a id="section-2-6"></a>
### 2.6 Python vs C++ — the right cut

The hardest part of this question is resisting two equally wrong instincts: "write everything in C++ for performance" and "write everything in Python because we can." Both are wrong. The right cut is empirical, not ideological.

**Stays Python forever:**

- Trade construction, schedule generation, day-count conventions. Bound by clarity, not by CPU.
- Calibration *orchestration* (the optimiser loop). The objective evaluation may be C++; the loop is Python.
- Serialisation, persistence, repositories, reporting.
- Visualisation, notebooks, the user-facing API.
- The registry, the dispatch table.
- Anything that runs O(1) per pricing call.

**Belongs in C++:**

- MC inner loops — path generation, payoff accumulation, variance reduction. The inner loop runs $10^6$ – $10^9$ times per pricing; a 5x speedup is real money.
- PDE solvers — the stencil operations, the tridiagonal solves, the ADI sweeps. Hot, vectorisable, and stable in form.
- AD tape recording and playback. The forward pass and the reverse pass are tight loops that are O(graph_size); they dominate when graphs are large.
- Curve evaluation primitives (`df(t)` called inside MC paths is hot).
- Low-level linear algebra not already covered by SciPy.

**Stays Python until profiled:**

- Bootstrap solvers. Usually called O(1) per day per curve. Optimise only if a profile says so.
- Greek computation by bumping. Embarrassingly parallel and dominated by the underlying pricer, which may already be C++.

**ABI management.** Use `pybind11` (or `nanobind` for new code). C++ exposes a narrow surface: a few classes (engines, surfaces, AD tape) with `.price(...)` style methods taking POD or NumPy arrays. **Do not** expose C++ inheritance hierarchies across the boundary; the maintenance cost is brutal. **Do** expose factory functions returning opaque handles.

The crucial discipline: the Python side defines the protocols (`Pricable`, `EuropeanIROptionPricer`); C++ implementations are just one more implementation of the protocol. From the Python side, you do not know which pricer is native and which is C++. This is what makes the port incremental — you migrate a hot pricer to C++ without touching its callers.

A point of leverage: write the *first* version of every model in Python (clarity wins, fast iteration, easy to debug). Profile. Port only the bottleneck. The Python version stays as a reference; the C++ version is tested against it.

---

<a id="section-3"></a>
## 3. Patterns vs anti-patterns

<a id="section-3-1"></a>
### 3.1 Industry references

**QuantLib (C++, open source).** The canonical reference. Patterns to imitate:

- The *Instrument / PricingEngine* split. An instrument is data; an engine prices it. This is the model/payoff cut, done well. Steal this idea, drop the inheritance.
- *Term structures* (discount, vol, hazard) as separate abstractions with a uniform evaluation interface. The right shape.
- *Observers/Observables* for market-data propagation. Don't steal this. It made sense in 2003 for desktop spreadsheets; in 2026 it produces invalidation bugs that take days to track down. Replace with immutable contexts and explicit re-pricing.
- Deep inheritance trees (`Instrument <- Bond <- FixedRateBond`). Don't steal this. It is what makes QuantLib's learning curve cliff-shaped. Use protocols and composition.
- The handle / `RelinkableHandle` pattern. Don't steal this. It exists because of the observer pattern; once you ditch the observer, handles become useless indirection.

**OpenSourceRisk Engine (ORE, C++, built on QuantLib).** A real production-shape system. Patterns to imitate:

- *Flat trade representation in XML/JSON*, with a single dispatcher that builds the engine. This is the right way to do persistence — the trade is data, the engine is constructed when needed. The on-disk format and the runtime object are decoupled.
- *Scenario sim market* — a market state that wraps a base market and applies scenario perturbations. The right shape for stress.
- *Configurable curve and model wiring via input files*, not via hard-coded `main()`. The right shape for ops.
- ORE inherits QuantLib's observer mess. Don't.

**OpenGamma Strata (Java).** The most architecturally clean of the three. Patterns to imitate, almost wholesale:

- *Calculation Runner* — a pure function from `(targets, measures, scenarios) -> results`. No state, no globals, parallel by construction. This is the right shape for batch risk.
- *MarketDataResult* with explicit failure handling per market data point. Failures do not crash the batch; they propagate as typed failures and are reported. This is how you build a system that runs overnight on 10,000 trades and gives you actionable diagnostics in the morning.
- *Immutable beans* throughout. Trade is immutable, market data is immutable, results are immutable. The whole system is functional in the small.
- *Function registry* keyed by measure name and trade type. Cleaner than QuantLib's engines because there is no instrument-side inheritance.

Strata is the closest open-source system to what a modern engine should look like. The drawback is Java verbosity (Joda-Beans boilerplate); Python with `attrs`/`pydantic` gets you the same shape with less typing.

**Numerix (proprietary).** Notable design choices visible from the outside:

- *Generic payoff scripting language.* Users write payoffs in a small DSL; the engine targets MC, PDE, or AAD as appropriate. The DSL is the right idea even if you do not need a full language — define an *algebra of payoffs* (sum, product, max, min, indicator, conditional, discount) and let instruments compose payoffs from primitives. This is what makes a system capable of "I have a new exotic, can you price it by Friday."
- *AAD as a first-class production capability.* Greeks for any instrument, any model, in time comparable to a single price. The architecture has to be designed around this from day one — retrofitting AAD onto a system not designed for it is a multi-year project.

**Murex, Calypso (vendor risk systems).** The lesson here is mostly negative: both grew over 20+ years and ended up with massive god-instruments, tight coupling between trade booking and pricing, and "configuration" surfaces that are de facto programming languages without type checking. The lesson: do not let the configuration surface become Turing-complete by accident.

<a id="section-3-2"></a>
### 3.2 Common pitfalls

**Pitfall 1: The god-class Trade.** A `Trade` accreting fields for every conceivable instrument — coupon, strike, barrier, callable_schedule, fixing_dates, autocall_levels — with most fields nullable. Symptoms: dataclasses with 40 optional fields; methods that switch on which fields are set; a `validate()` method that is 300 lines.

Argue against: a `Trade` per instrument type (a `FixedRateBond` is a separate class from a `Swap` from a `Swaption`). Polymorphism is structural — they all implement `Pricable` — not nominal. Each class has exactly the fields it needs. Validation is a constructor invariant per class. The "common Trade" is a `Protocol`, not a base class.

**Pitfall 2: Eager calibration.** A model's constructor takes market quotes and calibrates internally. Symptoms: `HestonModel(quotes)`; constructing the model is slow; you cannot inspect calibration diagnostics because they were discarded; you cannot reuse a calibration across pricing calls.

Argue against: calibration is a separate function returning a `CalibrationResult`. Model constructors take *parameters*. The calibration result carries provenance — what was fit, what residual, what optimiser path — and is serialisable. A pricing run that uses a calibrated model carries the `CalibrationResult.id` so you can later answer "where did these parameters come from?"

**Pitfall 3: Hidden globals.** Default valuation date is "today." Default RNG seed is "system time." Default solver tolerance is in a module constant nobody knows. Symptoms: tests are flaky depending on time of day; results differ between machines; reproducing a production number requires forensic archaeology.

Argue against: zero defaults for anything that affects numerical output. `PricingContext.valuation_date` is required. The MC engine takes a `seed` parameter and raises if not provided. Tolerances live in a `NumericalConfig` object on the context, serialised with every result.

**Pitfall 4: Tight coupling between calibration and pricing.** "To price a swaption you first call `model.calibrate_to_atm_swaptions(...)` and then `model.price(...)`." Symptoms: cannot price without calibrating; cannot share calibrations across instruments; calibration and pricing are tangled in test setup.

Argue against: calibration produces parameters; pricing consumes parameters. They are decoupled in time and in code. A `CalibrationResult` is a transferable artefact. You can calibrate at 9am and price all day. You can calibrate once for an entire portfolio. You can persist a calibration and re-price tomorrow against the same parameters to do attribution.

**Pitfall 5: Over-use of inheritance.** A 6-deep inheritance hierarchy where the leaf classes override 80% of the methods and the base class is full of `NotImplementedError`. Symptoms: every new instrument requires deciding where in the tree to insert; common functionality keeps getting pulled up and broken down; refactors of the base ripple through dozens of classes.

Argue against: inheritance for *implementation reuse* is almost always wrong in this domain. Composition: a `CallableBond` has a `FixedRateBond` plus a `CallSchedule`, not "is a" `FixedRateBond` extended. Inheritance for *interface declaration* is fine — but in Python, that means a `Protocol`, not an ABC. The exception: closed, well-understood hierarchies (a small enum-like family) can use ABCs. The default is composition.

**Pitfall 6: Under-use of dataclasses (or their equivalents).** Hand-rolled `__init__` with 20 positional arguments, manual `__eq__`, manual `__repr__`, manual `__hash__`. Symptoms: bugs in equality comparisons used in caching; equality semantics differ across classes; pickling breaks because someone forgot a slot.

Argue against: use `attrs` (or stdlib `dataclasses`, or `pydantic` if validation is needed) for *every* domain object. Frozen by default. `eq=True, frozen=True, slots=True`. The cost of writing the boilerplate by hand is not just verbosity; it is correctness bugs in the boilerplate.

**Pitfall 7: Lack of a calibration audit trail.** Calibrations are run, parameters are used, no record is kept. Symptoms: "why did the price move 20bp overnight" cannot be answered; regulatory inspectors get unhappy.

Argue against: every calibration writes a `CalibrationResult` with a unique id. Pricing results carry the calibration ids they depend on. The audit trail is `(price_id) -> (calibration_id, market_snapshot_id, code_version) -> (quotes, optimiser_log)`. This is implemented once, in the calibration framework, and free thereafter.

**Pitfall 8: Mutable market data.** Curves and surfaces with `set_quote(...)` methods. Symptoms: bumping mutates shared state; threading is impossible; "the curve changed under me" bugs in long-running processes.

Argue against: all market data is immutable. Bumping produces new objects. Structural sharing (curves under the hood share interpolation tables) keeps this cheap. The cost of an immutable bump is the cost of a few object allocations; the benefit is correctness.

<a id="section-3-3"></a>
### 3.3 Concrete shapes

**PricingContext.**

```python
@frozen
class PricingContext:
    valuation_date: date
    discount_curves: Mapping[Currency, DiscountCurve]
    fx_spots: Mapping[CurrencyPair, float]
    vol_surfaces: Mapping[VolKey, VolSurface]
    survival_curves: Mapping[Entity, SurvivalCurve]
    fixings: FixingHistory
    config: NumericalConfig

    def with_bump(self, bump: MarketBump) -> "PricingContext": ...
    def with_scenario(self, scenario: Scenario) -> "PricingContext": ...
```

Why this shape over alternatives:

- *Versus a flat list of curves keyed by string:* keyed by typed enum/class (`Currency`, `Entity`) so misspellings are compile-time errors, not runtime KeyErrors at midnight.
- *Versus exposing the underlying market quotes:* the context is the *interpreted* market — curves, surfaces, fixings. Quotes live one layer down and are referenced by id (in the calibration result), not by inclusion.
- *Versus mutability:* `with_bump` returns a new context. Bumping does not affect other threads or other risk runs.

**CalibrationResult.**

```python
@frozen
class CalibrationResult:
    id: UUID                          # stable artefact identity
    model_class: str                  # registry key, not a class object
    parameters: Mapping[str, float]   # named, not positional
    quotes_fitted: Sequence[QuoteRef] # by reference into a MarketSnapshot
    weights: Sequence[float]
    objective: ObjectiveKind          # SSE, weighted SSE, max-error, ...
    residuals: Sequence[float]        # per-quote, in input order
    rms_residual: float
    max_residual: float
    iterations: int
    optimiser: OptimiserSpec          # algorithm, tolerance, max_iter
    converged: bool
    diagnostics: CalibrationDiagnostics  # objective history, parameter path
    timestamp: datetime
    code_version: str                 # git sha
    market_snapshot_id: UUID
```

Why this shape:

- *Versus a `Tuple[float, ...]` of parameters:* named parameters survive refactors of the model.
- *Versus an opaque blob:* every consumer (pricing, audit, debug) needs a different slice; expose them all.
- *Versus skipping `converged`:* a calibration that did not converge is *not* an error to raise — it is a result with a flag and a residual. The caller decides whether to use it. Raising loses information.
- *Versus skipping `code_version`:* yesterday's calibration with today's code may not reproduce. The version is the only way to know.

**RiskRun.**

```python
@frozen
class RiskRun:
    id: UUID
    portfolio_id: UUID
    valuation_date: date
    measures: Sequence[RiskMeasureSpec]   # what was asked for
    results: Mapping[TradeId, Mapping[MeasureKey, RiskValue]]
    failures: Sequence[RiskFailure]       # typed failures, not exceptions
    context_id: UUID                      # market snapshot
    calibration_ids: Sequence[UUID]       # provenance
    elapsed: timedelta
    config: NumericalConfig
```

Why:

- *Failures are first-class.* A risk batch over 10,000 trades will have failures. They must be reported, not crash the batch. Each failure is typed (`MissingMarketData`, `CalibrationFailed`, `NumericalDivergence`, `Timeout`) with enough context to fix.
- *Results indexed by `(trade, measure)`.* Risk runs ask for multiple measures simultaneously (PV, DV01, vega, cross-gamma); the data structure must support that without separate runs.
- *Per-trade granularity preserved.* Aggregation (portfolio-level DV01) happens on top of the run, from the per-trade results. Aggregating at the bottom loses information.

**Scenario.**

```python
class Scenario(Protocol):
    name: str
    def apply(self, ctx: PricingContext) -> PricingContext: ...

# Concrete implementations:
@frozen
class ParallelRateShift(Scenario):
    name: str
    currency: Currency
    shift_bp: float
    def apply(self, ctx: PricingContext) -> PricingContext: ...

@frozen
class CompositeScenario(Scenario):
    name: str
    components: Sequence[Scenario]
    def apply(self, ctx: PricingContext) -> PricingContext:
        for c in self.components:
            ctx = c.apply(ctx)
        return ctx
```

Why:

- *Scenarios compose.* A composite scenario is a list of atomic scenarios applied in order. This is what lets you build a stress library (parallel + key-rate + vol + spread + FX) without combinatorial explosion of named scenarios.
- *Scenarios are pure functions on contexts.* They do not know about trades, portfolios, or risk. The same scenario applies to one trade or a portfolio of a million; the cost is the cost of pricing under the new context, not the cost of the scenario.
- *Scenarios are named.* The name is the link between the scenario object and the report — "2015 China devaluation" appears in the result, not "scenario_47".
- *Scenarios are serialisable.* You can ship a scenario library across teams, version it, regulate against it.

The alternative — scenarios as functions that take and return market data dictionaries — works for one-offs and rots immediately. Make scenarios first-class objects from the beginning; the cost is one decorator and one base protocol.

---

<a id="section-4"></a>
## 4. Pricebook through this lens

Pricebook at v0.874.0 is 23 sub-packages, 793 modules, 7 dependency layers (verified acyclic), ~11,600 tests. The architecture is not naive — it was iteratively shaped over several refactor cycles, and many of the principles above are already in place. This section reads pricebook *against* the reference design and sorts what we see into four buckets: match, extend, refactor, anti-pattern.

The exercise is honest, not flattering. The point is to surface mismatches whose fix is high-leverage. Items where pricebook already does the right thing are noted briefly because they are the foundations to build on; the bulk of the section is on where the shape diverges.

<a id="section-4-1"></a>
### 4.1 Where we match — keep as is

**Acyclic layering.** Pricebook has a verified-acyclic dependency graph with 7 layers (`ARCHITECTURE.md` carries the regen snippet). This is the foundational property the reference design demands. Several recent slices were specifically about preserving acyclicity (e.g., the `fixed_income → credit` edge severed via `TYPE_CHECKING` in v0.863). Continue doing this — it is the cheapest correctness guarantee in the library.

**`PricingContext` exists.** `core.pricing_context.PricingContext` is a real type with discount curves, projection curves, vol surfaces, credit curves, FX spots, and fixings. The reference's pass-in (not attached, not thread-local) decision is already what pricebook does — `instrument.pv_ctx(ctx)` is the canonical call. Keep this; it is the load-bearing decision that the rest of the design hangs from.

**Serialisation via decorator + registry.** `core.serialisable` defines `@serialisable("type_key", [fields])` plus a `_REGISTRY` populated at import time by decorator side-effects. This is exactly the L0-protocol + L3/L4-implementation pattern the reference recommends. The schema version field is missing (see §4.2) but the *shape* is right.

**Models mostly use Protocol-style composition.** `models.engine_protocol`, `models.char_func_protocol`, the various `*_via_engine` functions, and the engine registry together form a Protocol-shaped pricer dispatch system. There are still a few residual inheritance trees in older models, but the recent trajectory has been toward Protocols and composition — keep going that direction.

**Frozen dataclasses for conventions and most trade objects.** The `@serialisable_convention` decorator wraps frozen dataclasses (currency conventions, swap conventions, EM-FX, sovereign bonds). Most trade classes are also constructed as frozen-by-convention (no mutating methods on the public API). This is the reference's `@frozen`-everywhere pattern, applied non-uniformly but in the right direction.

**Numerical layer is mostly finance-free.** `pricebook.numerical` is, by inspection, a generic numerical toolkit: `_pde.py`, `_mc.py`, `_fourier.py`, `_optimize.py`, `_distributions.py`, `auto_diff.py`, etc. The L0-knows-nothing-about-finance principle is honoured here. There are a handful of exceptions (e.g., functions whose docstrings mention bonds), but the core discipline holds.

**Recent calibration work returns structured results.** `bond_hazard_bootstrap.HazardBootstrapResult` and `g2pp_calibration.G2PPCalibrationResult` are typed result objects with parameters, residuals, RMSE, max error, convergence flag, and (after Slice A) the regularisation strength and roughness. This is the *shape* of the reference's `CalibrationResult` — although it is one type per calibration kind rather than a unified abstraction. We can promote.

<a id="section-4-2"></a>
### 4.2 Where the shape is right but coverage incomplete

**Calibration results exist per family but not as a unified concept.** Each calibration family (hazard bootstrap, G2++, Hull-White, SABR, COS, etc.) has its own ad-hoc result type. Each carries some subset of `{parameters, residuals, RMSE, converged}`. None carries a UUID, a code version, a market snapshot reference, or a structured diagnostics object. This is the right kernel — just incomplete and not deduplicated.

*Promote to:* one `CalibrationResult` type at L6 with the full provenance contract. Per-family results become *subclasses* (carrying family-specific parameters) of the common type, or composition (a per-family `parameters` payload inside a shared `CalibrationResult` envelope).

**Scenarios exist but not as protocol-with-composition.** `risk.scenario` has parallel_shift, pillar_bump, vol_bump functions. They take and return market objects. They do not compose; there is no named-scenario library; there is no `Scenario` protocol. This is the reference's "scenarios as one-off functions that rot" anti-pattern — caught early but not fixed.

*Promote to:* a `Scenario` Protocol at L6 with `name` and `apply(ctx) -> ctx`. A `CompositeScenario` for chaining. A `ScenarioLibrary` of named historical scenarios (2008 GFC, 2015 China devaluation, 2020 COVID shock, ...).

**Failure handling is exception-driven, not result-driven.** When a calibration fails to converge, the various calibration entry points return a result with `converged=False`. But when a *pricer* fails (e.g., COS bounds violated, MC numerical issue), the failure mode is a Python exception or a NaN — there is no `PricingFailure` type that the caller can read. A risk batch over many trades that hits one failure today either crashes (exception) or silently propagates NaN.

*Promote to:* a `PricingFailure` typed result and a `RiskRun.failures` list. Risk batches over portfolios should be failure-tolerant by default (the Strata pattern).

**Schema versioning on serialised types.** The `@serialisable` registry stores `_SERIAL_TYPE` (the type tag) but no `_SERIAL_VERSION`. Today's serialisations cannot be migrated. The day a trade type adds a field, all stored representations of the old shape become un-loadable.

*Promote to:* `@serialisable("trade_type", version=1, [fields])`. Migration functions registered alongside.

**`NumericalConfig` is implicit.** Tolerances, MC paths, PDE grid sizes, RNG seeds — these live in default arguments scattered across the codebase. A reproducibility audit ("what tolerance was used for this number") requires reading code, not querying a config object.

*Promote to:* a single `NumericalConfig` type carried on the `PricingContext`. Defaults live in one place; bumping the default for a numerical investigation is a one-line change; the config is serialised with every result.

**No `MarketSnapshot` type separating quotes from curves.** Pricebook has a `core.market_data` module with `Quote` and `MarketDataSnapshot`, but `MarketDataSnapshot` is treated as ad-hoc data, not as the canonical L1 type that L2 curves are built from. Most code paths pass already-built `DiscountCurve` objects around without reference to which quotes they came from.

*Promote to:* `MarketSnapshot` as the canonical L1 type. Curves are built from snapshots with a recorded link. The link is what makes "this curve was fit to these quotes" provable rather than asserted.

<a id="section-4-3"></a>
### 4.3 Where the shape is wrong

These are the structural mismatches between pricebook and the reference. Each is high-leverage to fix, and each is more than a one-slice job.

**Risk lives at L3, parallel to instruments.** `pricebook/risk/` has 55 modules and sits at Layer 3 alongside `credit`, `crypto`, and `fixed_income`. The reference design says risk is at L6, depending only on the `Pricable` protocol and the `PricingContext`. The current placement *forces* risk modules to import from `fixed_income`, `options`, `credit`, etc., which makes the dependency graph wider than it should be and the risk modules tightly coupled to concrete instrument classes.

Concrete examples of this damage: `risk.greeks` has to know about specific Greek conventions per instrument family rather than being a generic bump-and-reprice over the protocol; `risk.xva` has special cases for swap-specific exposure profiles; `risk.cross_asset_greeks` has to thread per-asset-class logic. Each of these would shrink substantially if risk depended on `Pricable` instead of on concrete instruments.

*Refactor to:* risk at L7 (above all instrument layers), depending only on the `Pricable` protocol. The protocol lives in `core/`. Each instrument layer declares its concrete classes as `Pricable` via the registry. Risk modules see a uniform interface and shrink.

**Calibration is distributed across packages, not its own layer.** Calibrations live in `credit.bond_hazard_bootstrap`, `models.g2pp_calibration`, `models.hw_calibration`, `models.jump_calibration`, `models.lmm_calibration`, `curves.bootstrap`, `curves.global_solver`, `curves.multicurve_solver`, etc. Each is reasonable in isolation; collectively they have no shared abstraction, no shared result type, no shared diagnostics format, no shared optimiser configuration.

*Refactor to:* a `pricebook.calibration` package at L6. It re-exports the existing calibrators with a uniform interface (`Calibrator` protocol, `CalibrationResult` type, `OptimiserSpec`). The per-family implementations stay in their current locations as the concrete machinery; the calibration *layer* is the thin uniform front.

**Market data is conflated with curves.** `pricebook.curves` is L1 and contains bootstrap, NSS, Smith-Wilson, AAD curves. `pricebook.data` (6 modules, L2) handles loaders. There is no clean L1 that says "this is raw market data" separately from L2 "this is a fitted curve." The result: nothing in the dependency graph distinguishes a number that came from a quote from a number that came from a fit.

*Refactor to:* a real L1 = `pricebook.market_data` with `MarketSnapshot`, `Quote`, `QuoteId`, `FixingHistory`. L2 `curves` consumes snapshots; the link is recorded. The `data` package (loaders) becomes a thin shim in L8 (delivery, alongside reporting).

**`pe/` is at L0 alongside `core` and `numerical`.** The empirical dependency graph puts `pricebook.pe` at L0 — it imports nothing else and is imported by nothing. That is *technically* L0 in the topological sense, but it is *conceptually* L8: private equity is portfolio aggregation and reporting, far from a math kernel. Its L0 placement is an artefact of "imports nothing from pricebook," not a design decision.

*Refactor to:* move `pe/` to L7/L8 (portfolio/reporting). If it imports nothing from below, that is fine — but it should be *above* trades and portfolios in the conceptual layering, not below them.

**`registry.py` at the top level is a leaky abstraction.** The single-module `pricebook.registry` ties together solvers, pricers, and engines. It depends on `core, curves, models, numerical, risk, statistics` — a wide fan-out from a one-file module. Its existence is the right idea (a single dispatch point), but its placement (top-level module rather than a `pricing/` package with `pricing.registry`) makes it harder to find and harder to extend.

*Refactor to:* `pricebook.pricing.registry` at L5 (pricing layer). The dispatch table is per-instrument-type, populated by decorators in each instrument module.

<a id="section-4-4"></a>
### 4.4 Anti-patterns visible in the codebase

Cross-referencing with `MODULE_HEALTH.md` (the dual-critic audit from v0.866):

**Eager calibration in scattered places.** Several model constructors take market quotes and calibrate internally (e.g., the Heston model in some entry points, hazard rate models that build the deterministic shift inside `from_survival_curve`). The CIR++ pattern (`from_survival_curve(...)`) is closer to right because it takes an already-built curve, not raw quotes — but the underlying eager-calibration anti-pattern is still present in older modules.

**Hidden defaults that affect numerical output.** Several MC engines and PDE solvers have default seed parameters (e.g., `seed=42` baked into helper functions). The intent is reproducibility, but it is the *wrong* reproducibility: the user does not know that a default seed was used, and across runs of the same code the seed is silently the same. The reference design says zero defaults that affect numerical output.

**Mutable curves in a few places.** Most `DiscountCurve` operations return new objects (`bumped(...)`). A few helpers in `curves/curve_bumper.py` perform in-place modifications via private attributes. This is a small but real correctness exposure in long-running processes.

**A handful of god-class instruments.** Most trade classes are appropriately narrow (one trade type per class). A few — particularly older structured products, autocallables with many optional features, and certain CDS variants — have accreted optional fields with switch-on-which-are-set logic in their pricing methods. These are local debts, fixable per-class.

**Risk modules with switch-on-instrument-type logic.** The clearest anti-pattern visible in the audit: several risk modules have explicit `isinstance(trade, FixedRateBond)` / `isinstance(trade, Swaption)` / ... ladders. Each branch knows the specific Greek formula. This is exactly what the reference design's "risk depends on `Pricable`, not on concrete classes" rule is meant to prevent. Fixing this is what the L3 → L7 risk relocation in §4.3 buys us.

**Schema versioning absent.** As noted in §4.2. Today's serialised representations are committed to never changing — a real exposure for production storage.

**Audit trail for calibrations: absent.** As noted in §4.2. Yesterday's calibration cannot be reconstructed from today's code unless the entire input + output is stored separately, which it is not.

The audit report (`MODULE_HEALTH.md`) contains finer-grained findings — boundary conditions, off-by-one risks, division guards. Those are bug-level, not architectural; they are properly handled by the methodical audit (the next major task), not by this design document. Architecture changes the *shape*; module audit changes the *content*. Both are needed.

---

<a id="section-5"></a>
## 5. Delta list

The changes from where pricebook is today to where the reference design lives. Sorted by leverage (downstream impact ÷ implementation cost), not by size.

<a id="section-5-1"></a>
### 5.1 High-value adds

These are missing capabilities whose absence is a real exposure. Each is bounded in scope (one to a few slices).

**A1. `CalibrationResult` type with provenance.** Introduce `pricebook.calibration.CalibrationResult` carrying `(id, model_class, parameters, quotes_fitted, residuals, optimiser, converged, diagnostics, timestamp, code_version, market_snapshot_id)`. Each existing per-family result becomes a wrapper or subtype. Pricing results carry calibration ids. This unblocks audit, P&L attribution, regret analysis. **Effort: 3-4 slices.**

**A2. `MarketSnapshot` as the canonical L1 type.** Promote `core.market_data.MarketDataSnapshot` to a first-class type owned by a new `pricebook.market_data` package at L1. Curves consume snapshots; the link is recorded; the snapshot has a UUID and timestamp. **Effort: 3 slices (define + migrate curve constructors + migrate calibration entry points).**

**A3. `Scenario` Protocol + composable library.** Define `Scenario` as `name: str` + `apply(ctx) -> ctx`. Wrap existing parallel/key-rate/vol shock functions as concrete scenarios. Add a `CompositeScenario`. Add a `ScenarioLibrary` of historical scenarios. **Effort: 2 slices.**

**A4. `PricingFailure` and `RiskRun.failures` for failure-tolerant risk batches.** Define `PricingFailure` (typed: `MissingMarketData`, `CalibrationFailed`, `NumericalDivergence`, `Timeout`, ...). Risk batch entry points return `RiskRun` with `results` *and* `failures`. Single-trade calls still raise; batch calls accumulate. **Effort: 2-3 slices.**

**A5. `NumericalConfig` on `PricingContext`.** Define `NumericalConfig` (mc_paths, pde_grid, solver_tol, seed, ...). Carry on `PricingContext`. Defaults live in one place. Serialised with every result. **Effort: 2 slices.**

**A6. Schema versioning on `@serialisable`.** Add `version: int` to the decorator. Provide a `migrate` registration mechanism. Default version is 1 for all existing types. **Effort: 2 slices.**

<a id="section-5-2"></a>
### 5.2 Wrong-shape refactors

These are larger but the leverage is high — fixing the shape unblocks many downstream improvements.

**R1. Risk relocation: L3 → L7 (above instruments).** Risk modules depend on `Pricable` protocol only. Each instrument-specific Greek formula moves either into the instrument (if it is structural) or into the risk module (if it is a generic bump). The protocol lives in `core/`. **Effort: 6-10 slices (across the 55 risk modules + tests).**

**R2. Calibration as a layer (consolidate distributed calibrations).** Create `pricebook.calibration` package at L6. Define `Calibrator` protocol. Wrap existing calibrators. Migrate callers to use the protocol. Keep the per-family implementations where they live. **Effort: 4-5 slices.**

**R3. Market data L1 split.** As described in §4.3. Migrate curve construction signatures to take `MarketSnapshot`. **Effort: 4-5 slices (significant API surface affected).**

**R4. `pe/` relocation to L7/L8.** Move out of L0 into the portfolio/reporting layer. **Effort: 1 slice (it imports nothing from below; the move is mechanical).**

**R5. Consolidate `registry.py` into `pricing.registry`.** Move into a `pricebook.pricing` package. Decorator-based registration per instrument module. **Effort: 2 slices.**

<a id="section-5-3"></a>
### 5.3 Nice-to-haves

Useful but not load-bearing.

**N1. Generic payoff algebra.** A small DSL of payoff primitives (sum, product, max, min, indicator, conditional, discount) with which exotic structured products can be assembled without writing a new pricer. The Numerix idea. Not free — but bounded if scoped to a small algebra rather than a full scripting language. **Effort: 5-8 slices.**

**N2. `PricingSession` convenience wrapper.** A class that bundles `PricingContext` with helper methods (`session.price`, `session.risk`, `session.scenario`). Avoids context-threading verbosity without introducing thread-locals. **Effort: 1 slice.**

**N3. Repository pattern for persistence.** `TradeRepository`, `CalibrationResultRepository`, `MarketSnapshotRepository`. Thin wrappers over the existing `db.PricebookDB` + serialisation. **Effort: 2-3 slices.**

**N4. Test architecture per layer.** Reorganise `tests/` into `tests/L0_numerical/`, `tests/L1_market_data/`, ..., `tests/L8_reporting/` mirroring the layer cut. Test imports respect the layer rule. **Effort: 1 slice (mechanical) + ongoing discipline.**

<a id="section-5-4"></a>
### 5.4 Won't-fix

Conscious scope-outs. Listed so they do not haunt later.

**W1. Observer / observable pattern.** Pricebook does not use this. We will not add it. The reference design explicitly argues against it.

**W2. Handle / RelinkableHandle indirection.** Pricebook does not use this. The reason QuantLib has it (observer-driven invalidation) does not apply.

**W3. Pickle-based persistence as the production format.** Pickle is fine for caches but not for cross-version storage. The serialisation contract (dict + type tag + schema version) is the production format. Pickle stays as a private optimisation.

**W4. Full DSL for payoffs.** Pricebook is not Numerix. A bounded payoff algebra (N1) is in scope; a Turing-complete payoff language is not.

**W5. Multi-language bindings beyond Python and C++.** Java/C#/JavaScript bindings are not in scope. If a consumer needs them, they wrap the Python interface.

---

<a id="section-6"></a>
## 6. Roadmap

The roadmap has two axes. **Gates** are the narrative — each gate is a single user-visible promise (a thing the library does, or does better, after the gate). **Phases** inside each gate are the execution — each phase has one architectural focus and one bounded scope.

A gate is shippable: completing all phases inside it leaves the codebase in a usefully better state, even if no subsequent gate is started. Gates can be paused between, not within.

### Roadmap at a glance

| Gate | What the user gets | Phases inside | Slice count |
|---|---|---|---:|
| **G1 — Audit-ready** | The bottom-up module audit can measure against the right type contracts, not the current ad-hoc shapes | P1 Calibration unified + P2 Market data L1 + P3 NumericalConfig & versioning | 14-16 |
| **G2 — Production-grade** | Failure-tolerant risk batches; composable scenario library; auditable persistence | P4 Scenarios & failures + P5 Repositories & per-layer tests | 7-9 |
| **G3 — Architecturally clean** | Risk depends on `Pricable` not on concrete instruments; small structural cleanups land | P6 Cleanups + P7 Risk relocation (isolated) | 9-13 |
| **G4 — Capability-complete** | AAD as a protocol available to every model; bounded payoff algebra for exotics | P8 AAD protocol + P9 Payoff algebra | 10-13 |
| **G5 — Performant at scale** | C++ hot paths (MC, PDE, AAD tape, curve eval) | P10 C++ port | 19-28 |
| **Total** | | 10 phases | **59-79** |

At the historical pricebook slice rate of 5-15 per week (1-3/day), this is **6-15 weeks of focused work plus the open-ended C++ port**.

### Gate 1 — Audit-ready

The most important gate to land before anything else. The bottom-up module audit (the next major task after this document is accepted) is the largest piece of remaining work in the project; doing it against the current ad-hoc types will lock in those types as the de facto contract. Spending two weeks first on the right contracts saves months of audit rework.

**P1 — Calibration unified.** *(A1 CalibrationResult + R2 calibration as a layer; 6-7 slices)*
Define `pricebook.calibration.CalibrationResult` with full provenance: `(id, model_class, parameters, quotes_fitted, residuals, optimiser, converged, diagnostics, timestamp, code_version, market_snapshot_id)`. Define the `Calibrator` protocol at L6. Migrate the existing calibration entry points (`bond_hazard_bootstrap`, `g2pp_calibration`, `hw_calibration`, `jump_calibration`, `lmm_calibration`, `sabr` calibration, curve bootstrapping) to produce uniform results. Per-family parameter payloads stay in their current modules.

**P2 — Market data L1 + `MarketSnapshot`.** *(A2 MarketSnapshot + R3 market data L1 split; 4-5 slices)*
Create `pricebook.market_data` at L1. Define `MarketSnapshot`, `Quote`, `QuoteId`, `FixingHistory` as the canonical raw-data types. Move existing `core.market_data` content into the new package. Curve constructors take snapshots; the link is recorded. After this, the dependency graph distinguishes "this number came from a quote" from "this number came from a fit."

**P3 — `NumericalConfig` & schema versioning.** *(A5 NumericalConfig + A6 schema versioning; 4 slices)*
Define `NumericalConfig` (mc_paths, pde_grid, solver_tol, seed, ...) and carry it on `PricingContext`. Defaults live in one place; bumping is one line. Extend `@serialisable` with `version: int` and provide a migration registration mechanism — all existing types default to version 1.

**Gate exit criteria:** every calibration in the codebase returns a `CalibrationResult`; every curve carries a `MarketSnapshot` reference; `PricingContext` carries a `NumericalConfig`; every serialised type has a version field. The audit can begin.

### Gate 2 — Production-grade

Once the right types are in place, the things a desk would *actually notice* tomorrow: failure-tolerant risk batches, a real scenario library, persistence that survives schema migrations.

**P4 — Scenarios & failure handling.** *(A3 Scenario protocol + A4 PricingFailure & RiskRun; 4-5 slices)*
Define `Scenario` Protocol with `name: str` and `apply(ctx) -> ctx`. Wrap existing parallel/key-rate/vol-shock functions as concrete scenarios. Add `CompositeScenario` for chaining. Build a `ScenarioLibrary` of historical scenarios (2008, 2015, 2020, ...). In parallel: define `PricingFailure` typed family (`MissingMarketData`, `CalibrationFailed`, `NumericalDivergence`, `Timeout`). Risk batch entry points return `RiskRun` with `results` *and* `failures` — single-trade calls still raise, batches accumulate.

**P5 — Repositories & per-layer tests.** *(N3 repository pattern + N4 test architecture; 3-4 slices)*
Implement `TradeRepository`, `CalibrationResultRepository`, `MarketSnapshotRepository` as thin wrappers over `db.PricebookDB` + serialisation. Reorganise `tests/` into per-layer directories mirroring the production layering; test code imports respect the same layer rule as production code. This is partly mechanical; the discipline part is ongoing.

**Gate exit criteria:** a desk can run an overnight risk batch with named scenarios, see typed failures in the morning report, and reload yesterday's calibration result by id.

### Gate 3 — Architecturally clean

The structural refactors. After G3 the dependency graph matches the reference design at the layer level.

**P6 — Small structural cleanups.** *(R4 pe/ relocation + R5 registry consolidation; 3 slices)*
Move `pe/` from L0 (where it sits as an artefact of "imports nothing from below") to L7 portfolio/reporting. Consolidate `pricebook.registry` into `pricebook.pricing.registry` (with the `pricing` package created if it doesn't exist as a real package today). Mechanical, low-risk, but the cleanup is visible in `ARCHITECTURE.md` and improves discoverability.

**P7 — Risk relocation L3 → L7.** *(R1; 6-10 slices)*
The biggest single refactor in the roadmap. Define a `Pricable` protocol in `core/`. Each instrument layer declares its concrete classes as `Pricable` via the existing registry. Risk modules are migrated to depend only on the protocol — replacing `isinstance(trade, FixedRateBond) / isinstance(trade, Swap) / ...` ladders with a single uniform code path. Some Greek formulas may move into the instrument (if they are structurally specific to that instrument); the rest become generic bump-and-reprice. Done in sub-slices, with the risk module test suite serving as the regression net.

**Gate exit criteria:** `pricebook.risk` imports only from `core` (via the `Pricable` protocol) and `pricing`. The dependency graph regenerates with risk at L7 instead of L3.

### Gate 4 — Capability-complete

New things the library couldn't do before. Independent of G3 in principle (you could do G4 before G7 if you wanted), but the work is easier on a clean architecture.

**P8 — AAD as a protocol from L0.** *(P3.1; 5 slices)*
Promote `numerical/auto_diff.py` to the AD layer that models opt into via generic-scalar discipline. The model writer codes against `Numeric` (a Protocol over `float` and `Dual`), and AAD just works when an AD scalar is passed in. Risk runs that use AD compute Greeks in time comparable to a single price.

**P9 — Payoff algebra (bounded).** *(N1; 5-8 slices)*
A small algebra of payoff primitives — `Sum`, `Product`, `Max`, `Min`, `Indicator`, `Conditional`, `Discount`, `Forward`, `Spot`. Exotic structured products are assembled by composition rather than by writing a new pricer. Not a Turing-complete scripting language; explicitly bounded to keep maintenance honest.

**Gate exit criteria:** the SABR model in pricebook supports AAD Greeks via a one-line model attribute. An autocallable can be assembled from `Indicator`, `Discount`, `Max` primitives without writing a new pricer.

### Gate 5 — Performant at scale

The C++ port of hot paths. Open-ended and ongoing — the gate is "in progress" rather than "completed" for a long time, but it has a meaningful first-version milestone.

**P10 — C++ port.** *(P4 hot paths; 19-28 slices for the initial port)*
- *MC inner loop* — path generators and payoff accumulators (6-10 slices including pybind11 wrappers and ABI design)
- *PDE stencils* — Crank-Nicolson, ADI, Hundsdorfer-Verwer kernels (4-6 slices)
- *AAD tape and playback* — forward pass and reverse pass kernels (6-8 slices)
- *Curve evaluation hot path* — `df(t)` for C++ MC inner loops (3-4 slices)

The Python implementations stay as test oracles. Profile first, port the bottleneck, keep the Python version available.

**Gate exit criteria:** an MC pricing of a 10-year basket option with 100k paths is at least 5× faster than the pure-Python implementation at the same numerical tolerance, with the Python version still producing the same answer to numerical precision.

### Dependency graph between gates and phases

```
G1 — Audit-ready
  P1  Calibration unified ─────┐
  P2  Market data L1 ──────────┤── BOTTOM-UP AUDIT (the next major task)
  P3  NumericalConfig & ver. ──┘
  │
  ▼
G2 — Production-grade
  P4  Scenarios & failures
  P5  Repositories & tests
  │
  ▼
G3 — Architecturally clean
  P6  Cleanups (pe/, registry)
  P7  Risk relocation L3→L7  ◄─── biggest single refactor; do isolated
  │
  ▼
G4 — Capability-complete
  P8  AAD as protocol
  P9  Payoff algebra
  │
  ▼
G5 — Performant at scale
  P10 C++ port  ◄─── ongoing; ships in chunks
```

G2 and G4 are partly parallelisable with the gate before them — e.g., you could start P9 (payoff algebra) before all of G3 lands — but the simpler narrative is one gate at a time. The big precedence rules are:

- **G1 before the audit.** Hard constraint.
- **G3 P7 (risk relocation) after G1.** The risk modules need `CalibrationResult` and `MarketSnapshot` types to migrate cleanly.
- **G4 P8 (AAD protocol) after G3 P7.** AAD over a risk system that's tightly coupled to instruments is the wrong shape to attempt the port from.
- **G5 (C++) can start any time after G1**, in principle. In practice it's safer after G3 because the C++ interface should be cut against the clean Python protocols, not the messy current ones.

### Wall-clock estimates

At the pricebook slice rate (5-15 slices per week, 1-3 per day):

| Gate | Slices | Wall-clock (low-med-high) |
|---|---:|---|
| G1 Audit-ready | 14-16 | 1 wk – 2 wk – 3 wk |
| G2 Production-grade | 7-9 | 1 wk – 1.5 wk – 2 wk |
| G3 Architecturally clean | 9-13 | 1 wk – 2 wk – 3 wk |
| G4 Capability-complete | 10-13 | 1.5 wk – 2 wk – 3 wk |
| G5 Performant at scale | 19-28 (initial port) | 3 wk – 5 wk – 8 wk |
| **Total** | **59-79** | **7 wk – 12.5 wk – 19 wk** |

Estimates are commitment-quality only if sections 1-3 are accepted as written. The two load-bearing decisions remain the **9-layer cut** and the **calibration-as-its-own-layer placement** — change either and the roadmap re-scopes from G1 down.

### How to start: the first three slices of G1 P1

Concretely, the first three commits after this document is accepted:

1. **Slice 1.** New `pricebook.calibration` package skeleton + `CalibrationResult` dataclass + `Calibrator` Protocol. No migrations yet. Tests for the dataclass and the protocol shape.
2. **Slice 2.** Migrate `bond_hazard_bootstrap` (the most recently-touched calibration, fresh in everyone's memory) to return the new `CalibrationResult`. The `HazardBootstrapResult` becomes a thin compatibility wrapper. Tests for the migrated entry point.
3. **Slice 3.** Migrate `g2pp_calibration` and `hw_calibration` to the new result type. Tests.

After these three slices, the pattern is established; the remaining four-five calibrators (LMM, SABR, jump, curve bootstrap, multicurve) follow the same template. G1 P1 is done at slice 6-7.

---

*End of design document. Next step: pushback on sections 1-3 and the delta list, then G1 P1 slice 1 begins.*
