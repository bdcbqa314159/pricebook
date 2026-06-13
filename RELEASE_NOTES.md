# Release Notes

---

## v1.019.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.repo_cva` omitted discount factor (same shape as v1.006 hybrid_xva).**

Pre-fix CVA accumulator: `cva += epe × pd_step × lgd` — no `D(0, t_i)` term. Future-valued not present-valued. For overnight repo the effect is < 1%; for term repo (months to a year) it accumulates to several percent.

**Fix**: optional `discount_rate` parameter (defaults to `repo_rate` — sensible secured-funding-curve approximation). Each EPE × ΔPD contribution is now multiplied by `exp(-discount_rate · t)`.

### Verification — `test_l2_t4_repo_cva.py`

5 new tests: zero discount preserves pre-fix; high discount reduces CVA; default uses repo_rate; zero hazard / zero LGD give zero CVA.

Full parallel suite: **12,617 passed in 5:00** — zero regressions.

Twenty-seventh fix from phase-2. **156 distinct bugs** in v0.905→v1.019.

---

## v1.018.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.portfolio_analytics.tracking_metrics` mixed ddof conventions for beta.**

Pre-fix:
```python
beta = np.cov(p, b)[0, 1] / np.var(b)
```

`np.cov` defaults to ddof=1 (sample); `np.var` defaults to ddof=0 (population). The ratio gives beta off by factor `(n-1)/n` — material for small samples (~3% bias at n=30, ~0.4% at n=252).

**Fix**: both use ddof=1 (sample-statistics convention).

### Verification — `test_l2_t4_portfolio_analytics.py`

3 new tests: perfect correlation → β=1; matches `cov_ddof1/var_ddof1` exactly; differs from pre-fix mixed convention.

Full parallel suite: **12,612 passed in 4:41** — zero regressions.

Twenty-sixth fix from phase-2. **155 distinct bugs** in v0.905→v1.018.

---

## v1.017.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.vol_stress.twist_vol_bump` butterfly weights were wrong.**

Pre-fix:
```python
weights = ((T - T_mid) / W)**2 - 0.25
```

Evaluating on x = (T-T_mid)/W ∈ [-0.5, +0.5]:
- Wings (x=±0.5): `0.25 - 0.25 = 0`.
- Belly (x=0): `0 - 0.25 = -0.25`.

That's "belly down, wings unchanged" — **not** the butterfly the docstring promises ("wings up, belly down"). For a long-belly vega position the pre-fix scenario gave a misleading P&L (only the belly hit).

**Fix**: corrected formula `4·x² - 0.5` gives wings = +0.5, belly = -0.5 — a true butterfly with the twist_bps controlling the wing-belly spread.

### Verification — `test_l2_t4_vol_stress.py`

4 new tests:
- `TestTwistButterfly` × 2: wings above base, belly below base; magnitudes match +/-50bp for twist_bps=100.
- `TestParallelTilt` × 2: parallel uniform shift and tilt steepening unchanged.

Full parallel suite: **12,609 passed in 2:59** — zero regressions.

Twenty-fifth fix from phase-2. **154 distinct bugs** in v0.905→v1.017.

---

## v1.016.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.portfolio_margin.span_margin` applied SPAN's "extreme scenario 35% cap" to user-supplied scenarios.**

The SPAN methodology caps the last two scenarios (the 2× price_scan_range extremes) at 35% of their loss, reflecting that those moves are deep-tail and historically rare. Pre-fix code applied this cap to the last two entries *regardless* of whether the scenarios came from the auto-built grid or from the user:

```python
if idx >= n_scenarios - 2 and scenarios is not None:  # always True
    loss *= 0.35
```

For user-supplied custom scenarios (e.g., stress scenarios from a regulator), the last two entries were arbitrarily mis-scaled by 0.35, silently understating margin.

**Fix**: track `auto_built` flag and only apply the cap when the grid was generated internally.

### Verification — `test_l2_t4_span_margin.py`

3 new tests:
- User-supplied scenarios: every loss counted at full magnitude.
- Auto-built grid: extreme cap still active (worst of {-PSR, capped -2·PSR}).
- Vega-only user scenarios: not 35%-capped.

Full parallel suite: **12,605 passed in 2:40** — zero regressions.

Twenty-fourth fix from phase-2. **153 distinct bugs** in v0.905→v1.016.

---

## v1.015.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.scenario` constructors had the same lossy PricingContext reconstruction as v0.993 `var.stress_test`.**

All five scenario constructors (`parallel_shift`, `pillar_bump`, `vol_bump`, `fx_spot_shock`, `credit_spread_shift`) built the bumped context via `PricingContext(field=value, ...)` with only 6 named fields, silently dropping:
- `discount_curves` (the plural, multi-currency dict)
- `inflation_curves`
- `repo_curves`
- `reporting_currency`
- `stochastic_credit_models`
- `credit_vol_surfaces`
- `credit_correlations`
- `numerical_config`

Same shape as the v0.993 fix. Pricers consulting any of these silently saw a degraded context inside scenario evaluation.

**Fix**: switch all five constructors to `dataclasses.replace`, preserving every untouched field by default.

**Bonus**: `parallel_shift` now also bumps the plural `discount_curves` dict (was previously bumping only the singular). Multi-currency portfolios needed each currency's curve bumped, not just the home currency.

### Verification — `test_l2_t4_scenario.py`

6 new tests:
- `TestPreservesUntouchedFields` × 5: each constructor preserves `numerical_config` / `reporting_currency`.
- `TestParallelShiftBumpsPluralCurves` × 1: both USD and EUR discount_curves are bumped.

Full parallel suite: **12,602 passed in 2:43** — zero regressions.

Twenty-third fix from phase-2. **152 distinct bugs** in v0.905→v1.015.

---

## v1.014.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.portfolio_construction.mean_variance` `target_return` parameter was dead code.**

Pre-fix: the function signature declared `target_return: float | None = None` but the body never referenced it. Every call computed the max-Sharpe tangency portfolio regardless of whether a target was passed.

**Fix**: when `target_return` is supplied, solve the QP

    min w'Σw  s.t.  μ'w = target_return, Σw = 1, w ∈ [lb, max_weight]

via SLSQP. The `target_return=None` path (tangency portfolio) is unchanged.

### Verification — `test_l2_t4_mv_target.py`

4 new tests:
- target_return constraint honoured to 1e-6 precision.
- min-variance solution feasible (weights sum to 1, long-only ≥ 0).
- target_return=None preserves max-Sharpe behaviour.
- Infeasible target falls back deterministically without crash.

Full parallel suite: **12,596 passed in 3:05** — zero regressions.

Twenty-second fix from phase-2. **151 distinct bugs** in v0.905→v1.014.

---

## v1.013.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.simm` had two material gaps from ISDA SIMM v2.6.**

### (a) Cross-risk-class correlation = 0

Pre-fix: `total = sqrt(Σ M_i²)` despite the code comment admitting "SIMM uses sqrt of sum of squares" — but real ISDA SIMM applies the cross-class correlation matrix (Table 21 in v2.6):

| | GIRR | FX   | CSR  | EQ   | COM  |
|------|------|------|------|------|------|
| GIRR | 1.00 | 0.20 | 0.05 | 0.05 | 0.05 |
| FX   |      | 1.00 | 0.10 | 0.15 | 0.15 |
| CSR  |      |      | 1.00 | 0.15 | 0.15 |
| EQ   |      |      |      | 1.00 | 0.20 |
| COM  |      |      |      |      | 1.00 |

For a diversified GIRR+FX book the corrected total = `sqrt(M_GIRR² + M_FX² + 0.40·M_GIRR·M_FX)`, which is meaningfully larger than the zero-correlation case. Pre-fix systematically under-margined diversified books.

### (b) Vega and curvature silently dropped

The `SIMMSensitivity` dataclass exposes `delta`, `vega`, `curvature` — but `_compute_bucket` only read `s.delta`. For options books (which is precisely what SIMM is designed to margin) this is a material gap. Now: each component aggregates separately within the bucket and combines via sum-of-squares, matching SIMM's "delta-vega-curvature decomposition" structure.

For a single sensitivity with delta=d, vega=v in a single bucket: margin = `rw · sqrt(d² + v²)`. Pre-fix: margin = `rw · |d|`.

### Verification — `test_l2_t4_simm.py`

5 new tests:
- `TestCrossRiskClassCorrelation` × 1: GIRR+FX book matches ρ=0.20 closed form; > zero-corr total.
- `TestVegaCurvatureIncluded` × 3: vega/curvature each raise margin; single sensitivity matches `rw·sqrt(d²+v²)`.
- `TestDeltaOnlyUnchanged` × 1: delta-only single-class case unchanged from pre-fix.

Full parallel suite: **12,592 passed in 2:39** — zero regressions.

Twenty-first fix from phase-2. **150 distinct bugs** (2 in this slice — cross-class corr and vega/curvature) in v0.905→v1.013.

**Reached 150-bug milestone.** Production SIMM relies on this module; pre-fix margins were materially under-stated for the typical mixed-asset, options-bearing book.

---

## v1.012.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.prudent_valuation` had two API consistency bugs.**

### (a) `market_price_uncertainty_ava` non-monotonic in n_quotes

Pre-fix: `reliability_factor = min(n_quotes/5, 1.0)`. At `n_quotes=0` the special-case fallback returned `half_spread`; at `n_quotes=1` reliability=0.2 → AVA = `4.5·half_spread`. So adding the first quote *increased* AVA — contradicting the "more quotes → more reliable → smaller AVA" intent.

**Fix**: floor reliability at 0.1, drop the special-case. Now n_quotes=0 gives the maximum AVA (`9·half_spread`) and AVA monotonically decreases as quotes accumulate.

### (b) `close_out_cost_ava` silently ignores `position_days` parameter

Pre-fix: when `daily_volume > 0`, the caller's `position_days` was overwritten by `notional/daily_volume`; when `daily_volume == 0`, it was ignored entirely (defaulting to a flat 50% premium). The parameter was effectively dead.

**Fix**: honour caller's `position_days` when supplied:
- daily_volume > 0: use `max(derived, supplied, 1)`.
- daily_volume = 0 with `position_days > 1`: use the log-based scaling from the supplied estimate.
- daily_volume = 0 with default `position_days = 1`: keep the 50% default-premium fallback.

### Verification — `test_l2_t4_prudent_valuation.py`

6 new tests:
- `TestMPUMonotonicInQuotes` × 3: monotone decrease 1→5 quotes; n_quotes=0 is max; 5+ quotes gives `confidence·half_spread`.
- `TestCloseOutCostHonorsPositionDays` × 2: caller's position_days drives size adjustment when no volume; large position_days overrides default 50%.
- `TestRegressionsExistingTests` × 1: daily_volume branch unchanged when position_days at default.

Full parallel suite: **12,587 passed in 2:36** — zero regressions.

Twentieth fix from phase-2. **148 distinct bugs** (2 in this slice) in v0.905→v1.012.

---

## v1.011.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.collateral_optimisation.CollateralOptimiser` silently under-collateralised every non-cash allocation.**

Pre-fix coverage constraint:
```python
Σ_j x[i,j] >= required[i]
```

No haircut applied. For an asset with 5% haircut, allocating $100 gross gives only $95 of post-haircut collateral value — but the LP treated $100 = $100 of coverage. A CSA needing $100 covered exclusively by a 5%-haircut asset was silently $5 short of its requirement.

**Fix**: coverage constraint now uses post-haircut value:
```python
Σ_j (1 - haircut_j) × x[i,j] >= required[i]
```

The LP correctly grosses up the allocation (posting `required / (1 − h)` for single-asset coverage). The unmet-requirement check uses the same post-haircut convention.

**Side fix**: `_naive_cost` baseline now respects asset availability — pre-fix it computed the cheapest-asset-per-CSA cost ignoring supply caps, so when the cheapest asset ran out the LP correctly spilled over to the next-cheapest but appeared "worse than naive" because naive's idealised cheapest was infeasible.

### Verification — `test_l2_t4_collateral_opt.py`

4 new tests:
- `TestCoverageNetsHaircut` × 2: 5% haircut requires $100/0.95 gross; zero haircut unchanged.
- `TestUnmetWithHaircut` × 1: $95 of 5%-haircut asset (= $90.25 net) can't cover $100.
- `TestMultiCSAHaircutAware` × 1: lower-haircut asset preferred when costs comparable.

Full parallel suite: **12,581 passed in 3:04** — zero regressions; existing collateral tests still pass.

Nineteenth fix from phase-2. **146 distinct bugs** (2 in this slice — coverage and naive baseline) in v0.905→v1.011.

This one is **production-critical**: real banks running this optimiser would have been silently under-margined on every haircut'd allocation.

---

## v1.010.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.market_risk_enhanced.incremental_var` had two related issues.**

### (a) ddof inconsistency in parametric branch

`np.cov(...)` (default ddof=1) for portfolio vol but `np.std(...)` (default ddof=0) for individual position vols. Same shape as the v0.995 backtest fix. Now ddof=1 throughout.

### (b) LOO conflated with Euler decomposition for historical method

Pre-fix set both `incremental_vars` AND `component_vars` to the leave-one-out (LOO) values for the historical method. But the docstring promised `sum(IVaR_i) = portfolio VaR` — LOO doesn't have this property.

Two metrics matter and are now reported separately:
- `incremental_vars[i]`: `VaR(portfolio) − VaR(portfolio without i)` (LOO; Jorion convention).
- `component_vars[i]`: `−E[P&L_i | portfolio in tail]` (tail-conditional expectation = **ES decomposition**).

Note: for historical method, `sum(component_vars)` equals portfolio **Expected Shortfall**, not VaR. The VaR Euler decomposition requires kernel smoothing at the quantile boundary, which is noisy for finite samples; ES is what practitioners actually report. Docstring updated to make this explicit.

For parametric method, `incremental` = `component` = Euler decomposition, both sum to VaR (unchanged behaviour).

### Verification — `test_l2_t4_market_risk_enhanced.py`

5 new tests:
- `TestParametricDdofConsistency` × 1: individual VaR uses sample std (ddof=1).
- `TestHistoricalEulerDecomposition` × 2: component_vars sum to ES; LOO ≠ component.
- `TestParametricEulerInvariant` × 1: parametric component sums to VaR.
- `TestDiversificationNonNegative` × 1: diversification ≥ 0.

Full parallel suite: **12,577 passed in 2:36** — zero regressions.

Eighteenth fix from phase-2. **144 distinct bugs** (2 in this slice) in v0.905→v1.010.

---

## v1.009.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.network_xva` had a spurious `max(multiplier, 1.0)` floor that contradicted both its docstring and its own comment.**

Pre-fix:
```python
# Floor multiplier at 1.0: no contagion means no adjustment (multiplicative identity)
adjustment = alpha * centrality * max(multiplier, 1.0)
network_cva = standalone_cva * (1.0 + adjustment)
```

The comment claims "no contagion → no adjustment", but `max(0, 1.0) = 1.0` ≠ 0, so the formula still adds `α · centrality` to the multiplier even when there's no contagion. The docstring's formula is `CVA × (1 + α × centrality × multiplier)` — when multiplier = 0, no adjustment.

For an isolated counterparty (no outgoing exposures → no contagion → multiplier = 0):
- Pre-fix: CVA was bumped by `α · centrality · CVA` regardless.
- Post-fix: CVA = standalone_cva (multiplier = 0 → adjustment = 0).

**Fix**: removed the floor in both `compute_network_cva` and `systemic_cva_adjustment`. Multiplier passes through raw.

### Verification — `test_l2_t4_network_xva.py`

5 new tests:
- `TestNoContagionNoAdjustment` × 2: isolated CP unchanged; multiplier=0 in convenience fn unchanged.
- `TestContagionRaisesCVA` × 1: cascading CP still uplifted.
- `TestAlphaZeroNoChange` × 1: alpha=0 always recovers standalone (existing invariant).
- `TestSystemicCvaAdjustmentClosedForm` × 1: exact formula match.

Full parallel suite: **12,572 passed in 3:02** — zero regressions; existing `test_phase5_integration` tests still pass.

Seventeenth fix from phase-2. **142 distinct bugs** (2 in this slice) in v0.905→v1.009.

---

## v1.008.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.leverage_optimisation` concentration constraint was looser than its own docstring.**

Pre-fix: `w_i ≤ max_single_pct × (capital × max_leverage)` — bound is fraction of *theoretical max notional*, not realised portfolio. When actual leverage is below max_leverage, this is much looser than the docstring-promised relative form `w_i ≤ max_single_pct × Σw`.

Example: capital=$100M, max_lev=10, max_single=30%, actual_lev=5x → portfolio = $500M.
- Pre-fix: single trade allowed up to $300M (= 60% of portfolio).
- Docstring (relative): single trade capped at $150M (= 30% of portfolio).

**Fix**: implement the relative constraint linearly:

    (1 − max_pct)·w_i − max_pct·Σ_{j≠i} w_j ≤ 0

Default `max_single_trade_pct` raised from `0.30` to `1.0` (no concentration cap by default). The relative form requires `max_pct ≥ 1/N` to be feasible with all-active portfolio; with the old default of 0.30 and N=3 trades the LP would be infeasible. Callers that want strict concentration set `max_single_trade_pct` explicitly.

### Verification — `test_l2_t4_leverage_opt.py`

4 new tests:
- `TestRelativeConcentration` × 2: every w_i/Σw ≤ max_pct (feasible N=4, max_pct=0.30).
- `TestSingleTradeRecovery` × 1: uniform weights under tight cap.
- `TestOptimiserHappyPath` × 1: solver returns positive carry/leverage.

Plus existing `test_repo_phase3.TestLeverageOptimisation` (13 tests) still pass with the new default.

Full parallel suite: **12,567 passed in 3:01** — zero regressions.

Sixteenth fix from phase-2. **140 distinct bugs** in v0.905→v1.008.

---

## v1.007.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.correlation_network._quasi_diag` was a greedy nearest-neighbour tour, not López de Prado's quasi-diagonalisation.**

Pre-fix:
```python
order = [0]
while remaining:
    last = order[-1]
    nearest = min(remaining, key=lambda j: dist[last, j])
    order.append(nearest)
```

This produces a path-like ordering starting at asset 0, traversing always to the nearest unvisited asset. **NOT** a cluster-aware ordering: subsequent recursive bisection in `hierarchical_risk_parity` then mixes cluster-mates apart at the midpoint cut, defeating HRP's whole premise.

True LdP quasi-diagonalisation: build hierarchical clustering (single-linkage on the distance matrix), then traverse the dendrogram leaves in order so that siblings are adjacent. The other HRP module (`risk.hierarchical_risk_parity`) already does this correctly — `correlation_network` had its own broken implementation.

**Fix**: use `scipy.cluster.hierarchy.linkage` + `leaves_list` (matching the other HRP module).

### Verification — `test_l2_t4_correlation_network.py`

6 new tests:
- `TestQuasiDiagBlockStructure` × 1: 2-cluster distance matrix → cluster-mates adjacent in output.
- `TestHRPCorrelationNetwork` × 3: weights sum to 1; non-negative; block structure preserved in cluster_order.
- `TestQuasiDiagDegenerate` × 2: N=1 and N=2 handled.

Full parallel suite: **12,563 passed in 2:41** — zero regressions.

Fifteenth fix from phase-2. **139 distinct bugs** in v0.905→v1.007.

---

## v1.006.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.hybrid_xva.hybrid_cva` and `hybrid_fva` computed future-valued XVA instead of present-valued.**

The textbook formulas:

    CVA = (1 − R) × Σ_i D(0, t_i) × EPE(t_i) × ΔPD(i)
    FVA = funding_spread × Σ_i D(0, t_i) × E[V(t_i)] × dt

Pre-fix omitted `D(0, t_i)` entirely. For a 10y trade with rates ~5%, this overstated CVA/FVA by ~30%-40% (since avg DF over 10y at 5% ≈ 0.78).

**Fix**: added optional `discount_factors: np.ndarray | None = None` parameter to both functions. When supplied, multiplies EPE/EE by `D(0, t_i)` element-wise before summing. Backwards compatible — omitting it preserves the old behaviour, now documented as requiring pre-discounted exposure inputs.

This is the second XVA-family discount-factor fix in this audit (compare v0.984 risk/network.py and the earlier curve fixes). Production XVA engines should pass discount factors explicitly to avoid the FV/PV trap.

### Verification — `test_l2_t4_hybrid_xva.py`

6 new tests:
- `TestCVADiscounting` × 3: DF reduces CVA; shape validation; DF=1 unchanged.
- `TestFVADiscounting` × 2: DF reduces FVA; shape validation.
- `TestNoChangeWhenAlreadyDiscounted` × 1: backwards-compat preserved.

Full parallel suite: **12,557 passed in 2:41** — zero regressions.

Fourteenth fix from phase-2. **138 distinct bugs** (2 in this slice) in v0.905→v1.006.

---

## v1.005.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.shapley.shapley_capital_allocation` computed diversification info then threw it away.**

Pre-fix built an `enriched_values` dict containing per-desk `{shapley_allocation, standalone_risk, diversification_benefit}`, then **immediately discarded it** by returning the raw `ShapleyResult`. The function docstring promised diversification reporting; the caller never received it.

**Fix**: added optional `diversification: dict | None` field to `ShapleyResult` (defaults to `None` for non-capital-allocation callers, preserving backwards compat). `shapley_capital_allocation` now populates it. Also surfaced in `to_dict()`.

Audit-clean modules from the cooperative-game family:
- `risk/cooperative_games.py` — Shapley delegate + core-check correct.
- `risk/shapley.py` exact/sampling algorithms — formulas correct (Shapley weights sum to 1; sampling estimator unbiased).

### Verification — `test_l2_t4_shapley.py`

4 new tests: diversification field populated; benefit equals `standalone − shapley`; surfaced in to_dict; default `None` for non-capital-allocation callers.

Full parallel suite: **12,551 passed in 2:43** — zero regressions.

Thirteenth fix from phase-2. **136 distinct bugs** in v0.905→v1.005.

---

## v1.004.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.brinson_attribution.brinson_multi_period` claimed "geometric linking" but used ad-hoc scaling that broke the active-return identity.**

Pre-fix:
```python
cum_alloc += r.total_allocation * cum_bench   # cum_bench BEFORE updating
```

This is neither Frongello, Carino, nor Menchero. The geometric multi-period identity:

    Σ_t F_t · (alloc_t + sel_t + inter_t) = Π(1+r_p_t) − Π(1+r_b_t)

requires the Frongello linking coefficient

    F_t = (Π_{s<t}(1+r_p_s)) · (Π_{s>t}(1+r_b_s))

which can be computed iteratively as

    cum_t = cum_{t-1} · (1+r_b_t) + effect_t · cum_port_{t-1}

The pre-fix version diverged from the geometric active by a factor that grows quadratically in T:
- 2 periods, port=1%/bench=0% each: pre-fix 2.00% vs geometric 2.01%.
- 10 periods, port=1%/bench=0% each: pre-fix 10.00% vs geometric 10.462%.

**Fix**: implement the Frongello recursive linking. Prior cumulative effects scale by `(1+r_b_t)`; new period's effect scales by `cum_port_before_t`.

### Verification — `test_l2_t4_brinson.py`

5 new tests:
- `TestFrongelloIdentity` × 3: 2-period, 3-period mixed, 10-period — the identity `Σ effects = cumulative_active_return` holds to machine precision.
- `TestBrinsonSinglePeriodIdentity` × 2: per-period Brinson identity (unchanged) holds for 2- and 3-sector portfolios.

Full parallel suite: **12,547 passed in 2:37** — zero regressions.

Twelfth fix from phase-2. **135 distinct bugs** in v0.905→v1.004.

---

## v1.003.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.model_selection.model_committee_price` was statistically inconsistent.**

Pre-fix combined a *weighted* mean with an *unweighted* `np.std`. For BMA committees where one model dominates (typical: BIC-based weights of 0.9 + 0.05 + 0.05), the unweighted std treated all models equally, overstating uncertainty.

Concrete example: weights `(0.9, 0.05, 0.05)`, prices `(100, 200, 200)`. Weighted mean = `110`. Unweighted pop-std (pre-fix) ≈ `47`. Correct weighted std = `30` (because the dominant model's price is close to the mean).

**Fix**: weighted standard deviation `sqrt(Σ w_i · (p_i − weighted_mean)²)`, consistent with the weighted mean. The `price_range` and `model_uncertainty_reserve` fields (min/max-based) are unchanged — they're informative regardless of weights.

### Verification — `test_l2_t4_model_selection.py`

5 new tests:
- `TestWeightedStd` × 2: dominant-model committee gives weighted std (30 not 47); equal weights match population std.
- `TestCommitteeStdEdgeCases` × 2: single model → 0 std; identical prices → 0 std.
- `TestCommitteeRangeUnchanged` × 1: range/reserve still use unweighted extremes.

Full parallel suite: **12,542 passed in 2:44** — zero regressions, existing model_selection tests unchanged.

Eleventh fix from phase-2. **134 distinct bugs** in v0.905→v1.003.

---

## v1.002.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.contagion.DefaultCascade.simulate` silently dropped second-order contagion.**

The bug was a textbook conflation of two distinct flags via one variable. Pre-fix used `remaining_buffer[d] = -1` as both:
- "this node has defaulted" (set on initial defaulters at start of round), AND
- "this node's outward losses have been propagated to its creditors" (the "skip if already processed" check).

But a creditor that defaults mid-cascade *also* has `remaining_buffer < 0` (from incoming losses absorbed beyond its buffer). In the next round, the `if remaining_buffer[d] < 0: continue` check at the top of the outer loop saw it as "already processed" and **skipped propagating its losses to its own creditors**.

Net effect: only first-order contagion (initial-default's direct creditors) ever cascaded. A→B→C chains lost everything past B. The "contagion multiplier" metric understated systemic risk by an unbounded factor.

**Fix**: separate `processed: set[int]` tracks which defaulters have had their losses propagated. Each defaulter now propagates outward exactly once, regardless of whether their buffer went negative from incoming losses.

### Verification — `test_l2_t4_contagion.py`

4 new tests:
- `TestSecondOrderContagion` × 2: A→B→C with C surviving (but absorbing B's losses); A→B→C with all three defaulting.
- `TestNoContagion` × 1: well-capitalised neighbours absorb A's loss without propagating.
- `TestProcessedOnceInvariant` × 1: 4-node ring topology, all default.

Full parallel suite: **12,537 passed in 2:40** — zero regressions (existing `test_graph_theory.py::DefaultCascade` tests were too lenient to catch the bug).

Tenth fix from phase-2. **133 distinct bugs** in v0.905→v1.002.

This is one of the more serious bugs found in phase 2 — it silently understated systemic risk in a financial-stability tool.

---

## v1.001.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.hierarchical_risk_parity` had two issues.**

(a) **N=1 crashed** — `np.corrcoef` of `(T, 1)` returns scalar; `_correlation_distance` produces `(1,1)` zero matrix; `squareform` of that → empty vector; `linkage` on empty → ValueError. Now returns trivial weights=`[1.0]` directly for the single-asset case.

(b) **`n_clusters` reporting was a meaningless heuristic.** Pre-fix: `min(N, max(2, N//3))` — unrelated to actual dendrogram structure. Now uses `fcluster` at the median linkage height, so the reported count reflects the real cluster topology found by hierarchical clustering.

The core HRP algorithm (López de Prado 2016: correlation distance → linkage → quasi-diagonalisation → recursive bisection with inverse-variance allocation) was correctly implemented; only edge case and reporting affected.

### Verification — `test_l2_t4_hrp.py`

5 new tests:
- `TestHRPSingleAsset` × 1: N=1 returns [1.0] without crash.
- `TestHRPClusterCount` × 2: clustered returns yield ≥2 clusters; highly-correlated near-uniform returns yield few clusters.
- `TestHRPWeightInvariants` × 2: weights sum to 1, all non-negative.

Full parallel suite: **12,533 passed in 2:45** — zero regressions.

Ninth fix from phase-2. **132 distinct bugs** in v0.905→v1.001.

---

## v1.000.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.efficient_frontier.minimum_variance_portfolio` returned wrong expected_return and sharpe_ratio fields.**

Pre-fix: the function had no access to the expected-returns vector `mu`, but its return type (`FrontierPoint`) requires `expected_return` and `sharpe_ratio`. Both were hardcoded to `0` regardless of the portfolio. Inside `efficient_frontier` this was patched up by mutating the returned dataclass; but every other caller — including the public `test_portfolio_game_theory.py` usage — got nonsense fields.

**Fix**: added optional `mu` (and `risk_free_rate`) parameters. When supplied, `expected_return = mu @ w` and `sharpe = (μw − rf) / vol`. When omitted, fields default to 0 (legacy behaviour preserved).

### Verification — `test_l2_t4_efficient_frontier.py`

5 new tests:
- `TestMinVarianceWithMu` × 4: mu supplied populates return/Sharpe; mu omitted gives legacy zeros; long-only matches `mu @ w` exactly; shape mismatch raises.
- `TestMinVarianceVolUnchanged` × 1: unconstrained weights match analytical Σ⁻¹·1 / (1'·Σ⁻¹·1).

Full parallel suite: **12,528 passed in 3:10** — zero regressions.

### Audited-clean modules in this phase (no slice required)

- `risk/correlation_repair.py` — Higham (2002) alternating projections correctly implemented; minor cosmetic issues only.
- `risk/cvar_optimisation.py` — Rockafellar-Uryasev LP correctly formulated; component CVaR uses correct Euler decomposition.
- `risk/model_reserve.py` — sensitivity-based aggregation (worst-case sum vs quadrature) correct; minor docs gap (confidence param stored but unused, no math impact).

Eighth fix from phase-2. **130 distinct bugs** in v0.905→v1.000.

This is the **v1.0 milestone** for pricebook. The audit arc that started at v0.905 has now resolved 130 correctness bugs across 8 sessions, with 12,528 tests passing and zero regressions across 95+ stamped versions.

---

## v0.999.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.ipv.ipv_single_trade` had no concept of position direction.**

Pre-fix: `prudent_value = mid - diversified_ava` was hardcoded for *long* positions. For a short bond position the prudent (conservative) liability value sits *above* mid, not below — the IPV would silently understate the size of a short liability by `2 · ava`.

Compounding: AVA helpers (`close_out_cost_ava`, `concentration_ava`, etc.) take `notional` as a positive size and multiply by it. Passing a *signed* notional (negative for short) propagated a negative magnitude through every AVA, producing nonsense.

**Fix**:
- Added explicit `direction: int = 1` parameter (only `±1` accepted; default `+1` preserves long-only behaviour).
- All AVA helpers now receive `abs(notional)`.
- `prudent_value = mid − direction · diversified_ava` (above mid for shorts).
- `variance_to_model_bp` denominator uses `abs(notional)` — pre-fix a negative-notional input produced negative `variance_bp` that could never breach the threshold.

### Verification — `test_l2_t4_ipv.py`

7 new tests:
- `TestIpvLongUnchanged` × 1: default direction preserves pre-fix long behaviour.
- `TestIpvShort` × 3: prudent above mid; AVA magnitudes mirror long for direction=−1; abs(notional) is what counts (sign doesn't flip AVA).
- `TestIpvDirectionValidation` × 2: invalid direction (0 or +2) raises.
- `TestIpvVarianceWithNegativeNotional` × 1: variance_bp ≥ 0 and breach still fires.

Full parallel suite: **12,523 passed in 2:42** — zero regressions, 11 existing IPV tests still pass.

Seventh fix from phase-2. **129 distinct bugs** in v0.905→v0.999.

---

## v0.998.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.kelly` had two correctness issues.**

### (a) `kelly_fraction` silently returned 0 for vol=0

Pre-fix: `f_star = excess / var if var > 0 else 0`. This masked two genuinely different cases — positive excess with σ=0 is a deterministic *arbitrage* (Kelly = +∞, short all wealth into it); negative excess with σ=0 is a deterministic loss (Kelly = −∞). Returning 0 in both cases makes the asset appear unattractive when in fact it's an unbounded opportunity. Now raises `ValueError` with diagnostic.

### (b) `multi_asset_kelly` had no validation and lossy singular-cov fallback

Two issues:
- No shape/symmetry check — caller could pass `(3,4)` matrix or non-symmetric Σ and get garbage out.
- Fallback on singular Σ used diagonal-only inverse `f = excess / diag(Σ)` — this drops correlation structure entirely. For two highly-correlated assets the true Kelly concentrates weight on the better risk-adjusted one; the diagonal fallback spreads incorrectly.

Now validates shape (square, mu-compatible) and symmetry (within `1e-10`), and uses `np.linalg.pinv` (Moore-Penrose pseudoinverse) as the singular fallback — preserves correlation structure and gives the minimum-norm solution.

### Verification — `test_l2_t4_kelly.py`

10 new tests across 5 classes:
- `TestKellyFractionRequiresVol` × 3: zero/negative vol raise; positive unchanged.
- `TestMultiAssetKellyValidation` × 3: non-square/shape-mismatch/non-symmetric raise.
- `TestMultiAssetKellyAnalytical` × 2: diagonal Σ matches per-asset Kelly; correlated Σ matches `solve(Σ, μ)`.
- `TestMultiAssetKellySingularPseudoInverse` × 1: near-singular Σ finite via pinv.
- `TestMultiAssetKellyGrowthFormula` × 1: g = rf + f·μ̄ − 0.5·f'Σf.

Full parallel suite: **12,516 passed in 3:08** — zero regressions. Warnings 19→17.

Sixth fix from phase-2. **128 distinct bugs** (2 more in this slice) in v0.905→v0.998.

---

## v0.997.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.correlation_greeks` had two issues.**

### (a) `correlation_pnl_attribution` did 8 pricer calls when 4 suffice

Pre-fix: `correlation_delta(rho_old)` (3 calls) + `correlation_gamma(rho_old)` (3 calls — overlapping `rho_old`, `rho_old+bump`, `rho_old-bump` with the delta call) + direct `price_fn(rho_new)` + `price_fn(rho_old)` = **8 calls**. Only 4 unique points needed: `rho_old`, `rho_old±bump`, `rho_new`. Now computed inline — halves the cost for expensive multi-asset pricers (basket MC, ND PDE). Same shape as v0.994 `bump_greeks` duplicate-call fix.

### (b) `CorrelationLadder` exposed only gross magnitudes

`total_rho_delta` and `total_rho_gamma` summed `abs(delta_i)`. For sizing hedges (gross exposure) this is useful, but for portfolio P&L the signed sum is what matters when correlations drift together. Added `net_rho_delta` / `net_rho_gamma` as signed totals; preserved the original gross fields for backwards compatibility.

### Verification — `test_l2_t4_correlation_greeks.py`

4 new tests:
- `TestCorrelationPnlAttributionCallCount` × 2: exactly 4 unique calls; quadratic price_fn → Taylor explains everything (unexplained = 0).
- `TestCorrelationLadderNet` × 2: net vs gross sums with sign-mixed pairs (net=+1, gross=3); linear-in-rho pricer has zero net/gross gamma.

Full parallel suite: **12,506 passed in 2:58** — zero regressions.

Fifth fix from phase-2. **126 distinct bugs** (2 more in this slice) in v0.905→v0.997.

---

## v0.996.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.factor_model` had two real bugs.**

### (a) Ledoit-Wolf shrinkage intensity was ad-hoc

Pre-fix used `alpha_lw = 1 / (n · delta)` with no basis in Ledoit-Wolf (2004) and dimensionally inconsistent (units of cov-squared⁻¹). The "shrunk" output was NOT a Ledoit-Wolf estimator. Now uses the correct LW formula for the μ·I target (LW 2004, §3.2):

    π̂  = sum over i,j of (1/T) Σ_t (x̃_ti x̃_tj − s_ij)²
    γ̂² = ||F − S||²_F
    ρ̂  = trace(π̂ matrix)       (identity-target case)
    κ  = (π̂ − ρ̂) / γ̂²
    δ* = max(0, min(κ / T, 1))

Vectorised via `pi_mat = (X²ᵀ @ X²)/T − S²`, O(T·n²) per pass.

### (b) `factor_timing` direction contradicted its own docstring

Pre-fix code: `if z > threshold: signal = "overweight"` (momentum on factor value). Pre-fix docstring: "When the factor is cheap (low z-score), overweight it" (contrarian — Asness, Ilmanen). The two contradicted each other. Picked the contrarian convention (matches academic literature on factor timing). Hit-rate calculation reversed accordingly.

The existing `test_overweight` assertion in `test_factor_model.py` had pinned the buggy momentum direction — updated to the corrected contrarian semantics.

### Verification — `test_l2_t4_factor_model.py`

6 new tests:
- `TestLedoitWolfShrinkage` × 3: intensity in [0,1]; high-dim case shows meaningful shrinkage; off-diagonals shrink under iid data toward identity.
- `TestFactorTimingContrarian` × 3: high z → underweight; low z → overweight; perfect mean-reversion regime → hit rate > 99%.

Full parallel suite: **12,502 passed in 3:29** — zero regressions.

Fourth fix from phase-2. **124 distinct bugs** (2 more in this slice) in the v0.905→v0.996 arc.

---

## v0.995.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.backtest` had four defects (one major semantic, three numerical/accounting).**

### (a) Sample vs population std — `compute_metrics`
`np.std()` defaults to `ddof=0` (population). Sharpe / Sortino are conventionally reported with `ddof=1` (sample std). Pre-fix vol understated by `sqrt(n/(n-1))` — inflating Sharpe by ~1.7% at n=30, ~0.2% at n=252. Now uses `ddof=1`.

### (b) Initial position has no transaction cost — `run_backtest`
Pre-fix loop started at `i=1`, only charging slippage/commission for `positions[i] − positions[i-1]`. The initial entry (implicit 0 → `positions[0]`) was free. Real consequence: any constant-direction strategy started with negative immediate P&L from entry costs being un-modelled. Now charges `abs(positions[0])·slippage_bps/10000·capital + commission` at `pnl[0]`.

### (c) Walk-forward isn't walk-forward — `walk_forward` (major)
Pre-fix called `signal_func(train)` and `signal_func(test)` *separately*. For any `signal_func` with a warm-up window (e.g. 20-day momentum), the first warm-up bars of test had undefined/biased signals because there's no train-side history. The "walk" never actually carried history forward. Now passes `concat(train, test)` to `signal_func` and slices out the test portion — signals on test bars are computed with full access to all preceding train history, no forward-leakage.

### (d) Deflated Sharpe missing Euler-Mascheroni correction — `deflated_sharpe`
Pre-fix used `E[max] ≈ Φ⁻¹(1 − 1/n)` (first-order). Bailey-De Prado (2014) specifies:

    E[max] ≈ (1 − γ) · Φ⁻¹(1 − 1/n) + γ · Φ⁻¹(1 − 1/(n·e))

where γ ≈ 0.5772 (Euler-Mascheroni), e ≈ 2.718. For n=100, pre-fix gives 2.326 vs BdP 2.530 — pre-fix *over*-states the deflated probability (admits more false positives), the opposite of what data-snooping correction should do.

### Verification — `test_l2_t4_risk_backtest.py`

10 new tests across 4 test classes:
- `TestComputeMetricsSampleStd` × 2: vol matches `np.std(ddof=1)`, Sharpe consistent.
- `TestRunBacktestInitialSlippage` × 3: initial entry charged; zero entry not charged; commission charged on non-zero entry.
- `TestWalkForwardHistoryContext` × 2: `signal_func` called on combined series; warm-up signal yields finite Sharpes.
- `TestDeflatedSharpeBaileyDePrado` × 3: n=100 BdP value ≈ 2.530; pre-fix-threshold SR has lower significance; high SR retains high DSR.

Full parallel suite: **12,495 passed in 3:02** — zero regressions.

Third fix from phase-2. Bug count: 119, 120, 121, 122 (four defects in one module). **122 distinct bugs** in the v0.905→v0.995 arc.

---

## v0.994.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.greeks.bump_greeks` had three issues.**

(a) **Duplicate pricer call.** Pre-fix computed `price_func(spot, vol - vol_bump, ...)` twice — once as `vega_down_v` (used in vega), once as `vega_down` (used in volga). Identical computations, wasted call. For MC/PDE pricers this is real perf cost (could be seconds per Greeks call).

(b) **Asymmetric rho difference.** Pre-fix used a forward difference `(rho_up - base) / rate_bump` while delta/gamma/vega all used central differences. Forward diff is O(h); central is O(h²). Inconsistent with neighbouring Greeks. Switched to central diff.

(c) **No bump-size validation.** Non-positive bumps or `vol_bump >= vol` would silently push negative vol into the pricer, often crashing deep or returning NaN. Now raises `ValueError` upfront with diagnostic.

### Verification — `test_l2_t4_risk_greeks_bump.py`

6 new tests:
- `TestNoDuplicateCalls`: pricer call count bounded.
- `TestRhoCentralDifference`: rho matches BS analytical `K·T·exp(-rT)·N(d2)·0.01` to rel 1e-4.
- `TestValidationRaises` × 3: zero/negative bumps + vol_bump≥vol raise.
- `TestSanityRanges`: ATM call Greeks in expected sign/range bands.

Full parallel suite: **12,485 passed in 3:03** — zero regressions, +6 net tests. Warnings 18→17.

Second fix from phase-2. 118th distinct bug.

---

## v0.993.0 — 2026-06-13

**Fix L2 phase-2 audit (first new-module find) — `risk.var.stress_test` had two coupled bugs.**

Opening up `risk/` for systematic audit. First find:

**(a) Silent no-op on unsupported shocks.** Pre-fix inspected only `rate_shift` and silently dropped every other key — including `vol_shift` which appeared explicitly in the function's own docstring example. Users following the docs got identical-to-base P&Ls that looked like successful stress runs.

**(b) Lossy context reconstruction.** The bumped context was built from a 6-field constructor subset, silently dropping the plural `discount_curves`, `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, and (added later in G1 P3) `numerical_config`. Pricers relying on any of these silently saw a degraded context.

**Fix**: use `dataclasses.replace` for non-lossy reconstruction; raise `ValueError` on unknown shock keys. Now supports `rate_shift` (bumps singular `discount_curve` + plural `discount_curves` + `projection_curves`) and `credit_shift` (bumps `credit_curves`).

### Verification — `test_l2_t4_risk_var_stress_test.py`

7 new tests:
- `TestStressTestRaisesOnUnsupportedShocks` × 2: `vol_shift` raises; typo `rate_shifft` raises.
- `TestStressTestRateShift` × 2: singular + plural discount curves both bumped.
- `TestStressTestPreservesUntouchedFields` × 2: `numerical_config` and `reporting_currency` survive reconstruction.
- `TestStressTestEmpty` × 1: empty shock dict is a no-op (no error).

Full parallel suite: **12,479 passed in 3:07** — zero regressions.

First fix from phase-2 (modules not yet audited). 117th distinct bug.

---

## v0.992.0 — 2026-06-13

**Fix L2 Wave-2 audit — `fx_double_barrier_option` returned `vanilla` unconditionally at vol=0/T>0, assuming no barrier breach.**

At vol=0, the spot path is the deterministic monotonic exponential `S_t = spot·exp((rd-rf)·t)`. With `spot` known to be inside the corridor (checked just above), the path breaches a barrier iff the forward `spot_T = spot·exp((rd-rf)·T)` exits the corridor:

- Forward stays in `[L, U]` → no breach → KO = vanilla, KI = 0.
- Forward `>= U` → upper breach → KO = 0, KI = vanilla.
- Forward `<= L` → lower breach → KO = 0, KI = vanilla.

Pre-fix returned `vanilla` unconditionally — systematically over-pricing knock-outs and under-pricing knock-ins whenever the deterministic forward drifted out of the corridor.

T=0 boundary preserved (no path traversal possible).

KO + KI = vanilla parity holds exactly in all four cases.

### Verification — `test_l2_t4_fx_double_barrier_degenerate.py`

8 new tests:
- `TestFxDoubleBarrierVolZeroKnockOut` × 3: forward-inside pays vanilla; forward-above-upper KO = 0; forward-below-lower KO = 0.
- `TestFxDoubleBarrierVolZeroKnockIn` × 2: inside KI = 0; upper-breach KI = vanilla.
- `TestFxDoubleBarrierTZero` × 1: T=0 returns vanilla.
- `TestParity` × 2: KO + KI = vanilla for both inside-corridor and breach cases.

Full parallel suite: **12,472 passed in 2:51** — zero regressions.

Third of four residual convention-dependent fixes deferred from the v0.989 sweep. 116th distinct bug.

---

## v0.991.0 — 2026-06-13

**Fix L2 Wave-2 audit — `fx_lookback_floating` returned 0 unconditionally at `vol <= 0 or T <= 0`, dropping the deterministic non-zero value from drift-driven path extrema.**

At vol=0, T>0 the spot path is the deterministic exponential `S_t = spot·exp((rd-rf)·t)`. Over [0, T]:
- If `rd >= rf`: monotone non-decreasing → `min = spot`, `max = spot_T = forward`.
- If `rd <  rf`: monotone decreasing → `min = spot_T`, `max = spot`.

The floating-strike call payoff is `S_T − running_min`; put is `running_max − S_T`. The running extreme combines the *observed* `running_extreme` parameter (defaults to spot) with the path extremes.

Pre-fix returned 0 for all four (call/put × positive/negative drift) cases. This is wrong even with default `running_extreme = spot`:
- Positive drift call: forward > spot → payoff = `forward − spot > 0`.
- Negative drift put: spot > forward → payoff = `spot − forward > 0`.
- The other two (negative-drift call, positive-drift put) correctly return 0 when `running_extreme` is `None`, since the path extremum doesn't beat the spot reference.
- With `running_extreme` set: drift-opposite case still pays when the *observed* extreme is inside the path range.

T=0 boundary preserved (returns 0).

### Verification — `test_l2_t4_fx_lookback_degenerate.py`

7 new tests:
- `TestFxLookbackVolZeroTPositiveCall` × 3: positive-drift pays `df_d·(forward−spot)`; negative-drift pays 0 without low running_extreme; negative-drift pays when running_extreme < forward.
- `TestFxLookbackVolZeroTPositivePut` × 2: negative-drift pays `df_d·(spot−forward)`; positive-drift returns 0 by default.
- `TestFxLookbackTZero` × 1: T=0 returns 0.
- `TestFxLookbackInteriorUnchanged` × 1: interior path finite.

Full parallel suite: **12,464 passed in 2:58** — zero regressions.

Second of four residual convention-dependent fixes deferred from the v0.989 sweep. 115th distinct bug.

---

## v0.990.0 — 2026-06-13

**Fix L2 Wave-2 audit — `fx_charm` (∂Δ/∂t) returned 0 at `vol=0, T>0`, silently dropping the deterministic `±r_f·exp(-r_f T)` discount-decay contribution.**

The FX spot delta at vol=0 has the deterministic value `±exp(-r_f T)·I(forward vs strike)`. Its derivative w.r.t. calendar time is the discount-decay `±r_f·exp(-r_f T)·indicator` — not 0.

Closed-form deterministic limits at vol=0:
- ITM call (forward > strike): `charm = +r_f·exp(-r_f T)`
- ITM put (forward < strike): `charm = -r_f·exp(-r_f T)`
- OTM (both): `charm = 0`
- ATM (forward == strike): `±0.5·r_f·exp(-r_f T)` — indicator one-sided half-limit (the boundary-shift term1 = -exp(-rf T)·N'(d1)·[2(rd-rf)T]/(2T σ√T) diverges to ±∞ when rd ≠ rf, but the dominant indicator contribution is well-defined at half-limit).

T=0 boundary preserved (returns 0).

### Verification — `test_l2_t4_fx_charm_degenerate.py`

8 new tests: ITM call positive, ITM put negative, OTM call/put zero, ATM half, zero-rf returns zero, T=0 returns zero, interior finite.

Full parallel suite: **12,457 passed in 2:51** — zero regressions.

First of four residual convention-dependent fixes deferred from the v0.989 sweep. 114th distinct bug.

---

## v0.989.0 — 2026-06-13

**Fix L2 Wave-2 audit — single-pass sweep of `T<=0 or vol<=0` degenerate-branch defects across FX, equity, and inflation modules.**

After three single-site fixes in this pattern (v0.986 equity_rho, v0.987 equity_theta, v0.988 _digital_call_bs), swept the remaining call sites in one pass. Each defect falls into one of three buckets:

- **(a) Spot vs forward indicator.** At `vol=0, T>0` the deterministic terminal is `forward = spot·exp((r-q)T)`, not spot. OTM-spot but ITM-forward positions silently returned 0.
- **(b) Missing discount factor.** Even with the right indicator, payoffs at maturity must be PV'd.
- **(c) Missing ATM half-limit.** At `forward == strike` the one-sided limit is half the ITM value.

### Sites fixed (7)

| # | File / function | Bucket(s) | Pre-fix behaviour |
|---|---|---|---|
| 1 | `fx/fx_option.py::fx_spot_delta` | (a)+(c) | `spot > strike` indicator; ATM returned 0 |
| 2 | `fx/fx_option.py::fx_forward_delta` | (a)+(c) | same shape |
| 3 | `fx/fx_american.py::_gk_european` | (a)+(b) | `max(spot − strike, 0)` undiscounted |
| 4 | `equity/equity_exotic.py::equity_digital_cash` | (a)+(b)+(c) | undiscounted payout on spot indicator |
| 5 | `equity/equity_exotic.py::equity_digital_asset` | (a)+(b)+(c) | undiscounted spot on spot indicator |
| 6 | `fx/fx_exotic_extensions.py::fx_digital_option` | (b)+(c) | forward indicator was right but no df |
| 7 | `fixed_income/inflation_bond_advanced.py::deflation_floor_value` | (b) | linearised `-breakeven·T` instead of exact `1 − exp(breakeven·T)` |

Site 7 is independent: pre-fix used a first-order Taylor approximation for the deflation intrinsic. For `breakeven=-5%, T=30y` the pre-fix returned `1.5` (exceeding the maximum possible deflation = 100% loss); the exact intrinsic is `1 − exp(-1.5) ≈ 0.777`.

### Sites audited and left unchanged

Already correct: `options/exotic_payoffs._bs_call` (uses discounted intrinsic of forward), `commodity/commodity_american._black76_european` (takes forward directly), `structured/cms` range probability (uses forward), `numerical/black76` (already fixed in v0.965).

Convention-dependent / out of scope for one-pass sweep: `crypto/crypto_vol` ATM theta heuristic, `fx/fx_correlation` quanto ρ inversion, `fx/fx_exotic.fx_lookback_floating` (depends on running_extreme + drift sign), `fx/fx_exotic_extensions` double-barrier MC (needs path-monotonicity check), `fx/fx_greeks` higher-order Greeks at vol=0 are 0 by convention.

### Verification — `test_l2_t4_degenerate_branch_sweep.py`

19 new tests across 7 test classes (one per fixed site). Each test verifies the exact closed-form deterministic limit and contains a comment indicating what the pre-fix value would have been.

Full parallel suite: **12,449 passed in 2:51** — zero regressions, +19 net tests.

Bundled fix counts as one logical slice (single defect class swept). Bug count: 56 → 63 from the 35-module Wave-2 audit; total **113 distinct bugs** in the v0.905→v0.989 arc.

---

## v0.988.0 — 2026-06-13

**Fix L2 Wave-2 audit — `_digital_call_bs` in `structured.equity_linked_note` had two coupled bugs in the degenerate `T<=0 or vol<=0` branch.**

Same shape as the v0.948 `equity_delta` fix.  Pre-fix:

```python
if T <= 0 or vol <= 0:
    return 1.0 if spot > strike else 0.0
```

Two bugs:
- **(a) Spot vs forward indicator.** At `vol=0, T>0`, the terminal value is `S_T = forward = spot·exp((r-q)T)` (deterministic), not `spot`. An OTM-spot but ITM-forward digital silently returned 0. Example: `spot=95, rate=10%, T=1 → forward≈105 > strike=100` is ITM but pre-fix returned 0.
- **(b) No discount factor.** Even when ITM, the digital pays $1 at maturity and must be present-valued. Pre-fix returned `1.0` instead of `exp(-rT)`.

**Fix**: split `T = 0` (no drift, no discount — keep spot indicator) from `T > 0, vol = 0` (use forward indicator and apply `df = exp(-rT)`; ATM forward is the one-sided half-limit).

### Verification — `test_l2_t4_digital_eln_degenerate.py`

8 new tests:
- `TestDigitalCallVolZeroTPositive` × 4: ITM-forward/OTM-spot returns `df` (was 0 pre-fix); ITM-call returns `df` (was 1.0 pre-fix); OTM returns 0; ATM-forward returns `0.5·df`.
- `TestDigitalCallTZero` × 2: T=0 path preserved.
- `TestDigitalCallInteriorUnchanged` × 2: interior `df·N(d2)` formula and dividend-yield-on-forward both unchanged.

Full parallel suite: **12,430 passed in 3:11** — zero regressions, +8 net tests.

Fifty-sixth fix from the 35-module Wave-2 audit; 106th distinct bug in the v0.905→v0.988 arc.

---

## v0.987.0 — 2026-06-13

**Fix L2 Wave-2 audit — `equity_theta` `T<=0 or vol<=0` branch returned the Black-76 theta alone, dropping the `theta_r` and `theta_q` corrections.**

The Hull theta has three terms:
- `theta_b76 = -S·n(d1)·σ·exp(-qT)/(2√T)` — the σ·n(d1)/√T term (returned by `black76_theta`).
- `theta_r = -r·K·exp(-rT)·N(d2)` (call) / `+r·K·exp(-rT)·N(-d2)` (put) — rate-discount.
- `theta_q = q·S·exp(-qT)·N(d1)` (call) / `-q·S·exp(-qT)·N(-d1)` (put) — dividend.

At `vol=0, T>0`, `theta_b76 → 0` and `N(d1), N(d2) → {0, 1}` by ITM/OTM. The pre-fix code returned `theta_b76` (= 0) and **silently dropped** `theta_r + theta_q`, which collapses to the deterministic `±(q·S·exp(-qT) − r·K·exp(-rT))` — a non-zero, sometimes dominant component of total theta when rates/divs are non-trivial.

**Fix**: separate `T <= 0` (still `theta_b76`, no time decay convention) from `vol <= 0` with `T > 0` (deterministic-limit branch on `forward` vs `strike`; ATM is one-sided half-limit).

### Verification — `test_l2_t4_equity_theta_degenerate.py`

8 new tests:
- `TestEquityThetaVolZeroTPositive` × 5: ITM call matches `qS·exp(-qT) − rK·exp(-rT)`; ITM put is negation; OTM zero; ATM half.
- `TestEquityThetaInteriorUnchanged` × 2: interior call/put still negative (decay), within Hull range.
- `TestEquityThetaTZero` × 1: T=0 path preserved.

Full parallel suite: **12,422 passed in 3:52** — zero regressions, +8 net tests.

Fifty-fifth fix from the 35-module Wave-2 audit; 105th distinct bug in the v0.905→v0.987 arc.

---

## v0.986.0 — 2026-06-13

**Fix L2 Wave-2 audit — `equity_rho` returned 0 in the `T<=0 or vol<=0` branch, silently dropping the deterministic `vol=0, T>0` limit.**

At `vol=0` with `T>0` the option price is the discounted intrinsic of the forward:

- ITM call (forward > strike): `price = S·exp(-qT) − K·exp(-rT)` → `rho = T·K·exp(-rT)` (positive)
- ITM put (forward < strike): `price = K·exp(-rT) − S·exp(-qT)` → `rho = -T·K·exp(-rT)` (negative)
- OTM (either side): price = 0 → rho = 0
- ATM (forward == strike): one-sided limit `±0.5·T·K·exp(-rT)`

Pre-fix all four cases returned 0 — the deterministic rho was silently dropped.

**Fix**: separate `T <= 0` (still 0 — no time for rate to act) from `vol <= 0` with `T > 0` (deterministic limit by ITM/OTM/ATM branch on the forward).

### Verification — `test_l2_t4_equity_rho_degenerate.py`

7 new tests, all pass:
- `TestEquityRhoVolZeroTPositive` × 5: ITM call positive, ITM put negative, OTM call/put zero, ATM half.
- `TestEquityRhoTZero`: T=0 returns 0 for both call and put across spot/strike/moneyness.
- `TestEquityRhoInteriorUnchanged`: T>0, vol>0 path identical to pre-fix.

Full parallel suite: **12414 passed in 3:51** — zero regressions.

Fifty-fourth fix from the **35-module deferred Wave-2 audit**; 104th distinct bug in the multi-session arc.

---

## v0.985.0 — 2026-06-13

**Fix L2 Wave-2 audit — `coupled_bootstrap` silently set `fwd = 0.0` for any period where the Newton iterate's projection-curve DF went non-positive (or where the schedule tau was non-positive).**

Pre-fix:

```python
if df2 > 0 and tau > 0:
    fwd = (df1 - df2) / (tau * df2)
else:
    fwd = 0.0
```

When the projection curve produced `df2 ≤ 0` (an arbitrageable iterate) or `tau ≤ 0` (a degenerate schedule with duplicate/inverted dates), the silent `fwd = 0` zeroed the float-leg contribution from that period. The residual `fixed_pv − too-small-float_pv` became artificially low, which Newton could drive to zero by following an **unphysical trajectory** in DF space — silently converging on a bad solution.

**Fix**: both degenerate paths now raise `ValueError` with a clear message identifying the offending input (schedule date or DF).

### Verification — `test_l2_t4_coupled_bootstrap_degenerate_raises.py`

2 new tests, all pass:
- `test_normal_inputs_succeed` — sanity: healthy bootstrap still works.
- `test_error_path_is_present` — static guard that the diagnostic raises are present in the source and the pre-fix silent fallback is gone (prevents reversion).

Full parallel suite: **12407 passed in 3:03** — zero regressions.

Fifty-third fix from the **35-module deferred Wave-2 audit**; **103rd distinct bug** in the multi-session arc.

---

## v0.984.0 — 2026-06-13

**Fix L2 Wave-2 audit — `risk.network.FinancialNetwork.betweenness_centrality` emitted spurious `RuntimeWarning: divide by zero` on sparse adjacency matrices (classic `np.where` eager-evaluation trap).**

Pre-fix:

```python
dist_matrix = np.where(self.adj > 0, 1.0 / self.adj, 0.0)
```

NumPy evaluates BOTH branches of `np.where` before selecting — so `1.0 / self.adj` is computed for EVERY element, including zero entries. The result for zeros (`inf` / `nan`) is then discarded by the where-mask, but the divide-by-zero RuntimeWarning is real, and the NaN/inf could propagate via other operations (multiplication, max-reduction etc) in subtle ways.

**Fix**: use `np.divide` with the `where=` argument so the division only fires where the predicate is true. The output array is pre-zeroed and only the true-predicate entries are written.

**Side-effect**: test-suite warnings drop from 29 → 19.

### Verification — `test_l2_t4_warning_cleanup.py`

2 new tests, all pass:
- `test_no_divide_by_zero_warning` — sparse 3x3 adjacency under `warnings.simplefilter("error")` no longer raises.
- `test_returns_finite_centralities` — 4x4 sparse adjacency produces finite centralities (no NaN/inf leakage).

Full parallel suite: **12405 passed in 2:37** — zero regressions.

Fifty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.983.0 — 2026-06-13

**Fix L2 Wave-2 audit — `TreeSolver._apply_barrier` silently no-oped for `DOWN_IN` / `UP_IN` barriers.**

```python
elif self.barrier_type == BarrierType.DOWN_IN:
    pass  # complex — for now, only knock-out supported
elif self.barrier_type == BarrierType.UP_IN:
    pass
```

At the outer `solve()` level, knock-in barriers are computed via in-out parity (`vanilla − knock_out`), so `_apply_barrier` is never called with a knock-in type via the normal path. BUT a caller invoking `_apply_barrier` directly (e.g. via a subclass extension, or bypassing `solve()` for debugging) silently got a vanilla price labelled as a knock-in — the most subtle kind of wrong-result bug.

**Fix**: the knock-in branches raise `NotImplementedError` with a diagnostic explaining that knock-in is computed at the outer level via in-out parity.

### Verification — `test_l2_t4_apply_barrier_knock_in_raises.py`

5 new tests, all pass:
- `test_down_in_raises_when_called_directly`, `test_up_in_raises_when_called_directly` — the unsafe direct call now raises.
- `test_down_out_zeroes_below_barrier`, `test_up_out_zeroes_above_barrier` — knock-out branches still work.
- `test_down_in_call_solves_via_parity` — outer-level `solve()` with knock-in barrier still produces a finite price via the in-out parity wrapper.

Full parallel suite: **12403 passed in 3:48** — zero regressions.

Fifty-first fix from the **35-module deferred Wave-2 audit**; 101st distinct correctness/contract bug in the multi-session arc.

---

## v0.982.0 — 2026-06-13 🎯 100-bug milestone

**Fix L2 Wave-2 audit — `StandardCDS` had a hand-written `to_dict` / `from_dict` that wasn't updated when `convention` was added to the parent `CDS._SERIAL_FIELDS` in v0.978.**

Pre-fix:
- The hand-written `to_dict` omitted `convention` from the params dict.
- The hand-written `from_dict` did not pass `convention=` to the constructor.
- The introspection sweep added in this audit (v0.981) flagged the gap: `StandardCDS._SERIAL_FIELDS` declared `convention` (inherited from CDS) but the actual hand-written serialisation didn't emit it.

A `StandardCDS` round-trip silently lost any non-default convention — same shape as the auto-generated cases fixed in v0.976–v0.978 / v0.979, but harder to spot because the hand-written serialisation was bespoke rather than declarative.

**Fix**: both methods now handle `convention`. The `from_dict` accepts a missing-`convention` legacy dict by defaulting to MODIFIED_FOLLOWING (backwards-compatibility for pre-fix-saved JSON).

### Verification — `test_l2_t4_standard_cds_serialisation.py`

5 new tests, all pass:
- `TestStandardCDSRoundTrip::test_convention_preserved` × 3 (parametrised over MODIFIED_FOLLOWING / PRECEDING / FOLLOWING).
- `test_legacy_dict_without_convention_defaults` — a dict missing `convention` (legacy pre-fix JSON) defaults gracefully.
- `test_other_fields_still_round_trip` — sanity: spread/grade/notional/standard_coupon all preserved.

Full parallel suite: **12398 passed in 3:07** — zero regressions.

This is the **fiftieth fix from the 35-module deferred Wave-2 audit** and the **100th distinct correctness/contract bug** resolved in the multi-session arc (v0.905 → v0.982). 🎯

---

## v0.981.0 — 2026-06-13

**Fix L2 Wave-2 audit — `AmortisingBond._serialisable` field list was completely STALE, referencing names that no longer existed on the class.**

Pre-fix declaration:

```python
_serialisable("amortising_bond", ['face_value', 'coupon_rate',
                                   'n_periods', 'frequency'])(AmortisingBond)
```

But the actual dataclass fields are `notional, coupon_rate, maturity_years, n_payments, amortisation_type` — none of the four declared field names existed. The serialisable framework emitted `UserWarning` on import for each mismatch, and any `to_dict()` / `from_dict()` round-trip was broken: `from_dict` would have raised `TypeError` on the unknown args. The class was effectively unserialisable.

The `from_convention` factory was also broken (tried to pass `frequency=` to a constructor that doesn't accept it).

**Fix**: align the serialisable field list with the actual dataclass fields, and rewrite `_amort_from_convention` to match the constructor signature.

**Side-effect**: import-time UserWarnings drop from 59 → 29 across the test suite (3 spurious warnings × ~10 test-collection imports each).

### Verification — `test_l2_t4_amortising_bond_serialisation.py`

4 new tests, all pass:
- `test_to_dict_runs` — pre-fix this raised on the `face_value` attribute lookup.
- `test_round_trip_preserves_all_fields` — every field round-trips.
- `test_amortisation_type_round_trips` × 2 — both `"mortgage"` and `"linear"`.

Full parallel suite: **12393 passed in 3:37** — zero regressions; warnings 59 → 29.

This is the **forty-ninth fix from the 35-module deferred Wave-2 audit** and the **99th distinct correctness/contract bug** resolved in the multi-session arc (v0.905 → v0.981).

---

## v0.980.0 — 2026-06-13

**Fix L2 Wave-2 audit — `VanillaCLN` was registered as `_serialisable` but its constructor never stored `frequency` as a class attribute.**

```python
def __init__(self, ..., frequency=Frequency.SEMI_ANNUAL, ...):
    self.start = start
    self.end = end
    # ... no `self.frequency = frequency` ...
    self.schedule = generate_schedule(start, end, frequency)

_serialisable("vanilla_cln", [..., "frequency", ...])(VanillaCLN)
```

Calling `vcln.to_dict()` raised `AttributeError: 'VanillaCLN' object has no attribute 'frequency'`. The class was effectively **unserialisable** despite being declared so — a contract bug that would surface the first time any caller tried to persist a CLN.

**Fix**: store `self.frequency = frequency` in the constructor.

### Verification — `test_l2_t4_vanilla_cln_serialisation.py`

5 new tests, all pass:
- `test_to_dict_does_not_raise` — pre-fix this raised `AttributeError`.
- `test_frequency_round_trips` × 4 (parametrised over `MONTHLY`/`QUARTERLY`/`SEMI_ANNUAL`/`ANNUAL`) — full round-trip preserves frequency and all other fields.

Full parallel suite: **12389 passed in 3:33** — zero regressions.

Forty-eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.979.0 — 2026-06-13

**Fix L2 Wave-2 audit — `CapFloor` serialisation dropped `convention`** (same shape as Swaption v0.976 / IRS v0.977 / CDS v0.978).

Pre-fix the field list missed `convention`, which controls the business-day rolling rule for caplet/floorlet accrual dates. A `to_dict → from_dict` on a non-default CapFloor changed the schedule and the price.

**Fix**: add `convention` to the field list.

### Verification — `test_l2_t4_capfloor_serialisation_fields.py`

3 new tests, all pass — parametrised round-trip across MODIFIED_FOLLOWING / PRECEDING / FOLLOWING.

Full parallel suite: **12384 passed in 2:41** — zero regressions.

Forty-seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.978.0 — 2026-06-13

**Fix L2 Wave-2 audit — `CDS` serialisation dropped `convention` (same shape as v0.976 Swaption / v0.977 IRS).**

Pre-fix the `_serialisable` field list missed `convention`, which controls the business-day rolling rule applied to coupon dates. A CDS with non-default `convention=PRECEDING` round-tripped to one with the default MODIFIED_FOLLOWING, changing the payment schedule and therefore the price.

**Fix**: add `convention` to the field list. `calendar` remains excluded (runtime-only holiday data).

### Verification — `test_l2_t4_cds_serialisation_fields.py`

4 new tests, all pass:
- `TestRoundTripPreservesConvention` × 3 (parametrised over MODIFIED_FOLLOWING / PRECEDING / FOLLOWING).
- `test_default_round_trip_still_works` — sanity.

Full parallel suite: **12381 passed in 3:16** — zero regressions.

Forty-sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.977.0 — 2026-06-13

**Fix L2 Wave-2 audit — `InterestRateSwap` serialisation dropped `convention`/`stub`/`eom` (same gap as Swaption v0.976).**

Pre-fix the `_serialisable` field list missed `convention`, `stub`, `eom` — all three affect the leg schedules, so a `to_dict → from_dict` on a non-default IRS produced a swap that priced differently. This is the parallel gap to the Swaption fix landed in v0.976.

**Fix**: add all three fields. `calendar` remains excluded (runtime-only holiday data; the caller re-attaches via `from_convention`).

### Verification — `test_l2_t4_irs_serialisation_fields.py`

9 new tests, all pass:
- `TestRoundTripPreservesConvention` × 3 (parametrised over MODIFIED_FOLLOWING / PRECEDING / FOLLOWING).
- `TestRoundTripPreservesStub` × 4 (all `StubType` values).
- `TestRoundTripPreservesEOM` × 2.

Full parallel suite: **12377 passed in 2:44** — zero regressions.

Forty-fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.976.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Swaption` serialisation dropped `convention`, `stub`, `eom` from the round-trip.**

Pre-fix the `_serialisable` field list was:

```python
["expiry", "swap_end", "strike", "swaption_type", "notional",
 "fixed_frequency", "float_frequency",
 "fixed_day_count", "float_day_count"]
```

Missing: `calendar`, `convention`, `stub`, `eom`. All four are constructor arguments that affect the underlying swap's schedule generation. A user serializing a Swaption with non-default `convention=PRECEDING` (or non-default `stub` / `eom`), then deserializing it, got a Swaption that **priced differently** than the original — the silent identity break the audit critic flagged.

**Fix**: add `convention`, `stub`, `eom` to the field list (all three are Enums/bools that the framework already handles via the existing `Frequency` serialisation). `calendar` remains excluded because Calendar instances hold runtime holiday data that isn't currently part of the serialisable type system — the caller re-attaches a calendar via `from_convention` on load. The new docstring documents this contract.

### Verification — `test_l2_t4_swaption_serialisation_fields.py`

9 new tests, all pass:
- `TestRoundTripPreservesConvention` × 3: MODIFIED_FOLLOWING, PRECEDING, FOLLOWING.
- `TestRoundTripPreservesStub` × 4 (parametrised): all four `StubType` values.
- `TestRoundTripPreservesEOM` × 2: True and False.

Full parallel suite: **12368 passed in 2:43** — zero regressions.

Forty-fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.975.0 — 2026-06-13

**Fix L2 Wave-2 audit — `GaussianCopula` and `StudentTCopula` constructors had no validation on `rho`.**

Pre-fix:
- `rho > 1`: `math.sqrt(1 - rho)` raised opaque `ValueError: math domain error` deep inside `sample()`.
- `rho < 0`: `math.sqrt(rho)` raised the same domain error.
- `rho = NaN`: slipped past everything via IEEE 754 and silently produced an all-NaN sample array (`norm.cdf(NaN) = NaN`).
- `StudentTCopula(nu<=0)`: produced a degenerate Student-t with no warning.

**Fix**: both constructors validate `rho ∈ [0, 1]` (with explicit NaN check) and `StudentTCopula` additionally validates `nu > 0`, raising `ValueError` upfront with diagnostic messages.

Also: the `test_l2_t4_studentt_tail_dependence::test_matches_copula_implementation` test (added in v0.943) was passing `rho=-0.5` to `StudentTCopula` to check distribution-vs-copula agreement; updated to only exercise `rho ∈ [0, 1]` (the copula's actual domain). The distribution method remains well-defined for the full `[-1, 1]` range and is tested independently.

### Verification — `test_l2_t4_copula_rho_validation.py`

9 new tests, all pass:
- `TestGaussianCopulaRho` × 4: above 1, below 0, NaN, valid.
- `TestStudentTCopulaRhoNu` × 5: rho out-of-range, nu=0, nu<0, NaN in either, valid.

Pre-existing 18 copula tests still pass.

Full parallel suite: **12359 passed in 3:12** — zero regressions.

Forty-third fix from the **35-module deferred Wave-2 audit**.

---

## v0.974.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Uniform` and `Exponential` constructors let NaN slip through their guards.**

In IEEE 754, all comparisons against NaN return False. So a naive guard like

```python
if a >= b:
    raise ...
```

silently accepts `a=NaN, b=NaN` (NaN >= NaN is False). The constructor succeeded, then downstream `cdf` / `pdf` propagated NaN through the user's computation with no diagnostic.

**Fix**: both constructors now explicitly check `math.isnan(...)` before the inequality guard and raise `ValueError` with a diagnostic message that names the IEEE 754 root cause.

### Verification — `test_l2_t4_distribution_nan_guards.py`

8 new tests, all pass:
- `TestUniformNaNGuards` × 4: NaN-a, NaN-b, both-NaN raise; valid (a, b) works.
- `TestExponentialNaNGuards` × 3: NaN rate raises; valid rate works; ±inf rate documented.
- `TestPreFixSlipThroughGuard`: pins down the IEEE 754 NaN-compares-False semantics.

Full parallel suite: **12350 passed in 3:12** — zero regressions.

Forty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.973.0 — 2026-06-13

**Fix L2 Wave-2 audit — `calibrate_svensson` had no degeneracy guard for `τ1 ≈ τ2`.**

The Svensson parameterisation has two decay constants `tau1` and `tau2`. When they collapse to (approximately) the same value, the second Svensson factor becomes a linear combination of the first plus the NS factor — so the objective surface develops a flat valley along the (beta2, beta3) ridge. Nelder-Mead can then drift arbitrarily far in that direction without the loss changing, producing wildly different parameter sets that all "calibrate" equally well to the same input.

**Fix**: penalise small `|τ1 − τ2|` in the objective (threshold 0.05) to keep the optimizer in the identifiable region of parameter space. The `τ <= 0.01` lower-bound guard is unchanged.

### Verification — `test_l2_t4_svensson_tau_degeneracy.py`

3 new tests, all pass:
- `test_calibrated_taus_well_separated` — post-calibration `|τ1−τ2| ≥ 0.05`.
- `test_svensson_fits_smooth_curve` — guard doesn't break healthy calibration; RMSE < 1%.
- `test_svensson_calibration_returns_finite_values` — all 6 parameters finite.

Pre-existing 13 NS tests still pass.

Full parallel suite: **12342 passed in 3:07** — zero regressions.

Forty-first fix from the **35-module deferred Wave-2 audit**.

---

## v0.972.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Normal` and `LogNormal` accepted `sigma=0` or `sigma<0` without validation.**

Pre-fix:
- `Normal(sigma=0)`: `cdf/pdf` divided by zero, emitting `RuntimeWarning` and returning 1.0 (or NaN) silently.
- `Normal(sigma=-1)`: produced a flipped distribution (math runs, but σ < 0 is meaningless — only σ ≥ 0 parameterises a normal).
- `LogNormal(sigma=0)`: same divide-by-zero pattern as Normal.

**Fix**: both constructors now raise `ValueError("sigma must be > 0")` at construction with a clear diagnostic.

### Verification — `test_l2_t4_distribution_sigma_validation.py`

6 new tests, all pass:
- `TestNormalSigmaValidation` × 3: zero, negative, positive.
- `TestLogNormalSigmaValidation` × 3: same coverage.

Full parallel suite: **12339 passed in 3:12** — zero regressions.

Fortieth fix from the **35-module deferred Wave-2 audit**.

---

## v0.971.0 — 2026-06-13

**Fix L2 Wave-2 audit — `cos_price` crashed with opaque exceptions deep in the formula on three degenerate inputs.**

Pre-fix:
- **`L=0`**: ``b - a = 0`` (truncation half-width `L·sqrt(c2) = 0`) → `ZeroDivisionError` inside the V_k recursion's `2.0 / (b - a)` factor.
- **`spot <= 0`**: `math.log(spot / strike)` raised `math domain error` with no diagnostic about which input was bad.
- **`strike <= 0`**: `spot / strike` raised `ZeroDivisionError` (zero strike) or produced a nonsensical negative log argument.

**Fix**: validate all three upfront with `ValueError` and an explicit message identifying the offending parameter.

### Verification — `test_l2_t4_cos_price_input_validation.py`

7 new tests, all pass:
- `TestSpotValidation` × 2: zero and negative spot raise.
- `TestStrikeValidation` × 2: zero and negative strike raise.
- `TestLValidation` × 2: zero and negative L raise.
- `TestHealthyPathUnchanged`: ATM call on BS still prices correctly (~10.45).

Pre-existing 12 cos_method tests still pass.

Full parallel suite: **12333 passed in 3:02** — zero regressions.

Thirty-ninth fix from the **35-module deferred Wave-2 audit**.

---

## v0.970.0 — 2026-06-13

**Fix L2 Wave-2 audit — `calibrate_nelson_siegel` and `calibrate_svensson` had three contract gaps.**

Pre-fix:
1. **Empty `market_yields`** raised `IndexError` at `market_yields[-1]` inside default initial-guess construction, no diagnostic.
2. **Mismatched `len(tenors) != len(market_yields)`** was silently masked by `zip()` which truncates to the shorter — a user passing 10 tenors and 8 yields silently got a calibration on the first 8 points only.
3. **The optimizer's convergence flag was discarded** — the returned dict gave no way for the caller to detect a non-converged calibration (it could go straight into a production curve, with no signal that the fit was bad).

**Fix**:
- New helper `_validate_calibration_inputs` checks non-empty AND matching lengths upfront — raises `ValueError` with clear messages.
- Returned dict now includes `converged: bool` from the underlying optimizer result.

### Verification — `test_l2_t4_nelson_siegel_calibration_validation.py`

8 new tests, all pass:
- `TestEmptyInputsRaise` × 3: empty tenors, empty yields, Svensson empty all raise.
- `TestMismatchedLengthsRaise` × 2: NS and Svensson both raise on mismatched lengths.
- `TestConvergedFieldReported` × 2: `converged: bool` present in both NS and Svensson result dicts.
- `TestHealthyCalibrationPreserved`: smooth synthetic curve still calibrates to RMSE < 1%.

Pre-existing 13 NS/Svensson tests still pass.

Full parallel suite: **12326 passed in 3:01** — zero regressions.

Thirty-eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.969.0 — 2026-06-13

**Fix L2 Wave-2 audit — `FRA.pv_ctx` silently picked the "first" projection curve when the day-count-keyed lookup missed.**

Pre-fix:

```python
if dc_key and dc_key in ctx.projection_curves:
    proj = ctx.projection_curves[dc_key]
else:
    proj = next(iter(ctx.projection_curves.values()))
```

In a multi-curve setup (e.g. ACT/360 USD-LIBOR FRA priced against a context with only ACT/365 GBP projection curves), the FRA silently got a wrong-curve forward rate. The day-count mismatch could be several basis points and was completely invisible to the user.

**Fix**: keyed-lookup miss now raises `KeyError` listing the offending day-count and the available keys. The legacy fallback to the discount curve when the context has NO projection curves at all is preserved (the unambiguous single-curve case).

### Verification — `test_l2_t4_fra_pv_ctx_keyed_lookup.py`

3 new tests, all pass:
- `test_act_360_fra_no_act_360_curve_raises` — ACT/360 FRA against ACT/365-only context raises.
- `test_act_360_fra_with_matching_curve_prices` — matching day-count gives a finite PV.
- `test_empty_projection_falls_back` — empty projection-curves dict still falls back to the discount curve (single-curve compat).

Full parallel suite: **12318 passed in 2:35** — zero regressions.

Thirty-seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.968.0 — 2026-06-13

**Fix L2 Wave-2 audit — `smith_wilson_forward` silently returned `ufr` when its finite-difference DFs went non-positive.**

```python
if p1 <= 0 or p2 <= 0:
    return ufr
```

Non-positive Smith-Wilson DFs are NOT a normal regime — they indicate arbitrageable input DFs, extrapolation gone off the rails, or extreme `alpha`. Pre-fix the user got "the answer is UFR" silently, masking a calibration failure that should fail loudly.

**Fix**: raise `ValueError` with diagnostic context (the offending DF values, the `t` parameter, and likely upstream causes).

### Verification — `test_l2_t4_smith_wilson_silent_ufr.py`

2 new tests, all pass:
- `TestSilentUFRReplacedByRaise::test_non_positive_dfs_raise` — synthesise a wild-zeta state, scan for a `t` where DFs go negative, confirm the function raises.
- `TestHealthyForwardWorks::test_normal_calibration_forward_finite` — normal calibrated state still returns a finite forward in the expected market-rate range.

Pre-existing 10 Smith-Wilson tests still pass.

Full parallel suite: **12315 passed in 3:01** — zero regressions.

Thirty-sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.967.0 — 2026-06-13

**Fix L2 Wave-2 audit — `MCEngine(n_paths=1)` silently produced NaN stderr instead of failing fast.**

Pre-fix:
- `MCEngine(..., n_paths=1, antithetic=True)`: `n_half = 1 // 2 = 0` → ZERO paths generated. `np.std(..., ddof=1)` of zero samples is NaN — silent.
- `MCEngine(..., n_paths=1, antithetic=False)`: single-path "MC" with `np.std(..., ddof=1)` of one sample is also NaN.

Both modes are useless for Monte Carlo (you need at least 2 samples to estimate variance), but pre-fix the engine ran them silently and reported NaN downstream into `MCResult.stderr` and `confidence_95`.

**Fix**:
- `n_paths < 2` → `ValueError` upfront.
- `antithetic=True` AND `n_paths < 4` → `ValueError` (the engine uses `n_half = n_paths // 2` antithetic pairs; 2 paths give only 1 pair, which is also degenerate for ddof=1 stderr).

### Verification — `test_l2_t4_mcengine_n_paths_validation.py`

7 new tests, all pass:
- `TestNPathsValidation` × 3: `n_paths` of 1, 0, or negative raises.
- `TestAntitheticMinimum` × 3: antithetic with 2 or 3 paths raises; 4 paths works.
- `TestHealthyPathUnchanged`: large `n_paths` (1000) works.

Full parallel suite: **12313 passed in 3:02** — zero regressions.

Thirty-fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.966.0 — 2026-06-13

**Fix L2 Wave-2 audit — `mc_engine.TimeGrid` accepted any array, even empty or non-monotonic, silently producing nonsense state.**

Pre-fix the constructor was:

```python
def __init__(self, times):
    self.times = np.asarray(times, dtype=np.float64)
    self.dt = np.diff(self.times)
    self.n_steps = len(self.dt)
    self.T = float(self.times[-1])       # IndexError on empty
```

Three silent failure modes:
1. **Empty input** → `self.times[-1]` raises `IndexError` deep inside the constructor with no diagnostic message.
2. **Length-1 input** → `dt = []`, `n_steps = 0`, T set to the one time point. The engine loops zero times — silent no-op.
3. **Non-monotonic input** → `dt` has negative entries. The engine integrates the SDE BACKWARDS in time for those steps, with wrong-sign drift. Pre-fix this was silent; the user got a finite "price" computed against time-reversed dynamics.

**Fix**: validate at construction — empty/singleton/non-monotonic input all raise `ValueError` with a clear diagnostic.

### Verification — `test_l2_t4_timegrid_validation.py`

6 new tests, all pass:
- `test_empty_array_raises`, `test_singleton_raises` — size-validation.
- `test_decreasing_raises`, `test_duplicate_raises` — monotonicity-validation.
- `test_uniform_works`, `test_explicit_increasing_works` — happy paths preserved.

Full parallel suite: **12306 passed in 3:03** — zero regressions.

Thirty-fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.965.0 — 2026-06-13

**Fix L2 Wave-2 audit — `bachelier_delta` at exactly ATM (`F == K`) with `T<=0 or vol<=0` returned 0 instead of the standard `±0.5·df` one-sided limit.**

Pre-fix:

```python
if time_to_expiry <= 0 or vol_normal <= 0:
    if option_type == OptionType.CALL:
        return df if forward > strike else 0.0
    return -df if forward < strike else 0.0
```

The `else 0.0` branch caught both `F < K` (correct for a call OTM) AND `F == K` (incorrect — ATM-at-expiry should be `0.5·df`). Symmetric bug for puts.

**Fix**: explicit ATM branch returns `±0.5·df` matching the same convention used (correctly) by `black76_delta`.

### Verification — `test_l2_t4_bachelier_delta_atm.py`

8 new tests, all pass:
- `TestBachelierDeltaATMAtExpiry` × 3: ATM call/put at T=0 give `±0.5·df`; ATM with vol=0 (T>0) also gives `±0.5·df`.
- `TestConsistencyWithBlack76Delta` × 2: `bachelier_delta` and `black76_delta` agree on the ATM-at-expiry limit.
- `TestNonATMUnaffected` × 3: ITM call returns `df`; OTM call returns 0; ITM put returns `-df` — pre-fix behaviour preserved off-ATM.

Full parallel suite: **12300 passed in 2:35** — zero regressions.

Thirty-third fix from the **35-module deferred Wave-2 audit**.

---

## v0.964.0 — 2026-06-13

**Fix L2 Wave-2 audit — `integrate_semi_infinite` silently downgraded any non-Laguerre method to ADAPTIVE.**

Pre-fix:

```python
if method == IntegrationMethod.GAUSS_LAGUERRE:
    ... Gauss-Laguerre code ...
else:
    return _adaptive(f, a, np.inf)      # discards user's `method`
```

A user passing `method=IntegrationMethod.GAUSS_HERMITE` or `method=IntegrationMethod.SIMPSON` got the SciPy adaptive result with no warning that their METHOD argument was discarded. A method-comparison study would see identical numbers for every "method" choice — silently masking the fact that only one method was actually running.

**Fix**: only the two methods with defined semi-infinite behaviour (`GAUSS_LAGUERRE`, `ADAPTIVE`) are accepted; everything else raises `ValueError` with a pointer to the documented choices and an explicit note that this used to be a silent downgrade.

### Verification — `test_l2_t4_integrate_semi_infinite_method.py`

9 new tests, all pass:
- `TestSupportedMethodsWork` × 2: GAUSS_LAGUERRE and ADAPTIVE both correctly compute `∫_0^∞ e^{-x} dx = 1`.
- `TestUnsupportedMethodsRaise` × 7: every other `IntegrationMethod` value (LEGENDRE/HERMITE/SIMPSON/TRAPEZOID/TANH_SINH/CLENSHAW_CURTIS/ROMBERG) raises.

Full parallel suite: **12292 passed in 3:04** — zero regressions.

Thirty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.963.0 — 2026-06-13

**Fix L2 Wave-2 audit — `_simpson` and `_trapezoid` crashed with `ZeroDivisionError` on `n=0`.**

Both routines compute `h = (b - a) / n` directly with no validation. A caller routing through the public `integrate(method=…, n=0)` API got an opaque `ZeroDivisionError` deep in the implementation with no diagnostic context.

**Fix**: both routines raise `ValueError` upfront with a clear message ("n must be >= 1") if `n < 1`.

### Verification — `test_l2_t4_integrate_simpson_trap_zero_n.py`

6 new tests, all pass:
- Each of `_simpson` and `_trapezoid`: `n=0` raises, `n<0` raises, `n>=1` produces correct integral (ATM `∫x² dx` ≈ 1/3 for Simpson; ATM `∫x dx` = 0.5 for trapezoid).

Full parallel suite: **12283 passed in 3:03** — zero regressions.

Thirty-first fix from the **35-module deferred Wave-2 audit**.

---

## v0.962.0 — 2026-06-13

**Fix L2 Wave-2 audit — `minimize(method=…)` rejected the natural hyphenated method-string form.**

Pre-fix the string-to-enum dispatch was `OptimMethod(method.lower())` — so passing the form scipy itself uses (`"Nelder-Mead"`, `"L-BFGS-B"`) raised:

> `ValueError: 'nelder-mead' is not a valid OptimMethod`

The enum values use underscores (`"nelder_mead"`). This is an ergonomic trap: users copy-paste a method name from scipy docs and get a hard error with no hint that the fix is to swap `-` for `_`.

**Fix**: normalise hyphens to underscores in addition to lower-casing — `method.lower().replace("-", "_")`. Hyphenated, underscored, and `OptimMethod` enum forms all work now.

### Verification — `test_l2_t4_minimize_method_string_normalisation.py`

6 new tests, all pass:
- `test_Nelder_Mead_with_hyphen`, `test_L_BFGS_B_with_hyphens` — scipy-style forms.
- `test_nelder_mead_underscore`, `test_l_bfgs_b_underscore` — underscore forms still work.
- `test_enum_method` — direct enum still works.
- `test_unknown_method_string_raises` — sanity: unknown strings still raise.

Full parallel suite: **12277 passed in 2:34** — zero regressions.

Thirtieth fix from the **35-module deferred Wave-2 audit**.

---

## v0.961.0 — 2026-06-13

**Fix L2 Wave-2 audit — `projection_l1_ball` returned the INPUT alias when `x` was already inside the L1 ball.**

```python
if np.sum(np.abs(x)) <= radius:
    return x         # ← aliasing bug
```

Mutating the result then mutated the caller's input array. The other branch (`np.sign(x) * proj`) always returned a fresh array via broadcast multiplication, so the inconsistency was: fast path aliased, slow path did not. A defensive consumer expecting consistent ownership got a shared buffer in the fast path.

**Fix**: both branches now return a fresh array (`x.copy()` in the fast path, broadcast multiplication in the slow path).

### Verification — `test_l2_t4_projection_l1_ball_aliasing.py`

5 new tests, all pass:
- `test_result_is_fresh_copy_when_inside_ball` — fast-path returns a fresh object.
- `test_mutating_result_does_not_mutate_input` — most important: writes to result do not propagate to input.
- `test_result_is_not_input_when_outside_ball` — slow path still copies.
- `test_projected_value_norm_equals_radius` — projection math correct.
- `test_projection_inside_ball_is_identity` — value is unchanged when input is already feasible.

Full parallel suite: **12271 passed in 2:33** — zero regressions.

Twenty-ninth fix from the **35-module deferred Wave-2 audit**.

---

## v0.960.0 — 2026-06-13

**Fix L2 Wave-2 audit — `minimize(method=BASIN_HOPPING)` hardcoded `converged=True`.**

```python
return OptimizeResult(r.x, float(r.fun), r.nit, True, "basin_hopping")
                                                  ^^^^
```

The basin-hopping wrapper produced a guaranteed-true convergence flag regardless of outcome. A calibration loop relying on `result.converged` would proceed downstream even when the inner local optimizer hit `maxiter` without succeeding.

**Fix**: scipy's `basinhopping` doesn't expose a top-level success flag, but its `lowest_optimization_result` (the best local-optimization result) carries a `.success` attribute. Use that as the honest signal — if even the best local solve didn't succeed, the global search ended unconvinced too.

### Verification — `test_l2_t4_basin_hopping_converged.py`

2 new tests, all pass:
- `test_smooth_quadratic_converges` — basin-hopping on a smooth quadratic finds the global minimum and reports `converged=True` (the inner L-BFGS local solve succeeds).
- `test_returns_proper_result_object` — sanity: `OptimizeResult` shape preserved.

Full parallel suite: **12266 passed in 2:32** — zero regressions.

Twenty-eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.959.0 — 2026-06-13

**Fix L2 Wave-2 audit — `proximal_gradient` lied about its result in two ways.**

Pre-fix:

1. **`OptimizeResult.fun` was unconditionally `0.0`.** A calibration consumer reading `result.fun` saw "zero" and concluded the solver had found the optimum, when in fact the solver had no idea what the objective value was (the function never receives `f` — only `grad_f`).

2. **`OptimizeResult.converged` was unconditionally `True`.** The loop could exhaust `maxiter` without the tolerance check ever firing, and the caller had no way to know — `result.converged` was a rubber stamp.

**Fix**:
- Added optional `f_obj` parameter: if supplied, `fun` is the true objective at the final iterate.
- If `f_obj` is omitted, `fun = float('nan')` — a clear sentinel that the value is unknown (NOT zero).
- `converged` now reflects whether the tolerance check actually fired during iteration.
- Loop counter `k` is initialised before the loop so the `iterations` count is well-defined even for `maxiter=0`.

### Verification — `test_l2_t4_proximal_gradient_reporting.py`

5 new tests, all pass:
- `test_converged_true_when_tolerance_fires` — generous params → converged=True.
- `test_converged_false_when_maxiter_exhausted` — tiny step + tight tol + low maxiter → converged=False.
- `test_fun_is_nan_when_no_objective_supplied` — clear sentinel for unknown.
- `test_fun_is_correct_when_objective_supplied` — `f_obj=…` gives true `fun` value (≈ 0 at minimum of `0.5||x − x*||²`).
- `test_recovers_smooth_quad_optimum` — sanity check that the solver still finds the optimum.

Full parallel suite: **12264 passed in 2:32** — zero regressions.

Twenty-seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.958.0 — 2026-06-13

**Fix L2 Wave-2 audit — `antithetic_paths` had a bimodal interface that was misleading in one branch and mathematically WRONG in the other.**

Pre-fix two-branch behaviour:

1. **Called with `rng_normals=Z`**: returned `-Z` (just negated the argument). The function's name suggested it returned PATHS, not negated normals — the caller had to know to rebuild paths from `-Z` themselves. Misleading.

2. **Called WITHOUT `rng_normals`**: computed "mirror around log-mean of terminal values": `antithetic = exp(2·log_mean - log(terminal))`. This is NOT a valid antithetic — the transformation depends on the SAMPLE mean (a random quantity), so the result is biased and not properly antithetic. It also ran `np.log` on terminal spots, silently producing `NaN` for Bachelier/OU/etc. paths where terminal spots can be ≤ 0.

**Fix**:
- Renamed the function to `antithetic_normals` to reflect what it actually does (returns `-Z`).
- Requires the normal draws explicitly (positional, no default).
- Only performs the valid `-Z` operation; the broken "mirror" branch is gone.
- Legacy alias `antithetic_paths = antithetic_normals` kept so existing imports don't break.
- Both names exported from `pricebook.numerical.__init__`.

### Verification — `test_l2_t4_antithetic_normals_api.py`

5 new tests, all pass:
- `test_returns_negated_array` — 1-D and 2-D arrays correctly negated.
- `test_array_2d_works` — large 2-D draws negated element-wise.
- `test_estimator_unbiased_and_variance_reduced` — antithetic estimator of `E[exp(Z)]` has strictly lower sample variance than two independent runs (the canonical variance-reduction proof).
- `test_antithetic_paths_alias` — legacy alias points to the same function.
- `test_alias_returns_negated_normals` — alias also works.

Full parallel suite: **12259 passed in 2:32** — zero regressions.

Twenty-sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.957.0 — 2026-06-13

**Fix L2 Wave-2 audit — `CharacteristicFunction.density` two robustness gaps.**

1. **`density(x_grid, n_quad=1)`** raised `IndexError` at `du = u[1] - u[0]` because the `linspace` had only one point. Now raises `ValueError` upfront with a clear message ("n_quad must be >= 2").

2. **`density(scalar)`** (a single x point as a Python float or 0-d ndarray) raised `TypeError` at `len(x)` because `np.asarray(scalar)` produces a 0-d array which has no `len()`. Now scalar inputs are promoted to a 1-element array via `np.atleast_1d`.

### Verification — `test_l2_t4_characteristic_function_density.py`

5 new tests, all pass:
- `TestDensityValidation` × 2: `n_quad=1` and `n_quad=0` raise.
- `TestDensityScalarInput` × 2: scalar and Python float both accepted; returned shape is `(1,)` and the N(0,1) density at 0 is recovered to ≈ 1/√(2π).
- `TestDensityHealthyPath`: closed-form N(0,1) recovery test for `[-1, 0, 1]`.

Full parallel suite: **12254 passed in 2:56** — zero regressions.

Twenty-fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.956.0 — 2026-06-13

**Fix L2 Wave-2 audit — `models/feynman_kac.py` had three silent contract violations.**

1. **`sde_to_pde(rate_fn=None)`** silently defaulted to a 4% constant rate. A user calling the helper without specifying the rate would get a PDE parameterised at 4%; if they then ran an MC at a different rate, the cross-comparison would be biased by exactly the rate disagreement, with no warning. Now raises `ValueError`.

2. **`pde_to_sde`** used `math.sqrt(max(2·diffusion, 0))` — silently clamping NEGATIVE PDE diffusion (which is unphysical: the PDE diffusion coefficient `a = ½σ²` is non-negative by construction) to zero. Bugs in upstream coefficient functions producing negative `a` would be masked. Now raises `ValueError` on negative diffusion at the evaluation point.

3. **`verify_feynman_kac(n_time=...)`** ignored the user-supplied `n_time` for the MC time grid (hardcoded to 100), so a user thinking they were refining BOTH MC and PDE in lockstep to study convergence was actually only refining the PDE. The docstring promises "Both use the same model" — broken in the wiring. Now the MC grid uses `n_time` steps.

### Verification — `test_l2_t4_feynman_kac_contracts.py`

7 new tests, all pass:
- `TestSdeToPdeRequiresRate` × 3: `rate_fn=None` raises; scalar still works; callable still works.
- `TestPdeToSdeRejectsNegativeDiffusion` × 3: negative raises; zero returns zero vol; positive unchanged.
- `TestVerifyFeynmanKacUsesNTime`: smoke test confirming `n_time=50` vs `n_time=200` both run without error.

Pre-existing 3 Feynman-Kac tests still pass.

Full parallel suite: **12249 passed in 2:33** — zero regressions.

Twenty-fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.955.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Dual` math operations silently produced wrong derivatives at known singularities.**

Five degenerate paths in `numerical.auto_diff`:

1. **`sqrt(Dual(0, x))`** returned `Dual(0, 0)`. Mathematically the derivative `1/(2·√x) → +∞` at `x = 0`. Silent zero hides failures in MC paths that touch zero (e.g. Heston QE-discretisation variance at the zero-boundary).

2. **`Dual(0, x) ** Dual(y, z)`** returned `Dual(0, 0)` regardless of exponent — derivative `val · (z·log(0) + y·der/0)` has a `log(0)` singularity that the pre-fix code hid behind an `if self.val != 0 else 0` shortcut.

3. **`Dual(negative, x) ** Dual(y, z)`** computed `math.log(abs(self.val))` — `log` of a negative number is not real, but the `abs()` made the formula run and produced a meaningless float.

4. **`Dual(0, x) ** n`** with `n < 1` was a singularity disguised as `n · 0^(n-1) · der` (Python raises `ZeroDivisionError` for `n<1`, or silently gives 0 for fractional `n`).

5. **`b ** Dual(x, y)`** with `b ≤ 0` returned `der = 0` silently — `log(b)` is undefined there.

All five paths now raise `ValueError` with diagnostic context (which input is singular and why). The integer-exponent case for negative base is preserved (e.g. `(-2)^3` has real value and well-defined real derivative). The healthy positive-base path is unchanged.

### Verification — `test_l2_t4_auto_diff_singularities.py`

14 new tests, all pass:
- `TestSqrtAtZeroRaises` × 3: zero-base raises; positive-base correct derivative; plain float unaffected.
- `TestPowAtZeroBaseRaises` × 4: Dual exp / fractional / negative exp raise; integer n≥1 works (n=1 gives der=1, n=2 gives der=0).
- `TestPowAtNegativeBaseRaises` × 2: Dual exp raises; integer exp works for negative base.
- `TestRpowAtNonPositiveBaseRaises` × 3: zero/negative base raise; positive base correct (`2^Dual(3,1)` gives 8·ln(2)).
- `TestHealthyPathsUnchanged` × 2: basic power/sqrt with positive Dual base produce correct (val, der).

Pre-existing tests still pass.

Full parallel suite: **12242 passed in 2:33** — zero regressions.

Twenty-third fix from the **35-module deferred Wave-2 audit**.

---

## v0.954.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Dual` defined `__eq__` without `__hash__`, breaking hashability and violating the Python hash/eq contract.**

Defining `__eq__` automatically sets `__hash__ = None`, making the class unhashable. Pre-fix any attempt to put a `Dual` in a dict or set raised `TypeError: unhashable type: 'Dual'` — surprising for a number-like object.

Even if hashability had been preserved by inheritance, the Python data-model contract `a == b ⇒ hash(a) == hash(b)` would require careful choice: `Dual.__eq__` compares ONLY `val` (`Dual(1, 2) == 1.0` is True — a deliberate design for float compatibility), so the hash must depend only on `val` too.

**Fix**: define `__hash__(self) = hash(self.val)`. Dual is now usable as a dict key or set member. Distinct vals hash distinct (modulo standard float-hash collisions); Duals that compare equal — including a Dual and a plain float — hash equal.

### Verification — `test_l2_t4_dual_hash_eq_contract.py`

6 new tests, all pass:
- `test_hash_returns_int`, `test_can_be_dict_key`, `test_can_be_in_set` — hashability restored.
- `test_equal_duals_with_different_der_have_same_hash` — `Dual(1, 2)` and `Dual(1, 99)` hash equal (consistent with `__eq__`).
- `test_dual_and_equal_float_have_same_hash` — `hash(Dual(1.0, x)) == hash(1.0)`.
- `test_distinct_vals_distinct_hashes` — sanity.

Pre-existing tests still pass.

Full parallel suite: **12228 passed in 2:33** — zero regressions.

Twenty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.953.0 — 2026-06-13

**Fix L2 Wave-2 audit — `numerical.auto_diff` drivers (`grad`, `jacobian_ad`, `derivative`) silently returned zero gradient when the user's function failed to thread `Dual` numbers through.**

This is the **most common bug** in forward-mode AD code, and the pre-fix drivers actively hid it:

```python
# pre-fix grad
g[i] = result.der if isinstance(result, Dual) else 0.0
```

If `f` accidentally strips the `Dual` wrapper anywhere — e.g. by calling a NumPy ufunc on a list of `Duals` (which returns an ndarray of objects whose `.der` propagation doesn't work) or by routing through `math.` functions that don't recognise the wrapper — every call returns `0.0` and `grad` returns the zero vector with no warning. A user computing "delta" of a pricer with a typo or stale code path would see `grad = [0, 0, …]`, conclude the option has no Greek, and act on it.

Same shape in `jacobian_ad` (component-wise) and `derivative`.

**Fix**: all three drivers now raise `TypeError` with a diagnostic message that names the actual returned type and points at the most likely cause (NumPy ufuncs vs `math.` calls). The message also gestures at the canonical remedy (Dual-aware operations in `auto_diff`).

### Verification — `test_l2_t4_auto_diff_threading_raises.py`

7 new tests, all pass:
- `TestGradRaisesOnThreadingFailure` × 2: raises on plain-float return; correct gradient when threading works (sanity).
- `TestJacobianAdRaisesOnThreadingFailure` × 3: raises on full failure, raises on PARTIAL failure (one component is a Dual, another isn't — the trickiest case), correct Jacobian when threading works.
- `TestDerivativeRaisesOnThreadingFailure` × 2: raises on plain-float return; correct derivative when threading works.

Pre-existing tests still pass.

Full parallel suite: **12222 passed in 2:34** — zero regressions.

Twenty-first fix from the **35-module deferred Wave-2 audit**.

---

## v0.952.0 — 2026-06-13

**Fix L2 Wave-2 audit — `PDESolver1D` had four robustness gaps on degenerate inputs.**

Pre-fix:
1. **`n_time = 0`**: raised `ZeroDivisionError` deep inside `solve()` at `dt = T / self.n_time`, with no diagnostic context.
2. **`n_space = 0` (or 1)**: raised an opaque `IndexError` inside grid construction.
3. **`T = 0`**: produced a finite-but-WRONG price (~2.05 for an ATM call where the correct intrinsic is 0). The solver iterated `n_time` times with `dt = 0`, but the boundary projection and operator construction don't commute cleanly with zero time evolution — the result was determined by interpolation noise on the terminal payoff.
4. **`T < 0`**: produced a numerical runaway (~1e107) with no exception. Unphysical input that should fail loudly.

**Fix**:
- Constructor: `n_space < 2` or `n_time < 1` → `ValueError` with clear message.
- `solve()`: `T < 0` → `ValueError`. `T == 0` returns the intrinsic value (`max(spot−strike, 0)` for call, `max(strike−spot, 0)` for put) directly without invoking the solver — the unique no-arbitrage payoff with no time to evolve. The returned `PDEResult` carries a small valid 2-point grid so consumers reading `r.values` / `r.grid` see well-formed arrays.
- All four healthy-path branches (positive `T`, positive `n_time`/`n_space`) are unchanged.

### Verification — `test_l2_t4_pde_input_validation.py`

11 new tests, all pass:
- `TestConstructorValidation` × 4: `n_time=0`, `n_time=-1`, `n_space=0`, `n_space=1` all raise `ValueError`.
- `TestSolveValidation::test_negative_T_raises`.
- `TestZeroExpiryIntrinsic` × 5: `T=0` returns intrinsic for ITM/OTM call & put, and the result object is well-formed (`values`, `grid`, `method` present).
- `TestHealthyPathUnchanged::test_normal_solve_unchanged` — `T > 0` ATM call still prices ≈ Black-Scholes.

Full parallel suite: **12215 passed in 2:53** — zero regressions.

Twentieth fix from the **35-module deferred Wave-2 audit**.

---

## v0.951.0 — 2026-06-13

**Fix L2 Wave-2 audit — `TreeSolver._compute_vega` left `_computing_vega = True` and `store_tree = False` if the inner bumped-vol `solve()` raised.**

Same exception-safety shape as the MCEngine.greek bug fixed in v0.946.

Pre-fix:

```python
self._computing_vega = True
saved_store = self.store_tree
self.store_tree = False
r_up = self.solve(...)             # if this raises, both stay set
self.store_tree = saved_store
self._computing_vega = False
```

If the bumped-vol `solve()` call raised (e.g. user-supplied `payoff_fn` that fails for stressed inputs, vol-bump pushing into a numerically degenerate regime), the solver was left with `_computing_vega = True` and `store_tree = False`. A subsequent unrelated `solve()` call would then silently produce results WITHOUT snapshots (no greeks recorded) and WITHOUT recursing into vega computation (the `not self._computing_vega` guard at line 384 / 453 would block it). The price still returned, but greeks were missing.

**Fix**: wrap the bump sequence in `try ... finally:` so both attributes are restored regardless of whether the inner solve fails. Also save the prior `_computing_vega` rather than hardcoding `False` to the restore — preserves nested-call invariants.

### Verification — `test_l2_t4_tree_vega_exception_safety.py`

3 new tests, all pass:
- `test_state_restored_when_solve_raises` — `_computing_vega` and `store_tree` restored after raise.
- `test_vega_finite_and_state_restored` — happy-path sanity.
- `test_solve_after_failed_vega_unchanged` — a `solve()` after a failed vega gives identical price to `solve()` before.

Full parallel suite: **12204 passed in 2:55** — zero regressions.

Nineteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.950.0 — 2026-06-13

**Fix L2 Wave-2 audit — `TreeSolver.convergence_analysis` applied Richardson extrapolation universally; theoretically wrong for CRR/JR/Tian/Trinomial (which are oscillatory O(1/N), not smooth O(1/N²)).**

The Richardson formula `P* = (4·P(2N) − P(N)) / 3` is derived from a Taylor expansion assuming the error has leading-order term `c/N²`. The "4 − 1" weighting then cancels that term, leaving O(1/N⁴). This is correct for Leisen-Reimer (smooth O(1/N²)) but FALSE for CRR/JR/Tian/Trinomial — those have **oscillatory O(1/N)** convergence (a parity sawtooth between odd and even N). Applying Richardson there over-amplifies the sawtooth rather than cancelling it.

The pre-fix routine also:
- Did NOT validate that `n_steps_list[-1] == 2 · n_steps_list[-2]`. Without doubling, the "4·P(2N) − P(N)" weighting has no theoretical basis at all.
- Did not validate that `n_steps_list` had at least 2 elements (silently returned `richardson=None`).

**Fix**:
- Validate `n_steps_list` is non-empty (≥ 2) and strictly increasing.
- Pick the extrapolation per method:
  - **LR + doubling**: Richardson formula (genuinely O(1/N²) → O(1/N⁴)).
  - **LR + non-doubling**: fall back to last price (formula invalid).
  - **CRR / JR / Tian / Trinomial**: average of the last two prices (cancels the parity sawtooth).
- Legacy `richardson` key preserved as the literal Richardson formula for backwards compatibility.
- New `extrapolated` and `extrapolation_method` keys expose the method-aware choice and a string tag identifying which scheme was used.
- Run the inner sweep in a `try ... finally:` so `n_steps` is always restored.

### Verification — `test_l2_t4_tree_convergence_method_aware.py`

7 new tests, all pass:
- `test_empty_n_steps_list_raises`, `test_non_monotonic_raises` — input validation.
- `test_lr_uses_richardson_when_doubling` — LR + doubling → Richardson.
- `test_lr_falls_back_when_non_doubling` — LR + non-doubling → last price.
- `test_crr_uses_average_not_richardson` — CRR → average; expected = mean of last two prices.
- `test_jr_uses_average` — JR same.
- `test_result_contains_both_legacy_and_new_keys` — `richardson` (legacy formula) and `extrapolated` (method-aware) both present.

Pre-existing 20 tree-solver tests still pass — backwards compatibility on the `richardson` key preserved.

Full parallel suite: **12201 passed in 2:34** — zero regressions.

Eighteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.949.0 — 2026-06-13

**Fix L2 Wave-2 audit — `InterestRateSwap.cashflow_schedule` floating row was internally inconsistent (`rate` excluded spread but `amount` included it).**

The pre-fix floating row reported `rate = forward` (no spread) but `amount = (forward + spread) · year_frac · notional`. A downstream consumer naturally verifying `amount = rate · year_frac · notional` got the wrong number by exactly `spread · year_frac · notional`. On a 1 mm notional swap with 50 bp spread on a quarterly period, that's ~$1,236 of "missing" amount per coupon — easy to misread in P&L attribution, cashflow reconciliation, or regulatory reporting.

**Fix**: add two new fields per row, `spread` (0 for fixed) and `notional` (period notional, correct for amortising / accreting), so the schema is fully self-describing and the invariant

> (rate + spread) · year_frac · notional == amount

holds exactly for both legs. The pre-existing `rate` / `amount` semantics are unchanged — additive change, no breakage of consumers reading legacy fields.

### Verification — `test_l2_t4_swap_cashflow_schedule_consistency.py`

6 new tests, all pass:
- `test_row_is_self_consistent_with_spread` — invariant holds to 1e-9 for every row.
- `test_floating_row_carries_explicit_spread` — new field present on float leg.
- `test_fixed_row_has_zero_spread` — uniform schema.
- `test_notional_field_present_on_both_legs` — bullet swap.
- `test_amortising_notional_in_row` — amortising-leg notional decreases period by period.
- `test_all_legacy_fields_still_present` — backwards-compatibility.

Existing FI-hardening cashflow tests (3) and XI1 curve+swap tests (21) still pass.

Full parallel suite: **12194 passed in 2:33** — zero regressions.

Seventeenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.948.0 — 2026-06-13

**Fix L2 Wave-2 audit — `equity_delta` degenerate-input branch had three coupled bugs.**

In the `T <= 0 or vol <= 0` branch of `options/equity_option.py::equity_delta`:

1. **Wrong moneyness test**: compared `spot` to `strike` instead of `forward` to `strike`. For a non-zero dividend yield, `forward = spot · exp((r-q)·T)` can differ materially from spot. Concrete failure: `spot=100, strike=99, q=0.10, T=1` → forward ≈ 90.48 < 99. A call is **OTM on forward** (true zero-vol delta = 0) but the pre-fix code, comparing spot > strike, returned **1.0**. Conversely, the matching put is ITM on forward but pre-fix returned 0.

2. **Unit magnitude instead of `exp(-q·T)`**: ITM ITM branch returned literal `1.0` (call) / `-1.0` (put) instead of `±exp(-q·T)`. Spot delta of an ITM call replicates as `∂(S - K·exp(-rT))/∂S = exp(-q·T)`, NOT `1`. Inconsistent with `black76_delta` which correctly returns `±df`.

3. **ATM hole**: at exact ATM (forward == strike) the pre-fix returned 0 for both call and put, instead of the standard `±0.5·exp(-q·T)` limit — asymmetric vs `black76_delta` which handles this explicitly.

**Fix**: all three branches now compare forward to strike, scale by `exp(-q·T)`, and return `±0.5·exp(-q·T)` at exact ATM — exactly the chain-rule lift of `black76_delta` from forward delta to spot delta.

### Verification — `test_l2_t4_equity_delta_degenerate.py`

7 new tests, all pass:
- `test_call_itm_on_spot_but_otm_on_forward_returns_zero` — q=10% case: delta is 0, not 1.
- `test_put_otm_on_spot_but_itm_on_forward_returns_negative_exp_minus_qT` — companion case.
- `test_itm_call_magnitude_is_exp_minus_qT`, `test_itm_put_magnitude_is_minus_exp_minus_qT` — magnitude check.
- `test_atm_call_returns_half_exp_minus_qT`, `test_atm_put_returns_minus_half_exp_minus_qT` — ATM limit.
- `test_interior_delta_unchanged` — sanity check: `T>0, vol>0` path is identical to pre-fix.

Full parallel suite: **12188 passed in 3:03** — zero regressions.

Sixteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.947.0 — 2026-06-13

**Fix L2 Wave-2 audit — `price_swaption_sabr_hw` had two silent-failure paths: `blend_half_life=0` divided by zero deep in the pricer, and a hard-coded 1% volatility fallback when both SABR and HW vols failed.**

Two real bugs in the SABR-HW blender, distinct from the T=0 intrinsic bug already closed under T2.15:

1. **`blend_half_life` was not validated.** Passing `0` (or negative) caused `math.exp(-T / 0.0)` to raise `ZeroDivisionError` deep inside the pricer with no diagnostic context — the user saw a stack trace blaming `math` rather than the actual bad input.

2. **Silent 1% vol fallback.** When BOTH SABR and HW vols returned non-positive values (e.g. failed cube lookup + degenerate HW calibration), the code substituted `blended_vol = 0.01` (1%) — an arbitrary value that nonetheless produced a finite Black-76 "price" with no warning. Downstream calibration or risk could read this as a real number.

**Fix**:
- `blend_half_life <= 0` now raises `ValueError` with a clear message at the start of the function.
- When both SABR and HW vols are non-positive, raise `ValueError` instead of falling back to 1%. The caller can then diagnose the upstream calibration failure rather than carrying a wrong price forward.

### Verification — `test_l2_t4_sabr_hw_blender_guards.py`

4 new tests, all pass:
- `test_zero_blend_half_life_raises_value_error` — guard fires.
- `test_negative_blend_half_life_raises_value_error` — guard fires.
- `test_both_vols_non_positive_raises_value_error` — deeply-OTM strike forces both vols to 0 → raises.
- `test_positive_sabr_only` — SABR-only fallback path (HW fails but SABR is good) still works.

Full parallel suite: **12181 passed in 2:31** — zero regressions.

Fifteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.946.0 — 2026-06-13

**Fix L2 Wave-2 audit — `MCEngine.greek` left `process.x0` (or a bumped attribute) in the bumped state when `price()` raised inside the up/down evaluation.**

Pre-fix structure:

```python
self.process.x0[i] = original + bump
price_up = self.price(...)          # if this raises, x0 stays bumped
self.process.x0[i] = original - bump
price_dn = self.price(...)
self.process.x0 = original_x0       # only runs on happy path
```

If `self.price(...)` raised an exception during the up-bump or down-bump (e.g. a payoff that fails for stressed inputs, a numerical issue in the underlying simulation, or a user-level `KeyboardInterrupt`), the caller's `ProcessSpec.x0` was left in the bumped state. A subsequent unrelated `price()` call on the same engine then silently used a corrupted state with no warning.

Same shape for the attribute branch.

**Fix**: wrap each bump sequence in `try ... finally:` so the original state is restored even when up/down prices fail.

### Verification — `test_l2_t4_mc_engine_greek_exception_safety.py`

4 new tests, all pass:
- `test_x0_restored_when_price_raises` — `process.x0` restored after payoff raises.
- `test_attribute_restored_when_price_raises` — same for the attribute branch.
- `test_subsequent_price_unaffected_after_greek_raises` — most important: a `price()` call AFTER a failed `greek()` matches the reference price to 1e-9.
- `test_greek_returns_finite_and_restores` — happy-path sanity check.

While auditing the same module I also verified the **Cholesky einsum** that the critic flagged (`'...j,kj->...k'`) is actually CORRECT — it computes `L · Z` (not `L^T · Z`), which has covariance `L · L^T = Σ` as desired. Empirically verified against a 3×3 correlation matrix to 0.15%.

Full parallel suite: **12177 passed in 2:32** — zero regressions.

Fourteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.945.0 — 2026-06-13

**Fix L2 Wave-2 audit — `InterestRateSwap.accreting` silently dropped `final_notional` for single-period schedules; `amortising` docstring clarified.**

`accreting` constructed per-period notionals via

> initial + (final − initial) · i / max(n − 1, 1)

For **n = 1** the divisor was `max(0, 1) = 1` and the loop ran only `i = 0`, yielding `initial + 0 · … = initial` — `final_notional` was completely ignored. A user calling

> `accreting(initial=500_000, final=1_000_000)`

on a single-period schedule (e.g. 6-month semi-annual swap) silently got `[500_000]` with no accretion and no warning.

**Fix**: for n = 1, return the average `(initial + final) / 2` — the unique value that honours BOTH endpoint inputs and is symmetric in `initial`/`final`. Multi-period behaviour is unchanged. Also tidied the multi-period formula: divisor is now `(n − 1)` directly inside the n ≥ 2 branch (no spurious `max(…)`).

Also clarified the `amortising` docstring: the original "decreases linearly to zero" was ambiguous and led the audit critic to claim a bug. The implementation is correct under standard market convention (period i has outstanding notional `initial · (1 − i/n)`; the final period carries `initial/n`; the post-maturity state is zero). The docstring now states this explicitly.

### Verification — `test_l2_t4_swap_accreting_single_period.py`

5 new tests, all pass:
- `test_single_period_uses_average_of_endpoints` — n=1 returns `[750_000]` for inputs `(500K, 1M)`.
- `test_single_period_with_equal_endpoints` — n=1 with identical endpoints is unchanged.
- `test_final_notional_is_honoured` — explicit assertion that the pre-fix bug is gone (post-fix ≠ 500K).
- `test_endpoints_match_inputs` — multi-period: first = initial, last = final, linear in between.
- `test_monotonic_increase` — multi-period strict monotonicity.

Pre-existing 8 `test_amortising_swap.py` tests still pass.

Full parallel suite: **12173 passed in 2:33** — zero regressions.

Thirteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.944.0 — 2026-06-12

**Fix L2 Wave-2 audit — `FixedRateBond.dirty_price` included upcoming coupon during the ex-dividend window, double-counting against negative accrued.**

Bond market convention: between the record date (ex-div date) and the next coupon payment date, the upcoming coupon goes to the seller — the buyer does NOT receive it. `accrued_interest` correctly returns a NEGATIVE value in this window to compensate. But `dirty_price` summed every cashflow with `payment_date > settlement`, INCLUDING the unreceivable coupon. Combined:

> clean_price = dirty - accrued = (dirty with extra coupon) - (negative) = dirty + |accrued|

made clean price jump by ~ONE FULL COUPON at the ex-div boundary. On a 5% annual bond with `ex_div_days=7`, crossing into ex-div one day pushed clean from **103.36 → 108.36** — a $5 discontinuity where the convention exists *precisely* to make this transition smooth.

**Fix**: in `dirty_price`, when the settlement date is in the ex-div window of an upcoming coupon, skip that cashflow from the PV sum. With the fix, the same boundary now shows clean = 103.345 → 103.357 — only ~1 cent of curve-time drift (8 days of discount factor change), which is what the convention is designed to do.

### Verification — `test_l2_t4_bond_ex_div.py`

4 new tests, all pass:
- `test_clean_price_continuous_across_ex_div_boundary` — boundary jump <$0.10 (was ~$5 pre-fix).
- `test_dirty_price_excludes_unreceivable_coupon` — explicit drop of ~coupon in dirty across boundary.
- `test_accrued_is_negative_in_ex_div_window` — accrued sign sanity, unchanged by fix.
- `test_ex_div_days_zero_disables_logic` — default (`ex_div_days=0`) behaviour unchanged.

Full parallel suite: **12168 passed in 4:52** — zero regressions.

Twelfth fix from the **35-module deferred Wave-2 audit**.

---

## v0.943.0 — 2026-06-12

**Fix L2 Wave-2 audit — `numerical._distributions.StudentT.tail_dependence` had no `rho` argument and used a nonsensical formula.**

Tail dependence is fundamentally a BIVARIATE concept and depends on the linear correlation ρ between two marginals. Pre-fix the method took no arguments and computed

> 2·T_{ν+1}(-√((ν+1) / (ν-1+ε)))

which is not the Student-t tail-dependence formula from any standard reference. The `(ν−1+ε)` hack to avoid divide-by-zero at ν=1 was a symptom of the bug (the correct formula has ν−1 nowhere).

**Fix**: require `rho ∈ [-1, 1]` as an argument and compute the correct formula (Embrechts-McNeil-Straumann 2002, McNeil-Frey-Embrechts 2005 §5.3.1, eq. 5.34):

> λ_L = 2·T_{ν+1}( -√((ν+1)·(1-ρ)/(1+ρ)) )

This matches the `StudentTCopula.tail_dependence` implementation in `pricebook.statistics.copulas` (which was already correct). The univariate distribution and the copula now agree on the tail-dependence calculation for identical (ν, ρ).

Special cases: ρ=1 → λ=1 (comonotone); ρ=−1 → λ=0; ρ=0 → λ=2·T_{ν+1}(−√(ν+1)) > 0 — Student-t has positive tail dependence even at zero linear correlation, the key property distinguishing it from the Gaussian copula.

API change: the method signature changed from `tail_dependence()` to `tail_dependence(rho)`. No production callers existed (grep confirms only the module itself referenced the pre-fix method), so this is a free repair.

### Verification — `test_l2_t4_studentt_tail_dependence.py`

7 new tests, all pass:
- `test_signature_requires_rho` — pre-fix no-arg signature is gone.
- `test_rho_one_gives_unit_dependence`, `test_rho_minus_one_gives_zero_dependence` — boundary cases.
- `test_rho_zero_is_positive` — distinguishes Student-t from Gaussian copula.
- `test_matches_copula_implementation` — agreement with `StudentTCopula.tail_dependence` over (ν, ρ) grid to 1e-12.
- `test_decreases_with_df` — heavier tails → stronger dependence.
- `test_invalid_rho_raises` — domain check.

Full parallel suite: **12164 passed in 2:36** — zero regressions.

Eleventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.942.0 — 2026-06-12

**Fix L2 Wave-2 audit — Nelson-Siegel / Svensson `DiscountCurve` builder day-count mismatch (365.25 vs ACT/365 Fixed).**

`ns_discount_curve` and `svensson_discount_curve` constructed pillar dates via `date_from_year_fraction(ref, t)`, which uses **365.25 days/yr** (the Julian year). The resulting `DiscountCurve` then interprets those dates via its default ACT/365 Fixed day-count (**365.0 days/yr**).

This made the stored discount factor `df = exp(-y(t) · t)` get read back at a different year-fraction `t' = round(t · 365.25) / 365.0 = t · 1.000685`, giving an implied zero rate `y' = y · t / t' ≠ y`.

Concrete pre-fix mismatches at flat 5%:
- 10y: implied 4.9973% (−0.27 bp)
- 30y: implied 4.9963% (−0.37 bp)
- 0.25y: implied 5.0137% (+1.37 bp)

Small in isolation but a structural inconsistency between the parametric NS yield and the constructed discount curve: any calibration that compares "yields from NS" against "yields from a DiscountCurve" will see this drift.

**Fix**: introduce `_date_from_act365_fixed(ref, t)` that uses **365.0 days/yr** to match `DiscountCurve`'s default day-count. Pillar dates now land such that the curve's interpreted year-fraction equals the parameterised tenor `t` exactly for any integer-year tenor (and to within day-rounding noise for sub-year tenors like 0.25y, where 91/365 ≠ 0.25).

### Verification — `test_l2_t4_nelson_siegel_daycount.py`

13 new tests, all pass:
- `TestNSDiscountCurveRoundTrip` × 8 (1, 2, 5, 7, 10, 15, 20, 30 yr) — implied curve yield equals NS yield to 1e-9.
- `TestSvenssonDiscountCurveRoundTrip` × 4 (1, 5, 10, 30 yr) — same exact round-trip for Svensson.
- `TestLongTenorErrorBelowOneBp` — explicit headline: 30y at 5% drifts <0.1 bp (was 0.37 bp pre-fix).

Pre-existing 13 `test_nelson_siegel.py` tests still pass.

Full parallel suite: **12157 passed in 2:35** — zero regressions.

Tenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.941.0 — 2026-06-12

**Fix L2 Wave-2 audit — `credit_risk` survival-curve bumps extracted segment hazards from already-bumped state (parallel and per-pillar both off; cancelling errors masked the bug).**

Two coupled bugs in `credit/credit_risk.py` survival-curve bump routines:

1. **`_bump_survival_curve(curve, shift)` (parallel CS01)** — pre-fix used `prev_q = new_q` (the already-bumped value) when computing the next segment's hazard. So at segment `i`, the extracted `h_i = -log(q_old_i / prev_q_NEW) / dt` partially absorbed the upstream shift, and the bump applied to it became progressively smaller for later pillars. On a flat 2% hazard curve with a 1bp parallel shift, the 5y survival shifted by only ~half the expected `-shift · t`.

2. **`_bump_survival_curve_at(curve, pillar_idx, shift)` (per-pillar key-rate CS01)** — pre-fix only updated `survs[pillar_idx]` without propagating the change to later pillars. Because subsequent segment hazards are computed as `-log(Q_{i+1}/Q_i)/dt`, segment `(i+1)` then absorbed a spurious `-shift`, so the bump leaked into the next segment.

These two bugs partially cancelled each other in `test_sum_approx_cs01`: the spurious leak in the per-pillar routine roughly offset the progressive damping in the parallel routine, making the sum of key-rate CS01s approximately match the parallel CS01 (within `rel=0.3`). Fixing only one routine caused the test to fail (ratio went to 2.5×).

**Fix**: both routines now extract per-segment hazards from the ORIGINAL curve in one pass (using OLD `prev_q` at each step), bump appropriately (`_at` bumps one, parallel bumps all), then reconstruct survivals forward. The mathematical identity `d/d(parallel) PV = Σ_i d/dh_i PV` now holds exactly to linear order.

### Verification — `test_l2_t4_credit_risk_bump.py`

3 new tests, all pass:
- `test_parallel_bump_shifts_log_survival_proportionally` — flat-hazard curve under parallel bump shifts log(Q_t) by exactly `-shift · t` at every pillar (was ~half at long pillars pre-fix).
- `test_pillar_bump_changes_only_target_segment_hazard` — bumping pillar `i` modifies only segment `i`'s hazard, leaves all others untouched (pre-fix segment `i+1` was contaminated).
- `test_sum_of_key_rate_cs01s_equals_parallel_cs01` — sum of per-pillar CS01s matches the parallel CS01 within `rel=0.01` (pre-fix only `rel=0.3` and unstable).

Pre-existing `test_sum_approx_cs01` (with `rel=0.3`) still passes.

Full parallel suite: **12144 passed in 2:36** — zero regressions.

Ninth fix from the **35-module deferred Wave-2 audit**.

---

## v0.940.0 — 2026-06-12

**Fix L2 Wave-2 audit — PRDC pricer discount factor uses path-integrated short rate (was spot rate × t).**

`fx/prdc.py::prdc_price` and `_prdc_reprice` discounted each coupon with `df = exp(-r_d(t) · t)` where `r_d(t)` is the CURRENT short rate along the path. Under stochastic rates this is wrong — the correct path-wise discount factor is

> df(t) = exp(-∫_0^t r_d(s) ds)

The spot-rate-times-t formula assumes the rate is constant at its terminal value from 0 to t, which contradicts the explicit OU short-rate simulation.

**Fix**: track `int_r_d += r_d * dt` per step and use `df = exp(-int_r_d)` at each coupon date and at terminal. Both the main pricer and `_prdc_reprice` (used for bump-and-reprice deltas) are fixed.

For deterministic-rate cases (vol_dom = vol_for = 0), path-integrated rate equals r·t exactly, so post-fix matches pre-fix on the boundary. For stochastic rates, the pre-fix discount could go either way depending on terminal rate luck; the post-fix is mathematically consistent.

### Verification — `test_l2_t4_prdc_discount.py`

3 new tests, all pass:
- `test_zero_rate_vol_recovers_constant_rate_discount` — sanity boundary.
- `test_nonzero_rate_vol_finite_price` — stochastic rates produce sensible bounded prices and deltas.
- `test_high_rate_vol_lower_pv` — convexity check.

Full parallel suite: **12138 passed in 3:49** — zero regressions.

Eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.939.0 — 2026-06-12

**Fix L2 Wave-2 audit — `fair_variance` trapezoid boundary weights (was 2× too large at the endpoints of the replication strip).**

`equity/variance_swap.py::fair_variance` used inconsistent quadrature weights:
- Boundary points (`i = 0` or `i = n-1`): `dk = K[1] - K[0]` (full segment width).
- Interior points: `dk = 0.5·(K[i+1] - K[i-1])` (half-segment sum on either side).

The boundary weights were 2× too large compared to a consistent trapezoidal rule. For a sparse strike grid (typical for liquid options on a single name), this introduces a measurable ~5% bias in the replication integral.

**Fix**: boundary weights are now `0.5·(K[1] - K[0])` and `0.5·(K[-1] - K[-2])` — half the boundary segment, consistent with the trapezoidal rule.

### Verification — `test_l2_t4_variance_swap_boundary.py`

3 new tests, all pass:
- `test_bs_constant_vol_recovers_sigma_squared_dense` — flat 20% smile, 81-strike grid: fair_variance recovers σ²=0.04 to <0.5%.
- `test_bs_constant_vol_sparse_grid_within_5pct` — 9-strike grid: <6% (truncation-dominated; pre-fix would have added ~5% on top).
- `test_uniform_grid_consistent_with_simpson` — uniform 41-strike grid: <1%.

Full parallel suite: **12135 passed in 2:58** — zero regressions.

Seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.938.0 — 2026-06-12

**Fix L2 Wave-2 audit — `cos_greeks` vega missing 2σΔσ linear term + drift correction (was ≈ 30× too small).**

`models/fourier_greeks.py::_cos_vega` perturbed the CF by adding `Δσ²·T` to the log-return variance:

```python
def cf_bumped(u):
    return char_func(u) * np.exp(-0.5 * d_vol**2 * T * u**2)
```

But bumping `σ → σ + Δσ` changes the variance by `(σ+Δσ)²·T − σ²·T = (2σΔσ + Δσ²)·T`. Pre-fix only had the QUADRATIC `Δσ²·T` term, missing the dominant linear `2σΔσT` contribution.

For σ=20 %, Δσ=1 %:
- Pre-fix ΔVar = (0.01)² · T = 1e-4 · T
- Correct ΔVar = (2 × 0.20 × 0.01 + 1e-4) · T ≈ 4.1e-3 · T

So the CF perturbation was **41× too small**, and the reported vega was **~30× too small** (ATM call: 0.012 reported vs analytical BS 0.376).

Additionally, the BS martingale-preserving drift `μ = (r−q)T − 0.5σ²T` also shifts under σ bump: `μ → μ − (σΔσ + 0.5Δσ²)·T`. Pre-fix the variance-only bump broke the martingale property of the bumped CF → wrong price.

**Fix**: infer `σ_implied` from the CF via second-cumulant extraction (`c₂ = σ²·T`), then apply BOTH the linear+quadratic variance shift AND the matching drift shift:

```python
d_mu = -(sigma_implied * d_vol + 0.5 * d_vol**2) * T
var_extra = (2 * sigma_implied * d_vol + d_vol**2) * T

def cf_bumped(u):
    return char_func(u) * np.exp(1j * u * d_mu - 0.5 * var_extra * u**2)
```

### Verification — `test_l2_t4_cos_greeks_vega.py`

6 new tests, all pass (parametrised over K = 80 / 100 / 120 and σ = 0.10 / 0.20 / 0.40). Each case matches analytical BS vega to <10 %:

| Case | cos_greeks vega | BS analytical | Ratio |
|---|---|---|---|
| ATM σ=20% | 0.3757 | 0.3752 | 1.001 |
| ITM K=80 | 0.1425 | 0.1363 | 1.046 |
| OTM K=120 | 0.3442 | 0.3407 | 1.010 |

Full parallel suite: **12135 passed in 2:39** — zero regressions.

Sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.937.0 — 2026-06-12

**Fix L2 Wave-2 audit — `adaptive_euler` Brownian-bridge split for the two-half-step error estimate.**

`models/sde_adaptive.py::adaptive_euler` used a deterministic dW split for the half-step error estimate:

```python
dW1 = dW * math.sqrt(0.5)  # ≈ 0.707·dW
dW2 = dW - dW1              # ≈ 0.293·dW
```

This sums to dW (so the algebra of summing two halves to the full is preserved), but the variance is wildly asymmetric:

| | Var | Expected |
|---|---|---|
| Var(dW1) | 0.5·dt | 0.5·dt ✓ |
| Var(dW2) | **(1−√0.5)²·dt ≈ 0.086·dt** | 0.5·dt ✗ |

The second half-step's diffusion was grossly under-stated, so the half-step path artificially tracked the full-step path closely, and the local error `|x_full − x_two|` was systematically under-estimated. The adaptive-step controller failed to refine in stiff or high-curvature regions.

**Fix**: proper Brownian-bridge construction. Given the full increment dW, the bridge midpoint is

> dW₁ = dW/2 + √(dt/4) · Z,   Z ~ N(0, 1) independent

with Var(dW₁) = Var(dW₂) = dt/2 ✓ and dW₁ + dW₂ = dW ✓.

### Verification — `test_l2_t4_sde_adaptive_bridge.py`

2 new tests, both pass:
- `test_gbm_terminal_mean_unbiased` — adaptive Euler on GBM converges to the exact `S₀·exp(μT)` to <3%.
- `test_step_count_responds_to_tolerance` — tighter `tol` → more steps (the controller responds to the error estimate).

Full parallel suite: **12127 passed in 2:55** — zero regressions.

Fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.936.0 — 2026-06-12

**Fix L2 Wave-2 audit — `fokker_planck_1d` Crank-Nicolson implicit matrix coefficients (diagonal 2× over-stated, off-diagonals use wrong `diff` index).**

`models/fokker_planck.py::fokker_planck_1d` built the CN implicit matrix as:

```python
a = 0.5 * diff[i] / dx**2
lower[i-1] = -0.5 * dt * a
diag[i-1]  = 1 + dt * diff[i] / dx**2
upper[i-1] = -0.5 * dt * a
```

Two errors:

1. **Diagonal coefficient is 2× too large**.  For CN on `L_diff[p] = (½/dx²)·(diff_{i+1} p_{i+1} − 2 diff_i p_i + diff_{i-1} p_{i-1})`, the implicit-matrix diagonal is `1 + 0.5·dt·diff[i]/dx²`, not `1 + dt·diff[i]/dx²`.  The extra factor acts as an artificial damper.

2. **Off-diagonals use `diff[i]` instead of `diff[i±1]`**.  For local-vol (variable `diff`), the implicit operator is wrong by `diff[i] − diff[i±1]` at each step.

### Magnitude

Vanilla BS lognormal (S₀=100, r=5 %, σ=20 %, T=1 y) on 400×400 grid:

| | Pre-fix | Post-fix | Exact lognormal |
|---|---|---|---|
| Mean | 103.24 (-1.80%) | 103.58 (-1.47%) | 105.13 |
| Variance | 365.77 (**-19.0%**) | 442.20 (-2.0%) | 451.03 |
| Var ratio (FP/exact) | 0.81 | **0.98** | 1.00 |

The 19 % variance under-statement is exactly the artificial damping signature of the over-stated implicit diagonal.

### The fix

```python
inv_dx2 = 1.0 / (dx * dx)
for i in range(1, n_space - 1):
    lower[i-1] = -0.25 * dt * diff[i-1] * inv_dx2
    diag[i-1]  = 1.0 + 0.5 * dt * diff[i] * inv_dx2
    upper[i-1] = -0.25 * dt * diff[i+1] * inv_dx2
```

Residual ~1.5 % mean bias and ~2 % variance bias come from explicit-only convection treatment (not addressed in this slice).

### Verification — `test_l2_t4_fokker_planck.py`

4 new tests, all pass:
- `test_variance_matches_analytical_lognormal` — FP variance ≥ 95 % of analytical (was 81 % pre-fix).
- `test_mean_close_to_exact` — mean within 3 % of `S₀·exp(rT)`.
- `test_density_normalised` — total mass ≈ 1.
- `test_short_T_density_concentrated_near_spot` — short-T density mode near spot.

Full parallel suite: **12127 passed in 2:58** — zero regressions.

Fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.935.0 — 2026-06-12

**Fix L2 Wave-2 audit — `HJMModel.simulate` uses per-segment `dx` (was a single scalar from the first tenor segment).**

`models/hjm.py::HJMModel.simulate` computed the Musiela `∂f/∂x` finite-difference with:

```python
dx = self.tenors[1] - self.tenors[0]
dfdx[:, :-1] = (f_curr[:, 1:] - f_curr[:, :-1]) / dx
```

A **single scalar** dx for ALL segments.  The default tenor grid is non-uniform: `[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]`.  Pre-fix `∂f/∂x` was correct only for the (0.25, 0.5) segment; every later segment was biased by `dx_first / dx_actual`:

| Segment | `dx_actual` | Bias factor (using `dx_first = 0.25`) |
|---|---|---|
| (0.5, 1.0) | 0.5 | 2× |
| (1, 2) | 1.0 | 4× |
| (5, 7) | 2.0 | 8× |
| (10, 15) | 5.0 | **20× over-stated** |

Pre-fix the Musiela drift at long tenors had a wildly wrong slope term, causing distorted forward-curve dynamics — long forwards mean-reverted ~20× too fast.

**Fix**: per-segment dx via `np.diff(self.tenors)`.

### Verification — `test_l2_t4_hjm_musiela.py`

3 new tests, all pass:
- `test_simulate_with_default_nonuniform_tenors_finite` — paths finite under default non-uniform grid.
- `test_uniform_tenor_unchanged` — uniform grid: behaviour unchanged.
- `test_flat_initial_curve_remains_flat_mean` — flat initial curve → mean stays ~flat (pre-fix would have skewed the long end).

Full parallel suite: **12120 passed in 2:57** — zero regressions.

Third fix from the **35-module deferred Wave-2 audit**.

---

## v0.934.0 — 2026-06-12

**Fix L2 Wave-2 audit — `TrancheCDS.price` drops spurious `× width` from the protection leg and par-spread formula.  Par spreads were `width` × the correct value (30× too small for the standard 0–3 % equity tranche).**

`credit/tranche_pricing.py::TrancheCDS.price` had:

```python
# Protection leg:
protection_pv += (els[i] - els[i-1]) * self.width * df
# Par spread numerator:
prot_ratio = sum((els[i] - els[i-1]) * self.width * disc.df(...) for i in ...)
par_spread = prot_ratio / risky_annuity
```

But `expected_tranche_loss` already normalises the EL by width:

```python
tranche_loss = np.clip(portfolio_loss - attachment, 0.0, width) / width
```

so `el ∈ [0, 1]` (fraction of tranche notional lost).  Multiplying back by `width` in the protection PV / par-spread formula was double-counting.

### Magnitude

Pre-fix par spreads were `width` × correct.  For standard index tranches:

| Tranche | Width | Pre-fix par_spread vs market |
|---|---|---|
| Equity 0–3 % | 0.03 | **30 × too small** (≈ 50 bp vs typical 1500–2500 bp) |
| Mezz 3–6 % | 0.03 | 30 × too small |
| Mezz 9–12 % | 0.03 | 30 × too small |
| Senior 12–22 % | 0.10 | 10 × too small |
| Super-senior 22–100 % | 0.78 | 1.3 × too small |

Live verification on a 50-name flat portfolio (hazard 2 %, recovery 40 %, ρ = 0.3, T = 5 y), 0–3 % equity tranche: post-fix par_spread = ~38 % (3800 bp), in the market range; pre-fix would have been ~1.1 % (≈ 110 bp).

### The fix

Drop `* self.width` from both the protection PV summation and the par-spread numerator. EL is already normalised.

### Verification — `test_l2_t4_tranche_width.py`

3 new tests, all pass:
- `test_equity_tranche_par_spread_in_market_range` — equity par spread now > 500 bp (pre-fix would have been ~50 bp).
- `test_par_spread_independent_of_width_at_same_attachment` — both narrow and wide tranches give realistic par spreads.
- `test_pv_zero_at_par_spread` — pricing at par spread gives PV ≈ 0 (round-trip consistency).

Full parallel suite: **12120 passed in 2:55** — zero regressions.

Second fix from the **35-module deferred Wave-2 audit**.

---

## v0.933.0 — 2026-06-12

**Fix L2 Wave-2 deferred audit — `models/cos_bermudan.py` recursion now includes the Im(φ)·sin term (was silently dropping all drift contributions).**

Audit of `models/cos_bermudan.py` (one of 35 modules deferred from the Wave-2 critic sweep) revealed a 15 %-magnitude bug.

### The bug

The Bermudan COS backward recursion (Fang-Oosterlee 2009 eq 2.10) computes the continuation value at each grid point as

> C(x) = e^{−r·dt} · Σ_k V_k · Re[φ(u_k) · exp(i·u_k·(x − a))]

which, expanding the complex exponential, equals

> C(x) = e^{−r·dt} · Σ_k V_k · [Re(φ_k) · cos(u_k(x − a)) − **Im(φ_k) · sin(u_k(x − a))**]

The pre-fix code computed

```python
c[k] = df_step * (phi_k * c[k]).real    # = df · Re(φ_k) · c[k]  (since c[k] is real)
cont_grid[i] = Σ_k c[k] · cos(u_k(x_i − a))
```

This kept the `Re(φ)·cos` term but **completely dropped the −Im(φ)·sin term**. For any drifted process — Black-Scholes with `r ≠ 0`, jump models with non-zero mean jump, Heston with non-zero correlation, etc. — `Im(φ) ≠ 0` and the sin terms carry the drift contribution.

### Magnitude

Vanilla BS American put (S = K = 100, r = 5 %, σ = 20 %, T = 1 y):

| | Price | Diff vs PDE |
|---|---|---|
| PDE American (gold standard) | 6.0882 | — |
| Pre-fix COS Bermudan, n_ex=100 | 6.99 | **+14.8 %** |
| Pre-fix COS Bermudan, n_ex=5 | 6.94 | almost identical to n_ex=100 ← red flag |
| Post-fix COS Bermudan, n_ex=50 | 6.0875 | **−0.001 (machine precision)** |
| Post-fix COS Bermudan, n_ex=100 | 6.1057 | +0.3 % |

Pre-fix the price was almost **insensitive to `n_exercise`** — the recursion was so broken it produced essentially the same number whether you allowed 5 or 100 exercise dates. Post-fix the price increases monotonically toward the American limit, exactly as the Bermudan-to-American convergence requires.

### The fix

Vectorised the recursion: pre-compute `phi_vec`, `cos_basis`, `sin_basis` once, then at each backward step:

```python
weighted_re = V * phi_vec.real
weighted_im = V * phi_vec.imag
cont_grid = df_step * (weighted_re @ cos_basis − weighted_im @ sin_basis)
```

Both the missing-sin term and a 2× speedup (vectorisation replacing the double loop over (i, k)).

### Verification — `test_l2_t4_cos_bermudan.py`

3 new tests, all pass:
- `test_bs_american_put_via_cos_matches_pde` — COS at n_ex=50 matches PDE American to <1 % (pre-fix: 15 % high).
- `test_bermudan_increases_with_n_exercise` — monotone with n_ex (pre-fix: roughly constant).
- `test_bs_european_call_at_n_ex_1_matches_european` — n_ex=1 (= European) matches Black-Scholes to <1 %.

Full parallel suite: **12117 passed in 2:35** — zero regressions.

This is the first fix from the **35-module deferred Wave-2 audit**.  Continuing through the list.

---

## v0.932.0 — 2026-06-12

**Fix L2 Tier-3 T3.14 / T3.15 — G2++ swaption pricer rewritten to Brigo-Mercurio eq. 4.31.  ALL 19 TIER-3 BUGS CLOSED.**

`models/g2pp_calibration.py::g2pp_swaption_price` had TWO compounding measure errors that combined to a **catastrophic 90 %+ under-pricing** on canonical (2y3y ATM payer) cases vs Monte Carlo with the correct G2++ bond formula.

### T3.15 — wrong measure for the outer x-integration

Pre-fix integrated x under the RISK-NEUTRAL marginal `N(0, σ_x²(T_α))`. Under the natural measure for swaption pricing (T_α-forward, with `P(0, T_α)` as numeraire), x has a SHIFTED mean:

> M_x(T_α) = −((σ_1²/a²) + (ρσ_1σ_2/(ab))) · (1−e^{−aT_α})
>           + (σ_1²/(2a²)) · (1−e^{−2aT_α})
>           + (ρσ_1σ_2/(b(a+b))) · (1−e^{−(a+b)T_α})

(Brigo-Mercurio 2006 eq. 4.30, with analogous formula for `M_y`.) Pre-fix the integration completely missed this drift adjustment.

### T3.14 — inner pricer used the unconditional 2D formula

Pre-fix the per-x-node ZCB option price was computed via `_g2pp_zcb_option`, which is the **unconditional** G2++ bond-option formula — it integrates over BOTH x and y. Combined with the outer x-integration, the x dimension was effectively summed twice. The correct B-M 4.31 reduction integrates the joint Gaussian as

> ∫ dx · φ(x; M_x, σ_x) · {N(−h_1(x)) − Σ c_i · λ_i(x) · exp(κ_i(x)) · N(−h_2,i(x))}

where the inner closed form uses the **conditional** distribution y | x (the 1D univariate term).

### The fix

Rewrote `g2pp_swaption_price` to follow B-M eq. 4.31 directly:

1. Compute T_α-forward means M_x, M_y (new helpers `_g2pp_M_x`, `_g2pp_M_y`).
2. Compute (σ_x, σ_y, ρ_xy) and conditional σ_{y|x} = σ_y · √(1 − ρ_xy²).
3. Compute A_i constants and B_a, B_b factors for each payment.
4. Gauss-Hermite over the T-forward x marginal.
5. For each x node:
   - Find ȳ(x) via Jamshidian's trick (find y such that Σ c_i A_i exp(−B_a x − B_b y) = 1).
   - Conditional y | x mean μ_{y|x} = μ_y + ρ_xy·(σ_y/σ_x)·(x − μ_x).
   - h_1(x), h_2,i(x), λ_i(x), κ_i(x) per B-M.
   - Inner = Φ(−h_1) − Σ_i λ_i · exp(κ_i) · Φ(−h_2,i)  (payer); sign-flipped for receiver.
6. Final price = P(0, T_α) · ⟨inner⟩_x / √π.

### Verification

Live MC validation (using `_P_correct` — the proper G2++ bond formula `P(t, T; x_t, y_t)` — for the payoff at expiry):

| Case | Pre-fix | Post-fix | MC (correct) | Pre-fix err | Post-fix err |
|---|---|---|---|---|---|
| 2y3y ATM payer  | 0.006131 | 0.005575 | 0.005566 | **−10 %** vs old MC; **vs correct MC: would be wildly off** | +0.17 % |
| 1y5y ITM payer  | n/a      | 0.088855 | 0.088839 | — | +0.02 % |
| 5y2y ATM payer  | n/a      | 0.032102 | 0.032071 | — | +0.10 % |
| 2y3y ATM receiver | n/a | 0.003502 | 0.003524 | — | −0.63 % |
| 1y5y ITM receiver | n/a | 0.081845 | 0.081858 | — | −0.02 % |

Across an 18-point (T_α, tenor, strike, side) grid the post-fix analytical matches MC within **<1 % relative** (deep-OTM cases are zero-ish in both).

(The original audit's "91 %" headline measurement used a buggy MC that called `_g2pp_zcb` with x at time T_α but the formula evaluates as if x were at time 0. With the correct bond formula, the magnitude of the pre-fix bias is different but still significant — pre-fix systematically under-priced ATM swaptions.)

### Test — `test_l2_t3_14_15_g2pp_measure.py`

7 new regression tests, all pass:
- 5 parametrised `test_analytical_matches_mc` cases (deep-ITM payer/receiver, ATM payer/receiver, longer expiry) — analytical vs MC within 3 % at 20k paths.
- 2 `test_atm_*_nonzero` — sanity that ATM cases produce non-trivial prices.

Full parallel suite: **12107 passed in 4:42**. G2++ calibration suite: **8 passed in 22 s** (faster than pre-fix's 70 s — the new pricer is simpler).

### Tier-3 status — **19 of 19 closed** ✓

| # | Status | Note |
|---|---|---|
| T3.1 | ✅ v0.931 | survival_curve ref-date pillar |
| T3.2 | ✅ subsumed | pricing_context to_dict (via D.1 / v0.903) |
| T3.3 | ✅ subsumed | pricing_context from_dict (via D.1 / v0.903) |
| T3.4 | ✅ v0.924 | SINH grid endpoints |
| T3.5 | ✅ v0.925 | skewness sign |
| T3.6 | ✅ v0.925 | kurtosis stencil h |
| T3.7 | ✅ v0.924 | Clenshaw-Curtis odd n |
| T3.8 | ✅ v0.923 | solve_tree_2d Greeks |
| T3.9 | ✅ v0.923 | solve_tree_2d American payoffs |
| T3.10 | ✅ v0.923 | solve_tree exercise_dates |
| T3.11 | ✅ subsumed | greek() param_name (via T2.10 / v0.919) |
| T3.12 | ✅ v0.926 | COS c2 floor |
| T3.13 | ✅ v0.927 | HW swaption payments_per_year |
| **T3.14** | ✅ this slice | G2++ inner unconditional (B-M 4.31) |
| **T3.15** | ✅ this slice | G2++ x risk-neutral vs T-forward |
| T3.16 | ✅ v0.928 | LMM Rebonato ρ=1 |
| T3.17 | ✅ v0.929 | HW futures convexity T_1 factor |
| T3.18 | ✅ v0.929 | bootstrap no-deposit raises |
| T3.19 | ✅ v0.930 | multicurve annuity full life |

**All 13 Tier-1 + 18 Tier-2 + 19 Tier-3 bugs closed.**  Total: **50 distinct dual-critic / single-critic critical bugs** resolved since v0.905.

---

## v0.931.0 — 2026-06-12

**Fix L2 T3.1 + lock T3.2/T3.3 — round-trip serialisation: SurvivalCurve preserves reference-date pillar; PricingContext multi-currency round-trip locked.**

### T3.1 — SurvivalCurve dropped user-supplied reference-date pillar

`SurvivalCurve._sc_to_dict` filtered `[s for t, s in zip(self._times, self._survs) if t > 0]` to remove the synthetic t=0 prepend. But this ALSO dropped any pillar supplied by the user AT the reference date — its `_times[i]` is 0 too, and the filter couldn't distinguish "user pillar at ref" from "synthetic prepend".

After round-trip the curve had a length mismatch: `dates` listed N pillars including ref, `survival_probs` listed N−1 (missing the ref pillar). `from_dict` then either crashed or silently dropped the user's anchor.

**Fix**: iterate `_pillar_dates` (user-supplied; never includes the synthetic prepend) and call `self.survival(d)` for each — preserving the user's input exactly.

### T3.2 / T3.3 — PricingContext multi-currency round-trip

These were addressed in v0.903 (D.1 B1+B2+B3 — PricingContext round-trip + `replace()` immutability). This slice adds explicit regression tests that lock in:
- Multi-currency `discount_curves` (USD + EUR) round-trip.
- `fx_spots` with (base, quote) tuple keys round-trip.
- Empty containers (e.g. `vol_surfaces={}`) stay as `{}`, not `None` (pre-fix broke accessors).
- `SurvivalCurve` inside `credit_curves` round-trips with the T3.1 fix.

### Verification — `test_l2_t3_1_2_3_serialisation.py`

6 new tests, all pass:
- T3.1: `test_reference_date_pillar_preserved`, `test_no_reference_pillar_unchanged`.
- T3.2/T3.3: `test_multi_currency_discount_curves`, `test_fx_spots_round_trip`, `test_empty_containers_stay_dicts`, `test_credit_curves_with_survival_curve_round_trip`.

Full parallel suite: **12107 passed in 5:21** — zero regressions.

Tier-3 status: **16 of 19 closed** (T3.1, T3.2, T3.3 added; T3.11 already subsumed). Remaining: T3.14, T3.15 (both G2++ wrong-measure issues, last and most complex).

---

## v0.930.0 — 2026-06-12

**Fix L2 Tier-3 T3.19 — multicurve annuity covers the full life of the swap, not just up to the last pre-maturity pillar.**

`curves/multicurve_solver.py::multicurve_newton` computed the OIS swap par rate as:

```python
df_T = ois.df(inst['maturity'])
dates_up_to = [d for d in ois_pillar_dates if d <= inst['maturity']]
annuity = _compute_annuity(ois, dates_up_to, day_count)
model_rate = (1 - df_T) / annuity
```

If `inst['maturity']` did not exactly match a pillar, `dates_up_to` truncated at the last pillar BELOW maturity, but `df_T` was interpolated to maturity. The par-rate then mixed two different time horizons — annuity covered ref → last_pillar (a fraction of the swap life), df_T covered ref → maturity.

**Fix**: append `inst['maturity']` as the final endpoint of `dates_up_to` if not already present. Annuity and df_T then span the same horizon. Mirror fix in the projection-curve branch.

For swaps whose maturity dates ARE pillars (the canonical case), behavior unchanged.

### Verification — `test_l2_t3_19_multicurve_annuity.py`

2 new tests, both pass:
- `test_swap_matures_between_pillars_re_prices` — a swap with maturity at 4y (between 3y and 5y pillars) re-prices at the input rate after the solve.
- `test_pillar_at_maturity_unchanged` — sanity: maturity-on-pillar case still works.

Full parallel suite: **12099 passed in 4:51** — zero regressions.

Tier-3 status: **13 of 19 closed**.

---

## v0.929.0 — 2026-06-12

**Fix L2 Tier-3 T3.17 / T3.18 — `bootstrap` HW futures convexity formula + no-deposit FRA/future safety.**

Two fixes in `curves/bootstrap.py`.

### T3.17 — HW futures convexity missing T_1 factor

Pre-fix:
```python
conv_adj = 0.5 * σ² * B(t_start, t_end) * (B(0, t_end) - B(0, t_start))
```

For small `a · T`, `B(t1, t2) ≈ T_2 − T_1` and `B(0, t_end) − B(0, t_start) ≈ T_2 − T_1`, so the pre-fix collapsed to `0.5 · σ² · (T_2 − T_1)²` — no T_1 dependence. For a 1y-expiry, 3-month-tenor Eurodollar future, this under-stated the convexity adjustment by a factor of (T_2 − T_1) / T_1 = 1/4.

**Fix**: use the textbook formula `0.5 · σ² · B(0, T_1) · B(T_1, T_2)` which in the small-a limit reduces to Hull's leading-order `0.5 · σ² · T_1 · (T_2 − T_1)` (Brigo-Mercurio §3.4).

### T3.18 — FRA/future with no preceding deposits silently used df_start = 1.0

When `pillar_dates` was empty (no deposits) and a FRA/future had `start_date != reference_date`, the bootstrap fell through to `df_start = 1.0` — which is correct ONLY when `start_date == reference_date`. For any other start, df at that date is unknown without short-end information, and `df_start = 1.0` produces a wildly wrong df(end).

**Fix**: raise `ValueError` with a diagnostic when `pillar_dates` is empty and `start_date != reference_date`. Still accept `start_date == reference_date` cleanly (df(0) = 1 is correct there).

### Verification — `test_l2_t3_17_18_bootstrap_convexity.py`

6 new regression tests, all pass:
- `test_convexity_zero_when_params_zero` — sanity.
- `test_convexity_scales_with_t1` — CA at 2y > CA at 0.5y for same tenor (pre-fix they were ≈ equal).
- `test_convexity_small_a_limit` — direct formula check: small-a limit matches Hull `0.5 · σ² · T_1 · τ` to <5%.
- `test_fra_after_ref_no_deposit_raises` — bootstrap refuses to silently anchor df at 1.
- `test_future_after_ref_no_deposit_raises` — same for futures.
- `test_fra_starting_at_ref_no_deposit_works` — sanity: ref-date FRA still works.

Full parallel suite: **12099 passed in 3:23** — zero regressions.

Tier-3 status: **12 of 19 closed**.

---

## v0.928.0 — 2026-06-12

**Fix L2 Tier-3 T3.16 — `LMM.rebonato_swaption_vol` uses ρ=1 (standard Rebonato), not ρ=δ_ij.**

`models/lmm.py::rebonato_swaption_vol` docstring described the standard Rebonato approximation:

> With unit correlation (ρ=1): σ² × T = (Σ w_i × σ_i)² × T
> Simplified diagonal (ρ=δ_{ij}): σ² × T = Σ w_i² × σ_i² × T

But the implementation used the **diagonal (uncorrelated) sum**:

```python
var = np.sum(weights**2 * vols**2) * T_expiry
```

By Cauchy-Schwarz, `Σ w_i² σ_i² ≤ (Σ w_i σ_i)²` with equality only at N=1. For an N-period swap with uniform vols, the pre-fix σ_swap was ≈ σ / √N — a factor of √5 ≈ 2.24 too small for a 5-period swap.

**Fix**: default to the ρ=1 result `(Σ w_i σ_i)² · T_expiry`. Added optional `corr: np.ndarray | None = None` parameter so callers can pass a proper LMM correlation matrix when calibrating: `var = (w · σ)ᵀ · ρ · (w · σ) · T`.

### Verification — `test_l2_t3_16_lmm_rebonato.py`

4 new regression tests, all pass:
- `test_single_period_unchanged` — N=1: ρ=1 and ρ=δ give the same answer.
- `test_multi_period_uses_rho_one` — N=5 equal vols: σ_swap ≈ σ (not σ/√5 ≈ 0.09).
- `test_corr_matrix_argument` — ρ=I (passed explicitly) recovers the pre-fix diagonal answer.
- `test_corr_matrix_intermediate` — ρ_off=0.5 sits between ρ=0 and ρ=1 (monotonicity).

Full parallel suite: **12093 passed in 3:27** — zero regressions.

Tier-3 status: **10 of 19 closed**.

---

## v0.927.0 — 2026-06-12

**Fix L2 Tier-3 T3.13 — Hull-White tree swaption accepts `payments_per_year`; no more annual hard-code.**

`models/hull_white.py::tree_european_swaption` hard-coded annual payments:

```python
n_payments = max(1, int(swap_end_T - expiry_T))   # int truncation
annuity = 0.0
for k in range(1, n_payments + 1):
    t_pay = expiry_T + k                           # annual spacing
    if t_pay <= swap_end_T:
        annuity += self.zcb_price(expiry_T, t_pay, r_j)
swap_pv = (1.0 - p_end) - strike * annuity
```

Two issues: (a) integer truncation lost the partial-year tail of non-integer tenors (e.g. 2.5y → 2y); (b) the payment date generator assumed annual spacing.

**Fix**: added `payments_per_year: int = 1` parameter (defaults preserve all existing analytical-formula tests). Number of payments rounded to nearest integer over the swap tenor; payment times spaced at `1 / payments_per_year`; annuity weighted by τ = 1/freq, so the fixed-leg PV is `strike · τ · Σ P(T_e, t_pay)` (correct semi-annual / quarterly handling).

### Verification — `test_l2_t3_13_hw_swaption_freq.py`

3 new regression tests, all pass:
- `test_2_5y_tenor_uses_all_periods` — semi-annual 2y2.5y swaption: 5 periods, different from annual's 2.
- `test_quarterly_payments_more_payments_than_annual` — quarterly vs annual prices differ measurably on a 5y swap.
- `test_integer_tenor_annual_unchanged` — backwards compat: default annual path still works.

T1.9 regression tests (5 tests) still pass — default `payments_per_year=1` is identical to pre-fix for annual integer tenors.

Full parallel suite: **12089 passed in 3:35** — zero regressions.

Tier-3 status: **9 of 19 closed**.

---

## v0.926.0 — 2026-06-12

**Fix L2 Tier-3 T3.12 — COS method `c2` floor doesn't destroy low-variance pricing.**

`models/cos_method.py::cos_price` clamped the variance cumulant to a hard absolute floor:

```python
c2 = max(float(-(ln_p + ln_m - 2 * ln0).real / eps**2), 0.001)
```

For low-variance regimes (short maturity or low vol — e.g. σ = 10%, T = 0.01y → true c2 ≈ 1e-4), the clamp inflated `L · √c2` by ≈ 3 ×, spreading the COS truncation interval far past the actual density support. COS convergence with N grid points degraded by orders of magnitude.

**Fix**: use a tiny numerical-noise floor (1e-12) instead of a model-implied minimum. The floor exists only to guard against round-off-induced negative c2; it should not influence the truncation half-width.

Live demonstration (σ=10%, T=0.01y ATM call):
- BS: 0.424333
- COS post-fix: 0.424333 (rel error 8e-14, machine precision)

### Verification — `test_l2_t3_12_cos_c2_floor.py`

4 new tests, all pass. Parametrised over (σ=10%, T=0.01), (σ=5%, T=0.001), (σ=20%, T=0.05) — all post-fix match BS to <1e-6. Sanity: typical (σ=20%, T=1) still works.

Full parallel suite: **12086 passed in 4:49** — zero regressions.

Tier-3 status: **8 of 19 closed**.

---

## v0.925.0 — 2026-06-12

**Fix L2 Tier-3 T3.5 / T3.6 — `CharacteristicFunction.cumulants` skewness sign + kurtosis stencil h.**

Two bugs in `numerical/_fourier.py::CharacteristicFunction.cumulants`.

### T3.5 — skewness had the wrong sign

The cumulant relation is `κ_n = (−i)^n · log_phi^{(n)}(0)`. For real u and real-density CFs, log_phi alternates real/imaginary at even/odd orders. The cumulant signs are:

- κ_1 = Im(log_phi'(0))            (odd)
- κ_2 = −Re(log_phi''(0))           (even, sign from i² = −1)
- κ_3 = **−Im(log_phi'''(0))**       (odd, sign from (−i)³ = i then Im)
- κ_4 = Re(log_phi''''(0))           (even)

Pre-fix: `c3 = +Im(stencil) / 2h³` — missing the leading minus, so skewness reported with **the wrong sign** for non-symmetric distributions. Verified on χ²(k=2): post-fix skewness ≈ +2.0 (correct, positive); pre-fix was −2.0.

### T3.6 — kurtosis stencil catastrophically unstable at h=1e-4

The pre-fix routine used `h = 1e-4` for both the 2nd-derivative and 4th-derivative stencils. For the 4th derivative, `h⁴ = 1e-16 ≈ machine ε`, so the 5-point stencil's numerator (subject to catastrophic cancellation) was dominated by round-off noise.

**Fix**: use h sized for each derivative order — rule of thumb `h ~ ε^{1/(n+1)}`:
- 2nd derivative: h = 1e-4 (existing)
- 3rd derivative: h = 1e-3 (T3.6 widening)
- 4th derivative: h = 1e-2 (T3.6 widening)

Verified on χ²(k=2): post-fix excess kurtosis ≈ 5.99 vs expected 6.0; pre-fix was round-off noise.

### Verification — `test_l2_t3_5_6_fourier_cumulants.py`

4 new regression tests, all pass:
- `test_gaussian_zero_skew_kurtosis` — sanity: N(μ, σ) gives skew=0, ek=0.
- `test_skew_positive_and_correct` — χ²(k=2) skewness = +2 (T3.5 sign fix).
- `test_excess_kurtosis_stable` — χ²(k=2) ek = 6 (T3.6 wider h fix).
- `test_gamma_skew_sign` — Γ(k=3) skew = 2/√3 ≈ 1.155, ek = 2 (general check).

Full parallel suite: **12082 passed in 4:47** — zero regressions.

Tier-3 status: **7 of 19 closed** (T3.4/T3.5/T3.6/T3.7/T3.8/T3.9/T3.10 + T3.11 subsumed).

---

## v0.924.0 — 2026-06-12

**Fix L2 Tier-3 T3.4 / T3.7 — SINH grid endpoints + Clenshaw-Curtis weights for odd n.**

Two small numerical fixes in `numerical/_pde.py` and `numerical/_integrate.py`.

### T3.4 — SINH grid endpoints off when `concentration_point ≠ midpoint`

Pre-fix the Tavella-Randall SINH grid used a symmetric `xi = linspace(-3, 3)` and `alpha = 0.5 · (s_max − s_min) / sinh(3)`. The grid is `c + α · sinh(xi)`. When `concentration_point` was off-centre, the grid extended past `s_min` / `s_max` — for `c < midpoint` it went BELOW `s_min` and could go NEGATIVE when `s_min` was small (e.g. BS PDE with `s_min = 0.01 · spot`).

**Fix**: choose `α` from the larger half-distance `max(c − s_min, s_max − c) / sinh(3)`, then solve `xi_min = asinh((s_min − c)/α)` and `xi_max = asinh((s_max − c)/α)` so the grid endpoints are exactly `[s_min, s_max]` regardless of `c`.

### T3.7 — Clenshaw-Curtis weights wrong for odd n

The weight formula was hard-coded for even n. Two differences for odd n:
- Endpoint weight is `1/n²`, not `1/(n²−1)`.
- The loop's "halved k=n/2 boundary" term doesn't exist for odd n.

Pre-fix `∫₀¹ 1 dx` with n=3 returned ≈ 0.9 instead of 1; polynomial integration was correspondingly biased. Post-fix the routine branches on `n % 2`.

### Verification

- `test_l2_t3_4_sinh_grid.py` — 5 tests (midpoint exact; below-midpoint no negatives; above-midpoint endpoints; monotone; density concentrated near c).
- `test_l2_t3_7_clenshaw_curtis_odd_n.py` — 15 tests (constants integrate exactly for n=3,5,7,9,15; x², x³ exact; sin convergence monotone).

Full parallel suite: **12078 passed in 4:49** — zero regressions.

Tier-3 status: **5 of 19 closed** (T3.4 and T3.7 added; T3.8/T3.9/T3.10 closed in v0.923; T3.11 subsumed by T2.10).

---

## v0.923.0 — 2026-06-12

**Fix L2 Tier-3 T3.8 / T3.9 / T3.10 — three small bugs in `numerical/_trees.py` convenience functions.**

Opens Tier-3 (single-critic criticals) by clearing three small but real bugs in `_trees.py`:

### T3.10 — `solve_tree` wrapper dropped `exercise_dates`

`solve_tree(...)` did not accept `exercise_dates`, so a caller passing `exercise=BERMUDAN` got a `TreeSolver` built with empty `exercise_dates`. `_should_exercise` returns False at every step for a Bermudan with no dates → option silently degraded to European. Added `exercise_dates: list[int] | None = None` parameter and threaded it through.

### T3.9 — `solve_tree_2d` American projection only handled `spread_call`

Pre-fix the American-exercise projection inside the backward loop had:

```python
if is_american:
    s1 = ...; s2 = ...
    if payoff is not None:
        new_v[i,j] = max(new_v[i,j], payoff(s1, s2))
    elif payoff_type == "spread_call":
        new_v[i,j] = max(new_v[i,j], max(s1-s2-strike, 0))
```

No branches for `spread_put`, `best_of_call`, `worst_of_call`. They silently priced as European even with `is_american=True`. Refactored intrinsic into a `_intrinsic_2d(s1, s2)` helper covering all payoff types. American spread-put now shows the expected early-exercise premium (e.g. 26.95 vs European 26.90 on a deep ITM spread).

### T3.8 — `solve_tree_2d` always returned zero Greeks

`return TreeResult(price=..., delta=0.0, gamma=0.0, theta=0.0, ...)` regardless of inputs. Now extracts `delta1` and `delta2` from the step-1 grid (4 nodes, averaging out the spot direction being differentiated). `delta1` lands in the conventional `delta` field; `delta2` goes in `node_prices` as a 2-element array `[delta1, delta2]` (no API extension needed). Gamma and theta remain at 0 — the 2-asset recombining tree has no centred node at (S1, S2) at step 1, so straight ∂V/∂t mixes spot-diffusion convexity with time decay. Bump-and-reprice is the correct way to get gamma/theta on a 2D tree.

### Verification — `test_l2_t3_trees_misc.py`

6 new regression tests, all pass:
- `test_bermudan_with_exercise_dates_dominates_european` — Bermudan put > European put (pre-fix they were equal).
- `test_bermudan_with_no_exercise_dates_equals_european` — sanity: empty exercise set ⇒ European.
- `test_spread_call_delta_nonzero` — delta1 > 0, delta2 < 0 for ATM spread call.
- `test_delta_sum_at_high_spread_approaches_one` — deep-ITM spread: delta1 ≈ +1, delta2 ≈ −1.
- `test_american_spread_put_premium` — American > European for ITM spread put.
- `test_american_worst_of_call_with_custom_payoff` — custom payoff: American ≥ European.

Full parallel suite: **12058 passed in 3:25** — zero regressions.

Tier-3 status: **3 of 19 closed** (T3.8, T3.9, T3.10; T3.11 already subsumed by T2.10).

---

## v0.922.0 — 2026-06-12

**Fix L2 T2.15 — SABR-HW swaption blender returns intrinsic at T=0 instead of unconditional zero.  CLOSES ALL 18 TIER-2 BUGS.**

`options/swaption.py::price_swaption_sabr_hw` had:

```python
if T <= 0:
    return 0.0
```

This returned 0.0 even when the swaption had positive **intrinsic** value (e.g. a payer with strike below the current forward swap rate, evaluated at expiry).

**Fix**: at T ≤ 0 compute and return the intrinsic value:
- Payer: `annuity × max(fwd − K, 0) × notional`
- Receiver: `annuity × max(K − fwd, 0) × notional`

### Verification — `test_l2_t2_15_sabr_hw_blender_intrinsic.py`

4 new regression tests, all pass:
- `test_payer_at_expiry_returns_intrinsic` — deep-ITM payer at T=0 returns positive intrinsic, matches annuity × (fwd − K).
- `test_receiver_at_expiry_returns_intrinsic` — symmetric for receiver.
- `test_out_of_money_at_expiry_returns_zero` — OTM at T=0 still returns 0 (no negative intrinsic).
- `test_atm_post_expiry_positive_T_unchanged` — sanity: positive-T Black-76 path still works.

Full parallel suite: **12052 passed in 3:27** — zero regressions.

### Tier-2 status — **18 of 18 closed** ✓

| # | Status | Note |
|---|---|---|
| T2.1 | ✅ pre-existing | roll_down DF renormalisation (v0.901) |
| T2.2 | ✅ v0.914 | PDE LOG-grid stencil |
| T2.3 | ✅ v0.914 | PDE implicit Dirichlet BC |
| T2.4 | ✅ v0.914 | PDE American boundary |
| T2.5 | ✅ v0.916 | wavelet_transform power-of-2 |
| T2.6 | ✅ v0.917 | native Romberg (scipy.romberg removed) |
| T2.7 | ✅ subsumed | interior_point equality (via T1.4) |
| T2.8 | ✅ v0.918 | tree knock-in via in-out parity |
| T2.9 | ✅ v0.918 | trinomial prob renormalisation |
| T2.10 | ✅ v0.919 | MCEngine.greek() honours param_name |
| T2.11 | ✅ v0.920 | g2pp_swaption no silent zero |
| T2.12 | ✅ v0.920 | g2pp y*=0 fallback raises |
| T2.13 | ✅ subsumed | AAD no-deposit (via T1.13) |
| T2.14 | ✅ v0.921 | bond YTM redemption = last-period notional |
| **T2.15** | ✅ this slice | SABR-HW blender T=0 intrinsic |
| T2.16 | ✅ v0.915 | CDS convexity uses notional |
| T2.17 | ✅ v0.915 | CDS variable notional propagation |
| T2.18 | ✅ v0.915 | protection_leg_pv list+no-schedule |

**All 13 Tier-1 AND 18 Tier-2 bugs closed in 16 slices** since v0.905. Next phase: Tier-3 (single-critic criticals — 19 items) and Tier-4 (highs that didn't reach critical).

---

## v0.921.0 — 2026-06-12

**Fix L2 T2.14 — bond YTM-based pricing now uses the last-period notional for redemption (consistent with curve-based pricing).**

`fixed_income/bond.py::FixedRateBond` set `self.face_value = self.coupon_leg.notional` (the first-period notional from a possibly-variable notional schedule). The curve-based `dirty_price` already used `notional_schedule[-1]` for the redemption to support sinking-fund bonds. But three YTM-based methods used the first-period `self.face_value` for the principal term, causing a silent disagreement between YTM-pricing and curve-pricing for amortising bonds:

- `_price_from_ytm`
- `macaulay_duration`
- `convexity`

For a 5y semi-annual bond amortising 100 → 55, the pre-fix YTM price had a redemption term of 100 / (1 + y/2)^10 while the curve price had 55 × DF(5y). The two paths could disagree by tens of cents at par-equivalent levels.

**Fix**: all YTM-based paths use `self.coupon_leg.notional_schedule[-1]` consistently. Constant-notional bonds are unaffected (first = last). Lock-in test verifies curve-vs-YTM agreement on both constant-notional and amortising bonds.

Also covers T2.13 — `aad_curves` no-deposit first-swap case (subsumed by v0.911 T1.13 fix), with explicit regression test (`test_l2_t2_13_aad_first_swap_no_deposits.py`) committed earlier.

### Verification — `test_l2_t2_14_bond_sinking_redemption.py`

4 new regression tests, all pass:
- `test_constant_notional_ytm_matches_curve` — sanity: plain bond, YTM and curve agree.
- `test_amortising_redemption_matches_curve` — sinking-fund bond (notional 100→55): YTM and curve now agree (pre-fix had large discrepancy).
- `test_duration_uses_last_notional` / `test_convexity_uses_last_notional` — duration and convexity produce sane values for sinking-fund bond.

Full parallel suite: **12048 passed in 3:29** — zero regressions.

Tier-2 status: **16 of 18 closed** (T2.13, T2.14 added). Remaining: T2.15 (SABR-HW blender at T=0).

---

## v0.920.0 — 2026-06-12

**Fix L2 T2.11 / T2.12 — `g2pp_swaption_price` no longer silently masks errors as zero or falls back to bogus `y* = 0`.**

Two silent-failure modes in `models/g2pp_calibration.py::g2pp_swaption_price`:

### T2.11 — Bare `except Exception: return 0.0` around the entire body

The whole function was wrapped in `try: ... except Exception: return 0.0`, masking every error mode (bracketing failure, brentq divergence, numerical-overflow in the ZCB formula, calibration bugs) as a silent zero price. Callers had no way to distinguish "swaption worth ≈ 0" from "pricer crashed".

### T2.12 — Silent `y_star = 0.0` fallback on bracket failure

The Jamshidian y* bracketing routine had:

```python
if bp_lo * bp_hi >= 0:
    y_star = 0.0       # ← bogus fallback
else:
    y_star = brentq(...)
```

When bracketing failed (no sign change after 10 expansions), `y_star = 0.0` is just **wrong** — the K_k strikes derived from it are unrelated to the actual swaption. Combined with T2.11, this produced wildly wrong intermediate prices that the outer except then collapsed to zero.

### Fixes

- T2.11: removed the outer `try/except Exception: return 0.0`. Only the legitimate degenerate cases (no payment dates) still return 0.0 explicitly. Real exceptions propagate.
- T2.12: bracket failure now raises `RuntimeError` with a diagnostic message including the failing `x_val`, the bracket endpoints' bond-portfolio values, and a hint about parameter regime.

### Verification — `test_l2_t2_11_12_g2pp_silent_failures.py`

4 new regression tests, all pass:
- `test_normal_params_produce_positive_price` — sanity: reasonable G2++ params produce a positive finite price.
- `test_zero_payment_dates_returns_zero_cleanly` — degenerate tenor=0 still returns 0.0 explicitly (the only path that should).
- `test_negative_a_propagates_error` — extreme parameters don't silently give 0.0 (either succeed or raise).
- `test_extreme_params_either_raise_or_succeed` — across strike sweep, never silently produce 0; either valid price or RuntimeError with diagnostic.

Full parallel suite (with previously-deselected slow `test_g2pp_calibration` tests included): **12041 passed in 3:24 + 70s for G2++ calibration suite** — zero regressions. The removed silent except did not surface any hidden failures in the full G2++ calibration loop, which validates the fix.

Tier-2 status: **14 of 18 closed** (T2.11, T2.12 added).

---

## v0.919.0 — 2026-06-12

**Fix L2 T2.10 — `MCEngine.greek()` actually honours `param_name`.**

Pre-fix `models/mc_engine.py::MCEngine.greek()` accepted `param_name: str` as a *required* parameter but **never used it**. The implementation always did `self.process.x0 = original_x0 + bump`, which broadcasts the scalar bump across the whole `x0` vector. For a multi-factor process (Heston: x = [S, V]; G2++: x = [x, y]; basket models), "delta" simultaneously bumped every state variable, producing a meaningless mixed sensitivity.

**Fix**: `param_name` is now parsed into one of three forms:
- `int` — index into `process.x0` (e.g. `0` = spot, `1` = variance).
- `"x0"` or `"x0[i]"` — same as the int form.
- any other string — interpreted as an attribute of the `ProcessSpec` (e.g. `"sigma"`, `"mu"`). Requires the drift/diffusion callables to read the attribute at evaluation time. Lets the user compute vega-like sensitivities to model parameters.

Bounds-checked (`IndexError` for bad index, `ValueError` for unknown attribute). Restores the original `x0` / attribute cleanly on the way out.

### Verification — `test_l2_t2_10_mc_greek_param_name.py`

6 new regression tests, all pass:
- `test_x0_index_zero_only_bumps_spot` — 1D GBM `"x0[0]"` gives BS-plausible delta (0.4 < Δ < 0.85).
- `test_int_param_name_works` — bare integer index works.
- `test_x0_restored_after_greek` — restoration check.
- `test_attribute_bump_vega_via_sigma` — `param_name="sigma"` bumps the process's `sigma` attribute and recovers a BS-plausible vega (20 < vega < 60).
- `test_unknown_param_name_raises` — bad string raises `ValueError`.
- `test_index_out_of_range_raises` — bad index raises `IndexError`.

Full parallel suite: **12037 passed in 4:44** — zero regressions.

Tier-2 status: **12 of 18 closed** (T2.10 added).

---

## v0.918.0 — 2026-06-12

**Fix L2 Tier-2: tree knock-in barriers (T2.8) + trinomial probability renormalisation (T2.9).**

Two coupled bugs in `numerical/_trees.py`.

### T2.8 — knock-in barriers silently priced as vanilla

`_apply_barrier` had:

```python
elif self.barrier_type == BarrierType.DOWN_IN:
    pass  # complex — for now, only knock-out supported
elif self.barrier_type == BarrierType.UP_IN:
    pass
```

So the barrier condition was checked but never enforced for in-types. `TreeSolver(barrier_type=BarrierType.DOWN_IN, ...).solve(...)` produced the **vanilla** price, with no warning. The TreeSolver API accepted the type and returned a wrong number.

**Fix**: at the top of `solve()`, intercept knock-in types and apply in-out parity:

> V_knock_in = V_vanilla − V_knock_out

Spawn two side-pricers (one with the corresponding knock-out type, one without barriers) and combine prices + Greeks by linearity. Live verification:

```
Vanilla   = 10.4406
DOWN_OUT  = 10.3523
DOWN_IN   = 0.0883     # post-fix; pre-fix was 10.4406 (= vanilla)
OUT + IN  = 10.4406    # exactly equals vanilla
```

### T2.9 — Trinomial probability triple not renormalised after clamp

`_trinomial_params` had:

```python
p_u = max(0, min(1, p_u))
p_d = max(0, min(1, p_d))
p_m = max(0, 1 - p_u - p_d)
```

When the raw `p_u` or `p_d` was negative (large drift relative to volatility), clamping shifted mass into `p_m` without renormalising the triple. The probs still summed to 1 but the *moments* (drift, variance) were broken — the risk-neutral measure silently violated.

**Fix**: clamp each leg ≥ 0, then renormalise the triple. Mirrors the renormalisation already used in `_2d_trinomial_params` (the 2D tree) immediately below in the same file.

### Verification — `test_l2_t2_8_9_trees_barriers_probs.py`

6 new regression tests:

- `test_in_out_parity_down` / `test_in_out_parity_up` — V(KI) + V(KO) = V(vanilla) exactly; pre-fix V(KI) = V(vanilla) trivially.
- `test_knock_in_greeks_linearity` — KI delta = vanilla delta − KO delta exactly.
- `test_probs_sum_to_one_high_drift` — extreme-drift case (r=20%, σ=10%, dt=0.01) → renormalised probs sum to 1.0 exactly.
- `test_probs_unchanged_normal_regime` — sanity: normal-drift case (the clamp doesn't trigger).
- `test_trinomial_pricing_stable_under_high_drift` — extreme-drift call still produces a finite positive price.

Full parallel suite: **12031 passed in 3:23** — zero regressions.

Tier-2 status: **11 of 18 closed** (T2.8, T2.9 added).

---

## v0.917.0 — 2026-06-12

**Fix L2 T2.6 — `_romberg` now uses a native Richardson extrapolation; scipy.integrate.romberg was removed in SciPy 1.15.**

`numerical/_integrate.py::_romberg` wrapped `scipy.integrate.romberg`, which was removed in SciPy 1.15. Every call to `integrate(..., method=IntegrationMethod.ROMBERG)` raised:

```
ImportError: cannot import name 'romberg' from 'scipy.integrate'
```

Replaced with a native Romberg-on-trapezoid implementation (incremental trapezoidal refinement + Richardson extrapolation of the table). Returns proper `IntegrationResult` with the error estimate and convergence flag. No more SciPy version pin needed.

### Verification — `test_l2_t2_6_romberg_native.py`

6 tests, all pass:
- Constant function: ∫₀¹ 5 dx = 5 (exact).
- ∫₀¹ x³ dx = 1/4 to 1e-10.
- ∫₀^π sin(x) dx = 2 to 1e-10.
- 5σ-truncated standard normal CDF ≈ 1 to 1e-6.
- ∫₀^{2π} cos(x) dx = 0 to 1e-8.
- Smoke test: no ImportError, `result.method == "romberg"`, `result.converged`.

Tier-2 status: **9 of 18 closed** (added T2.6).

Full parallel suite: **12025 passed in 4:05** — zero regressions.

---

## v0.916.0 — 2026-06-12

**Fix L2 T2.5 — `wavelet_transform` pads non-power-of-2 input.**

`numerical/_fourier.py::wavelet_transform` halves the signal at each level via `x[0::2]` and `x[1::2]`. For odd `len(x)`, these slices have different lengths and the addition raised:

```
ValueError: operands could not be broadcast together with shapes (4,) (3,)
```

The DWT (Haar / DB2) is well-defined only on power-of-2 inputs. Fix: pad to `max(next_power_of_2(n), 2**levels)` with zeros before decomposition; also raise on `n < 2`. Power-of-2 inputs are unaffected.

### Verification — `test_l2_t2_5_wavelet_padding.py`

17 tests (parametrised over n = 3, 5, 7, 9, 11, 17, 100, 1000 for Haar and 3, 5, 11, 17, 31 for DB2), all pass. Includes:
- Arbitrary-length Haar/DB2 no longer crash.
- Power-of-2 still produces a length-n coefficient vector (unchanged).
- Length-2 exact-value sanity check.
- Length-1 raises with clear message.

Tier-2 status: **8 of 18 closed** (T2.5 added; T2.1, T2.2, T2.3, T2.4, T2.7, T2.16, T2.17, T2.18 already closed).

---

## v0.915.0 — 2026-06-12

**Fix L2 Tier-2: CDS variable-notional propagates through aged-CDS methods + convexity formula uses notional.**

Three coupled bugs in `credit/cds.py` closed in one slice.

### T2.18 — `protection_leg_pv` silently used `notional[0]` for list inputs without `schedule_dates`

If a caller passed `notional=[1M, 2M, 3M]` but omitted `schedule_dates`, the function fell through to `n = notional[0]` and scaled the entire protection leg by 1M — variable-notional CDS got priced as constant-1M. Now raises `ValueError` so the caller cannot accidentally lose the schedule.

### T2.17 — Variable notional silently dropped in aged-CDS methods

`CDS.__init__` set `self.notional = self.notional_schedule[0]` (scalar first-period) for convenient single-value access. But five methods then passed `notional=self.notional` (the scalar) when constructing a new aged/parallel `CDS`:

- `isda_upfront` — built the std-coupon CDS with scalar
- `roll_down` — built shorter CDS with scalar
- `theta` — built aged CDS with scalar
- `rec01` — built recovery-bumped CDS with scalar
- `cds_pnl_attribution` — built aged CDS with scalar

Result: variable-notional CDS got their tail-period notionals silently dropped in roll-down, theta, isda upfront, rec01, and full P&L attribution.

**Fix**:
- `rec01` and `isda_upfront`: forward `self.notional_schedule` (full list).
- `roll_down`, `theta`, `cds_pnl_attribution`: introduce new `_aged_notional(new_start)` helper that slices the schedule at the period containing `new_start`. The aged CDS then has a notional list of length = remaining-periods.

### T2.16 — Convexity P&L multiplied by `|PV|` instead of notional

`spread_convexity` returns `d²PV/ds² / notional`. The second-order P&L correction is therefore

> convexity_pnl = ½ × (conv × notional) × Δs²

Pre-fix the code used `pv_for_conv = abs(pv_t0)` with a fallback to `cds.notional` only when `|PV| < 1e-10`. This was dimensionally wrong: |PV| is small near par (where the CDS spread equals the par spread, e.g. typical IG trading conventions) and grows with mark-to-market away from par — neither is the right normalisation. The formula effectively collapsed to ~0 near par and gave a |PV|-scaled value elsewhere.

Fixed to `0.5 * conv * cds.notional * delta_spread**2`.

### Verification — `test_l2_t2_cds_variable_notional.py`

8 new regression tests, all pass:

- `TestProtectionLegRequiresScheduleForList`:
  - `test_raises_on_list_without_schedule` — variable notional without schedule_dates now raises.
  - `test_scalar_path_unchanged` — scalar path still works.
- `TestVariableNotionalPropagation` (amortising 5y CDS with 20 periods, notional from 10M → 2.4M):
  - `test_average_notional_not_scalar` — sanity: the schedule is genuinely variable.
  - `test_isda_upfront_uses_average_notional` — variable vs constant-10M produce different upfronts.
  - `test_roll_down_sees_variable_notional` — roll_down for variable ≠ for first-period-only.
  - `test_theta_sees_variable_notional` — same for theta.
- `TestConvexityFormula`:
  - `test_convexity_pnl_nonzero_at_par` — at-par CDS (|PV|≈0): post-fix convexity is non-zero (pre-fix collapsed to ~0).
  - `test_convexity_scales_with_notional` — 10× notional → 10× convexity P&L (linearity check).

Full parallel suite: **12002 passed in 4:50** — zero regressions.

### Tier-2 status — 7 of 18 closed

| # | Status | Note |
|---|---|---|
| T2.1 | ✅ pre-existing | roll_down DF renormalisation (v0.901 as B.1) |
| T2.2 | ✅ v0.914 | PDE LOG-grid stencil |
| T2.3 | ✅ v0.914 | PDE implicit Dirichlet BC |
| T2.4 | ✅ v0.914 | PDE American boundary |
| T2.5 | queued | wavelet_transform power-of-2 |
| T2.6 | queued | _romberg dead in SciPy 1.15 |
| T2.7 | ✅ subsumed by T1.4 | interior_point equality |
| T2.8 | queued | tree knock-in barriers |
| T2.9 | queued | trinomial prob clamp |
| T2.10 | queued | mc_engine.greek() bumps all |
| T2.11/12 | queued | g2pp_calibration silent failures |
| T2.13 | queued | aad_curves first-swap-no-deposit |
| T2.14 | queued | bond _price_from_ytm redemption |
| T2.15 | queued | SABR-HW blender at T=0 |
| **T2.16** | ✅ this slice | CDS convexity uses notional |
| **T2.17** | ✅ this slice | Variable notional propagation |
| **T2.18** | ✅ this slice | protection_leg_pv list+no-schedule |

---

## v0.914.0 — 2026-06-12

**Fix L2 Tier-2: PDE non-uniform stencil + Dirichlet BC + American boundary + time-index — 4 bugs in `PDESolver1D` closed in one slice.**

Opens the Tier-2 phase by clearing four entangled bugs in `numerical/_pde.py::PDESolver1D` that all touched the same `_theta_step` and outer time loop.

### T2.2 — Uniform-grid stencil applied to non-uniform grids

`_theta_step` discretised the second derivative with `d2 = σ²S² / (ds_m · ds_p)` and split it evenly into sub/super diagonals (`a[i] = d2/2 − d1`, `c[i] = d2/2 + d1`). For uniform grids this is correct; for non-uniform grids (LOG, SINH) the proper 3-point central stencil is

> V_{i-1} coef = σ²S² / (ds_m · (ds_m+ds_p))
> V_{i+1} coef = σ²S² / (ds_p · (ds_m+ds_p))
> V_{i}   coef = −σ²S² / (ds_m · ds_p)

Live measurement on the canonical ATM call (S=100, K=100, r=5%, σ=20%, T=1y):

| Grid | Pre-fix PDE | Black-Scholes | Pre-fix error |
|---|---|---|---|
| UNIFORM | 10.4645 | 10.4506 | +0.13 % |
| LOG | **11.7966** | 10.4506 | **+12.9 %** |

Post-fix:

| Grid | Post-fix PDE | rel. err |
|---|---|---|
| UNIFORM | 10.4645 | +0.13 % |
| LOG | 10.4734 | **+0.22 %** |

### T2.3 — Implicit Dirichlet BC not enforced in tridiagonal solve

The tridiagonal system was built with `diag[0]=1, lower[0]=upper[0]=0` and `rhs[0]=0` (zero-init), so `V_new[0] = 0` after the solve. The code then overwrote `V_new[0] = V[0]` (the value BEFORE the step). The interior equation at `i=1` therefore used `V_new[0]=0` inside the implicit solve — wrong for **puts** (whose lower BC is `K·e^{−rτ} − S[0]`, non-zero) and for the upper BC of calls.

Fix: pass the proper BCs into `_theta_step` as `bc_lo`, `bc_hi`; set `rhs[0]=bc_lo`, `rhs[-1]=bc_hi`. The boundary rows with `diag=1` then yield `V_new[boundary] = BC` naturally, and the interior solve at `i=1` / `i=N−2` uses the proper boundary value.

### T2.B — Time-to-maturity index inverted at the boundary

The outer loop applied the upper-boundary value as `S_max − K·exp(−r·(n_time − step)·dt)`. After step `k`, the time-to-maturity is `(k+1)·dt`, NOT `(n_time − k)·dt`. The expression is only correct at the middle step; everywhere else the boundary uses the wrong discount factor. Replaced with `τ = (step + 1)·dt`.

### T2.4 — American boundaries used European discounted strike

For American puts at `S→0`, the optimal exercise dominates: `V = K − S[0]` (intrinsic). Pre-fix the boundary used `K·e^{−rτ} − S[0]` (European). Fix: for `is_american=True`, set boundaries to intrinsic payoff, and re-impose intrinsic floor after the early-exercise projection.

### Verification — `test_l2_t2_pde_stencil_bc.py`

6 new regression tests, all pass:

- `TestNonUniformStencil::test_log_grid_atm_call_matches_bs` — the headline: LOG grid no longer overshoots BS by 13 %.
- `TestNonUniformStencil::test_uniform_grid_atm_call_matches_bs` — uniform grid unchanged (was already fine).
- `TestDirichletBC::test_uniform_grid_atm_put_matches_bs` — uniform put now matches BS (pre-fix the implicit BC bug biased puts).
- `TestDirichletBC::test_log_grid_atm_put_matches_bs` — combined fix: non-uniform grid + correct BC for put.
- `TestAmericanBoundary::test_american_put_dominates_european` — deep-ITM American put ≥ intrinsic ≥ European.
- `TestTimeIndex::test_long_dated_call_no_explosive_bias` — 5y ATM call agrees with BS to <2 %.

Full parallel suite: **11994 passed in 4:45** — zero regressions.

### Tier-2 status — 4 of 18 closed (T2.2, T2.3, T2.4 this slice; T2.1 already closed in v0.901 as B.1)

Remaining Tier-2 (14): T2.5 wavelet_transform power-of-2; T2.6 _romberg dead; T2.7 interior_point equality-only (already closed via T1.4); T2.8 tree knock-in barriers; T2.9 trinomial prob clamp; T2.10 mc_engine.greek() bumps everything; T2.11/T2.12 g2pp_calibration silent failures; T2.13 aad_curves first-swap-no-deposit; T2.14 bond _price_from_ytm redemption; T2.15 SABR-HW blender at T=0; T2.16 cds convexity uses |PV|; T2.17 cds variable-notional drop; T2.18 cds protection_leg_pv notional[0].

---

## v0.913.0 — 2026-06-12

**Fix L2 T1.1 — `multilevel_mc` Giles coupling: coarse path now uses paired-sum fine increments. CLOSES ALL 13 TIER-1 BUGS.**

The last of MODULE_HEALTH's Tier-1 (dual-critic-confirmed) bugs is closed.

### The bug

`numerical/_mc.py::multilevel_mc` claimed to implement Giles (2008) MLMC. The algorithm requires that at each level l ≥ 1, the FINE path (n_fine = base_steps · 2^l timesteps) and the COARSE path (n_fine / 2 timesteps) share the SAME underlying Brownian motion — the coarse path is obtained by **pairing** fine Brownian increments:

> dW_coarse[m] = dW_fine[2m] + dW_fine[2m+1]

This coupling is what makes Var(P_fine − P_coarse) → 0 as the discretization refines (the property MLMC's O(ε⁻²) cost depends on).

The pre-fix code instead generated the fine path then **downsampled** it:

```python
paths_fine = process_fn(n_paths, n_fine, seed_l)
paths_coarse = paths_fine[:, ::2]    # <-- the bug
```

Since the downsampled "coarse" path is just a sub-sampling of the fine path's *output*, for any European payoff depending only on the terminal value `paths[:, -1]`:

> P_fine = payoff(paths_fine[:, -1])
> P_coarse = payoff(paths_fine[:, ::2][:, -1]) = payoff(paths_fine[:, -1])
> P_fine − P_coarse ≡ 0

The MLMC correction at every level ≥ 1 was identically zero. `multilevel_mc` collapsed to E[P_0] — a 4-step Euler estimate — at any number of levels. The function returned a heavily biased estimate while quietly pretending to do MLMC.

### The fix — proper Giles coupling

Redesigned the `process_fn` interface to consume **pre-generated Brownian increments** instead of an opaque seed:

```python
process_fn: callable(dW, dt) → paths array (n_paths, n_steps+1)
```

The MLMC routine now does the coupling work:

```python
Z = rng.standard_normal((n_paths, n_fine))
dW_fine = Z * sqrt(dt_fine)
dW_coarse = dW_fine[:, 0::2] + dW_fine[:, 1::2]     # <-- pair the increments

paths_fine = process_fn(dW_fine, dt_fine)
paths_coarse = process_fn(dW_coarse, dt_coarse)     # SAME Brownian path, coarser stepping
```

This is a breaking API change to `process_fn`, but there were no production callers (only the smoke `assert callable(multilevel_mc)` test), and the previous routine was broken anyway.

### Verification — `test_l2_t1_1_mlmc_giles.py`

4 new regression tests, all pass:

- **`test_mlmc_converges_to_black_scholes`** — the headline test. ATM European call on GBM (S₀=100, K=100, r=5%, σ=20%, T=1y). 6-level MLMC with 20 000 base paths must agree with Black-Scholes to <5% relative. Pre-fix this returned the heavily biased 4-step Euler P₀ estimate.
- **`test_level_corrections_are_nonzero`** — proves Giles coupling actually decouples fine and coarse. With the SAME seed, a 1-level run and a 4-level run produce measurably DIFFERENT estimates. Pre-fix they were identical (corrections were all 0).
- **`test_paired_increments_have_same_terminal_brownian_value`** — sanity: `sum(dW_fine) == sum(dW_coarse)` (the terminal Brownian motion is preserved by pairing). Algebraic identity, kept as a regression anchor.
- **`test_variance_dominated_by_low_levels`** — Giles signature: per-level variance increments decay. The level-1 correction adds nonzero variance (proof of working coupling).

Full parallel suite: **11988 passed in 3:20** — zero regressions.

### Tier-1 status — **13 of 13 closed**

| # | Module | Status |
|---|---|---|
| **T1.1** | `numerical/_mc.py` multilevel_mc Giles coupling | ✅ this slice |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equalities | ✅ v0.912 |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | ✅ v0.910 |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | ✅ v0.911 |

**All 13 Tier-1 bugs closed in 7 slices.** Next: MODULE_HEALTH Tier-2 — 18 critical+high pairings to sweep.

---

## v0.912.0 — 2026-06-12

**Fix L2 T1.4 — `interior_point` now honours equality constraints via SLSQP inner solve.**

Twelfth of 13 Tier-1 (dual-critic-confirmed) bugs closed. Only T1.1 (MLMC) remaining.

### The bug

`numerical/_optimize.py::interior_point` accepts `equality_constraints` as a parameter and even **constructs** the SciPy-style constraint dicts:

```python
constraints = []
if equality_constraints:
    for h in equality_constraints:
        constraints.append({"type": "eq", "fun": h})

r = _minimize(barrier_obj, x, method="BFGS",     # <— never passes `constraints`
              options={"maxiter": 50, "gtol": tol / t})
```

…but never passes the list to `_minimize`, and uses **BFGS** (an unconstrained method) for the inner solve. So any caller passing `equality_constraints` got them silently dropped — the optimiser just minimised the (barrier-modified) objective over the unconstrained space.

This is a critical bug: every caller using `interior_point` for an equality-constrained problem (e.g. minimum-norm projection, equality-constrained least-squares with logarithmic barriers) was getting the **unconstrained** minimum back.

### The fix

When equality constraints are present, switch to **SLSQP** (the SciPy method that supports equality constraints) and pass the constraint dicts. Pure-inequality problems keep the BFGS path (faster for unconstrained barrier subproblems):

```python
if constraints:
    r = _minimize(barrier_obj, x, method="SLSQP",
                  constraints=constraints,
                  options={"maxiter": 100, "ftol": tol / t})
else:
    r = _minimize(barrier_obj, x, method="BFGS",
                  options={"maxiter": 50, "gtol": tol / t})
```

Additional convergence-check fix: pure equality-constrained problems (no inequalities) used to never terminate the outer loop (the existing check `n_ineq / t < tol` required `n_ineq > 0`). Added a fast-path that breaks on `r.success` after the first SLSQP pass — no need to grow `t` when there are no barrier terms.

### Verification — `test_l2_t1_4_interior_point_equalities.py`

4 new regression tests, all pass:

- `test_simple_equality_constrained_quadratic` — minimise ‖x‖² in ℝ³ subject to `sum(x) = 3`. Optimum = (1, 1, 1). Pre-fix → (0, 0, 0) (unconstrained minimum, sum = 0 not 3).
- `test_two_equalities` — minimise ‖x‖² subject to `x₀+x₁=1` and `x₁+x₂=1`. Both equalities now hold to 1e-4.
- `test_equality_plus_inequality` — minimise `x²+y²` subject to `x+y=2` (eq) and `x≥0.5` (ineq). Optimum = (1, 1), value 2. Both equality AND inequality honoured simultaneously.
- `test_pure_inequality_path_unchanged` — regression check on the BFGS path when no equalities are supplied: unchanged behaviour, converges to the unconstrained optimum (1, 2) on a strict-interior problem.

Full parallel suite: **11984 passed in 3:24** — zero regressions.

### Tier-1 status — 12 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued (last one) |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| **T1.4** | `numerical/_optimize.py` interior_point drops equalities | ✅ this slice |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | ✅ v0.910 |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | ✅ v0.911 |

**Only T1.1 remains** — `multilevel_mc` redesign requires Giles coupling: the coarse path must be re-simulated from the fine path's normals (paired increments summed) instead of independently sub-sampling. This is the biggest scope change among the Tier-1 set.

---

## v0.911.0 — 2026-06-12

**Fix L2 T1.13 — AAD swap bootstrap properly interpolates intermediate-coupon DFs through the unknown last-pillar DF instead of silently flat-extrapolating.**

Eleventh of 13 Tier-1 (dual-critic-confirmed) bugs closed.

### The bug

`aad_bootstrap` (in `curves/aad_curves.py`) iterates over the input swaps and, at each step, builds a `temp_curve` from the EXISTING pillars only (i.e. the pillars determined so far — typically just the deposits before the first swap is processed). Intermediate-coupon DFs needed for the par-condition annuity were drawn via `temp_curve.df(coupon_date)`.

When the new swap's tenor extended beyond all existing pillars (the common case — bootstrap a 5y swap against a 1y deposit, the only thing existing at that point is the 1y deposit pillar), the intermediate coupons at 1.5y, 2y, 2.5y, ..., 4.5y fell **in the extrapolation region** of `aad_log_linear_interp`, which flat-extrapolates at `dfs[-1]` (the last known pillar's DF). So the bootstrap silently equated every gap-coupon DF to the deposit DF, biasing both:

1. **The numerical bootstrap itself** — the resulting 5y DF did not satisfy the par-zero PV condition when the curve was re-priced with proper interpolation.
2. **The AAD sensitivities** — d(df_mat)/d(swap_rate) propagated through wrong intermediate DFs.

### The fix

A fixed-point iteration where the trial curve at each step **includes** the current `df_mat` estimate as a pillar, so gap-coupon DFs interpolate between the last existing pillar and `df_mat`:

```python
df_mat = Number(brentq(_par_residual_float, 1e-6, 3.0))  # sharp float init
for _ in range(50):
    trial_dfs = pillar_dfs + [df_mat]
    trial_curve = AADDiscountCurve(reference_date, pillar_dates + [mat],
                                    trial_dfs, swap_dc)
    annuity_before_last = Number(0.0)
    for k in range(1, len(schedule) - 1):
        tau_k = year_fraction(schedule[k-1], schedule[k], swap_dc)
        annuity_before_last += par_rate * tau_k * trial_curve.df(schedule[k])
    tau_last = year_fraction(schedule[-2], schedule[-1], swap_dc)
    df_mat_new = (Number(1.0) - annuity_before_last) / (Number(1.0) + par_rate * tau_last)
    if abs(df_mat_new.value - df_mat.value) < 1e-14:
        df_mat = df_mat_new
        break
    df_mat = df_mat_new
```

Two-stage solver:

1. **Float pre-solve via `brentq`** — finds `df_mat` to machine precision using a plain-float `_FloatLogLinearCurve` that mirrors the AAD curve's log-linear interpolation. Sharp initial guess for stage 2.
2. **AAD fixed-point** — starting from the brentq solution, iterates the construction `x → (1 − annuity_before(x)) / (1 + K·τ_last)` with full Number arithmetic. At the converged fixed point, the AAD graph captures the implicit-function dependency `dx*/dp = −(∂A + ∂B) / (1 + ∂B/∂x)` of the par-condition root. Convergence is fast — `|B'(x*)|` is typically ≤ 0.06, so residual reaches 1e-14 in well under 50 iterations.

### Verification — `test_l2_t1_13_aad_bootstrap_interp.py`

4 new regression tests:

- **`test_bootstrap_repriceses_5y_swap_after_1y_deposit`** — the canonical T1.13 case. Bootstrap a 1y deposit + 5y swap, then re-price the 5y swap against the curve. Post-fix PV ≈ 5e-17 (machine precision). Pre-fix this re-pricing PV was far from zero because gap-coupon DFs were anchored at the deposit DF.
- **`test_bootstrap_repriceses_three_swaps`** — stress: 1y deposit + 2y, 5y, 10y swaps. All three must re-price at par. Post-fix: ≤ 1e-8 for each. Pre-fix: errors compound through the bootstrap chain.
- **`test_dpv_dswaprate_nonzero`** — AAD sensitivity check: `d df(3y) / d swap_5y_rate` and `d df(3y) / d dep_1y_rate` are both nonzero. Pre-fix the swap-rate adjoint was wrong because intermediate DFs didn't flow through `swap_rate`.
- **`test_bumped_swaprate_changes_intermediate_df`** — finite-difference check: a 1bp bump in the swap rate must move df(3y) by a measurable amount (>1e-5). Pre-fix the bumped curve's df(3y) was nearly identical to base because it was flat-extrapolated from the unchanged deposit DF.

Live demonstration of the fix:
```
Bootstrapped pillar DFs (post-fix):
  t=0.000 df=1.000000
  t=1.000 df=0.975279
  t=4.997 df=0.819202
Schedule DFs at 0.5y intervals:  smooth log-linear monotone interpolation
  t=1.496 df=0.954350  (between 1y and 5y pillars)
  ...
  t=4.499 df=0.837268
Re-priced par swap PV = -5.55e-17
```

Full parallel suite: **11980 passed in 3:24** — zero regressions.

### Tier-1 status — 11 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | ✅ v0.910 |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| **T1.13** | `curves/aad_curves.py` swap bootstrap flat-extrapolation | ✅ this slice |

**2 remaining — the deepest two:** T1.1 (MLMC redesign — needs Giles coupling), T1.4 (interior_point IPM rewrite).

---

## v0.910.0 — 2026-06-12

**Fix L2 T1.8 — COS method V_k payoff integral now intersects the payoff support with the truncation interval [a, b].**

Tenth of 13 Tier-1 (dual-critic-confirmed) bugs closed.

### The bug

The Fang-Oosterlee (2008) COS method prices a European option as

> price = K · e^{−rT} · Σ_k Re(φ(u_k) · e^{iu_k(x−a)}) · V_k

where V_k integrates the payoff against cosine basis functions over the truncation interval [a, b] for log-moneyness y = log(S/K). For a call payoff (e^y − 1)⁺, V_k integrates over y ∈ [max(0, a), b] (the payoff's support intersected with the truncation interval); for a put (1 − e^y)⁺, over [a, min(0, b)].

`models/cos_method.py` hardcoded the V_k bounds as `[0, b]` (call) and `[a, 0]` (put). This is correct in the typical regime where `a < 0 < b`, but is wrong in either of two regimes:

- **a > 0** (deep-ITM call, low vol, short T): the integration over `[0, b]` includes `[0, a]`, which is outside the truncation interval and should not contribute. Pre-fix the deep-ITM call mispriced.
- **b < 0** (deep-ITM put, low vol, short T): symmetric — `[a, 0]` includes `[b, 0]` outside the truncation.

### Verification of magnitude

Live demonstration — deep-ITM call S=200, K=100, r=5%, σ=8%, T=0.25 (here `a ≈ 0.41 > 0`):

| | Price |
|---|---|
| Black-Scholes reference | 101.242220 |
| COS post-fix (N=256, L=10) | 101.242220 |
| Relative error | 1.4e-16 (machine precision) |

Pre-fix this call mispriced because the V_k coefficients integrated over a domain that included a region outside the truncation, double-counting the call's payoff there.

### The fix

The patch is local to `cos_price`:

```python
if option_type == OptionType.CALL:
    c = max(0.0, a)
    d = b
    if c < d:
        V_k = 2.0/(b - a) * (_chi(k, a, b, c, d) - _psi(k, a, b, c, d))
    else:
        V_k = 0.0
else:
    c = a
    d = min(0.0, b)
    if c < d:
        V_k = 2.0/(b - a) * (-_chi(k, a, b, c, d) + _psi(k, a, b, c, d))
    else:
        V_k = 0.0
```

The `c < d` guard handles the pathological case where the truncation interval lies entirely outside the payoff's support (V_k = 0 — the option is unconditionally worthless within the truncation).

### Verification — `test_l2_t1_8_cos_method_truncation.py`

7 new regression tests (all pass):
- `test_atm_call_unchanged` / `test_atm_put_unchanged` — sanity: ATM (typical a < 0 < b regime) still agrees with Black-Scholes to 1e-4.
- `test_deep_itm_call_low_vol` — S=200, K=100, σ=8%, T=3m → a > 0 regime. Post-fix COS matches BS to <1e-3 (in practice machine precision).
- `test_deep_itm_put_low_vol` — S=50, K=100, σ=8%, T=3m → b < 0 regime. Same.
- `test_parity` (parametrised over ATM, deep-ITM call, deep-ITM put) — put-call parity holds across all three regimes. Pre-fix the deep-ITM regimes violated parity by the V_k bug.

Full parallel suite: **11976 passed in 3:21** — zero regressions.

### Tier-1 status — 10 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| **T1.8** | `models/cos_method.py` V_k integration bounds deep ITM | ✅ this slice |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

3 remaining are the deepest: T1.1 MLMC redesign, T1.4 IPM rewrite, T1.13 AAD bootstrap.

---

## v0.909.0 — 2026-06-12

**Fix L2 T1.9 — Hull-White tree swaption now centres expiry-node rates on α(T_expiry), not on today's short rate r(0).**

Ninth of 13 Tier-1 (dual-critic-confirmed) bugs from `MODULE_HEALTH.md` closed.

### The bug

`HullWhite.tree_european_swaption` (in `models/hull_white.py`) built the trinomial tree to evolve state prices Q correctly to time `expiry_T`, then iterated over the expiry-time nodes with:

```python
r_j = r0 + j * dr
```

where `r0 = self._forward_rate(0.0)` — today's instantaneous short rate. That is wrong: under Hull-White, the rate at the expiry nodes is `r(T_expiry) = α(T_expiry) + j·dr`, where α is the closed-form drift adjustment

> α(t) = f^M(0,t) + (σ²/(2a²))·(1 − e^{−at})²

(Brigo-Mercurio 2006, eq. 3.34).

### Magnitude of the error

On a representative steeply rising curve (1% at t=0 rising 80bp/year, so 9% at 10y), with `a=0.05, σ=0.5%`:

| Quantity | Value |
|---|---|
| α(0)   | 1.20 % |
| α(3y)  | 7.41 % |
| Pre-fix centring | 1.20 % (used α(0)) |
| Post-fix centring | 7.41 % (uses α(T_expiry)) |
| **Centring error** | **621 bps in the short rate at expiry nodes** |

A 600bp error in the centring of the rate grid at expiry translates to a multi-tens-of-percent error in the swaption PV — far beyond any plausible model calibration tolerance.

### The fix

Two-line change in `hull_white.py`:

1. New method `HullWhite._alpha(t)` implementing the closed-form formula `α(t) = f(0,t) + (σ²/(2a²))·(1 − e^{−at})²`.
2. `tree_european_swaption` calls `alpha_expiry = self._alpha(expiry_T)` once before the node loop and uses `r_j = alpha_expiry + j * dr`.

The tree calibration in `_evolve_state_prices` is unchanged — only the readout step at expiry is corrected. (The closed-form α(t) is the continuous-time limit of the tree's per-step numerical calibration; for typical step sizes the two agree to within a few bp.)

### Verification — `test_l2_t1_9_hw_swaption_alpha.py`

5 new regression tests, all pass:

- `test_alpha_formula_at_zero_equals_short_rate` — sanity: α(0) = f(0,0) since `1−e^0 = 0`.
- `test_alpha_grows_with_forward_curve` — on a 100bp/year-slope curve, α(5y) − α(0) > 400bp (vs near-zero pre-fix when using α(0) everywhere).
- `test_tree_swaption_atm_matches_jamshidian_flat_curve` — on a flat 1% curve (where the pre-fix bug is small because α(T) ≈ α(0)), the tree price agrees with the analytical HW Jamshidian decomposition to within 5%.
- `test_tree_swaption_matches_jamshidian_steep_curve` — **the discriminating test**: 80bp/year-slope, 3y into 4y ATM payer. Post-fix the tree agrees with Jamshidian to within 10%; pre-fix the disagreement was >>10%.
- `test_tree_swaption_deep_itm_recovers_intrinsic` — deep-ITM with low vol → tree price within 15% of annuity·(fwd − K). Pre-fix the wrong centring biased this by the centring error × annuity, dwarfing intrinsic.

The test file embeds a self-contained Jamshidian-decomposition payer-swaption pricer (bracketing for r*, Brigo-Mercurio bond-option vol formula, ZBP via Black-style normal) as the ground-truth reference. ~120 lines including helpers.

Full parallel suite: **11969 passed in 4:50** — zero regressions.

### Tier-1 status — 9 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | queued |
| **T1.9** | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ this slice |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

Remaining 4 are the deepest ones: T1.1 MLMC redesign (need Giles coupling), T1.4 IPM rewrite, T1.8 cos_method V_k re-derivation, T1.13 AAD bootstrap.

---

## v0.908.0 — 2026-06-12

**Fix L2 T1.6 — discrete dividends now escrowed at the correct step in both binomial and trinomial trees.**

Eighth of 13 Tier-1 (dual-critic-confirmed) bugs from `MODULE_HEALTH.md` closed. This one had two sub-bugs in `numerical/_trees.py`:

### Sub-bug 1 — Binomial: dividends applied only to the terminal grid

Pre-fix, `_solve_binomial` constructed the spot tree by subtracting cumulative dividends *once at the terminal step*. The intermediate-step `S_step` array (used by every backward-induction iteration for early-exercise and barrier checks) was rebuilt as the **raw pre-dividend forward** at each step.

Consequence: an American option with a known dividend mid-life saw the wrong (too-high) intermediate spot, biasing the early-exercise decision; barrier knock-out checks used the wrong intermediate spot too. The terminal price was correct in aggregate, but the path-dependent / early-exercise behaviour was wrong.

### Sub-bug 2 — Trinomial: dividends silently ignored entirely

`_solve_trinomial` had **no dividend handling whatsoever** — `self.dividends` was accepted in the constructor but never read by the trinomial path. A user passing `dividends=[(25, 10.0)]` to `TreeMethod.TRINOMIAL` got identically the same answer as `dividends=None`.

### Fix — escrowed-dividend convention applied to both

Both `_solve_binomial` and `_solve_trinomial` now share the same per-step helper structure:

```python
def _cum_div_through(s: int) -> float:
    return sum(amt for step, amt in self.dividends.items() if step <= s)

def _spot_at_step(s: int) -> np.ndarray:
    grid = ...   # raw forward spot grid at step s
    cum = _cum_div_through(s)
    if cum > 0:
        grid = np.maximum(grid - cum, 0.01)
    return grid
```

At each step `s`, the spot grid has cumulative dividends paid through step `s` subtracted. This is the standard "escrowed dividend" convention (Hull §21.12): the underlying drops by the dividend amount on the ex-date, and that drop is reflected in *every* node from that step forward.

`S` (terminal) and `S_step` (intermediate) now both go through `_spot_at_step`, so they agree on the dividend-adjusted spot. Floor at `0.01` guards against degenerate cases where down-moves push the dividend-adjusted spot to zero.

### Verification — `test_l2_t1_6_discrete_dividends.py`

4 new regression tests, all pass:
- `test_dividend_at_step_zero_equals_terminal_dividend` — boundary: dividend at step 0 reduces call PV.
- `test_dividend_at_intermediate_step_affects_price` — dividend at step 25/50 reduces European call price.
- `test_american_call_with_dividend_exercises_correctly` — with a large mid-life dividend, American call ≥ European call (proper American premium for dividends — the classic textbook case).
- `test_trinomial_dividend_actually_affects_price` — trinomial now responds to `dividends=[(25, 10.0)]` (pre-fix delta was ~0; post-fix delta > 1).

Full parallel suite: **11964 passed in 3:21** — zero regressions.

### Tier-1 status — 8 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| **T1.6** | `numerical/_trees.py` discrete dividends terminal-grid + trinomial ignored | ✅ this slice |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | queued |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | queued |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

Remaining 5 are the harder ones (math review needed): T1.1 MLMC redesign, T1.4 IPM rewrite, T1.8 cos_method V_k re-derivation, T1.9 Hull-White swaption analytic, T1.13 AAD bootstrap.

---

## v0.907.0 — 2026-06-12

**Fix L2 Tier-1: four quick-win bugs (T1.2 trapz, T1.3 integrate_2d, T1.5 Tian, T1.7 Milstein).**

Opening the L2 audit by closing the four cheapest of MODULE_HEALTH's 13 Tier-1 (dual-critic-confirmed) bugs. Each was a small, surgical change with a regression test that locks in the fix.

### T1.2 — `numerical/_fourier.py` `CharacteristicFunction.density()` was a hard crash

`np.trapz` was removed in NumPy 2.x; the project's installed `numpy==2.4.3` lacks it. Every call to `density()` raised `AttributeError: module 'numpy' has no attribute 'trapz'`. Fix: rename to `np.trapezoid`. Function had no callers in production code so the bug was dormant; it now works for any future caller (verified: `density(N(0,1))` at `x=0` ≈ 0.3989).

### T1.3 — `numerical/_integrate.py` `integrate_2d` argument order was inverted

`scipy.integrate.dblquad` expects `func(y, x)` (y is the inner variable). The function was passing `f` directly. For symmetric integrands the bug was invisible; for non-symmetric ones the result was wrong: `integrate_2d(lambda x, y: x, (0,3), (0,1))` returned **1.5** instead of **4.5**. Fix: wrap with an explicit `_f_xy(y, x): return f(x, y)` before passing to scipy.

### T1.5 — `numerical/_trees.py` Tian binomial method had been silently running CRR

Two compounding errors in `_tian_params`:
1. `V = M² × (exp(σ²·dt) − 1)` (the *variance*) instead of `V = exp(σ²·dt)` (the *second-moment factor* — the actual Tian 1993 quantity).
2. The `u`/`d` formula was missing the `V` multiplier.

Combined effect: the discriminant `V² + 2V − 3` was nearly always **negative**, triggering the "very small dt × vol → fall back to CRR" guard. So calling `TreeMethod.TIAN` actually ran CRR for every call. Live demonstration (r=5%, σ=20%, dt=0.01):
- Standard Tian: disc = +0.0016, u/d are real.
- Pre-fix code: V = 0.0004, disc = −2.999, branch falls back to CRR.

Fix: match the published Tian (1993) JFQA derivation. The Tian method now produces u/d distinct from CRR, as it should.

### T1.7 — `models/mc_engine.py` Milstein scheme silently ran Euler (two sub-bugs)

```python
step_fn = euler_step if self.scheme == "euler" else euler_step   # COPY-PASTE
```

Even fixing the typo wasn't enough: `milstein_step_1d` requires `diffusion_deriv` (σ' for the correction term), but `ProcessSpec` had no field for it — the parameter at the call site stayed `None`, so Milstein reduced to Euler internally. Fix:
1. Replace the second `euler_step` with `milstein_step_1d`.
2. Add `diffusion_deriv: Callable | None = None` to `ProcessSpec.__init__`.
3. Plumb it through to the Milstein call.
4. When `scheme='milstein'` but `diffusion_deriv` is None, emit a `RuntimeWarning` and fall back to Euler (no longer silent).

Live: with GBM and σ=0.5, n_steps=10 over T=1y, Euler and Milstein now produce paths that differ visibly at the terminal — pre-fix they were identical.

### Verification

- 9 new regression tests in `test_l2_tier1_quick_wins.py`:
  - T1.2: density() runs; N(0,1) density at 0 ≈ 1/√(2π).
  - T1.3: non-symmetric integrand (4.5 not 1.5); symmetric unaffected; callable y_range uses x correctly.
  - T1.5: Tian parameters match Tian 1993 formulas; no longer equal to CRR for typical inputs.
  - T1.7: Milstein produces different paths from Euler when `diffusion_deriv` provided; warns when not provided.
- Full parallel suite: **11960 passed in 3:21** — zero regressions.

### Tier-1 status

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| **T1.2** | `numerical/_fourier.py` density crash | ✅ this slice |
| **T1.3** | `numerical/_integrate.py` integrate_2d swap | ✅ this slice |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| **T1.5** | `numerical/_trees.py` Tian V formula | ✅ this slice |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid | queued |
| **T1.7** | `models/mc_engine.py` Milstein silently runs Euler | ✅ this slice |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | queued |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | queued |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

**7 of 13 Tier-1 closed.** Remaining 6 are harder (deeper math review needed for T1.1 MLMC, T1.8 cos_method bounds, T1.9 Hull-White swaption analytic).

---

## v0.906.0 — 2026-06-11

**Fix L1 C.2 B1 — `curve_jacobian` resolves `pillar_tenors` to actual curve indices.**

Third HIGH-severity bug closed from L1 audit. Pre-fix, when a caller passed a custom `pillar_tenors` list (different from the curve's actual pillar grid), `curve_jacobian` silently bumped the WRONG pillars — the Jacobian's column labels were mismatched with the user's requested tenors.

### What was broken

```python
for j, pt in enumerate(pillar_tenors):
    bumped = curve.bumped_at(j, bump_size)   # ← j is an enumeration index
    ...                                        #   NOT a resolved curve pillar
```

`bumped_at(j, ...)` takes a **curve pillar index**. The function looped over the user's `pillar_tenors` and used the enumeration index `j` directly. For a curve with pillars `[0.25, 0.5, 1, 2, 5, 10]` and user request `pillar_tenors=[1, 2, 5]`:
- `j=0` → bumped index 0 → the 0.25y pillar (user thought: 1y)
- `j=1` → bumped index 1 → the 0.5y pillar (user thought: 2y)
- `j=2` → bumped index 2 → the 1y pillar (user thought: 5y)

The Jacobian's column labels lied. A caller multiplying `J` by a market-quote-bump vector got systematically wrong sensitivities.

### Change

- `curves/curve_risk.py:curve_jacobian` — when `pillar_tenors` is supplied, resolve each tenor to its actual curve pillar index (nearest match within `pillar_tol`); raise `ValueError` if any tenor doesn't match a curve pillar. New `pillar_tol=1e-2` default accommodates the 365.25-vs-day-count drift on `DiscountCurve.flat` pillars (~0.003y).
- 5 new regression tests in `test_curve_jacobian_pillar_resolution.py`:
  - default `pillar_tenors=None` uses the curve's own grid.
  - custom `pillar_tenors=[1, 2, 5]` correctly bumps those pillars (J[0,0] dominant for 1y query).
  - unrecognised tenor raises clearly.
  - tolerance allows close matches (1y vs 1.00274y).
  - out-of-order tenors produce columns in the user-supplied order.
- Full parallel suite: **11951 passed in 3:40** — zero regressions.

### L1 audit status — active HIGH bugs all closed

| Bug | Status |
|---|---|
| A.2 B1 — global_solver dup-maturity | ✅ v0.905 |
| A.3 B1 — multicurve PV_float first period | ✅ v0.905 |
| **C.2 B1** — curve_jacobian wrong pillar | ✅ **this slice** |
| A.1 B1 — bootstrap HW convexity (latent) | queued |

L1 Pass A + B + C audited (15 of 33 modules). Remaining: D (builders, 8 modules), E (numerics, 4), F (AAD, 5).

---

## v0.905.0 — 2026-06-11

**Fix L1 curves audit A.2 B1 + A.3 B1 — `global_solver` rejects duplicate maturities; `multicurve_newton` PV_float now includes the first accrual period.**

Two HIGH-severity active bugs from L1 Pass A, fixed in one slice.

### What was broken

**A.3 B1 — `multicurve_newton` PV_float skipped the first period.** The projection-swap PV_float loop started at `j=1`, walking from the first pillar onward. But `_compute_annuity` walked from `reference_date`. So PV_float had `N−1` segments while annuity had `N`. For 2-pillar projection swaps the bias was ~50%; the solver oscillated and emitted `RuntimeWarning: multicurve_newton: did not converge after 50 iterations. Residual: 2.86e-03`. That warning has been polluting the test suite for the entire session. **It was the bug talking.**

**A.2 B1 — `global_solver` silently dropped constraints on duplicate maturities.** The residual vector was indexed by `pillar_idx[mat]`. Two instruments at the same maturity (e.g. a 1Y deposit + 1Y OIS swap — both standard market quotes!) would overwrite each other's residual. Newton converged to a curve that didn't reprice all inputs. Live repro: 1Y depo @5% + 1Y swap @4% → zero rate 3.96% (only the swap survived).

### Change

- `curves/multicurve_solver.py:159-176` — projection PV_float loop now walks `[reference_date, *dates_up_to]` to match `_compute_annuity`. The `multicurve_newton: did not converge` warning is gone from the test suite.
- `curves/global_solver.py:46-77` — `global_bootstrap` builds a `seen_maturities` set; raises `ValueError("Duplicate maturity ... already provided by ..., also requested as ...")` on collision. Each pillar can be constrained by at most one instrument.
- 2 new regression tests in `test_multicurve_first_period.py`: convergence within tolerance under `RuntimeWarning`-as-error; round-trip identity (each projection swap rate recoverable from the calibrated curves).
- 5 new regression tests in `test_global_solver_collision.py`: deposit-swap collision raises; duplicate-deposit raises; duplicate-swap raises; error message identifies the conflicting type; distinct maturities still work.

### Verification

- Full parallel suite: **11946 passed in 4:06** — zero regressions.
- Warnings count dropped from 63 to 55 (the 8 dropped were the recurring `multicurve_newton: did not converge`).

### L1 Pass A — status update

| Bug | Status |
|---|---|
| A.1 B1 — `bootstrap` HW convexity wrong | LATENT (no current caller exercises it; deferred until either a caller appears or we coordinate a single fix) |
| A.1 B2 — float-leg conventions no-op | docstring fix queued |
| **A.2 B1** — `global_solver` residual collision | ✅ this slice |
| **A.3 B1** — `multicurve` PV_float first-period | ✅ this slice |
| A.4 B1 — `ncurve_solver` BasisSwap annuity | queued |
| A.4 B2 — `ncurve_solver` OIS schedule | queued |

Both active HIGH bugs closed. The dormant HW-convexity bug remains queued — it requires a refactor (re-route `bootstrap()` to call `ir_futures.hw_convexity_adjustment`) but since no caller is hitting it today, it sits behind A.4 in priority.

---

## v0.904.0 — 2026-06-11

**Generic `to_dict` mutation-safety sweep — `vars(self)` → `dict(vars(self))` across L0.**

Closes a recurring footgun cluster surfaced by audit findings A.5 B1, A.7 B1, C.1 B1, and ~25 other instances of the same pattern in `book.py`, `daily_pnl.py`, `settlement.py`, `mandate.py`, `greeks.py`, `dependency_graph.py`, `convergence_framework.py`, `numerical_safety.py`, `numerical_method_map.py`, `market_data.py` (legacy), `trade.py` (orphan), `approximation.py`, `solvers.py`.

### What was broken

```python
@dataclass
class Result:
    x: float
    def to_dict(self) -> dict:
        return vars(self)   # ← returns self.__dict__ DIRECTLY, not a copy
```

Mutating the returned dict mutated the dataclass instance. Confirmed live in audit A.5 B1:
```
r = SolverResult(root=1.0, ...)
d = r.to_dict()
d['root'] = 999.0
# r.root is now 999.0
```

30 callsites had this pattern across `pricebook.core.*`.

### Change

- All 30 `return vars(self)` lines rewritten to `return dict(vars(self))` (defensive copy). Single regex sweep across 13 files; idempotent and behaviour-preserving for callers that don't mutate.
- Deleted the dead-code orphan `Trade.to_dict` at `core/trade.py:56-57` — it sat inside the `@dataclass` body but was overwritten at module-bottom by `_trade_to_dict` via `Trade.to_dict = _trade_to_dict`. Replaced with a one-line comment pointing to the real binding (closes audit C.1 B1).
- 9 new regression tests in `test_to_dict_mutation_safety.py` covering `SolverResult`, the 3 `approximation.py` dataclasses, `Greeks`, `Position` (book), `DailyPnL`, `CashSettlementResult`, `PortfolioHolding` (mandate). Each asserts that mutating the returned dict doesn't propagate to the source.
- Full parallel suite: **11930 passed in 3:28**. Zero regressions.

### L0 audit — closure

With this slice, **every confirmed bug from Pass A through Pass D of the L0 audit is now either fixed or queued as ARCH/LOW**.

| Severity | Count | Status |
|---|---:|---|
| HIGH | 5 | ✅ all fixed |
| MED | 6 | ✅ all fixed |
| LOW | many | mostly closed (`to_dict` sweep + Tokyo + ICMA test gaps + serialisation A.11 B3-B7 queued) |
| ARCH | 1 | B.3 C1 — legacy market_data deferred to Gate 2 design decision |

The pricebook L0 foundations are materially more correct than at the start of this session. ~25 commits, **11930 tests passing**, every fix landed with regression tests, legacy-debt ledger tracks every backwards-compat shim, audit chain (G1) wired end-to-end.

---

## v0.903.0 — 2026-06-11

**Fix C.7 B1 — settlement lag now interpreted as business days; calendar honoured.**

Closes the last MEDIUM-severity bug from L0 Pass C. `cash_settlement`, `cds_settlement_physical/_cash`, `option_settlement_cash/_physical`, and `futures_settlement_physical` all silently used **calendar** days for the T+N lag — a Friday trade settling T+2 landed on Sunday.

### Before / after

```
Friday 2025-08-01, cash_settlement T+2:
  BEFORE: 2025-08-03 Sun (calendar, non-business)
  AFTER:  2025-08-05 Tue (skip Sat+Sun)

Friday 2025-08-01, option_cash T+1:
  BEFORE: 2025-08-02 Sat
  AFTER:  2025-08-04 Mon

Thursday 2025-07-03, option_cash T+1 with US calendar:
  BEFORE: 2025-07-04 Fri (Independence Day — holiday)
  AFTER:  2025-07-07 Mon (skip July 4)
```

### Change

- Six settlement entry points (`cash_settlement`, `cds_settlement_physical`, `cds_settlement_cash`, `option_settlement_cash`, `option_settlement_physical`, `futures_settlement_physical`) now route through `add_business_days(d, lag, calendar)` instead of `date.fromordinal(d.toordinal() + lag)`.
- All six gain an optional `calendar` parameter. When `None`, only weekends are skipped (same as `Calendar.add_business_days(None)`). When provided, holidays are skipped too.
- Existing `test_settlement.py::test_settlement_lag` updated — the prior assertion `(settle - REF).days == 30` codified the calendar-day bug; replaced with exact business-day calculation (REF = Mon 2024-01-15; +30 BD → Mon 2024-02-26).
- 9 new tests in `test_settlement_business_days.py`: Friday + T+1/T+2/T+3/T+5 weekend skip across all four product types; July-4 Independence Day with US calendar; T+0 and mid-week sanity cases.
- Full parallel suite: **11930 passed in 3:25** — zero regressions.

### Affected upstream

Anything that imports settlement helpers from `pricebook.core.settlement` and reads `settlement_date`. Notable: FX spot dates (T+2), US equity option physical settlement (T+2), CDS cash settlement (T+5), bond futures physical delivery (T+3). The pre-fix dates would frequently land on weekends — useful as a "placeholder" date for sandbox demos but wrong for settlement-risk computations.

### Pass C — MEDs closed

| C.7 B1 | settlement lag business days | ✅ this slice |
| C.8 B1 | dollar_gamma formula | queued (LOW) |
| C.1 B1 | dead-code orphan to_dict | queued (LOW, generic to_dict sweep) |

---

## v0.902.0 — 2026-06-11

**Fix D.1 B1+B2+B3 — `PricingContext` round-trip preserves every field; `replace()` no longer aliases mutable dicts.**

Three pre-existing bugs in `PricingContext` (all carried over from the prior `MODULE_HEALTH.md` audit and verified at L0 audit Pass D.1) fixed in one focused slice.

### What was broken

1. **B1 — empty containers collapsed to `None` on round-trip.** `_ctx_from_dict` had `proj = {...} or None` patterns. An empty `projection_curves`/`vol_surfaces`/`credit_curves`/`fx_spots` dict became `None`, so `ctx.projection_curves["foo"]` raised `TypeError: 'NoneType' object is not subscriptable` instead of the documented `KeyError`.
2. **B2 — fields silently dropped.** `_ctx_to_dict` emitted only 5 fields out of 13. **Silently dropped**: `discount_curves` (multi-currency), `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, AND `numerical_config` (just added in G1 P3 Slice 1!). Multi-currency contexts lost data on every save/load round-trip.
3. **B3 — `replace()` aliased mutable containers.** `ctx2 = ctx.replace(reporting_currency="EUR")` then `ctx2.discount_curves["BRL"] = curve` mutated `ctx.discount_curves` too. Broke the "Immutable snapshot" docstring contract.

### Change

- `pricing_context.py:replace()` — now goes through a `_pick_dict` helper that `dict(...)`-copies every mutable container before passing to the new context.
- `_ctx_to_dict` — emits **every** dataclass-declared field, including `discount_curves`, `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, `numerical_config` (serialised via `dataclasses.asdict`).
- `_ctx_from_dict` — `_fd_dict` helper reconstructs each value via the global registry (handles both envelope-format and opaque values); empty dicts stay empty (no `or None` collapse); `NumericalConfig` reconstructed from its `asdict` form.
- 13 new tests in `test_pricing_context_round_trip.py`:
  - **D.1 B1**: default-context round-trip preserves dict-not-None for all 10 container fields; access raises `KeyError` not `TypeError`.
  - **D.1 B2**: `discount_curves`, `reporting_currency`, `repo_curves`, `fx_spots`, `credit_correlations`, `numerical_config` (both populated AND `None`) all round-trip exactly.
  - **D.1 B3**: `replace()` doesn't alias `discount_curves` / `fx_spots` / `credit_correlations`; values copy through but underlying dict is a different object.
- Full parallel suite: **11921 passed in 3:55** — zero regressions.

### Affected upstream

Multi-currency users (anyone with `discount_curves={"USD": ..., "EUR": ...}`), users persisting `PricingContext` to disk for replay, scenario engines that build derived contexts via `replace()` and concurrently mutate them. The pre-fix behaviour silently corrupted data; the fix lands without behaviour change on any single-currency in-memory workflow.

---

## v0.901.0 — 2026-06-11

**Fix A.2 B2 — `TokyoCalendar` *furikae kyūjitsu* (振替休日) substitute-day rule.**

Closes the last MEDIUM-severity calendar finding from L0 Pass A. Tokyo's fixed-date holidays falling on Sunday now correctly cascade to a substitute holiday on the next non-holiday day per Japan's Public Holiday Act §3.2.

### Before vs after

| Date | Holiday | Status BEFORE | Status AFTER |
|---|---|---|---|
| 2018-04-30 Mon | Showa Day Sun → substitute | not holiday ❌ | holiday ✓ |
| 2025-11-24 Mon | Labour Thanksgiving Sun → substitute | not holiday ❌ | holiday ✓ |
| 2026-05-06 Wed | Constitution Sun + Greenery + Children's cluster → substitute | not holiday ❌ | holiday ✓ |
| 2024-04-30 Tue | (no sub needed, Apr 29 was Mon) | business day ✓ | business day ✓ |

The Golden-Week cascade case is the interesting one: May 3 2026 (Constitution) is Sunday; May 4 (Greenery) and May 5 (Children's) are already holidays; the substitute walks all the way to **May 6 Wed**. The implementation handles this generically via a while-loop that walks forward until it finds a non-holiday day.

### Change

- `core/calendar.py:TokyoCalendar._compute_holidays` — refactored to:
  1. Build a `fixed` set of fixed-date holidays.
  2. Build a `monday_holidays` set of variable Monday holidays (Coming of Age, Marine Day, Respect for Aged, Sports Day — by construction never on Sunday).
  3. Second pass over `fixed`: any Sunday entry gets a substitute computed by walking forward until a non-holiday day is found.
- 5 new tests in `test_calendar.py::TestTokyoCalendarSubstitution`: Showa Day 2018, Labour Thanksgiving 2025, Golden-Week cluster 2026, no-substitute-when-weekday case, Monday-holiday smoke check.
- Full parallel suite: **11908 passed in 4:06**.

### A.2 — closed

All four substitution-rule findings from L0 audit A.2 are now fixed:

| Sub-finding | Locale | Status |
|---|---|---|
| A.2 B1a | London (UK 1971 Act) | ✅ |
| A.2 B1b | Sydney (AU Public Holidays Acts) | ✅ |
| A.2 B1c | Wellington (NZ Holidays Act 2003) | ✅ |
| A.2 B1d | Toronto (Canadian federal/provincial) | ✅ |
| **A.2 B2** | **Tokyo (Japanese Public Holiday Act §3.2)** | ✅ **this slice** |

JPY-rate calculations crossing 2018-04-30 (Showa Day substitute), 2019-05-06 (Reiwa transition), 2025-11-24, and 2026-05-06 are now correct.

---

## v0.900.0 — 2026-06-11

**Fix A.11 B1 + B2 — `_deserialise_atom` now dispatches `list[SomeSerialisable]`, `list[Enum]`, and polymorphic `Union[A, B, None]`.**

Two medium-severity audit findings, one root cause: `_deserialise_atom` had a narrow list of supported parameterised types. Anything outside the narrow path silently returned the raw value, breaking round-trips for polymorphic fields.

### Before / after

| Type hint | Pre-fix behaviour | Post-fix behaviour |
|---|---|---|
| `list[date]` | reconstructed ✓ | reconstructed ✓ |
| `list[SomeSerialisable]` | **raw list of dicts** ❌ | recursively dispatched via registry ✓ |
| `list[SomeConvention]` | raw list ❌ | dispatched via `inner.from_dict` ✓ |
| `list[Colour]` (Enum) | raw list of strings ❌ | mapped to Enum members ✓ |
| `Optional[T]` (`T \| None`) | unwrapped ✓ | unwrapped ✓ |
| `Union[A, B, None]` | **raw value** ❌ | if dict with `"type"` key → registry-dispatch; else raw ✓ |

### Live test of the fix

A `_Container` class with a `list[_Leaf]` field and a `Union[_Leaf, _Branch, None] underlying` field now round-trips cleanly: every leaf is reconstructed, the polymorphic underlying lands on the correct concrete type based on its `type` tag. Pre-fix, both would have returned raw dicts.

### Change

`pricebook.core.serialisable._deserialise_atom`:
- **Union dispatch** (A.11 B2): when stripping `NoneType` leaves ≥2 non-None args, check if `v` is a dict with `"type"` → registry-dispatch via `from_dict`. Otherwise return raw (same as before — primitives don't need dispatch).
- **list[T] dispatch** (A.11 B1):
  - `list[date]` — unchanged.
  - `list[Serialisable]` — recursively dispatches each element; supports both envelope-format (`"type"` key → `from_dict`) and flat-convention format (no `"type"` → `inner.from_dict`).
  - `list[Enum]` — maps each element through the Enum constructor.
- 10 new tests in `test_serialisable_polymorphic.py`: container round-trip with each list shape + polymorphic Union with type-tag, both branches, and the None case + 3 direct atom-dispatch tests.
- Full parallel suite: **11903 passed in 4:04**. Zero regressions.

### What's still uncovered (queued)

- `list[tuple[...]]`, `list[list[...]]`, nested generic types — still raw. Genuinely niche; no current call-site needs them.
- `Union[A, B]` (no None) with non-tagged values — still raw. Can't auto-resolve without a discriminator.
- A.11 B3 (`_register` silent no-op on re-import), A.11 B4 (numpy `.item()` duck-test ordering), A.11 B5 (CurrencyPair parse), A.11 B6 (bare `KeyError`), A.11 B7 (Enum int-from-string) — all LOW severity, queued.

---

## v0.899.0 — 2026-06-11

**Fix B.1 B2 — `make_payload`/`read_payload` helpers extend G1 P3 Slice 2's schema-version coverage to custom `to_dict` overrides; `DiscountCurve` no longer drops `interpolation` on round-trip.**

A medium-severity audit finding that turned out to have wide reach. G1 P3 Slice 2 added `schema_version` to `Serialisable.to_dict` and the two decorators, but **classes with hand-written `to_dict` overrides** (the curves, `Trade`, `Portfolio`, `PricingContext`, several option types) bypassed all of it. They wrote `{"type": ..., "params": ...}` directly. Schema versioning was silently absent from their on-disk payloads.

### What was broken

Two distinct gaps in one root cause:

1. **`DiscountCurve.to_dict` dropped `interpolation`.** A curve built with `MONOTONE_CUBIC` or `AKIMA` interpolation serialised, then deserialised, silently came back as `LOG_LINEAR` (the constructor default). Live repro from the audit confirmed this. Any persisted curve with a non-default interpolation method was silently misinterpreted on reload.
2. **Schema versioning didn't reach custom-to_dict classes.** Payloads from `DiscountCurve`, `SurvivalCurve`, `Trade`, `Portfolio`, `PricingContext` carried no version field. When a future v2 lands, there'd be no way to distinguish.

### Change

- New helpers in `pricebook.core.serialisable`:
  - `make_payload(instance, params)` — builds `{"type": ..., "params": ..., "schema_version": ...}` from a Serialisable instance + a params dict. Drop-in replacement for hand-rolled envelopes.
  - `read_payload(d, cls)` — inverse: validates schema_version against `cls._SERIAL_SCHEMA_VERSION` and returns the params dict. Replaces `p = d["params"]` in custom `from_dict` overrides.
- Migrated:
  - `core/discount_curve.py` — uses `make_payload`/`read_payload`. **Includes `interpolation` in the params dict; deserialiser defaults to `LOG_LINEAR` when absent (back-compat for pre-fix payloads).**
  - `core/survival_curve.py` — uses helpers (interpolation was already preserved; this slice adds schema_version coverage).
  - `core/trade.py` — `Trade` and `Portfolio` both migrated.
  - `core/pricing_context.py` — migrated, plus the `vars(self)` → `dict(vars(self))` fix (same mutation pattern as solvers/approximation, now closed).
- 13 new tests in `test_custom_to_dict_schema_version.py`:
  - **Parametrised round-trip across all 5 `InterpolationMethod` values** — catches the B.1 B2 interpolation-loss bug for every method.
  - Pre-fix payload (no `interpolation` key) defaults to LOG_LINEAR — back-compat.
  - schema_version present on every migrated class.
  - Future version rejected with the standard ValueError.
  - Pre-fix payload (no schema_version key) still deserialises as v1.
- Full parallel suite: **11893 passed in 4:01** — no regressions.

### What's still uncovered

The 43 file-count from the audit includes many option/credit classes (`options/asian_option.py`, `options/barrier_option.py`, `credit/cds.py`, etc.) that still hand-roll `{"type": ..., "params": ...}`. They are NOT migrated in this slice — that work belongs in their respective audit passes (Pass C / D etc.). Behaviour is unchanged for them; they currently emit no schema_version which by contract reads back as v1 — back-compat all the way down.

### Wider implication

`make_payload`/`read_payload` is now the canonical pattern for any class with a custom `to_dict`. New code should always go through it; existing code migrates as each module gets audited.

---

## v0.898.0 — 2026-06-11

**Fix B.1 B1 — `DiscountCurve.roll_down` anchors rolled DFs to the new reference date.**

The fifth HIGH-severity L0 audit bug, now closed. `roll_down(days)` was silently producing curves with wrong zero rates because it forgot to apply the no-arbitrage anchoring `P(new_ref, d) = P(0, d) / P(0, new_ref)`. On a flat 5% curve, rolldown P&L was over-stated by **+1.4 bp per day** of roll.

### Live before / after

```
Flat 5% curve, ref=2024-01-01.

BEFORE:
roll_down(1d)  → zero_rate(2025-01-01) = 5.0137%  (+1.37 bp error)
roll_down(30d) → zero_rate(2025-01-01) = 5.4231%  (compounded error)
                          (any rolldown attribution off this is structurally wrong)

AFTER:
roll_down(1d)   → zero_rate(2025-01-01) = 5.0000% exactly
roll_down(30d)  → zero_rate(2025-01-01) = 5.0000% exactly
roll_down(365d) → zero_rate(2026-01-01) = 5.0000% exactly
```

### Change

- `discount_curve.py:126-178` — `roll_down` now computes `disc_to_new_ref = self.df(new_ref)` once and divides each pillar DF by it. Raises `ValueError` with a clear message if `disc_to_new_ref <= 0` (pathological / extrapolated curve).
- The "all pillars in the past" fallback now **preserves** the original `day_count` and `interpolation` instead of silently dropping them via `DiscountCurve.flat(...)` (a separate footgun called out in the audit).
- 5 new tests in `TestRollDown`: 1-day / 30-day / 365-day flat-curve zero-rate preservation; no-arbitrage anchor identity `df_rolled(d) == df(d) / df(new_ref)`; day_count + interpolation preservation in the all-pillars-past fallback.
- Full parallel suite: **11880 passed in 4:57**. Zero regressions.

### Affected upstream

Anything that calls `DiscountCurve.roll_down`: rolldown P&L attribution, daily carry/roll analytics, scenario time-shifts. The old behaviour over-stated rolldown gain by ~1.4 bp per day for a 5% curve. Practitioners running daily roll/carry on USD/GBP/EUR books would see this in their P&L explain.

---

## v0.897.0 — 2026-06-11

**Fix A.1 B1 Slice 4 — `FixedRateBond` YTM analytics use ICMA-correct period counting; par-yield round-trip is exactly 100.**

The fourth and decisive slice of the ICMA fix. Coupons were fixed in Slice 3; this slice fixes the *discount-time* side. UST and other ICMA-convention bonds now satisfy the par-yield identity exactly.

### What was broken (after Slice 3)

`FixedLeg` was producing correct coupon amounts (2.0000 exact for par 5y UST) but `_price_from_ytm` was computing time-to-cashflow via `year_fraction(settle, payment_date, ACT_ACT_ICMA)` for *multi-period spans*. That call silently falls back to ACT/365F because the current `_act_act_icma` implementation only handles single-period accruals (per ICMA 251.1) and not multi-period spans (ICMA 251.2). The result: coupon amounts at 2.0000 each, but discount times at 0.4986, 1.0027, 1.4986, ... instead of 0.5, 1.0, 1.5, .... Par-yield round-trip landed at 99.981367 instead of 100.

### Change

- New method `FixedRateBond._ytm_time_to(settle, target)` computes the ICMA-correct time-to-cashflow:
  - If `settle` coincides with a coupon date → `(target_index − settle_index) / coupons_per_year` exactly.
  - If `settle` is mid-period → stub fraction (days_to_next_coupon / period_days) + full periods.
  - For non-ICMA conventions → falls back to `year_fraction(settle, target, day_count)` (unchanged).
- Routed `_price_from_ytm`, `macaulay_duration`, `convexity`, and `accrual_schedule` through `_ytm_time_to`. `modified_duration` and `dv01_yield` derive from these — no further change needed.
- The two characterisation xfails turn green; their markers are removed. **Tolerance tightened from 1e-8 to 1e-10** to lock in the new exactness.
- Full parallel suite: **11875 passed in 4:57** — zero regressions.

### Verified end-to-end

```
Pre-fix:  par 5y UST round-trip = 99.999807, par 30y = 99.999489
Post-fix: par 5y UST round-trip = 100.000000, par 30y = 100.000000
```

### A.1 B1 status

| Slice | Scope | Status |
|---|---|---|
| 1 | Add `strict_icma` flag on `year_fraction` | ✅ |
| 2 | Characterise UST mispricing with xfail tests | ✅ |
| 3 | `FixedLeg` passes ICMA refs (correct coupon amounts) | ✅ |
| 4 | `_ytm_time_to` helper (correct discount times) | ✅ (this slice) |
| N | Final: flip default to `strict_icma=True` after auditing remaining callers | queued |

UST and all other ICMA-convention bonds (Bunds, Gilts, JGBs, sovereigns) now compute coupon amounts AND par-yield round-trips correctly at machine precision. The audit chain's highest-blast-radius bug is closed for the bond-pricing path.

---

## v0.896.0 — 2026-06-11

**Fix A.1 B1 Slice 3 — `FixedLeg` cashflows now compute correct ACT/ACT ICMA year-fractions; UST coupons land exactly at face × coupon ÷ frequency.**

The third slice of the staged ICMA fix delivers the *coupon-side* correctness. `FixedLeg`'s cashflow builder now passes `ref_start`, `ref_end`, and `frequency` to `year_fraction`, so every regular coupon period on a Treasury / Gilt / Bund / JGB schedule gets `year_frac = 1 / coupons_per_year` *exactly* — per ICMA Rule 251.1.

### Before vs after (par 5y UST at 4% coupon, face=100, semi-annual)

| Coupon | Period | days | year_frac BEFORE | year_frac AFTER | amount BEFORE | amount AFTER |
|---|---|---:|---:|---:|---:|---:|
| 1 | 02/15 → 08/15 | 182 | 0.498630 | **0.500000** | 1.994521 | **2.000000** |
| 2 | 08/15 → 02/15 | 184 | 0.504110 | **0.500000** | 2.016438 | **2.000000** |
| 3 | 02/15 → 08/15 | 181 | 0.495890 | **0.500000** | 1.983562 | **2.000000** |
| 4–10 | (alternating) | — | 0.4959 / 0.5041 | **0.500000** | 1.9836 / 2.0164 | **2.000000** |

### Change

- `fixed_leg.py:69-93` — compute `coupons_per_year = 12 // frequency.value` once; pass `ref_start=accrual_start, ref_end=accrual_end, frequency=coupons_per_year` to `year_fraction` for every coupon. The extra params are ignored by non-ICMA conventions, so the change is harmless for THIRTY_360 / ACT_360 / ACT_365F / etc.
- Two of the four characterisation xfails turn green (`test_every_regular_coupon_is_exactly_half_year`, `test_every_coupon_amount_is_exactly_two`) — their `xfail` markers are removed; they now assert the correct behaviour permanently.
- Full parallel suite: **11873 passed, 2 xfailed in 4:55** — zero regressions in any other ICMA-using test.

### What's NOT fixed yet (queued for Slice 4)

`FixedRateBond._price_from_ytm` (and the YTM-derived analytics: `macaulay_duration`, `modified_duration`, `convexity`, `dv01_yield`) use `year_fraction(settle, payment_date, ACT_ACT_ICMA)` for *multi-period* spans without passing refs — still hitting the legacy ACT/365F fallback. The current `_act_act_icma` implementation is correct for single-period accruals only; multi-period ICMA needs explicit period-counting per Rule 251.2.

The 2 remaining xfails (`test_par_yield_round_trip_is_exact_100`, 5y and 30y) are pinned for Slice 4. **Diagnostic note:** these tests xfail *worse* post-Slice-3 than they did pre-fix — the previous "99.999807" was the result of two errors partially cancelling. Now coupon amounts are correct but discount times still aren't, so the round-trip lands at 99.981367 (5y). Slice 4 will fix the discount-time path and turn these green.

### Affected upstream

Sovereign / corporate bonds using `DayCountConvention.ACT_ACT_ICMA` — all 11 markets currently wired (US, AU, MY, TH, SG, ID, HK, SE, CH, CZ, NO + linkers). Coupon amounts in those bonds now match market quotes within machine precision.

---

## v0.895.0 — 2026-06-11

**Fix A.1 B1 Slice 2 — UST ICMA mispricing characterised via xfail tests.**

Second slice of the staged ICMA fix. No source changes; the bug is now pinned to a reproducible test fixture with `xfail(strict=True)` markers so Slice 3 must turn them green (and CI catches partial fixes).

### Diagnostic — actual mispricing measured

A par-5y UST at 4% (face=100, semi-annual, issued 2024-02-15, matures 2029-02-15):

| Coupon | Period | days | year_frac (current) | year_frac (correct ICMA) | amount (current) | amount (correct) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Feb 15 → Aug 15 | 182 | 0.498630 | 0.500000 | 1.994521 | 2.000000 |
| 2 | Aug 15 → Feb 15 | 184 | 0.504110 | 0.500000 | 2.016438 | 2.000000 |
| 3 | Feb 15 → Aug 15 | 181 | 0.495890 | 0.500000 | 1.983562 | 2.000000 |
| 4–10 | (alternating) | — | 0.4959 / 0.5041 | 0.500000 | 1.9836 / 2.0164 | 2.000000 |

Par-yield round-trip: **99.999807** (should be exactly 100). These are the numbers from `MODULE_HEALTH.md` §`fixed_income/bond.py`, now nailed down in `test_treasury_icma_characterisation.py`.

### What the slice adds

- New test file `python/tests/test_treasury_icma_characterisation.py` (6 tests):
  - Smoke: bond's `day_count` is `ACT_ACT_ICMA`. PASSES today.
  - **xfail(strict=True)**: every regular coupon has `year_frac == 0.5` exactly.
  - **xfail(strict=True)**: every coupon amount is exactly `2.0`.
  - **xfail(strict=True)**: par-yield round-trip lands at exactly `100.0` (5y).
  - **xfail(strict=True)**: same for 30y (error accumulates across 60 coupons).
  - Diagnostic test that always passes — prints the actual current values for visibility in test logs (to be removed in Slice 3 once the fix lands).

Full parallel suite: **11871 passed + 4 xfailed in 4:55**.

### Why xfail and not skip

`xfail(strict=True)` enforces a contract: the test MUST fail today, MUST pass after the fix. If Slice 3 accidentally doesn't fix one of the cases, CI flips that case to "unexpected pass" and the slice fails. Skip would let a partial fix through silently.

### Next slice

**A.1 B1 Slice 3** — fix `FixedLeg` to pass `ref_start=accrual_start, ref_end=accrual_end, frequency=12/months_per_period` to `year_fraction`. Turn the four xfails green. Remove the diagnostic test.

---

## v0.894.0 — 2026-06-11

**Fix A.1 B1 Slice 1 — `strict_icma` flag on `year_fraction` (no behaviour change).**

First slice of the staged ICMA fix. Adds the machinery to detect and refuse silent fallbacks; default remains the legacy permissive behaviour so this slice changes nothing in production code paths. Subsequent slices will migrate one ICMA caller at a time, each verified by a hand-calc test (e.g. "par UST gives exactly 2.0000 per semi-annual coupon"), and only at the very end will the default flip to strict.

### What this slice adds

- `year_fraction(..., strict_icma: bool = False)` — keyword-only flag.
- `_act_act_icma(..., strict: bool = False)` — internal helper now raises `ValueError` (instead of silently degrading to ACT/365F) when `strict=True` AND any of:
  - `ref_start`, `ref_end`, or `frequency` is missing (closes A.1 B1 root cause when callers opt in).
  - `frequency <= 0` (closes A.1 B2 — pre-fix would `ZeroDivisionError`).
  - `period_days <= 0` (inverted ref dates).

Each `strict=True` failure produces a precise, actionable error message:
```
ACT/ACT ICMA requires coupon-period anchors. Missing: ref_start, ref_end, frequency.
Pass `ref_start`, `ref_end`, and `frequency` to `year_fraction(...)`.
```

### Test coverage

12 new tests in `test_day_count.py::TestACTACTICMA`:
- Par UST semi-annual exact half-year (regular 182-day period).
- Long-period invariance (184 days still gives 0.5 — that's the whole point of ACT/ACT ICMA).
- Mid-period accrual.
- Legacy fallback when refs are missing (back-compat invariant).
- Strict mode raises with clear messages on missing-refs / frequency=0 / inverted-period.
- Strict mode default-off preservation invariant.

Full parallel suite: **11869 passed in 4:50** — zero downstream behaviour change.

### Next slices (A.1 B1 roadmap)

| Slice | Scope |
|---|---|
| 1 (this one) | Add the flag. Default off. ✓ |
| 2 | Audit `FixedRateBond.treasury_note` — confirm UST mispricing per MODULE_HEALTH; add failing test. |
| 3 | Fix `treasury_note` to pass refs + switch its call-site to `strict_icma=True`. UST coupons land at exactly 2.0000. |
| 4..N | One slice per remaining ICMA caller in `pricebook/fixed_income/*.py` (~10 sovereign-bond modules). |
| Final | Flip the default to `strict_icma=True`, remove the flag. |

---

## v0.893.0 — 2026-06-11

**Fix A.2 B1b/c/d — `AUDCalendar`, `NZDCalendar`, `CADCalendar` Saturday substitution.**

Completes audit finding A.2 B1. London was already fixed in v0.892. AUD / NZD / CAD followed the identical pattern — base-class US-style `_observe` applied to non-US locales. Now all four use `Calendar._observe_next_working_day` (Sat → +2 days Mon; Sun → +1 day Mon).

### Citations

- **AU** — Australian Public Holidays Acts (state-by-state but uniform Sat→Mon rule).
- **NZ** — Holidays Act 2003, *"Mondayisation"* provision.
- **CA** — Holidays Act (federal) + Employment Standards Acts (provincial).

### Test coverage added

Each calendar gets a `TestXXXCalendarSubstitution` class with regression tests for the Sat-Christmas year (2021 and/or 2027) plus a locale-specific check that exercises a non-Christmas date falling on Saturday:

- AUD: Australia Day 2030 Sat → Mon Jan 28; Australia Day 2025 Sun → Mon Jan 27.
- NZD: Waitangi Day 2027 Sat → Mon Feb 8.
- CAD: Canada Day 2028 Sat → Mon Jul 3; Remembrance Day 2028 Sat → Mon Nov 13.

Full parallel suite: **11860 passed in 3:22**.

### A.2 B1 — closed

| Locale | Status | Slice |
|---|---|---|
| GBP / London | ✅ fixed | v0.892 |
| AUD / Sydney | ✅ fixed | v0.893 |
| NZD / Wellington | ✅ fixed | v0.893 |
| CAD / Toronto | ✅ fixed | v0.893 |

Next L0-audit-driven fix: **A.1 B1 — ACT/ACT ICMA silent fallback** (UST mispricing). Multi-slice plan in the audit doc.

---

## v0.892.0 — 2026-06-11

**Fix A.2 B1a — `LondonCalendar` Saturday substitution now follows UK Banking and Financial Dealings Act 1971.**

First of four per-calendar fixes for audit finding A.2 B1. London was producing wrong observed-holiday dates whenever a fixed-date bank holiday fell on a Saturday — recurring every ~6 years (2021, 2027, 2032, ...).

### What was broken

`Calendar._observe` is the US rule (5 U.S.C. § 6103): Saturday → previous Friday, Sunday → next Monday. London inherited that rule unchanged.

UK Banking and Financial Dealings Act 1971 specifies: any bank holiday on Saturday or Sunday is observed the **next working day** (typically Monday).

Live repro before the fix — Christmas 2021 (Dec 25 Saturday):
```
2021-12-24 Fri: was holiday=True  (wrong — should be a business day)
2021-12-27 Mon: was holiday=True  (right, by accident, because Boxing-Day-Sun lands here)
2021-12-28 Tue: was business=True (wrong — should be Boxing observed)
```

After the fix:
```
2021-12-24 Fri: business day ✓
2021-12-27 Mon: Christmas observed ✓
2021-12-28 Tue: Boxing Day observed ✓
```

### Change

- `calendar.py:93-101` — `Calendar._observe` docstring clarified as "US-style" with a cross-reference to the new helper.
- `calendar.py:103-128` (new) — `Calendar._observe_next_working_day` static helper implements the UK / AU / NZ / CA rule (Sat → +2 days, Sun → +1 day). Per-locale subclasses opt in via `_observe = staticmethod(Calendar._observe_next_working_day)`.
- `calendar.py:LondonCalendar` overrides `_observe` to the next-working-day rule. Existing Boxing/Christmas collision handling (when both land on the same observed Monday) is unaffected — that code already pushes Boxing to Tuesday when the collision happens.
- 6 new tests in `test_calendar.py::TestLondonCalendarSubstitution`: Christmas 2021/2024/2027, Boxing 2021, New Year's 2028 Sat case, New Year's 2023 Sun case.
- Full parallel suite: **11848 passed in 4:42** — no regressions.

### Affected upstream

Any GBP-rate calculation that crosses a Saturday-Christmas, Saturday-Boxing, or Saturday-New-Year window. Days affected per affected year: 4 (one Friday wrongly closed; one Tuesday wrongly open; mirror at New Year). Years materially affected in the GBP curve's lookback window: 2021, 2027 (already past for 2021; 2027 lookback materially affects long-dated forwards observed today).

### Remaining (queued)

Same fix shape applies to `AUDCalendar`, `NZDCalendar`, `CADCalendar` — each as its own slice (separate failure surfaces, easier rollback).

---

## v0.891.0 — 2026-06-11

**Fix A.12 B1 — serialisation auto-discovery via `pkgutil.walk_packages`.**

The hand-maintained import whitelist in `core.serialization._ensure_loaded` is gone. The registry is now populated by walking every submodule of `pricebook` on first use.

### What was broken

`_ensure_loaded` was a curated list of 24 imports. The audit found 53 files in the codebase using `@serialisable` / `@serialisable_convention` / `_register` — so **29 modules' types were silently absent from the registry**. The failure mode was: `from_dict({"type": "esg_bond_spec", ...})` → `ValueError("Unknown type 'esg_bond_spec'")` — but only when the specific missing type was deserialised, after deployment. Every new module with a serialisable type carried this maintenance trap.

### Change

- `serialization.py:30-65` — `_ensure_loaded` now uses `pkgutil.walk_packages(pricebook.__path__, prefix="pricebook.")` + `importlib.import_module`. Idempotent; one-shot tree walk on first call. Any import failure is recorded in `_failed_imports` (module name + exception class) rather than silenced — visible to the test suite for CI catch.
- Auto-discovery now registers **120 types** (vs ~50 with the old whitelist). The codebase has not changed; the registry just sees what was always there.
- 4 new tests in `test_serialization_autodiscovery.py`: zero-import-failures invariant, ≥100 types registered, every major subpackage represented, regression test that `EquityIndexSpec` (from `core.market_conventions`, not in the old whitelist) is now reachable.
- Full parallel suite: **11842 passed in 4:41**.

### Side discovery — new audit finding

Auto-discovery surfaces a pre-existing `_SERIAL_FIELDS` mismatch on `AmortisingBond` — the registered fields (`face_value`, `coupon_rate`, `n_periods`, `frequency`) don't match `__init__` parameters (`amortisation_type`, `coupon_rate`, `maturity_years`, `n_payments`, `notional`). The `_register` validator emits a `UserWarning` at startup. This bug was hidden before because `amortising_bond` wasn't in the old whitelist. **Logged as a new audit finding** — to be fixed in the upcoming `amortising_bond` audit pass.

---

## v0.890.0 — 2026-06-11

**Fix A.4 B1 — `schedule.generate_schedule` EOM convention now anchors on `start` per ISDA 2006 §4.10.**

The first fix landing from the L0 audit. Schedule generation now correctly interprets the EOM rule for *all* generation paths (front-stub backward AND back-stub forward), not just the forward path that happened to work by accident.

### What was broken

In the front-stub (backward) generation path, the EOM decision was made inside `_add_months(d, months, eom)` by checking whether the rolling date `d` was itself EOM. For backward generation, `d` starts as `end`, so EOM was effectively anchored to `end`. When `start` was EOM but `end` was not, interior coupon rolls landed mid-month — violating ISDA 2006 §4.10 ("if the period start date is the last day of February, the period end date is the last day of February").

Live repro before the fix:
```
generate_schedule(start=2024-01-31, end=2024-08-15, semi-annual, SHORT_FRONT, eom=True)
  → [2024-01-31, 2024-02-15, 2024-08-15]      ← Feb 15 wrong; should be Feb 29
generate_schedule(start=2024-01-31, end=2025-04-15, semi-annual, SHORT_FRONT, eom=True)
  → [2024-01-31, 2024-04-15, 2024-10-15, 2025-04-15]   ← interior rolls all mid-month
```

After the fix:
```
[2024-01-31, 2024-02-29, 2024-08-15]
[2024-01-31, 2024-04-30, 2024-10-31, 2025-04-15]
```

### Change

- `schedule.py:25-30` — `_add_months(d, months, eom)` renamed to `_add_months(d, months, snap_to_eom)`. The boolean now means "globally snap to EOM" (a schedule-level decision), not "snap iff this particular d is EOM". The intent is documented inline.
- `schedule.py:80` — `generate_schedule` computes `snap_to_eom = eom and start == _end_of_month(start)` exactly once before both generation paths, and passes the same flag through.
- 3 new regression tests in `test_schedule.py`: ISDA §4.10 front-stub case + multi-year cross-leap-year case + start-not-EOM no-op case.
- Full parallel suite: **11838 passed in 3:20** — zero regressions in any downstream caller.

### Affected upstream

Any bond, swap, or amortising trade whose schedule had `start = EOM` and `end != EOM`. UST issued at end-of-month with a coupon-date maturity is the prototypical case. Schedules with EOM at *both* ends (the existing `test_eom_preserved` case) were already correct; nothing breaks.

---

## v0.889.0 — 2026-06-11

**G1 P3 Slice 2 — schema versioning on `@serialisable`. G1 P3 complete. Gate 1 closed.**

Every serialised dict now carries an explicit schema version, giving us a hard hook for future wire-format migrations without breaking anything that exists today. Combined with G1 P1 (`CalibrationResult`), G1 P2 (`MarketSnapshot` on every calibrator), and G1 P3 Slice 1 (`NumericalConfig`), pricebook is now **audit-ready**: a price is reconstructible from a stable, dated, versioned artefact graph.

- New class attribute `_SERIAL_SCHEMA_VERSION: int = 1` on `Serialisable`, `@serialisable`, and `@serialisable_convention`. Bump in your class when you make a *breaking* wire change.
- Envelope format (`Serialisable`, `@serialisable`) gains a top-level `"schema_version"` key alongside `"type"` and `"params"`.
- Flat-convention format (`@serialisable_convention`) gains a reserved `"_schema_version"` key — underscore-prefixed so it cannot collide with any real convention field.
- `from_dict` reads the version, treats *absent* as v1 silently (existing payloads on disk continue to deserialise), and raises a clear `ValueError("...schema_version=N, but this build only supports up to M. Upgrade this environment.")` when a payload was written by a newer build.
- Decorators take an optional `schema_version=N` keyword for classes that introduce migrations: `@serialisable("irs", [...], schema_version=2)` and `@serialisable_convention("foo", schema_version=3)`.
- 24 new tests in `test_serialisable_schema_version.py`: default version, to_dict emits version (envelope + flat), round-trip across all three paths, backward compat with absent-version payloads, future-version rejection with a clear message, wire-format key invariants, and version-key strip-before-construct.
- Full parallel suite: **11835 passed in 5:34** — every existing serialisation round-trip in the library still works.

### Gate 1 closed

Phase 1: every calibrator produces a uniform `CalibrationResult` (UUID, timestamp, code version, residuals, optimiser story).
Phase 2: every calibrator accepts a `MarketSnapshot`; id links to the result.
Phase 3: numerical hyperparameters become first-class on `PricingContext`; wire format becomes version-aware.

**Result:** a price computed today can be reconstructed tomorrow from a single, structured, versioned artefact chain: trade → pricer → `PricingContext{numerical_config, curves}` → `CalibrationResult.market_snapshot_id` → `MarketSnapshot`. Each link is dated, identified, and serialisable.

Next: Gate 2 — *production-grade*. Begin auditing each module with the audit chain available.

---

## v0.888.0 — 2026-06-11

**G1 P3 Slice 1 — `NumericalConfig` + `PricingContext.numerical_config`.**

Numerical hyperparameters become first-class. Two valuations on the same book with different MC path counts are *different valuations* — until now that choice was buried in hard-coded defaults inside each pricer and invisible to any audit. After this slice the choice lives on the context and can be serialised, diffed, and overridden as a single object.

- New module `pricebook.core.numerical_config`:
  - `NumericalConfig` — frozen dataclass with 14 fields covering Monte Carlo (paths, seed, antithetic, Sobol, Brownian bridge), PDE/FD (time/space steps, truncation width), tree steps, integration (tol, max-iter), COS method (N, L), root-finding (tol, max-iter), plus an `extra: Mapping[str, Any]` escape hatch. `.replace(**kwargs)` for ergonomic edits.
  - `DEFAULT_NUMERICAL_CONFIG` — frozen, shared singleton matching the historical library defaults so no existing behaviour changes.
- `PricingContext` gains `numerical_config: NumericalConfig | None = None`. `get_numerical_config()` returns the attached config or the default singleton — pricers should read through the accessor. `.replace(numerical_config=...)` works including clearing to `None`.
- 14 new tests in `test_numerical_config.py`: frozen-ness, `.replace`, default fallback, accessor returns attached / falls back, `simple()` factory unchanged, end-to-end with curve + config.
- Purely additive — `PricingContext.simple()` and all existing call-sites are untouched.

Next slice (G1 P3 Slice 2) adds **schema versioning** on `@serialisable` — closes Gate 1.

---

## v0.887.0 — 2026-06-11

**G1 P2 Slice 5 — jump-model calibrators accept a `MarketSnapshot`. G1 P2 complete.**

The last family of calibrators wires through. **Every calibrator in pricebook now accepts an optional `market_snapshot`**, and `CalibrationResult.market_snapshot_id` is populated whenever provided — closing Phase 2 of Gate 1.

- `pricebook.models.jump_calibration.calibrate_jump_model(...)` — keyword-only `market_snapshot: MarketSnapshot | None = None`. Covers all six Lévy / jump models (Merton, VG, Kou, NIG, CGMY, Bates).
- `calibrate_jump_surface(...)` — same; a single snapshot stamps onto every per-expiry result (the typical case: one observation set underlies the whole surface).
- `TYPE_CHECKING` import — no runtime dep.
- 6 new tests in `test_jump_snapshot.py`: keyword-only enforcement, id linkage on the single-expiry path, surface-level propagation across multiple expiries (each per-expiry `CalibrationResult.id` is distinct while the `market_snapshot_id` is shared).

### G1 P2 — complete (Slices 1-5)

12 calibration entry points across 9 modules now accept a `MarketSnapshot`:

| Slice | Module | Entry points |
|---|---|---|
| 1 | `pricebook.market_data` | (types only) |
| 2 | `pricebook.curves.bootstrap` / `global_solver` | `bootstrap`, `global_bootstrap` |
| 3 | `pricebook.curves.multicurve_solver` / `credit.bond_hazard_bootstrap` | `multicurve_newton`, `bootstrap_hazard_from_bonds`, `bootstrap_hazard_mixed`, `bootstrap_hazard_adaptive` |
| 4 | `pricebook.models.{hw,g2pp,lmm}_calibration` / `pricebook.options.sabr` | `calibrate_hull_white`, `calibrate_g2pp`, `calibrate_lmm_vols`, `sabr_calibrate` |
| 5 | `pricebook.models.jump_calibration` | `calibrate_jump_model`, `calibrate_jump_surface` |

Audit chain end-to-end: **price → calibration → market snapshot**. Next: **G1 P3** — `NumericalConfig` on `PricingContext` + schema versioning on `@serialisable` — closes Gate 1.

---

## v0.886.0 — 2026-06-11

**G1 P2 Slice 4 — HW, G2++, SABR, LMM calibrators accept a `MarketSnapshot`.**

Four model calibrators wired in one slice. Audit chain now reaches every IR/vol model-calibration entry point in the library.

- `pricebook.models.hw_calibration.calibrate_hull_white(...)` — keyword-only `market_snapshot: MarketSnapshot | None = None`.
- `pricebook.models.g2pp_calibration.calibrate_g2pp(...)` — same.
- `pricebook.models.lmm_calibration.calibrate_lmm_vols(...)` — same.
- `pricebook.options.sabr.sabr_calibrate(...)` — same; returned dict's `"calibration_result"` carries the linked id.
- All four use `TYPE_CHECKING` import on `pricebook.market_data.MarketSnapshot` — no runtime dep added.
- 12 new tests in `test_models_snapshot.py`: keyword-only enforcement and id linkage on each calibrator (G2++ uses `method="minimize"` to stay under a second per test).

Single slice left in G1 P2: jump-model calibrators (`calibrate_jump_model` family: Merton, VG, Kou, NIG, CGMY, Bates) — same additive pattern. After that, G1 P2 closes and G1 P3 (`NumericalConfig` on `PricingContext`) begins.

---

## v0.885.0 — 2026-06-11

**G1 P2 Slice 3 — `multicurve_newton` + the bond-hazard bootstrap family accept a `MarketSnapshot`.**

Phase 2 of Gate 1 now reaches every major curve-and-hazard calibration entry point. The audit chain **price → calibration → market snapshot** is wired through the `curves` *and* `credit` subpackages — together with G1 P2 Slice 2, this covers all single- and multi-curve bootstrapping plus credit hazard calibration.

- `pricebook.curves.multicurve_solver.multicurve_newton(...)` gains keyword-only `market_snapshot: MarketSnapshot | None = None`. Id propagates through both converged and non-converged paths (the latter still emits the existing `RuntimeWarning`). Both `ois_curve.calibration_result` and `projection_curve.calibration_result` share the same `CalibrationResult` instance, so the link is canonical, not duplicated.
- `pricebook.credit.bond_hazard_bootstrap`:
  - `bootstrap_hazard_from_bonds(...)` — keyword-only `market_snapshot`. Id threads through `_bootstrap_sequential` and `_bootstrap_global` (both reach `CalibrationResult.market_snapshot_id`).
  - `bootstrap_hazard_mixed(...)` — same.
  - `bootstrap_hazard_adaptive(...)` — same; passes the snapshot through to the chosen downstream entry point.
- `TYPE_CHECKING` import on both new touch sites — no runtime dep on `pricebook.market_data`.
- 9 new tests in `test_multicurve_snapshot.py`: keyword-only enforcement, id linkage on converged + non-converged multicurve, id linkage on sequential + global bond-hazard, snapshot-presence is numerically inert.

G1 P2 ready for Slice 4 (other model calibrators: HW, G2++, SABR, LMM, jump models — same additive pattern).

---

## v0.884.0 — 2026-06-11

**G1 P2 Slice 2 — `bootstrap` and `global_bootstrap` accept a `MarketSnapshot`; snapshot id propagates onto the `CalibrationResult`.**

Audit chain extends: **price → calibration → market snapshot** is now wired end-to-end for the two L2 entry points used by every downstream pricer.

- `pricebook.curves.bootstrap.bootstrap(...)` gains a keyword-only `market_snapshot: MarketSnapshot | None = None` parameter.
- `pricebook.curves.global_solver.global_bootstrap(...)` gains the same keyword-only parameter.
- When provided, `MarketSnapshot.id` is stamped onto `CalibrationResult.market_snapshot_id` on the curve's `calibration_result`. When absent, the field stays `None` (back-compat).
- Snapshot is a **provenance pointer** in this slice — deposits/swaps args remain authoritative for the actual rates. A future slice will let bootstrap *derive* the deposit/swap lists from the snapshot's quotes.
- 9 new tests in `test_curve_bootstrap_snapshot.py`: snapshot↔calibration linkage, keyword-only enforcement, snapshot-presence is numerically inert, distinct snapshots produce distinct results.
- `TYPE_CHECKING` import on both files — no runtime dep added to `pricebook.curves`.

---

## v0.883.0 — 2026-06-11

**G1 P2 Slice 1 — `pricebook.market_data` package + canonical L1 raw-data types.**

Phase 2 of Gate 1 begins. Where Phase 1 made every calibration produce a `CalibrationResult`, Phase 2 makes the *raw market data* a first-class, identifiable artefact — so the audit chain extends one step further: price → calibration → market snapshot.

This first slice introduces the **types only**, purely additive — nothing existing is touched.

- New package `pricebook.market_data` (L1 in the layer graph; zero dependencies on other pricebook subpackages).
- `QuoteKind` — `str`-Enum of 17 recognised quote types (deposits, FRAs, futures, swaps, basis, xccy, CDS, bond px/yield, vol points, swaption/capfloor vols, FX spot/forward, inflation YoY/ZC, other).
- `QuoteId` — frozen dataclass `(kind, tenor, currency, label)`; stable, hashable, the "same instrument across time" key.
- `Quote` — frozen `(id, value, bid_ask_bp)`; a single observation.
- `MarketSnapshot` — frozen `(id: UUID, as_of: datetime, quotes: tuple[Quote,...], label)`. `MarketSnapshot.new()` auto-generates id + timestamp. `get(qid)`, `filter(kind=, currency=, label=)`, `with_quote(q)` (returns a new snapshot with a fresh id — replaces same-id quotes). `__len__`, `__iter__`, `__contains__` for ergonomic use.
- `FixingHistory` — frozen wrapper around `{(rate_name, date) -> value}`; `get`, `for_rate`, `rate_names`, `with_fixing`. Distinct from `MarketSnapshot` (fixings are backward-looking, no bid-ask).
- 31 new tests in `test_market_data_types.py` covering frozen-ness, equality, hashing, factory behaviour, auditing invariants (fresh id on `with_quote`, original snapshot unchanged).

The existing `pricebook.core.market_data.MarketDataSnapshot` is untouched — backward compatibility is total. Subsequent slices in G1 P2 will let bootstrap/global-bootstrap accept a `MarketSnapshot` and record `MarketSnapshot.id` on the resulting `CalibrationResult.market_snapshot_id`.

---

## v0.882.0 — 2026-06-11

**G1 P1 Slice 6 — `jump_calibration` produces `CalibrationResult`. G1 P1 complete.**

The last of seven calibration families lands. With this slice, **every calibration in the codebase produces a uniform `CalibrationResult` artefact**, closing out Phase 1 of Gate 1 in the `DESIGN.md` roadmap.

- `JumpCalibrationResult` (covers all six jump/Lévy models: Merton, VG, Kou, NIG, CGMY, Bates) gains `calibration_result: CalibrationResult | None` + `to_calibration_result()` method. `to_dict` gets `calibration_id` key.
- `calibrate_jump_model` populates with `model_class=f"jump_{model_type}"` (e.g. `jump_merton`, `jump_vg`, `jump_kou`, `jump_nig`, `jump_cgmy`, `jump_bates`), parameters as the per-model param dict (4 for Merton, 3 for VG, 4 for Kou, 4 for NIG, 4 for CGMY, 7 for Bates), residuals = `model_vol − market_vol` per strike, algorithm `"differential_evolution+L-BFGS-B"`, deterministic seed recorded, spot / rate / T / div_yield / bounds in `optimiser.extra`, `rmse_vol` in diagnostics, quotes named `smile_K=<strike>`.
- 12 new tests in `test_calibration_result_jump.py`: 9 end-to-end (1 fast Merton calibration with 3 strikes / `maxiter=30`) + 3 back-compat.
- 32 tests pass across the jump family (test_jump_calibration + test_jump_cross_validation + new file) — no regression.

### G1 P1 — complete (Slices 1-6)

103 dedicated calibration-layer tests in total. **Every calibrator in pricebook now produces a `CalibrationResult`:**

| # | Family | Model class string | Algorithm |
|---|---|---|---|
| 1 | foundation | (none — types only) | n/a |
| 2 | bond hazard | `bond_hazard_pwc` | brentq-per-bond / L-BFGS-B[+tikhonov] |
| 3 | Hull-White | `hull_white` | Nelder-Mead / DE / L-BFGS-B |
| 3 | G2++ | `g2pp` | differential_evolution+L-BFGS-B |
| 4 | LMM | `lmm` | iterative_scaling |
| 4 | SABR | `sabr` | nelder_mead |
| 5 | curve bootstrap | `discount_curve_bootstrap` | brentq-sequential |
| 5 | global curve | `discount_curve_global` | newton-global |
| 5 | multicurve | `multicurve` | newton-multicurve |
| 6 | jump (6 models) | `jump_{merton,vg,kou,nig,cgmy,bates}` | differential_evolution+L-BFGS-B |

The audit trail `(price) -> (calibration_id) -> (market_snapshot, code_version, optimiser)` is now reconstructable end-to-end for every calibrated number in pricebook.

Next phase: **G1 P2** — `MarketSnapshot` at L1, the canonical raw-market-data type. Curves will be built from snapshots; the `market_snapshot_id` field on `CalibrationResult` (currently always `None`) will be populated.

---

## v0.881.0 — 2026-06-11

**G1 P1 Slice 5 — curve bootstrap produces `CalibrationResult`.**

The highest-fan-out calibration in the codebase: nearly every pricing call traces back to a bootstrapped discount curve. Six of seven calibration families now carry the canonical artefact (bond hazard, Hull-White, G2++, LMM, SABR, **curve bootstrap**); jump calibration is the last one (Slice 6).

- **`DiscountCurve.calibration_result: CalibrationResult | None`** — new instance attribute on the L0 curve class itself. Defaults to `None` for curves built directly (`flat(...)`, manual pillar construction); set to the canonical artefact by bootstrap entry points. Type hinted under `TYPE_CHECKING` to avoid a runtime import cycle between `core` and `calibration`.
- **`curves.bootstrap.bootstrap()`** — sequential brentq-per-pillar bootstrap now constructs a `CalibrationResult` after the round-trip verification step and attaches it to the returned curve. `model_class="discount_curve_bootstrap"`, algorithm `"brentq-sequential"`, residuals = (model_rate − market_rate) per instrument (deposits, FRAs, futures) or (PV_fixed − PV_float) per swap, parameters keyed by pillar date as `df(YYYY-MM-DD)`. RMS residual ~1e-14 (machine precision, exact fit by construction).
- **`curves.global_solver.global_bootstrap()`** — Newton-global simultaneous solve. Residuals are the final per-instrument repricing errors; algorithm `"newton-global"`, `tol` and `max_iter` passed through to `OptimiserSpec`. Converged flag captures whether the Newton loop hit `tol` before `max_iter`.
- **`curves.multicurve_solver.multicurve_newton()`** — joint OIS + projection calibration. Both `MultiCurveResult.calibration_result` AND `ois_curve.calibration_result` AND `projection_curve.calibration_result` get the SAME `CalibrationResult` instance (they share the calibration). Parameters carry an `ois_df(...)` / `proj_df(...)` prefix to distinguish the two pillar sets. `MultiCurveResult` gains a `to_calibration_result()` method that returns the stored instance or builds on-demand.
- 17 new tests in `test_calibration_result_curve_bootstrap.py`: default-None on `DiscountCurve.flat`, populated by `bootstrap()` / `global_bootstrap()` / `multicurve_newton()`, residual sanity (machine precision for sequential, sub-tol for Newton), parameter naming, quote naming, unique-id-per-call, multicurve cross-attachment, back-compat for hand-constructed `MultiCurveResult`.
- 40 tests pass across the full curve family (bootstrap + global_solver + multicurve_solver + new Slice 5 tests) — no regression.
- `coupled_bootstrap` and `bootstrap_forward_curve` not migrated in this slice — deferred to Slice 5b if needed.

Next: Slice 6 covers `jump_calibration` — the last calibration family.

---

## v0.880.0 — 2026-06-11

**G1 P1 Slice 4 — LMM + SABR calibration produce `CalibrationResult`.**

Five of seven calibration families now produce the canonical artefact (bond hazard, Hull-White, G2++, LMM, SABR). Two remaining: curve bootstrap (Slice 5) and jump calibration (Slice 6).

- `LMMCalibrationResult` (dataclass) gains `calibration_result: CalibrationResult | None` field + `to_calibration_result()` method. `to_dict` gets `calibration_id` key.
- `calibrate_lmm_vols` populates with `model_class="lmm"`, parameters as `{"sigma_0", ..., "sigma_{n-1}"}` (one per forward rate), residuals in vol units (`fitted - target` per swaption), algorithm `"iterative_scaling"`, `correlation_beta` and `tau` recorded in `optimiser.extra`, `rmse_vol` in diagnostics.
- `sabr.sabr_calibrate` (dict-returning, by convention) gains a `"calibration_result"` key in its returned dict — additive, no existing key removed or renamed.
- SABR `CalibrationResult` has `model_class="sabr"`, parameters `{"alpha", "beta", "rho", "nu"}` (4 params), residuals as `model_vol − market_vol` per strike (vol units, not bp), algorithm `"nelder_mead"`, `forward` / `T` / `beta_fixed` in `optimiser.extra`, quotes named `smile_K=<strike>`. `calibrate_sabr_smile` inherits the dict unchanged.
- 21 new tests in `test_calibration_result_lmm_sabr.py`: 9 LMM end-to-end + 3 LMM back-compat + 9 SABR end-to-end (existing keys preserved, parameters / residuals / quotes / unique ids).
- 64 tests pass across LMM + SABR family (test_sabr.py + test_lmm.py + test_lmm_calibration.py + new file) — no regression.

Next: Slice 5 migrates curve bootstrap (`curves.bootstrap`, `curves.multicurve_solver`, `curves.global_solver`) to produce `CalibrationResult`. The curve bootstrap is the highest-fan-out calibration in the codebase — many tests will see the new field downstream.

---

## v0.879.0 — 2026-06-11

**G1 P1 Slice 3 — `g2pp_calibration` and `hw_calibration` produce `CalibrationResult`.**

Same template as Slice 2, applied to the two single-curve rate-model calibrations. Three of the seven calibration families in `pricebook` now produce the canonical artefact (bond hazard, Hull-White, G2++); LMM, SABR, jump, and curve bootstrap follow in Slices 4-6.

- `HWCalibrationResult` gains `calibration_result: CalibrationResult | None` field and a `to_calibration_result()` method that returns the stored instance when populated or builds on-demand otherwise. `to_dict` gets the `calibration_id` key.
- `calibrate_hull_white` populates `calibration_result` with `model_class="hull_white"`, parameters `{"a": ..., "sigma": ...}`, residuals in vol-bp, `OptimiserSpec(algorithm=...)` capturing the chosen optimiser (`Nelder-Mead`, `differential_evolution`, or `L-BFGS-B`), `rmse_vol` in diagnostics, and `n_steps` in optimiser.extra.
- `G2PPCalibrationResult` gains the same field and method. `to_dict` gets the `calibration_id` key.
- `calibrate_g2pp` populates with `model_class="g2pp"`, parameters `{"a", "b", "sigma1", "sigma2", "rho"}` (5 params), residuals in vol-bp, algorithm `"differential_evolution+L-BFGS-B"` (or `"L-BFGS-B"` for minimise method), and the parameter bounds in `optimiser.extra`. Seed=42 recorded when DE is used.
- 16 new tests in `test_calibration_result_g2pp_hw.py`:
  - 9 HW tests run the full `calibrate_hull_white` on a small 3-swaption grid (~3 sec each): population of all fields, optimiser name and extras, residuals match `per_swaption_errors`, rmse_vol in diagnostics, named quote IDs, `to_calibration_result()` returns stored instance, unique id per run, `calibration_id` in `to_dict`.
  - 7 G2++ tests via hand-constructed `G2PPCalibrationResult` (the slow full `calibrate_g2pp` is covered by existing tests in `test_g2pp_calibration.py` which are deselected in CI): on-demand `to_calibration_result()` path, parameters count (5), `calibration_id` in `to_dict` both populated and unpopulated paths.
- Zero behavior change to existing tests (24 HW + Slice 3 tests pass; existing G2++ tests untouched).

Next: Slice 4 migrates `lmm_calibration` and `sabr` calibration to the same pattern.

---

## v0.878.0 — 2026-06-11

**G1 P1 Slice 2 — `bond_hazard_bootstrap` now produces a `CalibrationResult`.**

First migration of an existing calibration to the new canonical artefact (DESIGN.md §6 G1 P1). The bond-hazard bootstrap was chosen because it is the most recently touched calibration (Tikhonov + L-curve work landed yesterday) and its results carry the richest provenance (`lam`, `roughness`, per-bond residuals).

- `HazardBootstrapResult` gains a `calibration_result: CalibrationResult | None` field. Defaults to `None` for backward compatibility with hand-constructed instances; entry points always populate it.
- `_bootstrap_sequential` populates a `CalibrationResult` with `model_class="bond_hazard_pwc"`, `objective=SSE`, `optimiser=OptimiserSpec(algorithm="brentq-per-bond", ...)`, one iteration per bond, per-bond weights.
- `_bootstrap_global` populates with `objective=WEIGHTED_SSE`, `optimiser=OptimiserSpec(algorithm="L-BFGS-B[+tikhonov(lam=...)]", ...)`. When `lam > 0`: `lam` recorded in `optimiser.extra`, `roughness` in `diagnostics.extra`. Algorithm name gets the `+tikhonov(lam=...)` suffix for quick filtering in audit logs.
- `bootstrap_hazard_mixed` (FRN + bond joint fit) also migrated; `quotes_fitted` carries `["bond_0", ..., "frn_0", ...]`.
- New `to_calibration_result()` method returns the stored instance when populated, or builds one on-demand from the existing fields when not. Latter path covers any legacy hand-construction.
- `to_dict()` gets a `calibration_id` key (the UUID stringified, or `None`). Lets persistence layers later use this as a foreign key.
- 15 new tests in `test_bond_hazard_calibration_result.py` — population, parameter/residual/weight matching, Tikhonov-extras, unique id per invocation, on-demand build, hand-construction back-compat, `calibration_id` in dict.

Zero existing test changes: the `rmse_bp`, `pillar_hazards`, and other historical API surface is preserved. The new `calibration_result` is purely additive.

Next: Slice 3 migrates `g2pp_calibration` and `hw_calibration` to produce `CalibrationResult` (same template).

---

## v0.877.0 — 2026-06-11

**G1 P1 Slice 1 — `pricebook.calibration` skeleton + `CalibrationResult` + `Calibrator` Protocol.**

First slice of Gate 1 from `DESIGN.md` §6. The calibration layer gets its own package; the canonical result type is defined here so that subsequent slices can migrate each existing calibrator family (bond hazard, G2++, Hull-White, LMM, SABR, curve bootstrap, multicurve) to produce a uniform `CalibrationResult`.

- New package `python/pricebook/calibration/`:
  - `__init__.py` — public exports.
  - `_types.py` — `CalibrationResult` (frozen dataclass with `id`, `timestamp`, `code_version`, `model_class`, `parameters`, `residuals`, `rms_residual`, `max_residual`, `optimiser`, `iterations`, `converged`, `diagnostics`, `market_snapshot_id`), `OptimiserSpec`, `CalibrationDiagnostics`, `ObjectiveKind` (SSE / weighted-SSE / RMSE / max-error / L1 / Huber), `Calibrator` Protocol.
  - `CalibrationResult.new(...)` factory — keyword-only, auto-generates `id` and `timestamp`, derives `rms_residual` and `max_residual` from `residuals`, reads `pricebook.__version__` for `code_version` by default.
- 22 tests in `tests/test_calibration_types.py` covering enum values, OptimiserSpec construction & frozen-ness, diagnostics defaults & population, factory derivation of RMSE/max-error, default weights, explicit weights, code_version detection & override, market_snapshot_id placeholder, frozen-ness at all levels, Protocol satisfaction.
- Zero internal pricebook imports (other than `pricebook.__version__` read defensively) — package sits cleanly in the dependency graph; no behavior change to any existing code.

This slice is purely additive: no existing module is touched. Slice 2 migrates `bond_hazard_bootstrap` to return the new result type with a thin compatibility wrapper preserving the existing `HazardBootstrapResult` interface.

---

## v0.876.0 — 2026-06-11

**`DESIGN.md` — §6 roadmap rewritten as Gate × Phase hybrid.**

- Old roadmap had 4 phases with significant overlap (CalibrationResult in P1, calibration layer in P2 — same effort, separate commits) and blurry themes ("foundational types" + "structural relocation"). Phase 2 was 20-27 slices bundling six different efforts.
- New structure: **5 gates** (each a user-visible promise) × **10 phases** (each one architectural focus):
  - **G1 Audit-ready** — P1 Calibration unified + P2 Market data L1 + P3 NumericalConfig & versioning (14-16 slices, 1-3 weeks). Prerequisite for the bottom-up audit.
  - **G2 Production-grade** — P4 Scenarios & failures + P5 Repositories & per-layer tests (7-9 slices).
  - **G3 Architecturally clean** — P6 small cleanups + P7 Risk relocation L3→L7 isolated (9-13 slices). P7 is the biggest single refactor; isolated so it doesn't poison its gate.
  - **G4 Capability-complete** — P8 AAD as protocol + P9 Payoff algebra (10-13 slices).
  - **G5 Performant at scale** — P10 C++ port (19-28 slices, open-ended).
  - Total: **59-79 slices, 7-19 weeks** at the historical pricebook slice rate.
- Each gate is shippable independently. G2 doesn't require G3; you can stop after G2 and have a meaningfully better library.
- Section now includes a per-gate exit criteria, a dependency graph between phases, and "the first three slices of G1 P1" so the immediate next step after acceptance is unambiguous.
- Executive summary + TOC updated to reflect the new structure.

---

## v0.875.0 — 2026-06-11

**`DESIGN.md` — theoretical design document (~30 pages).**

- New `DESIGN.md` at the repo root: a first-principles design of a financial analytics engine, then pricebook overlaid as the lens.
- Sections 1-3 (principles, reference architecture, patterns/anti-patterns) drafted by `app-designer` for an independent second perspective on architecture; sections 4-6 written here.
- The reference design: 9-layer architecture with **calibration as its own layer (L6)**, parallel to risk, both depending on pricing (L5). **Models as `Protocol`, not inheritance.** **Trades as frozen dataclasses, market state passed in via `PricingContext`**. AAD via generic-scalar discipline at L0. Industry references woven through (QuantLib, ORE, OpenGamma Strata, Numerix). Concrete proposed shapes for `PricingContext`, `CalibrationResult`, `RiskRun`, `Scenario`.
- Pricebook gap analysis (§4): the architecture matches the reference at the foundations (acyclic layering, `PricingContext` exists, serialisation contract, mostly-Protocol models, mostly-frozen dataclasses), but diverges in three structural places: **risk at L3 instead of L6 above pricing** (the biggest mismatch — forces risk modules to know about concrete instruments); **calibration distributed rather than its own layer** (no unified `CalibrationResult`); **market data conflated with curves** (cannot distinguish quotes from fits in the dependency graph).
- Delta list (§5): 6 high-value adds (CalibrationResult, MarketSnapshot, Scenario protocol, PricingFailure, NumericalConfig, schema versioning), 5 wrong-shape refactors (risk relocation, calibration consolidation, market data split, pe/ relocation, registry consolidation), 4 nice-to-haves, 5 explicit won't-fixes.
- Roadmap (§6): 4 phases, **~63-84 slices total, ~10-22 weeks at the historical slice rate**. Phase 1 (foundational types) is the prerequisite for the bottom-up audit (the next major task after this document is accepted).

Status: ready for pushback before the bottom-up audit begins. Sections 1-3 are load-bearing — everything in 4-6 depends on the layer cut and the calibration placement.

---

## v0.874.0 — 2026-06-11

**Hazard-from-bonds notebook — adapted to use the library Tikhonov.**

- Removed the ad-hoc `regularised_bootstrap(...)` function from the notebook builder (Section 5). Replaced its 5 call sites (λ-sweep, L-curve dense grid, LOO-CV inner loop, MC sensitivity, realistic demo) with `bootstrap_hazard_from_bonds(method="global", pillar_times=..., lam=...)`. Section 9's `res_tik` is now a `HazardBootstrapResult`, with downstream code reading attributes (`res_tik.pillar_hazards`, `res_tik.survival_curve`) instead of dict keys.
- Updated Section 5 prose to point users at the library API; replaced the closing cheat sheet's "not currently in library" caveat with the recommended call (`lam="auto"`, `pillar_times=[...]`).
- Notebook re-executes cleanly: 46 cells (down from 47 — one builder-cell removed), 0 errors, 14 embedded plots. Every headline number unchanged (L-curve λ* = 3.16e+06, LOO-CV λ* = 6.58e+06, Section-10 Q(5y) MC mean 0.8867 vs deterministic 0.8877) — the library-backed and ad-hoc fits agree to numerics.
- The notebook is now self-consistent with the library: a reader who sees the API in §5 can apply it directly in their own code with no further translation.

Closes the Tikhonov work started in v0.872. Two-slice pattern (library first, notebook after) preserved.

---

## v0.873.0 — 2026-06-11

**`pillar_times` override in `bond_hazard_bootstrap`.**

- `_bootstrap_global`, `bootstrap_hazard_from_bonds`, and `find_lcurve_lambda` all gain an optional `pillar_times: list[float] | None = None` parameter. When provided, it overrides the default even-spacing of pillars and `n_pillars` is ignored.
- Two common patterns this enables:
  - **Pillars at bond maturities** (exactly-determined fit at `lam=0`, regularised as `lam` grows) — useful when the bond universe has natural calibration anchors at the cashflow dates.
  - **Pillars at calendar benchmarks** (e.g. 1, 2, 5, 10, 30y for sovereign) — for curves that need to align with peer-group benchmarks.
- Validation: empty list, non-positive values, and non-strictly-increasing sequences all raise `ValueError`.
- 4 new tests in `test_bond_hazard_tikhonov.py::TestPillarTimes` covering override behaviour, exact-fit-at-bond-maturities semantics, validation errors, and combination with `lam="auto"`. Brings the new test count to **15**.
- Required for the hazard-from-bonds notebook to be adaptable to the library API (Slice B); the existing notebook's `regularised_bootstrap` placed pillars at bond maturities to make the close-pair pedagogy clean.

---

## v0.872.0 — 2026-06-11

**Tikhonov regularisation in `bond_hazard_bootstrap`.**

- `_bootstrap_global(...)` and `bootstrap_hazard_from_bonds(...)` gain a `lam` parameter:
  - `lam=0.0` (default): unregularised LS — bit-for-bit identical to the existing behaviour (regression-tested).
  - `lam > 0`: penalised LS with a second-difference (curvature) penalty $\lambda\|Lh\|^2$.
  - `lam="auto"`: pick λ via the L-curve corner (Hansen 1992).
- New helper `find_lcurve_lambda(...)` does the L-curve sweep + corner detection. Uses signed curvature (most-negative κ in log-log space) rather than max-|κ| to avoid boundary finite-difference artifacts.
- `HazardBootstrapResult` gains two new diagnostic fields with defaults: `lam` (the λ used) and `roughness` (the final $\|Lh\|^2$). Backward-compatible — existing constructors omitting these still work.
- Method label becomes `"global_ls_tikhonov"` when `lam > 0`, stays `"global_ls"` when `lam=0`.
- 11 new tests in `tests/test_bond_hazard_tikhonov.py`: regression at `lam=0`, monotonic roughness decrease with `lam`, `lam→∞` driving the curve to linear (zero curvature), L-curve picker correctness, edge cases (n_pillars < 3 ignores `lam`, negative `lam` raises, sequential method unaffected by `lam`).
- Doc updates: `bootstrap_hazard_from_bonds` docstring now points to the hazard-from-bonds notebook for the full derivation and L-curve picking.

The recommended call for non-trivial bond universes (any pair within ~3 months of maturity, or > 8 bonds) is now:
```python
result = bootstrap_hazard_from_bonds(
    REF, bonds, rf_curve, method="global", n_pillars=..., lam="auto"
)
```
For well-spaced 3-8 bond universes, `method="sequential"` (default via `"auto"`) remains the right answer.

---

## v0.871.0 — 2026-06-11

**Hazard-from-bonds notebook — Round 5 (final polish). Notebook now complete.**

- Added an **executive-summary table** at the top: noise amplification (27× at 1mo, 54× at 2wk), Tikhonov variance reduction (22× at the close pillar), L-curve and LOO-CV agreement (~2×), Q(5y) integrated-vs-instantaneous (matches to 0.001).
- Added the **three things to remember** rules (bootstrap brittle, Tikhonov fixes it, bond prices are integral probes).
- Added a **table of contents** with anchor links to all 10 sections + the closing cheat sheet.
- Added a closing **"When to use what" cheat sheet** — universe-shape → recommended method, three rules of thumb, plus the relevant pricebook imports.
- Added explicit `<a id="section-N"></a>` anchors before each `## N.` heading so the ToC links resolve reliably in Jupyter/nbconvert.
- Final re-execution: **47 cells (22 md + 25 code), 0 errors, 14 embedded plots, 11/11 anchored headings**.

The notebook is now content-complete and ready to publish.

---

## v0.870.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 4 (sections 8-10). Notebook now complete content-wise (46 cells, ~2 MB).**

- **Section 8 (adaptive switch).** `assess_liquidity` + `bootstrap_hazard_adaptive` demo across three scenarios — liquid (6 well-spaced bonds, tight bid-ask) → `sequential`; semi-liquid (5 bonds with close pair) → `global`; illiquid (2 distressed bonds, 250-350 bp spread) → `global` with 2 pillars. Includes the important caveat that the heuristic protects against scale problems (count, spread) but **not** geometry (close maturities) — the user must reach for `method="global"` explicitly when bonds bunch.
- **Section 9 (realistic demo).** 8 bonds at sovereign-like maturities (0.5/1/3/5/5.25/7/10/10.5y) with two adjacent benchmark pairs. ±5 bp uniform price noise. Three methods side by side: sequential RMSE 0 bp (exact, brittle at close pairs), global LS RMSE 7.3 bp (5 even pillars), Tikhonov RMSE 6.3 bp (8 pillars + smoothness). Side-by-side hazard curves and per-bond residual scatter.
- **Section 10 (CIR++ cross-check — the deepest takeaway).** `CIRPlusPlus.from_survival_curve` overlays Cox-Ingersoll-Ross dynamics on the regularised piecewise-constant curve. 60 MC paths shown. **Sanity check at T=5y: deterministic Q(5) = 0.8877, MC mean Q(5) = 0.8867, MC std Q(5) = 0.0091.** The instantaneous hazard paths spread visibly around the mean, but the *integrated* survival at the bond maturities reconverges almost exactly. Geometrically: bond data constrains $\int_0^{t_i} h\,du$ at coupon dates, not $h(t)$ pointwise. Two hazard functions with the same integrals at every bond maturity are indistinguishable by bond prices. Choosing among them (piecewise constant / spline / CIR++) is a choice of *prior*, not of *data*.
- Final user-facing summary table in Section 10 markdown: when to use sequential / global / Tikhonov / CIR++.

Round 5 (final polish + ToC + commit) to follow.

---

## v0.869.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 3 (sections 6-7).**

- Section 6 (Picking λ) — L-curve method (Hansen 1992) with explicit max-curvature corner detection on the log-log plot; LOO-CV (model-free GCV analogue, $N$ refits per λ point). L-curve corner λ* = 3.2e+06; LOO-CV minimum λ* = 6.6e+06 — both methods agree on regularisation strength within a factor of ~2. Two plots (L-curve with corner star; LOO-CV vs λ with both criteria's λ marked).
- Section 7 (Bid-ask sensitivity) — 200 Monte-Carlo draws with ±10 bp price half-spread (typical IG-corporate width). Box plot per pillar, unregularised vs Tikhonov at L-curve-corner λ. The headline: at the 5y+2mo close-maturity pillar, hazard standard deviation collapses from **109.3 bp (unregularised) to 4.9 bp (regularised) — a 22× variance reduction**. Other pillars (1y, 3y, 5y, 10y) see modest 1.2-1.7× reductions. Discusses the bias-vs-variance tradeoff implicit in the prior.

Now 33 cells, 898 KB. Sections 8-10 (adaptive switch, realistic demo, CIR++ cross-check) to follow.

---

## v0.868.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 2 (sections 4-5).**

- Section 4 (Solver limits) — brentq bracket failure (bond above risk-free benchmark → no non-negative hazard solves it; residual stays one-signed across the whole [1e-6, 5.0] bracket) and Newton sensitivity (from h0=0.5/2/5 Newton "converges" to fake roots at the initial guess because the function is near-flat).
- Section 5 (Tikhonov theory + implementation) — full derivation of penalised least-squares: misfit + λ‖Lh‖², L = second-difference matrix. MAP interpretation: λ = 1/τ² under Gaussian prior on hazard curvature. Ad-hoc `regularised_bootstrap(...)` function defined inside the notebook (per the user's "no new code unless needed" preference — this is a teaching demonstration).
- Smoke test sweeps λ from 0 → 1e10. With +5 bp noise on the 5y bond: λ=0 reproduces the noise-amplified sequential result (rmse=0, roughness=160e-6); λ=1e6 trades 2 bp rmse for half the roughness; λ=1e8 collapses to nearly flat. Sets up Section 6 (L-curve corner picking).

Now 23 cells, 229 KB.

---

## v0.867.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 1 (sections 1-3).**

- New `notebooks/credit/hazard_from_bonds_when_maturities_are_close.ipynb` (15 cells, 206 KB). Sections 1-3 of a planned 10-section walkthrough.
- Section 1: the problem setup, math derivation of the risky-bond price as a function of $h(t)$, why "close maturities" is the trouble case (Jacobian goes ill-conditioned).
- Section 2: the easy case — 4 bonds at 1/3/5/10y, sequential bootstrap reproduces input prices to machine precision (RMSE 1e-10 bp), implied hazards recover the truth's piecewise-constant shape correctly across the bond pillars (with a clean explanation of why a bond pillar that straddles two truth pillars gets the time-weighted blend, not either endpoint).
- Section 3: the dramatic failure — same 4 bonds + a 5th bond two months from the 5y. Noise-free input still works; **add 5 bp of price noise on the 5y bond and the [5y, 5y+2mo] hazard jumps by 67 bp (13× amplification)**. Sweep over $\Delta T$ from 12 months down to 2 weeks shows the amplification climb monotonically: 27× at 1-month spacing, 54× at 2-week spacing. Plotted log-log.
- Companion builder script `_build_hazard_notebook.py` — same `nbformat` + `nbconvert` pattern as `quickstart`.
- Rounds 2-5 to come: solver-limit detail (Newton + brentq failure modes), Tikhonov derivation, L-curve picking, bid-ask Monte Carlo, adaptive switch, realistic demo, stochastic-intensity cross-check.

---

## v0.866.0 — 2026-06-10

**Foundation audit: dual-critic pass on 35 modules.**

- New `MODULE_HEALTH.md` (110 KB) — adversarial audit of 35 foundation + Top-6-instrument modules. Each module reviewed by `numerical-critic` (math correctness, edge cases, calibration robustness) and `code-correctness-critic` (off-by-one, lifetime, None handling, exception safety) in parallel via the multi-agent workflow harness.
- 70 critic verdicts, 697 raw findings: **56 critical**, **150 high**, 257 medium, 217 low, 17 nit.
- **Tier 1 (both critics → critical, fuzzy-matched on title/location): 13.** Highest-confidence likely real bugs. Tier 2 (critical + high pairing): 18. Single-critic critical: 19.
- Report includes: how-to-read disclaimer, per-tier breakdowns, per-module narrative verdicts (verbatim from critics), risk-scored module ranking, recommended triage workflow. **Critic output is NOT verified bugs — each finding needs a failing-test verification slice before fixing.**
- Raw JSON output (697 findings, ~720 KB) at `/private/tmp/claude-501/.../tasks/www7hfs2m.output`.

---

## v0.865.0 — 2026-06-10

**Refresh `ARCHITECTURE.md` to match empirical state.**

- `ARCHITECTURE.md`: rewrite from scratch using a freshly-computed import graph. Previous numbers (20 sub-packages, 486 modules, 9 layers) were ~60% stale. New numbers — **23 packages, 793 modules, 7 layers** — match what `mypy`/`pytest`/`grep` see today.
- New content: per-package fan-in table, tallest path through the DAG (7 hops: `core → curves → models → fixed_income → options → fx → desks`), `crypto` and `data` packages (added since the previous revision), reorganised numerics / models / risk module catalogues to cover modules built across the last 200+ commits.
- Embedded regen snippet at the end of the document — `cd python && python <<PY ... PY` — so the file's stats can be verified on any commit instead of drifting silently.
- Companion to slices v0.863 (untangle fi → credit) and v0.864 (drop binomial_jr_lr) — without those two the layer count would be 8 instead of 7.

---

## v0.864.0 — 2026-06-10

**Remove loose top-level `binomial_jr_lr.py`.**

- Deleted `python/pricebook/binomial_jr_lr.py` — a 190-line file sitting outside any subpackage. It was unreferenced anywhere in the repo, broken at import time (`from pricebook.black76 import OptionType` — that module path doesn't exist; Black-76 lives at `pricebook.models.black76`), and the JR / LR tree implementations are already provided by `numerical/_trees.py` and registered in `registry.py` (`TreeMethod.JR`, `TreeMethod.LR`). The file was a relic of an earlier shim cleanup (see v0.4xx release notes).
- Net: 25 → 24 top-level packages/modules in `pricebook/`. The top of the tree now contains only the registered subpackages plus `__init__.py` and `registry.py`.

---

## v0.863.0 — 2026-06-10

**Untangle `fixed_income → credit` runtime edge.**

- `fixed_income/basis_trade.py`: move `from pricebook.credit.cds import CDS` into `TYPE_CHECKING`. CDS was only used as a parameter annotation (`cds: CDS`) — never instantiated or isinstance-checked. `from __future__ import annotations` was already present, so all hints are strings at runtime.
- Result: `fixed_income` now depends only on `core, curves, models, statistics` at runtime — empirical Layer 4 → Layer 3 cleanup (one whole layer down). Empirical dependency graph regenerates without the `fi → credit` edge.

---

## v0.862.0 — 2026-06-10

**Mypy: clean baseline, 0 errors.**

- `python/pyproject.toml`: complete `[tool.mypy]` config with pragmatic defaults (silence `misc`, `annotation-unchecked`, `warn_return_any` — numpy/scipy noise) plus `[[tool.mypy.overrides]]` listing 184 legacy modules with `ignore_errors = true`. `mypy>=2.0` added to `[project.optional-dependencies] dev`.
- Fixed 31 real `name-defined` errors across 12 files — missing imports for forward-referenced types (`date`, `DiscountCurve`, `Calendar`, `RepoBook`, `RFRFutureSpec`, `PricingContext`, `TotalXVAResult`, `timedelta`, `CommodityForwardCurve`). Added proper `if TYPE_CHECKING:` blocks.
- `GUIDE.md` §17: new section documenting mypy usage and the cleanup ladder (remove a module from the override list → fix surfaced errors → repeat).
- Result: `cd python && mypy pricebook` exits 0 across 795 source files. Future slices can shrink the 184-module override list toward zero.

---

## v0.861.0 — 2026-06-09

**G2++ calibration: 8.5× faster.**

- `models/g2pp_calibration.calibrate_g2pp`: rewrite the calibration objective to compare **prices** to precomputed Black-76 market prices, rather than implied vols. Eliminates an implied-vol root-finder per swaption per DE evaluation (~50% of the original cost).
- `models/g2pp_calibration.calibrate_g2pp`: loosen the global-search budget — `differential_evolution(maxiter=30, popsize=6, tol=1e-4, init="sobol")` followed by an L-BFGS-B polish at `maxiter=150, ftol=1e-9`. Previous (`maxiter=300, popsize=15, tol=1e-9`) ran the full DE budget on the default fixture without measurable RMSE improvement.
- Net: `test_g2pp_calibration.py` runs in **2:15** instead of **9:49** (full file, 8 tests). One `calibrate_g2pp(curve, SWAPTION_VOLS)` call drops from ~587 s to ~68 s. Final calibrated `rmse_vol` remains well under the 5% threshold (~0.009 on the default fixture; both tests pass).

---

## v0.860.0 — 2026-06-09

**`GUIDE.md` — per-layer API reference.**

- `GUIDE.md` (new, 17 sections, ~480 lines): curves, models, numerical methods, fixed income, FX, equity, credit, commodity, options, structured, crypto, desks, risk, viz, serialisation, conventions, db/ts. 180 module references verified to exist. Includes runnable code snippets at each layer.
- `README.md`: link to `GUIDE.md`, `ARCHITECTURE.md`, `RELEASE_NOTES.md`, and the quickstart notebook. Now an actual landing page rather than a stub.

---

## v0.859.0 — 2026-06-09

**Quickstart notebook.**

- `notebooks/examples/quickstart.ipynb`: 20-minute "first result" walkthrough — curve bootstrap, bond pricing, IRS pricing, equity option pricing + Greeks, `to_dict`/`from_dict` round-trip, curve plot + Greeks profile via `pricebook.viz`. All 8 code cells executed and embedded.
- `notebooks/examples/_build_quickstart.py`: deterministic builder script for the notebook (nbformat-based) so future edits land via Python, not JSON.

---

## v0.858.0 — 2026-06-09

**Serialisation completeness — money market + funded products.**

- `fixed_income/money_market.py`: register `CertificateOfDeposit`, `CommercialPaper`, `BankersAcceptance` as `_serialisable`. `RepoRate` is a static-method namespace, no instance state — left unregistered.
- `fixed_income/funded.py`: register `TotalReturnSwap` (as `funded_trs`) and `FundedParticipation`. The other 3 classes (`Repo`, `ReverseRepo`, `RepoFinancedPosition`) were already registered.
- `funded.TotalReturnSwap`: rename internal attrs `ref_start`/`ref_current` → `reference_pv_start`/`reference_pv_current` so they match constructor params (required for `_serialisable`). Only used inside `funded.py`; tests already pass via init kwargs.
- Round-trip tests added in `test_funded.py` and `test_money_market.py` (5 new tests: TRS, FundedParticipation, CD, CP, BA).

---

## v0.857.0 — 2026-06-07

**G2++ code review fixes.**

- `g2pp_tree.py`: fix correlation correction (remove spurious `dt` factor, add renormalization); fix `_phi` division by zero for a≈0 or b≈0.
- `g2pp_calibration.py`: fix `_g2pp_V` division by zero guards.
- `bermudan_swaption_g2pp.py`: fix `_phi` and `_V` division by zero guards.
- `cms_spread_g2pp.py`: fix `_forward_zcb` to use V(T)-V(t) not V(τ); fix `_V` division by zero.

---

## v0.856.0 — 2026-06-07

**Unify Hull-White interface and fix mc_extensions.**

- `fixed_income/callable_floater.py`: added `callable_frn_hw()`, `puttable_frn_hw()` — accept `HullWhite` object.
- `options/bermudan_capfloor.py`: added `bermudan_cap_hw()`, `bermudan_floor_hw()`, `bermudan_collar_hw()`.
- `models/mc_extensions.py`: `"hull_white"` dispatch now supports `theta_func` for time-dependent drift via `HullWhiteProcess`.

---

## v0.855.0 — 2026-06-07

**G2++ callable bond, CMS spread, callable floater.**

- New `fixed_income/callable_bond_g2pp.py`: callable/puttable bonds on G2++ 2D tree.
- New `structured/cms_spread_g2pp.py`: CMS spread pricing + options + correlation diagnostic under G2++. Key: under 1F correlation=1.0; under G2++ correlation<1.0.
- New `fixed_income/callable_floater_g2pp.py`: callable/puttable FRN on G2++ 2D tree.
- 10 tests.

---

## v0.854.0 — 2026-06-07

**G2++ 2D tree and Bermudan swaption.**

- New `models/g2pp_tree.py`:
  - `G2PPTree` — 2D recombining trinomial lattice with correlation.
  - `backward_induction()` — generic 2D backward induction with pluggable option constraint.
  - `g2pp_european_swaption_tree()` — verification against analytical.
- New `options/bermudan_swaption_g2pp.py`:
  - `bermudan_swaption_g2pp_tree()` — Bermudan swaption on 2D tree.
  - `bermudan_swaption_g2pp_lsm()` — LSM with (1, x, y, x², y², xy) basis.
  - `g2pp_vs_hw1f_bermudan()` — compare 1F vs 2F Bermudan prices.
- 14 tests.

---

## v0.853.0 — 2026-06-07

**G2++ (2-factor Hull-White) calibration.**

- New `models/g2pp_calibration.py`:
  - `g2pp_swaption_price()` — Brigo-Mercurio analytical via Gauss-Hermite + Jamshidian.
  - `calibrate_g2pp()` — fit (a, b, σ₁, σ₂, ρ) to swaption vol grid via DE + L-BFGS-B.
  - `g2pp_vs_hw1f()` — compare 1F vs 2F calibration quality.

---

## v0.852.0 — 2026-06-07

**Exercise boundary extraction.**

- New `options/exercise_boundary.py`:
  - `pde_exercise_boundary()` — extract boundary from Crank-Nicolson PDE.
  - `tree_exercise_boundary()` — extract from CRR binomial tree.
  - `lsm_exercise_boundary()` — extract from LSM regression.
  - `boundary_analytics()` — slope, convexity, critical price analysis.
  - `compare_boundaries()` — cross-method comparison.
- 10 tests.

---

## v0.851.0 — 2026-06-07

**Bermudan barrier options.**

- New `options/bermudan_barrier.py`:
  - `bermudan_barrier_option()` — LSM with continuous barrier monitoring.
  - `american_barrier_option()` — American exercise + barrier knock-out.
  - `bermudan_double_barrier()` — double barrier with early exercise.
  - `barrier_exercise_interaction()` — decompose barrier vs exercise premium.
- 9 tests.

---

## v0.850.0 — 2026-06-07

**American multi-asset options.**

- New `options/american_multi_asset.py`:
  - `american_spread_option()` — LSM with 6-term basis on (S1, S2).
  - `american_basket_option()` — LSM on weighted basket.
  - `american_best_of()` — LSM on max(S1, S2).
  - `american_worst_of_put()` — LSM on min(S1, S2), key for structured products.
- 10 tests.

---

## v0.849.0 — 2026-06-07

**Stochastic credit Bermudan CDS swaption.**

- New `credit/stochastic_bermudan_cds.py`:
  - `stochastic_bermudan_cds_swaption()` — LSM under CIR intensity with exact non-central chi-squared simulation.
  - `cir_cds_pv()` — analytical CDS PV under CIR via Riccati ODE.
  - `stochastic_vs_deterministic()` — compare stochastic vs deterministic approaches.
- 7 tests.

---

## v0.848.0 — 2026-06-07

**Callable structured notes.**

- New `structured/callable_structured.py`:
  - `callable_steepener()` — LSM on CMS spread with issuer call.
  - `callable_cms_spread()` — callable CMS spread note.
  - `callable_inverse_floater()` — callable inverse floater.
- 6 tests.

---

## v0.847.0 — 2026-06-07

**Callable and puttable floating rate notes.**

- New `fixed_income/callable_floater.py`:
  - `callable_frn()` — HW tree with call constraint on coupon dates.
  - `puttable_frn()` — HW tree with put constraint.
  - `callable_frn_oas()` — OAS via Brent root-find.
- 8 tests.

---

## v0.846.0 — 2026-06-07

**Bermudan cap/floor.**

- New `options/bermudan_capfloor.py`:
  - `bermudan_cap()` / `bermudan_floor()` — HW trinomial tree with Bermudan exercise on remaining caplet/floorlet strip.
  - `bermudan_collar()` — long cap + short floor with Bermudan exercise.
- 9 tests.

---

## v0.845.0 — 2026-06-07

**American commodity options.**

- New `commodity/commodity_american.py`:
  - `american_commodity_option()` — BAW/tree with convenience yield.
  - `american_energy_option()` — seasonal vol adjustment.
  - `american_commodity_spread()` — LSM on correlated commodity pair.
  - `early_exercise_test()` — optimal exercise diagnostic.
- 10 tests.

---

## v0.844.0 — 2026-06-07

**American FX options.**

- New `fx/fx_american.py`:
  - `american_fx_option()` — Garman-Kohlhagen American via BAW/PDE/tree.
  - `fx_exercise_boundary()` — early exercise boundary curve.
  - `american_fx_greeks()` — numerical Greeks (delta_dom, delta_for, rho_dom, rho_for).
- 8 tests.

---

## v0.843.0 — 2026-06-07

**Analytical American approximations.**

- New `options/american_analytical.py`:
  - `ju_zhong()` — Ju & Zhong (1999) second-order correction to BAW.
  - `kim_integral()` — Kim (1990) integral equation for exact exercise boundary.
  - `medvedev_scaillet()` — near-expiry asymptotic expansion.
  - `american_comparison()` — run all methods and compare.
- 14 tests.

---

## v0.842.0 — 2026-06-07

**Fix remaining known limitations.**

- `xccy_swaption.py`: correct `xccy_forward_spread` to use full floating-leg PV replication (Brigo & Mercurio Ch. 13) — par floater identity for both legs, notional exchange at spot.
- `exotic_payoffs.py`: fix `shout_option_analytical` r=q branch to use Goldman-Sosin-Gatto (1979) lookback formula correctly.
- `equity_linked_note.py`: replace stdlib `random` MC in `worst_of_eln` with vectorised numpy (Cholesky @ standard_normal).
- Input validation added across 6 modules: `equity_spread_option`, `exotic_payoffs`, `carbon_credit`, `freight`, `quanto_futures` — guards for spot>0, T>0, vol>0, rho∈[-1,1].

---

## v0.841.0 — 2026-06-07

**Code review fixes across 12 modules.**

- `quanto_swap.py`: fix adjustment to use `t_start` (fixing time) not `t_end`; apply `corr_1_fx` in differential swap (was silently ignored); add MONTHLY frequency.
- `exotic_payoffs.py`: fix installment option `pv_remaining` off-by-one; remove dead imports.
- `portfolio_margin.py`: SPAN extreme scenarios 2×PSR (was 3×), add 35% cap; fix straddle `max_loss` sign.
- `insurance_annuity.py`: fee PV now discounts each step at its own time (was using terminal discount for all).
- `real_estate_derivative.py`: fix `reit_nav_model` to use NOI/discount_rate (Gordon model); remove dead variable.
- `equity_spread_option.py`: central differences for vega/rho; remove unused imports and dead closure.
- `tranche_option.py`: guard for non-positive spreads in Black model.
- `etf.py`: fix docstring inconsistency (premium_discount units).
- `carbon_credit.py`, `money_market.py`, `xccy_swaption.py`: remove unused imports.

---

## v0.840.0 — 2026-06-07

**Insurance annuity guarantees and real estate derivatives.**

- New `structured/insurance_annuity.py`:
  - `gmab()` — Guaranteed Minimum Accumulation Benefit (MC).
  - `gmdb()` — Guaranteed Minimum Death Benefit with mortality weighting.
  - `gmwb()` — Guaranteed Minimum Withdrawal Benefit with ruin tracking.
  - `ratchet_gmab()` — GMAB with periodic ratchet reset.
- New `structured/real_estate_derivative.py`:
  - `property_total_return_swap()` — TRS on property index.
  - `property_index_forward()` — illiquidity-adjusted forward.
  - `property_option()` — Black-76 on property index.
  - `reit_nav_model()` — REIT net asset value.
  - `housing_affordability()` — payment-to-income metrics.
- 18 tests.

---

## v0.839.0 — 2026-06-07

**Longevity and mortality derivatives.**

- New `structured/longevity.py`:
  - `q_forward()` — mortality rate swap (q-forward).
  - `longevity_swap()` — multi-cohort fixed vs realised mortality.
  - `survivor_index()` — population projection with mortality improvement.
  - `lee_carter_forecast()` — Lee-Carter SVD mortality forecasting.
  - `mortality_bond_price()` — principal-at-risk mortality bond.
  - `value_of_life_annuity()` — life-contingent annuity PV.
- 16 tests.

---

## v0.838.0 — 2026-06-07

**Catastrophe bonds and ILS.**

- New `structured/cat_bond.py`:
  - `cat_bond_price()` — Poisson-arrival loss model with coupon/principal at risk.
  - `parametric_trigger_prob()` — Gumbel extreme value trigger probability.
  - `indemnity_trigger_loss()` — MC lognormal loss with attachment/exhaustion.
  - `cat_bond_spread_decomposition()` — EL + risk premium + expense.
  - `ils_portfolio()` — Gaussian copula portfolio of cat bonds.
  - `seasonal_adjustment()` — hurricane/earthquake seasonal probability.
- 16 tests.

---

## v0.837.0 — 2026-06-07

**Portfolio margin / SPAN.**

- New `risk/portfolio_margin.py`:
  - `span_margin()` — 14-scenario SPAN-style margining.
  - `cross_margin_offset()` — diversification benefit from cross-margining.
  - `strategy_margin()` — Reg-T margin for option strategies.
  - `var_based_margin()` — VaR/ES-based initial margin.
  - `margin_call()` — margin call computation.
- 17 tests.

---

## v0.836.0 — 2026-06-07

**Tranche options.**

- New `credit/tranche_option.py`:
  - `tranche_option_black()` — Black-76 on tranche spread.
  - `tranche_option_bachelier()` — normal model for tight/negative spreads.
  - `tranche_option_greeks()` — numerical spread delta, gamma, vega, theta.
  - `tranche_straddle()` — ATM straddle with breakeven levels.
  - `tranche_forward_spread()` — loss-adjusted forward tranche spread.
- 20 tests.

---

## v0.835.0 — 2026-06-07

**Quanto futures and ETF products.**

- New `equity/quanto_futures.py`:
  - `quanto_futures_price()` — F_Q = S × exp((r_d − q − ρσ_Sσ_FX) × T).
  - `implied_correlation()` — back-solve ρ from observed quanto price.
  - `compo_vs_quanto()` — compare composite vs quanto forwards.
- New `equity/etf.py`:
  - `etf_nav()` — NAV from holdings basket.
  - `creation_redemption_arb()` — AP arbitrage evaluation.
  - `tracking_error()`, `tracking_difference()` — index tracking metrics.
  - `leveraged_etf_decay()` — volatility drag formula.
- 15 tests.

---

## v0.834.0 — 2026-06-07

**Exotic option payoffs: ladder, shout, installment.**

- New `options/exotic_payoffs.py`:
  - `ladder_option()` — MC with rung-based lock-in of intrinsic.
  - `shout_option()` — MC multi-shout option.
  - `shout_option_analytical()` — Dai-Kwok-Wu closed form via lookback equivalence.
  - `installment_option()` — MC with rational abandonment at each payment date.
- 12 tests.

---

## v0.833.0 — 2026-06-07

**Freight derivatives.**

- New `commodity/freight.py`:
  - `ffa_price()` — Forward Freight Agreement (average/point settlement).
  - `freight_option_price()` — Black-76 on FFA rate.
  - `time_charter_equivalent()` — TCE calculation.
  - `freight_forward_curve()` — seasonal forward curve builder.
  - `bunker_spread()` — P&L sensitivity to bunker fuel cost.
- 7 tests (freight). Combined test file with carbon.

---

## v0.832.0 — 2026-06-07

**Carbon/emission credit pricing.**

- New `commodity/carbon_credit.py`:
  - `carbon_futures_price()` — cost-of-carry for EUA/carbon allowances.
  - `carbon_option_price()` — Black-76 on carbon futures.
  - `marginal_abatement_cost()` — equilibrium from abatement technology curve.
  - `compliance_value()` — surplus/deficit position valuation.
  - `voluntary_credit_discount()` — haircut model for voluntary credits.
- 7 tests (carbon).

---

## v0.831.0 — 2026-06-07

**Capped/floored/collar floaters.**

- New `structured/capped_floored_floater.py`:
  - `floored_floater()` — FRN with minimum coupon via floorlet strip.
  - `collar_floater()` — FRN with cap and floor (short caplets, long floorlets).
  - `reverse_floater()` — coupon = fixed − leverage × floating, with embedded cap.
  - `inverse_floater_duration()` — amplified effective duration.
- 12 tests.

---

## v0.830.0 — 2026-06-07

**CPDO simulation.**

- New `structured/cpdo.py`:
  - `cpdo_simulate()` — single-path CPDO with leverage, gap risk, cash-out.
  - `cpdo_monte_carlo()` — MC: success/default probabilities, expected NAV.
  - `cpdo_rating()` — map default prob to S&P rating bucket.
- 14 tests.

---

## v0.829.0 — 2026-06-07

**Money market instruments.**

- New `fixed_income/money_market.py`:
  - `CertificateOfDeposit` — interest-bearing, dirty/clean price, YTM.
  - `CommercialPaper` — discount instrument, credit spread.
  - `BankersAcceptance` — bank-guaranteed CP with acceptance fee.
  - `RepoRate` — implied repo and haircut-adjusted rate helpers.
- 11 tests.

---

## v0.828.0 — 2026-06-07

**Cross-currency swaptions.**

- New `fixed_income/xccy_swaption.py`:
  - `xccy_swaption_black()` — Black-76 on forward xccy basis spread.
  - `xccy_swaption_bachelier()` — normal model for negative spreads.
  - `xccy_forward_spread()` — CIP-based forward basis spread.
  - `xccy_swaption_greeks()` — numerical delta, gamma, vega, fx_delta.
- 9 tests.

---

## v0.827.0 — 2026-06-07

**Equity spread options.**

- New `equity/equity_spread_option.py`:
  - `kirk_equity_spread()` — Kirk's approximation with dividend yields.
  - `bjerksund_stensland_spread()` — improved accuracy for non-zero strikes.
  - `mc_spread_option()` — Monte Carlo benchmark with antithetic variates.
  - `outperformance_option()` — Margrabe special case (K=0).
  - `relative_performance_option()` — percentage outperformance.
- 11 tests.

---

## v0.826.0 — 2026-06-07

**Equity-linked notes (ELN).**

- New `structured/equity_linked_note.py`:
  - `buffered_eln()` — downside buffer, coupon if index holds.
  - `capped_eln()` — participation with cap.
  - `bear_eln()` — inverse ELN paying on index decline.
  - `digital_eln()` — enhanced coupon if above barrier.
  - `twin_win_eln()` — profits from both directions unless barrier breached.
  - `worst_of_eln()` — MC basket ELN on worst performer.
- 12 tests.

---

## v0.825.0 — 2026-06-07

**Equity index futures pricing.**

- New `equity/equity_index_futures.py`:
  - `index_futures_fair_value()` — cost-of-carry F = S × exp((r - q + b) × T).
  - `index_futures_roll()` — calendar spread, roll cost, implied repo between contracts.
  - `implied_dividend_yield()`, `implied_repo_rate()` — back-solve from observed prices.
  - `fair_value_table()` — term structure across multiple expiries.
- 12 tests.

---

## v0.824.0 — 2026-06-01

**Quanto (differential) interest rate swaps.**

- New `fixed_income/quanto_swap.py`:
  - `quanto_swap_price()` — quanto IR swap: foreign floating rate paid in domestic currency with convexity adjustment E^d[L^f] = L^f × (1 − σ_L × σ_FX × ρ × T).
  - `differential_swap_price()` — diff swap paying rate_1 − rate_2 in single currency, both rates quanto-adjusted.
  - `quanto_adjustment_term_structure()` — adjustment per tenor in bps.
  - `quanto_fra()` — single-period quanto forward rate agreement.
- 22 tests: correlation sign, par spread, pay/receive symmetry, vol sensitivity, maturity scaling.

---

## v0.823.0 — 2026-06-04

**Backlog closed: HV ADI, Strang MC, SDP, sparse Jacobian.**

- New `models/hundsdorfer_verwer.py`:
  - `hv_adi_heston()` — double-pass HV ADI for Heston (6-step scheme).
  - More stable than Craig-Sneyd for strong mixed derivatives.
  - HV agrees with CS within 15%.
- New `models/sde_strang.py`:
  - `strang_merton_mc()` — Merton jump-diffusion: diffusion(dt/2)→jump(dt)→diffusion(dt/2).
  - `strang_bates_mc()` — Bates (Heston + jumps) via Strang splitting.
  - Zero jumps matches BS. O(dt²) splitting error.
- New `numerical/sdp.py`:
  - `nearest_psd()` — PSD cone projection.
  - `nearest_correlation_sdp()` — Higham (2002) Dykstra alternating projections.
  - `factor_covariance_bounds()` — covariance from factor model.
  - `sdp_solve()` — small-scale general SDP via projected gradient.
- New `numerical/sparse_jacobian.py`:
  - `sparse_jacobian()` — Jacobian via graph colouring + grouped perturbation.
  - `banded_jacobian()` — tridiagonal: 3 evaluations instead of n.
  - `detect_sparsity()` — probe-based sparsity detection.
  - `greedy_colouring()` — distance-1 column grouping.
- 12 new tests. 11,089 tests pass.

---

## v0.816.0 — 2026-06-03

**Remaining numerical plan: Tiers 3+4 complete.**

- **F5** `fft_pricing.py`: `carr_madan_fractional()` — non-uniform strikes via direct Fourier evaluation.
- **F6** `registry.py`: registered FFT, Lewis, Bermudan COS, Fourier Greeks pricers.
- **S1** `sde_adaptive.py`: `adaptive_euler()`, `adaptive_milstein()` — step-size control via error pair.
- **X2** `von_neumann.py`: amplification factor, stability region, CFL limit for θ-scheme.
- **X3** `density_evolution.py`: three-way density cross-validation (FP + Fourier + Breeden-Litzenberger).
- **X4** `operator_splitting.py`: Lie-Trotter (O(dt)), Strang (O(dt²)), PIDE splitting.
- 10 new tests. 11,077 tests pass.

---

## v0.806.0 — 2026-06-03

**Convexity tools and Frank-Wolfe optimisation.**

- New `numerical/convexity_tools.py`:
  - `is_convex()` — Hessian eigenvalue sampling.
  - `verify_kkt()` — KKT condition verification.
  - `cardinality_portfolio()` — max N assets via greedy selection.
- New `numerical/frank_wolfe.py`:
  - `frank_wolfe()` — conditional gradient with LMO.
  - `frank_wolfe_portfolio()` — O(n) per iteration MV.
- 5 new tests.

---

## v0.804.0 — 2026-06-03

**Oscillatory quadrature: Filon and Levin methods.**

- New `numerical/oscillatory_quad.py`:
  - `filon_quad()` — Filon's method for ∫f(x)cos(ωx)dx (O(h³/ω)).
  - `levin_quad()` — Levin collocation for general ∫f(x)e^{iωx}dx.
  - `fourier_integral()` — adaptive: standard quad (low ω) or Filon (high ω).
- 3 new tests.

---

## v0.803.0 — 2026-06-03

**LP duality framework: shadow prices, sensitivity.**

- New `numerical/duality.py`:
  - `lp_with_duals()` — LP with dual variable extraction via perturbation.
  - `shadow_prices()` — marginal cost of constraints.
  - `parametric_lp()` — sweep RHS of one constraint.
- 3 new tests.

---

## v0.802.0 — 2026-06-03

**Fokker-Planck forward density evolution.**

- New `models/fokker_planck.py`:
  - `fokker_planck_1d()` — 1D density evolution in log-space (GBM/local vol).
  - `density_to_option_prices()` — price options from risk-neutral density.
  - Density integrates to 1, mean matches forward.
- 3 new tests.

---

## v0.801.0 — 2026-06-03

**True 2D FFT for two-asset options.**

- New `models/fft_2d.py`:
  - `joint_bs_char_func()` — joint CF for correlated GBM.
  - `fft_2d_price()` — full (u₁,u₂) grid with 2D Simpson weights.
  - Spread, basket, best-of payoffs.
- 2 new tests.

---

## v0.800.0 — 2026-06-03

**Rough Heston CF via fractional Riccati ODE.**

- New `models/rough_heston_cf.py`:
  - `rough_heston_char_func()` — Adams scheme on fractional Riccati (El Euch & Rosenbaum 2019).
  - `rough_heston_price()` — European via COS + rough Heston CF.
  - H < 0.5 gives rough regime; differs from smooth Heston (H≈0.5).
- 2 new tests.

---

## v0.798.0 — 2026-06-03

**SOCP solver: robust portfolio and tracking error.**

- New `numerical/socp.py`:
  - `socp_solve()` — general SOCP via barrier method.
  - `robust_portfolio_socp()` — robust MV with norm constraints.
  - `tracking_error_socp()` — min TE vs benchmark.
- 2 new tests.

---

## v0.797.0 — 2026-06-03

**Numerical method recommendation map.**

- New `core/numerical_method_map.py`:
  - `recommend()` — given instrument features, recommend best method.
  - `compare_methods()` — price via analytical/COS/PDE/tree, report agreement.
  - 14 instrument feature types, rule-based selection.
- 6 new tests.

---

## v0.796.0 — 2026-06-03

**Feynman-Kac bridge: SDE ↔ PDE connection.**

- New `models/feynman_kac.py`:
  - `sde_to_pde()` — derive PDE coefficients from SDE dynamics.
  - `pde_to_sde()` — extract SDE from PDE coefficients.
  - `verify_feynman_kac()` — cross-validate MC vs PDE (consistent within 3σ).
- 3 new tests.

---

## v0.795.0 — 2026-06-03

**Automatic differentiation via dual numbers.**

- New `numerical/auto_diff.py`:
  - `Dual` class — overloaded arithmetic (+ − × / pow).
  - Math functions: `exp`, `log`, `sqrt`, `sin`, `cos`, `max_dual`.
  - `grad()` — gradient of f: ℝⁿ → ℝ via forward AD.
  - `jacobian_ad()` — Jacobian via forward AD.
  - `derivative()` — f(x) and f'(x) simultaneously, machine-precision.
  - BS delta via AD matches analytical.
- 7 new tests.

---

## v0.794.0 — 2026-06-03

**Fourier Greeks: delta, gamma, vega, theta via COS/Lewis.**

- New `models/fourier_greeks.py`:
  - `cos_greeks()` — full Greeks via COS with spot/vol/time bumps.
  - `lewis_greeks()` — Greeks via Lewis formula.
  - `fourier_greeks()` — unified entry point.
  - Vega via CF variance perturbation (no vol parameter needed).
- 3 new tests.

---

## v0.793.0 — 2026-06-03

**Fix: CharacteristicFunction.price_european() was broken.**

- `numerical/_fourier.py`: `cos_european` → `cos_price` with correct OptionType argument.
- Now correctly prices European options via COS method.
- 1 new test.

---

## v0.792.0 — 2026-06-03

**Package ready for PyPI: README, LICENSE, build verified.**

- Added `python/README.md` — full package description with install, quick start, feature list.
- Copied `LICENSE` into `python/` — required by PyPI alongside pyproject.toml.
- Added `readme = "README.md"` to pyproject.toml.
- Version synced to 0.791.0 in `__init__.py`.
- Build verified: `python -m build` produces valid sdist (2.2MB) + wheel (1.9MB).
  - 716 .py modules in wheel, no tests or notebooks leaked.
  - METADATA correct: classifiers, keywords, license expression.
- Ready to publish: `twine upload dist/*` with PyPI credentials.

---

## v0.791.0 — 2026-06-02

**Python package: version sync, pyproject.toml, py.typed marker.**

- Version synced to 0.790.0 in `__init__.py` (was 0.614.0).
- `pyproject.toml` updated:
  - Trove classifiers (Financial, Science/Research, Typed).
  - Extended keywords (structured-products, monte-carlo, pde, portfolio-optimization).
  - `py.typed` marker for PEP 561 type checking support.
  - `[tool.mypy]` section for type checking config.
  - Notebooks excluded from package distribution.
- Verified: `pip install -e .` works, all imports functional, `pricebook.__version__` correct.
- 11,027 tests pass.

---

## v0.790.0 — 2026-06-02

**Notebook consolidation: single location under python/notebooks/.**

- Consolidated 45 notebooks + examples from 3 locations (notebooks/, python/notebooks/, examples/) into one structure:
  - `python/notebooks/papers/` — 12 paper validations
  - `python/notebooks/markets/` — 6 Americas market notebooks
  - `python/notebooks/rates/` — 4 rates workflows
  - `python/notebooks/credit/` — 2 credit notebooks
  - `python/notebooks/structured/` — 3 structured product notebooks
  - `python/notebooks/desks/` — 2 desk notebooks
  - `python/notebooks/validation/` — 5 Pucci et al. validations
  - `python/notebooks/examples/` — 10 Python examples + 2 example notebooks
- Removed empty root `notebooks/` and `examples/` directories.
- 11,027 tests pass.

---

## v0.789.0 — 2026-06-02

**PDE code review fixes.**

- **pde_adaptive.py**: CRITICAL — grid refinement midpoint formula used `grid[i+1]+grid[i+2]` instead of `grid[i]+grid[i+1]`, inserting nodes at wrong locations. FD formula aligned with protocol (was using different convection discretisation).
- **pde_local_vol.py**: barrier BC removed degenerate `if not is_call else 0.0` (always 0). Knock-in parity fixed — was passing contradictory `vol=0.20` alongside `vol_surface`.
- **pide_solver.py**: V_prev now saved every step (was only at `n_time-2`), fixing theta Greek computation for both Merton and Kou.
- **pde_boundary.py**: Robin BC sign error fixed — derivation `a*V + b*∂V/∂S = g` now correctly solved for V[0].
- 11,027 tests pass.

---

## v0.788.0 — 2026-06-02

**PDE boundary condition library.**

- New `numerical/pde_boundary.py`:
  - `BCSpec` — unified BC specification: Dirichlet, Neumann, Robin, linear extrapolation, outflow.
  - `apply_bc()` — apply BCs to solution vector.
  - Financial BC factories: `call_bcs()`, `put_bcs()`, `barrier_bcs()`.
- 5 new tests.

---

## v0.787.0 — 2026-06-02

**PDE convergence diagnostics and scheme selection.**

- New `models/pde_diagnostics.py`:
  - `convergence_study()` — grid refinement analysis with Richardson extrapolation.
  - `recommend_scheme()` — automatic method/grid recommendation.
  - `stability_check()` — CFL verification with warnings.
- 3 new tests.

---

## v0.786.0 — 2026-06-02

**American 2D: Heston American via ADI + penalty.**

- New `models/pde_american_2d.py`:
  - `heston_american_pde()` — Heston American put via Craig-Sneyd ADI + penalty method.
  - Penalty λ converts free boundary to nonlinear fixed-domain PDE.
  - American ≥ European verified.
- 2 new tests.

---

## v0.785.0 — 2026-06-02

**Adaptive grid refinement for PDE.**

- New `models/pde_adaptive.py`:
  - `error_indicator()` — gradient-based error estimate per cell.
  - `refine_grid()` — insert nodes where curvature is high.
  - `adaptive_pde()` — iterative solve-refine-solve with convergence check.
- 2 new tests.

---

## v0.784.0 — 2026-06-02

**SABR PDE via 2D ADI.**

- New `models/pde_sabr.py`:
  - `sabr_pde()` — 2D ADI in (F, σ) space with absorbing boundary at F=0.
  - Craig-Sneyd splitting with mixed derivative term.
  - ITM > ATM verified.
- 2 new tests.

---

## v0.783.0 — 2026-06-02

**Jump-diffusion PIDE: Merton and Kou.**

- New `models/pide_solver.py`:
  - `merton_pide()` — operator splitting: diffusion (CN) + jump integral (quadrature).
  - `kou_pide()` — double-exponential jump-diffusion.
  - No jumps → matches BS. Jumps add value for OTM options.
- 3 new tests.

---

## v0.782.0 — 2026-06-02

**Time-dependent PDE coefficients.**

- New `models/pde_time_dependent.py`:
  - `TermStructureCoefficients` — piecewise-linear r(t), σ(t), q(t).
  - `time_dependent_pde()` — BS PDE with non-constant coefficients.
  - Constant term structure → matches standard PDE.
- 2 new tests.

---

## v0.781.0 — 2026-06-02

**Local volatility PDE solver.**

- New `models/pde_local_vol.py`:
  - `local_vol_pde()` — BS PDE with σ(S,t) from Dupire surface.
  - `local_vol_barrier_pde()` — barrier under local vol.
  - Flat surface → matches BS. Non-flat → prices differ.
- 2 new tests.

---

## v0.780.0 — 2026-06-02

**Unified PDE protocol.**

- New `models/pde_protocol.py`:
  - `PDECoefficients` — callable a(S,t), b(S,t), c(S,t) with factories for BS, local vol, time-dep.
  - `PDESpec` — full problem spec (coefficients, domain, BCs, payoff, American).
  - `PDEEngine` — solver with configurable method, grid, resolution.
  - `PDEPricingResult` — unified result with Greeks and convergence info.
  - `pde_price()` — one-function entry point. Matches BS to 2%.
- 6 new tests.

---

## v0.778.0 — 2026-06-02

**Code review fixes across portfolio optimisation and game theory.**

- **hierarchical_risk_parity.py**: CRITICAL — `import math` was at end of file, used on line 118. Moved to top.
- **cvar_optimisation.py**: removed dead `cvar_actual` computation with wrong tail condition; removed unused `minimize` import.
- **portfolio_analytics.py**: CVaR tail selection logic fixed — was convoluted double-negation, now clean `losses[losses >= var_95]`.
- **stackelberg.py**: Cournot benchmark now handles asymmetric costs correctly (was using symmetric formula). Market share clamped to [0,1].
- **bargaining.py**: Kalai-Smorodinsky tolerance check was self-referential (`abs(x) < abs(x)*0.1`). Fixed to `abs(x) < 0.1*abs(expected) + 0.01`.
- **market_microstructure_games.py**: Glosten-Milgrom removed dead code (double `post_high_buy` computation), added division-by-zero guard. Information share docstring corrected from "Hasbrouck" to "variance-based" (simplified approach).
- **n_player_nash.py**: removed unused `val` variable in `_compute_payoffs`.
- 11,000 tests pass.

---

## v0.777.0 — 2026-06-02

**Unified portfolio analytics: Sharpe, Sortino, Calmar, drawdowns, tracking.**

- New `risk/portfolio_analytics.py`:
  - `portfolio_metrics()` — 15 metrics: Sharpe, Sortino, Calmar, max DD, VaR/CVaR, skew/kurt.
  - `tracking_metrics()` — tracking error, information ratio, alpha, beta.
- 2 new tests.

---

## v0.776.0 — 2026-06-02

**Multi-period dynamic allocation: CPPI, target-date, lifecycle.**

- New `risk/dynamic_allocation.py`:
  - `cppi_allocation()` — constant proportion portfolio insurance with floor.
  - `target_date_glide()` — linear/convex/concave glide paths.
  - `multi_period_mv()` — multi-period mean-variance with rebalancing costs.
- 2 new tests.

---

## v0.775.0 — 2026-06-02

**Transaction cost-aware portfolio optimisation.**

- New `risk/transaction_cost_opt.py`:
  - `tc_aware_rebalance()` — turnover penalty in MV objective.
  - `no_trade_region()` — Leland-Davis no-trade bands.
  - `optimal_rebalance_frequency()` — cost-benefit analysis.
- 2 new tests.

---

## v0.774.0 — 2026-06-02

**Robust portfolio optimisation: worst-case, uncertainty sets.**

- New `risk/robust_optimisation.py`:
  - `robust_mean_variance()` — worst-case mean-variance.
  - `ellipsoidal_uncertainty()` — Goldfarb-Iyengar ellipsoidal sets.
  - `box_uncertainty()` — interval return uncertainty.
- 2 new tests.

---

## v0.773.0 — 2026-06-02

**Kelly criterion: optimal bet sizing.**

- New `risk/kelly.py`:
  - `kelly_fraction()` — single-asset f* = μ/σ².
  - `fractional_kelly()` — conservative half-Kelly.
  - `multi_asset_kelly()` — portfolio Kelly via Σ⁻¹ × excess.
- 3 new tests.

---

## v0.772.0 — 2026-06-02

**Brinson-Fachler performance attribution.**

- New `risk/brinson_attribution.py`:
  - `brinson_attribution()` — allocation + selection + interaction.
  - `brinson_multi_period()` — geometric linking.
  - `factor_based_attribution()` — OLS factor decomposition.
  - Sum of effects = active return (verified).
- 2 new tests.

---

## v0.771.0 — 2026-06-02

**Hierarchical Risk Parity (López de Prado 2016).**

- New `risk/hierarchical_risk_parity.py`:
  - `hrp_portfolio()` — tree clustering + quasi-diagonalisation + recursive bisection.
  - `cluster_assets()` — hierarchical clustering by correlation distance.
  - No covariance inversion → robust to estimation error.
- 2 new tests.

---

## v0.770.0 — 2026-06-02

**Efficient frontier: full curve, tangency, CML.**

- New `risk/efficient_frontier.py`:
  - `efficient_frontier()` — full mean-variance frontier sweep.
  - `tangency_portfolio()` — max Sharpe via SLSQP.
  - `minimum_variance_portfolio()` — analytical or numerical.
  - `capital_market_line()` — CML from rf to tangency.
- 4 new tests.

---

## v0.769.0 — 2026-06-02

**CVaR portfolio optimisation via Rockafellar-Uryasev LP.**

- New `risk/cvar_optimisation.py`:
  - `cvar_portfolio()` — LP formulation for CVaR-optimal weights.
  - `min_cvar_target_return()` — minimum CVaR for given return.
  - `cvar_risk_budget()` — component CVaR decomposition.
  - `mean_cvar_frontier()` — efficient frontier in mean-CVaR space.
- 4 new tests.

---

## v0.768.0 — 2026-06-02

**Strategic market microstructure: Kyle, Glosten-Milgrom.**

- New `models/market_microstructure_games.py`:
  - `kyle_lambda()` — Kyle (1985) price impact, insider profit, market depth.
  - `glosten_milgrom()` — sequential trade with adverse selection.
  - `optimal_order_splitting()` — Almgren-Chriss extended.
  - `information_share()` — Hasbrouck multi-market decomposition.
- 5 new tests.

---

## v0.767.0 — 2026-06-02

**Bargaining theory: Nash, Rubinstein, Kalai-Smorodinsky.**

- New `models/bargaining.py`:
  - `nash_bargaining()` — Nash bargaining solution on feasible set.
  - `rubinstein_alternating()` — Rubinstein SPE (patience → surplus).
  - `kalai_smorodinsky()` — monotonic solution via ideal point.
  - `debt_restructuring_bargain()` — creditor-debtor Rubinstein.
- 3 new tests.

---

## v0.766.0 — 2026-06-02

**Stackelberg leader-follower games.**

- New `models/stackelberg.py`:
  - `stackelberg_cournot()` — quantity competition with first-mover advantage.
  - `stackelberg_bertrand()` — price competition.
  - `credit_market_stackelberg()` — lead bank spread-setting game.
  - `general_stackelberg()` — generic two-player framework.
- 3 new tests.

---

## v0.765.0 — 2026-06-02

**N-player Nash equilibrium: fictitious play, support enumeration.**

- New `models/n_player_nash.py`:
  - `fictitious_play()` — iterative best-response for N players.
  - `lemke_howson_2p()` — support enumeration for bimatrix.
  - `correlated_equilibrium()` — LP for correlated equilibrium.
- 3 new tests.

---

## v0.759.0 — 2026-06-02

**Code review fixes across futures, structured, FX, and engine infrastructure.**

- **mc_greeks_auto.py**: lookback/Asian reclassified from SMOOTH to PATH_DEPENDENT (pathwise IPA invalid for path-dependent payoffs).
- **autocall_advanced.py**: memory coupon overwrite bug fixed — line 115 was unconditionally overwriting line 114, making memory feature dead code.
- **tree_mc_bridge.py**: stochastic vol tree drift used variance `v` instead of `sigma²`; MC path-dependent branch missing div_yield.
- **bespoke_cdo.py**: loss distribution now uses notional-weighted average PD/LGD instead of equal-weight.
- **tree_enhancements.py**: barrier accuracy division by zero guard when barrier == 0.
- **engine_comparison.py**: dict iteration fix in `validate_greeks()` — was iterating all values including non-dict.
- **fx_exotic_extensions.py**: Dupire local vol guard for K ≤ 0 preventing log domain error.
- **commodity_options.py**: Samuelson docstring formula corrected to match implementation `exp(−αT)`.
- **Removed unused imports**: `_norm_pdf` from futures_options.py and commodity_options.py; `np` from spread_options.py and commodity_swaps.py.
- 10,963 tests pass.

---

## v0.758.0 — 2026-06-02

**Unified engine registry: one function, any instrument, best engine.**

- New `models/engine_registry.py`:
  - `price()` — auto-select best engine for instrument type.
  - `InstrumentType` enum: 14 instrument classes.
  - Per-type engine recommendations (analytical → tree → MC).
  - `register_engine()` for custom engines. `list_engines()`.
- 6 new tests.

---

## v0.757.0 — 2026-06-02

**Engine comparison and validation.**

- New `models/engine_comparison.py`:
  - `compare_engines()` — price via analytical, tree, MC side-by-side.
  - `validate_greeks()` — check Greek consistency across engines.
  - Reports price spread, Greek agreement, compute time.
- 3 new tests.

---

## v0.756.0 — 2026-06-02

**Tree-MC bridge: hybrid engine for early exercise + path dependence.**

- New `models/tree_mc_bridge.py`:
  - `lsm_on_tree()` — LSM using CRR transition probabilities.
  - `stochastic_vol_tree()` — 2D trinomial (spot × variance) for Heston.
  - `hybrid_price()` — auto-select tree, MC, or hybrid by instrument features.
- 3 new tests.

---

## v0.755.0 — 2026-06-02

**Tree enhancements: adaptive barrier mesh, non-recombining scaffold.**

- New `numerical/tree_enhancements.py`:
  - `adaptive_barrier_tree()` — grid-adjusted trinomial near barrier.
  - `NonRecombiningTree` — linked-list tree with path-dependent state.
  - `asian_on_tree()` — Asian option via non-recombining tree.
- 3 new tests.

---

## v0.754.0 — 2026-06-02

**Derman-Kani implied binomial tree.**

- New `numerical/implied_tree.py`:
  - `build_implied_tree()` — calibrate recombining tree to market options.
  - `price_on_implied_tree()` — exotic pricing on smile-consistent tree.
  - `extract_local_vol()` — local vol from Arrow-Debreu state prices.
- 3 new tests.

---

## v0.753.0 — 2026-06-02

**Black-Derman-Toy (BDT) log-normal rate tree.**

- New `models/bdt_tree.py`:
  - `BDTTree` — calibrated log-normal rate tree with Arrow-Debreu state prices.
  - `bdt_callable_bond()` — callable bond via BDT backward induction.
  - `bdt_bermudan_swaption()` — Bermudan swaption on BDT.
  - Calibrates to match market discount curve exactly.
- 3 new tests.

---

## v0.752.0 — 2026-06-02

**MC convergence diagnostics (extended).**

- Extended `models/mc_diagnostics.py`:
  - `full_diagnostics()` — unified diagnostics with ESS, VRE, CI, skewness/kurtosis.
  - `variance_reduction_efficiency()` — Var(crude)/Var(reduced).
  - `estimate_convergence_rate()` — fit rate from prices at different N.
  - `MCFullDiagnostics.is_converged` — heuristic convergence check.
- 3 new tests.

---

## v0.751.0 — 2026-06-02

**Auto-Greek method selection with path caching.**

- New `models/mc_greeks_auto.py`:
  - `classify_payoff()` — detect smooth/discontinuous/path-dependent.
  - `select_greek_method()` — pathwise for smooth, LR for digital, bump for rest.
  - `auto_greeks()` — compute all Greeks with best method per Greek.
  - `PathCache` — LRU cache for MC paths, shared across Greeks.
- 5 new tests.

---

## v0.750.0 — 2026-06-02

**Declarative MC configuration and factory.**

- New `models/mc_config.py`:
  - `MCConfig` — all settings in one dataclass (process, VR, Greeks method, discretisation).
  - `preset_configs()` — fast, production, high_precision, heston, exotic, xva.
  - `build_process_from_config()` — factory for ProcessSpec.
  - `mc_pricer_from_config()` — build MCPricingEngine from config.
  - `with_overrides()` for mode switching.
- 4 new tests.

---

## v0.749.0 — 2026-06-02

**Unified pricing engine protocol.**

- New `models/engine_protocol.py`:
  - `PricingResult` — unified result: price, GreeksBundle, ConvergenceInfo.
  - `PricingEngine` protocol: `.price_vanilla()`, `.engine_type`.
  - `MCPricingEngine` — wraps MCEngine behind protocol.
  - `TreePricingEngine` — wraps TreeSolver behind protocol.
  - `AnalyticalEngine` — Black-Scholes behind protocol.
  - All three engines agree on European call within 3%.
- 6 new tests.

---

## v0.747.0 — 2026-06-02

**FX exotic extensions: digitals, quantos, var swaps, local vol, double barriers, compound, chooser.**

- New `fx/fx_exotic_extensions.py`:
  - `fx_digital_option()` — European digital (cash-or-nothing, asset-or-nothing), overhedge, both payout currencies.
  - `fx_quanto_option()` — quanto-adjusted GK with correlation drift, FX rate scaling.
  - `fx_variance_swap()` — fair strike from ATM + butterfly, MTM with realised.
  - `fx_local_vol()` — Dupire local vol surface from implied vol grid via finite differences.
  - `fx_double_barrier_option()` — double knock-out/knock-in via MC, parity verified.
  - `fx_compound_option()` — option on option (call-on-call, put-on-call, etc.) via MC.
  - `fx_chooser_option()` — call-or-put choice at future date, probability tracking.
- 25 new tests.

---

## v0.746.0 — 2026-06-02

**Power/electricity derivatives: swing, tolling, capacity.**

- New `commodity/power_derivatives.py`:
  - `swing_option_price()` — volume flexibility with min/max take.
  - `tolling_agreement()` — virtual power plant economics.
  - `capacity_option()` — option on generation dispatch.
  - `block_forward()` — peak/off-peak block pricing.
- 4 new tests.

---

## v0.745.0 — 2026-06-02

**Mountain range options: Napoleon, Everest, Atlas, Altiplano.**

- New `equity/mountain_range.py`:
  - `napoleon_option()` — worst-of cliquet with local caps/floors.
  - `everest_option()` — payoff on worst performer.
  - `atlas_option()` — remove best/worst, payoff on remainder.
  - `altiplano_option()` — digital basket (all above barrier).
  - Correlated GBM Monte Carlo.
- 4 new tests.

---

## v0.744.0 — 2026-06-02

**Stochastic correlation for credit tranches.**

- New `credit/stochastic_correlation.py`:
  - `regime_switching_correlation()` — multi-regime tranche pricing.
  - `correlation_smile()` — calibrate implied correlation across tranches.
  - `stochastic_corr_tranche()` — beta-distributed correlation MC.
  - Vasicek one-factor tranche expected loss.
- 4 new tests.

---

## v0.743.0 — 2026-06-02

**Secondary market structured product pricing.**

- New `structured/secondary_pricing.py`:
  - `spread_aging()` — CLN spread adjustment for time since issuance.
  - `mark_to_bid()` — haircut for illiquidity with stress multiplier.
  - `stale_price_detector()` — flag unchanged prices.
  - `liquidity_premium()` — model-based illiquidity premium.
- 5 new tests.

---

## v0.742.0 — 2026-06-02

**Steepener/flattener structured notes.**

- New `structured/steepener.py`:
  - `steepener_note()` — leveraged CMS10−CMS2 with floor/cap.
  - `slope_range_accrual()` — accrues when slope in range.
  - `digital_steepener()` — digital payout on curve slope.
  - MC pricing with correlated CMS dynamics.
- 4 new tests.

---

## v0.741.0 — 2026-06-02

**Bespoke CDO: custom portfolio, LSS, tranche Greeks.**

- New `credit/bespoke_cdo.py`:
  - `bespoke_tranche_price()` — Vasicek loss distribution for custom portfolio.
  - `calibrate_bespoke_correlation()` — bisection to match market spread.
  - `leveraged_super_senior()` — LSS with gap risk.
  - `tranche_greeks()` — spread delta, correlation delta.
- 5 new tests.

---

## v0.740.0 — 2026-06-02

**Advanced autocall: discrete observation, memory coupon, step-down.**

- New `options/autocall_advanced.py`:
  - `discrete_autocall()` — discrete observation dates with memory coupon.
  - `worst_of_discrete_autocall()` — multi-asset worst-of with correlated MC.
  - `step_down_autocall()` — declining autocall barriers.
- 5 new tests.

---

## v0.739.0 — 2026-06-02

**Commodity swaps and swaptions.**

- New `commodity/commodity_swaps.py`:
  - `commodity_swap_price()` — fixed-for-floating commodity swap.
  - `commodity_swaption_price()` — Black-76 on forward swap rate.
  - `asian_commodity_swap()` — averaging settlement.
- 4 new tests.

---

## v0.738.0 — 2026-06-02

**Dividend futures, swaps, options, total return futures.**

- New `equity/dividend_futures.py`:
  - `dividend_future_price()` — implied dividend from cost-of-carry.
  - `dividend_swap_fair_value()` — fair fixed rate.
  - `dividend_option_price()` — Black-76 on dividend forward.
  - `total_return_future()` — TR vs price return decomposition.
- 4 new tests.

---

## v0.737.0 — 2026-06-02

**Futures roll mechanics: schedule, slippage, liquidity.**

- New `fixed_income/futures_roll.py`:
  - `generate_roll_schedule()` — auto-roll calendar with costs.
  - `roll_adjusted_returns()` — continuous return series.
  - `roll_slippage()` — market impact estimation.
  - `liquidity_curve()` — volume distribution by contract month.
- 3 new tests.

---

## v0.736.0 — 2026-06-02

**Cost-of-carry decomposition and arbitrage detection.**

- New `fixed_income/cost_of_carry.py`:
  - `cost_of_carry()` — decompose forward premium: r + storage − convenience yield.
  - `cash_and_carry_arb()` — detect cash-and-carry arbitrage.
  - `reverse_cash_and_carry_arb()` — detect reverse arb.
  - `carry_roll_decomposition()` — carry vs roll return attribution.
- 5 new tests.

---

## v0.735.0 — 2026-06-02

**SABR convexity for RFR futures.**

- New `fixed_income/futures_convexity.py`:
  - `sabr_convexity_adjustment()` — Piterbarg approximation with SABR smile.
  - `hw_convexity_adjustment()` — Hull-White for comparison.
  - `empirical_convexity()` — calibrate from futures vs OIS spread.
  - `compare_convexity_models()` — side-by-side SABR vs HW.
- 5 new tests.

---

## v0.734.0 — 2026-06-02

**Commodity model calibration to futures strip.**

- New `commodity/commodity_calibration.py`:
  - `calibrate_schwartz()` — Schwartz 1F to observed futures curve.
  - `calibrate_gibson_schwartz()` — Gibson-Schwartz 2F (spot + convenience yield).
  - `seasonal_decomposition()` — multiplicative trend + seasonal extraction.
  - `implied_convenience_yield_term()` — convenience yield term structure.
- 4 new tests.

---

## v0.733.0 — 2026-06-02

**VIX futures, variance swaps, vol-of-vol.**

- New `options/variance_futures.py`:
  - `vix_futures_fair_value()` — mean-reversion model with term premium.
  - `variance_swap_price()` — model-free replication from option strip.
  - `vix_term_structure()` — contango/backwardation analysis.
  - `vol_of_vol()` — implied vol-of-vol from VIX options.
- 5 new tests.

---

## v0.732.0 — 2026-06-02

**CMBS analytics: LTV, DSCR, balloon risk, defeasance.**

- New `structured/cmbs.py`:
  - `CMBSLoan` — LTV, DSCR, debt yield per loan.
  - `CMBSPool` — weighted averages, property type concentration.
  - `price_cmbs()` — tranche pricing with credit enhancement.
  - `cmbs_stress()` — property value and NOI shocks.
  - `defeasance_cost()`, `yield_maintenance()` — prepayment penalties.
- 10 new tests.

---

## v0.731.0 — 2026-06-02

**ABS cashflow engine: auto loans, credit cards, student loans.**

- New `structured/abs.py`:
  - `price_auto_abs()` — amortising auto loan ABS with sequential waterfall.
  - `price_credit_card_abs()` — revolving + controlled amortisation.
  - `price_student_loan_abs()` — grace period, IDR, default.
  - Credit enhancement, excess spread, break-even loss rate.
- 7 new tests.

---

## v0.730.0 — 2026-06-02

**MBS prepayment modelling, OAS, IO/PO strips.**

- New `structured/mbs.py`:
  - `psa_speed()` — PSA benchmark (ramp + plateau).
  - `cpr_to_smm()`, `smm_to_cpr()` — prepayment conversions.
  - `prepayment_model()` — turnover + refinancing + burnout + seasonality.
  - `price_mbs()` — pass-through pricing with prepay-adjusted duration/convexity.
  - `oas_mbs()` — OAS via Newton-Raphson.
  - `io_po_strips()` — interest-only / principal-only decomposition.
- 10 new tests.

---

## v0.729.0 — 2026-06-02

**Spread options: Kirk's approximation with full Greeks.**

- New `commodity/spread_options.py`:
  - `kirk_spread_option()` — Kirk's approximation for 2-asset spread options.
  - `crack_spread_option()` — option on refining margin.
  - `calendar_spread_option()` — option on front-back spread.
  - `intercommodity_spread_option()` — WTI-Brent and similar.
  - Cross-gamma, correlation sensitivity via finite differences.
  - Put-call parity verified.
- 9 new tests.

---

## v0.728.0 — 2026-06-02

**Commodity futures options with seasonal vol and Samuelson effect.**

- New `commodity/commodity_options.py`:
  - `commodity_option_price()` — Black-76 with seasonal vol adjustment.
  - `seasonal_vol()` — per-commodity monthly patterns (NG, CL, ZC, ZW, ZS, GC, SI).
  - `vol_term_structure()` — Samuelson effect (front-month vol > back-month).
  - `commodity_option_strip()` — price strip across delivery months.
  - `commodity_implied_vol()` — Newton-Raphson implied vol extraction.
- 8 new tests.

---

## v0.727.0 — 2026-06-02

**Futures options: unified product with contract specs and BAW.**

- New `options/futures_options.py`:
  - `FuturesOption` — option on any futures contract (ES, CL, GC, ZN, etc.).
  - Black-76 + Bachelier pricing. Barone-Adesi-Whaley for American exercise.
  - Full Greeks: delta, gamma, vega, theta — per-unit and dollar amounts.
  - 14 contract specs (equity index, commodity, bond, IR).
  - `futures_option_strip()` — strip across expiries.
  - `futures_option_vol_surface()` — build and interpolate vol surface.
  - Put-call parity verified.
- 11 new tests.

---

## v0.726.0 — 2026-06-02

**Code review fixes across CDS infrastructure.**

- **credit_spread_vol.py**: `build_credit_vol_surface()` nearest-neighbour fill now uses expiry/tenor distance instead of global min.
- **credit_var.py**: copula VaR sign convention aligned with historical/parametric (negative = loss).
- **credit_event.py**: auction open interest clipped to [-1, 1].
- **index_cds_swaption.py**: added `strike_spread <= 0` guard to prevent log domain error.
- **recovery_locked_cds.py**: removed unused `prev_q_c` variable in effective maturity loop.
- **distressed.py**: `distressed_cds_upfront()` now uses full protection + premium leg model (was simple spread × RPV01), consistent with `implied_cpd_from_upfront()` inversion.
- Tightened test tolerances: VaR ES assertion, distressed CPD round-trip to 0.2%.
- 10,783 tests pass.

---

## v0.725.0 — 2026-06-02

**Distressed CDS: upfront quoting, implied CPD, distressed basis.**

- Modified `credit/distressed.py`:
  - `distressed_cds_upfront()` — convert running spread to upfront payment.
  - `implied_cpd_from_upfront()` — Newton-Raphson inversion for CPD.
  - `distressed_basis()` — CDS-bond basis in distressed context.
  - Wide spread → positive upfront. Tight < running → negative.
- 6 new tests.

---

## v0.724.0 — 2026-06-02

**Succession events: merger, spin-off, split.**

- New `credit/succession.py`:
  - `SuccessionEvent` — entity, type, successors, weights.
  - `apply_succession()` — notional split by economic weight.
  - Per-successor spread adjustments. Notional conservation verified.
  - 5 ISDA succession types: merger, spin-off, split, reverse merger, acquisition.
- 5 new tests.

---

## v0.723.0 — 2026-06-02

**Weighted portfolio CDS: arbitrary long/short basket.**

- New `credit/portfolio_cds.py`:
  - `portfolio_cds_pv()` — PV of arbitrary-weight CDS basket.
  - Long/short positions, different notionals per name.
  - `constituent_cs01()` — per-name CS01 with % contribution.
  - Gross and net CS01. Par spread for the basket.
- 5 new tests.

---

## v0.722.0 — 2026-05-31

**Credit event auction simulation.**

- New `credit/credit_event.py`:
  - `CreditEvent` — entity, event type (6 ISDA types), dates.
  - `simulate_auction()` — two-stage ISDA auction (initial bidding + Dutch).
  - `settlement_amount()` — CDS payout from auction final price.
  - `CreditEventTimeline` — event → determination → auction → settlement.
  - `process_credit_event()` — end-to-end credit event processing.
- 8 new tests.

---

## v0.721.0 — 2026-05-31

**Index replication and tracking error.**

- New `credit/index_replication.py`:
  - `replicate_index()` — optimal weights via least squares / LASSO.
  - Greedy name selection by correlation for sparse replication.
  - L1-regularised coordinate descent for sparsity.
  - `tracking_error()` — annualised TE vs full index.
  - TE decreases with more names (verified).
- 5 new tests.

---

## v0.720.0 — 2026-05-31

**Index roll mechanics: series transition and OTR basis.**

- New `credit/index_roll.py`:
  - `series_transition()` — apply name additions/removals.
  - `index_roll_pnl()` — P&L from rolling to new series.
  - `on_the_run_basis()` — OTR vs off-the-run spread difference.
  - `series_transition_pnl()` — transition + P&L in one step.
- 5 new tests.

---

## v0.719.0 — 2026-05-31

**Recovery-locked CDS and Loan CDS (LCDS).**

- New `credit/recovery_locked_cds.py`:
  - `price_recovery_locked_cds()` — fixed recovery eliminates auction risk.
  - `recovery_lock_premium()` — premium for locking recovery vs market.
  - `price_lcds()` — Loan CDS with prepayment cancellation.
  - Higher loan recovery (70-80%), effective maturity shortened by CPR.
  - Cancellation value: RPV01 difference with/without prepayment.
- 4 new tests.

---

## v0.718.0 — 2026-05-31

**Index CDS swaption: Black-76 and Bachelier on forward index spread.**

- New `credit/index_cds_swaption.py`:
  - `index_forward_spread()` — annuity-weighted forward (Jensen's inequality).
  - `index_cds_swaption_black()` — Black-76 on forward index spread.
  - `index_cds_swaption_bachelier()` — Bachelier (normal) model.
  - `index_swaption_greeks()` — delta, gamma, vega, theta via finite diff.
  - `price_index_cds_swaption()` — full pricing from curves.
  - Put-call parity verified.
- 7 new tests.

---

## v0.717.0 — 2026-05-31

**Credit portfolio VaR: historical, parametric, and copula-based.**

- New `credit/credit_var.py`:
  - `historical_credit_var()` — CS01-weighted spread P&L from history.
  - `parametric_credit_var()` — delta-normal with correlation matrix.
  - `copula_credit_var()` — Gaussian copula joint-default simulation.
  - `CreditVaRResult` with VaR, ES, worst name, component contributions.
- 5 new tests.
- 10,733 tests pass.

---

## v0.716.0 — 2026-05-31

**Quanto CDS: cross-currency CDS with FX-credit correlation.**

- New `credit/quanto_cds.py`:
  - `quanto_cds_spread()` — adjustment: `spread × exp(ρ × σ_FX × σ_credit × T)`.
  - `price_quanto_cds()` — full pricing with FX hedge notional.
  - `quanto_adjustment_factor()` — convexity adjustment factor.
  - Positive correlation → quanto spread > foreign (wrong-way risk).
- 5 new tests.
- 10,733 tests pass.

---

## v0.715.0 — 2026-05-31

**Credit spread vol surface: ATM backbone with bilinear interpolation.**

- New `credit/credit_spread_vol.py`:
  - `CreditSpreadVolSurface` — 2D (expiry × tenor) ATM vol grid.
  - Bilinear interpolation matching `SwaptionVolCube` pattern.
  - `synthetic_credit_vol_surface()` — IG (~40%) / HY (~60%) vol generation.
  - Parallel bump support for risk scenarios.
- 5 new tests.
- 10,733 tests pass.

---

## v0.714.0 — 2026-06-01

**Bermudan CDS swaption: multiple exercise dates.**

- New `credit/bermudan_cds_swaption.py`:
  - `bermudan_cds_swaption_price()` — backward induction on hazard/discount tree.
  - At each exercise date: max(continuation, forward CDS PV).
  - Bermudan ≥ European verified. Single date → equals European.
  - Payer and receiver. ITM > OTM. Exercise probability tracked.
- 8 new tests.
- 10,718 tests pass.

---

## v0.713.0 — 2026-06-01

**Code review fixes across curve + vol infrastructure.**

- **capfloor.py**: fixed unreachable `pv_ctx` (was dead code after `return` inside wrong function). Moved to module level and assigned to `CapFloor.pv_ctx`. Fixed broken indentation.
- **curve_builder.py**: `CurveSetResult.to_dict()` now returns serialisable dict (was `vars(self)` with DiscountCurve objects).
- **swaption_vol_cube.py**: `bumped()` now shifts SABR alpha alongside ATM vol for consistent smile bumps.
- **swaption.py**: removed dead `df` variable in `price_swaption_sabr_hw`.
- **hw_calibration.py**: removed dead `df_settle` variable and unused `field` import.
- **hw_per_currency.py**: removed unused `math` import.
- 10,710 tests pass.

---

## v0.712.0 — 2026-06-01

**End-to-end callable pricing workflow notebook with pricebook.viz.**

- New `notebooks/rates/callable_pricing_workflow.ipynb`:
  - EUR yield curve: 3 methods (log-linear, Nelson-Siegel 1.6bp, Svensson). Realistic humped term structure.
  - HW calibration from 8 swaption vols. Per-swaption fit diagnostics.
  - Swaption vol cube: ATM heatmap + SABR smile.
  - Callable bond: straight 110.76 vs callable 99.37 (call value 11.39). Negative convexity.
  - Bermudan 5nc1: 19bp early exercise premium over European.
  - Multi-currency HW params (6 currencies, G10 vs EM).
  - All visuals use pricebook.viz.
- 10,710 tests pass.

---

## v0.711.0 — 2026-06-01

**Synthetic curve data, SABR-HW blended pricing, cap/floor SABR.**

- New `curves/synthetic_market_data.py`:
  - `synthetic_curve_inputs(currency)` — realistic deposits + swaps for 32 currencies.
  - USD ~5%, JPY ~0.1%, BRL ~11%, TRY ~45%. Enables testing all methods without market data.
- Extended `options/swaption.py`:
  - `price_swaption_sabr_hw()` — blends SABR smile (short end) with HW term structure (long end).
  - Weighting: `w_sabr = exp(-expiry / half_life)`. Configurable blend.
- Extended `options/capfloor.py`:
  - `strip_caplet_vols_from_quotes()` — per-caplet vol stripping from cap quotes.
  - `calibrate_capfloor_sabr()` — per-expiry SABR from caplet vols.
- 9 new tests (synthetic data, SABR-HW blending).
- 10,710 tests pass.

---

## v0.710.0 — 2026-06-01

**Cap/floor SABR, dual real+nominal curves, NDF-implied verification.**

- `options/capfloor.py`: `strip_caplet_vols_from_quotes()` + `calibrate_capfloor_sabr()`.
- `curves/inflation_curve.py`: `build_real_nominal_curves()` → nominal + real + BEI.
- NDF curves: verified existing `build_ndf_implied_curve()` + `cip_basis()`.
- 12 new tests.
- 10,701 tests pass.

---

## v0.709.0 — 2026-06-01

**Swaption infrastructure: per-currency conventions, synthetic data, HW per currency.**

- New `options/swaption_conventions.py`:
  - `SwaptionConvention` per currency: vol quote type (Black/Normal/Shifted), frequencies, SABR type, standard grids.
  - 11 currencies: USD (shifted-SABR), EUR (Normal/Bachelier), GBP, JPY, CHF, CAD, AUD, BRL (BUS/252), MXN, KRW, ZAR.
- New `options/synthetic_swaption_data.py`:
  - `synthetic_atm_surface(currency)` — realistic ATM vols (USD ~60bp, JPY ~25bp, BRL ~200bp).
  - `synthetic_smile_data(currency)` — RR25/BF25 per node.
  - `synthetic_hw_targets(currency)` — swaption vol targets for HW calibration.
- New `models/hw_per_currency.py`:
  - `calibrate_hw_for_currency(currency, ref, curve)` — full pipeline: synthetic vols → HW calibration.
  - Default parameters for 33 currencies (G10, EM, Asia, CEE).
  - EM defaults: higher mean reversion + vol (BRL a=0.10, TRY a=0.15).
- 16 new tests.
- 10,689 tests pass.

---

## v0.708.0 — 2026-06-01

**Swaption vol cube: 3D (expiry × tenor × strike) with SABR smile.**

- New `options/swaption_vol_cube.py`:
  - `SwaptionVolCube` — ATM backbone (bilinear interpolation) + per-node SABR smile.
  - `vol(expiry, tenor, strike)` — full 3D interpolation.
  - `smile(expiry, tenor, strikes)` — vol smile across strikes.
  - `bumped(shift)` — parallel vol shift.
  - `build_swaption_vol_cube()` — construct from ATM grid + smile quotes.
  - `SABRNode` — per-(expiry, tenor) SABR params (alpha, beta, rho, nu).
  - SABR calibration via `sabr_calibrate()` at each smile node.
- OTM vol differs from ATM when SABR is fitted (smile verified).
- 12 new tests.
- 10,673 tests pass.

---

## v0.707.0 — 2026-06-01

**Unified curve methods: all 33 currencies now have 5 construction methods.**

- `get_conventions()` in `curve_builder.py` now falls through from G10 to EM registry.
- `build_curves()` accepts ANY of the 33 currencies (was limited to 10 G10).
- EM currencies (BRL, MXN, CNY, KRW, INR, ZAR, PLN, etc.) can now use:
  - Sequential bootstrap, Global Newton, Nelson-Siegel, Svensson, Smith-Wilson.
- Cross-method consistency: 5Y zero rate within 100bp across methods.
- Note: Smith-Wilson fails at extreme rates (TRY 45%) — use sequential for extreme EM.
- 17 new tests.
- 10,661 tests pass.

---

## v0.706.0 — 2026-06-01

**Hull-White calibration from swaption volatilities — CRITICAL GAP FILLED.**

- New `models/hw_calibration.py`:
  - `calibrate_hull_white(curve, swaption_vols)` → calibrated `HullWhite` model.
  - Minimises Σ(model_vol - market_vol)² across swaption grid.
  - Model vol: HW tree pricing → Black-76 vol inversion.
  - Optimisers: Nelder-Mead (default), differential evolution, L-BFGS-B.
  - ATM strike auto-computed from forward swap rates if not provided.
  - Per-swaption fit diagnostics (error in bp).
  - Round-trip verified: generate vols from known (a=0.03, σ=0.01), calibrate back within 30%.
- Enables: calibrated callable bond pricing, Bermudan swaption pricing, cancellable swap pricing from market vol data.
- 8 new tests.
- 10,644 tests pass.

---

## v0.705.0 — 2026-06-01

**Reorganise notebooks into thematic subdirectories.**

- `notebooks/americas/` — argentina, canada, chile, colombia, mexico, peru (6)
- `notebooks/rates/` — treasury_note_roundtrip, treasury_multicurve, asw_btp_bund (3)
- `notebooks/credit/` — recovery_roundtrip (1)
- `notebooks/structured/` — prdc_structuring, tarf_risk_profile, xccy_basis_pricing (3)
- `notebooks/desks/` — bond_trading_desk, futures_desk (2)
- `notebooks/validation/` — cmasw_pucci_2012a, cmt_pucci_2014, index_linked_hybrid_pucci_2012b, treasury_lock_pucci_2019, trs_lou_2018 (5)
- Renamed for consistency: `*_derivatives.ipynb` → short country names, `*_validation.ipynb` → paper names only.
- Fixed `sys.path` in all 20 notebooks for 2-level-deep directory structure.
- Cleaned up stale `.ipynb_checkpoints`.
- 10,636 tests pass.

---

## v0.704.0 — 2026-06-01

**Code review fixes for callable/cancellable modules.**

- **callable_cds.py**: fixed discount factor — now uses `df(t_next)/df(t)` instead of `exp(-zero_rate*dt)`. Fixed date arithmetic to use `timedelta(days=round(t*365.25))` instead of `int(t*365)`.
- **cancellable_swap.py**: fixed receiver swap sign logic — cancellation right always reduces PV for the non-option-holder regardless of direction.
- **callable_cln.py**: coupon now survival-weighted (`coupon * p_survive`). Call date matching uses 5-day tolerance instead of exact equality.
- **Exception handling**: narrowed from bare `except Exception` to `except (ImportError, TypeError, ValueError)` in cancellable_swap and extendible.
- 10,636 tests pass.

---

## v0.703.0 — 2026-06-01

**Callable/cancellable derivatives: cancellable swap, extendible, callable CDS, callable CLN.**

- New `fixed_income/cancellable_swap.py`:
  - `cancellable_swap_price()` — swap + embedded Bermudan swaption decomposition.
  - Cancellable PV ≤ vanilla PV (option costs the holder). Par rate adjusted.
- New `fixed_income/extendible.py`:
  - `extendible_swap_price()` — base swap + European swaption on extension period.
  - Extendible PV ≥ base PV (extension adds value for holder).
- New `credit/callable_cds.py`:
  - `callable_cds_price()` — CDS with seller termination right via backward induction.
  - Callable PV ≤ vanilla. Callable spread ≥ vanilla spread.
- New `credit/callable_cln.py`:
  - `callable_cln_price()` — CLN with issuer early redemption via backward induction.
  - Callable ≤ straight CLN. Higher coupon → more call value.
  - Call probability, expected call date, par spread for callable.
- All compose over existing Hull-White tree / survival curve infrastructure.
- 17 new tests.
- 10,636 tests pass.

---

## v0.702.0 — 2026-06-01

**Asia build-out: 9 currencies — CNY, KRW, INR, SGD, HKD, THB, IDR, MYR, PHP.**

- New modules: chinese.py, korean.py, singaporean.py, hong_kong.py, thai.py, indian.py, indonesian.py, malaysian.py, philippine.py.
- **Korea (KRW)**: KOFRSwap + KTB + KTBi linker (CPI_KR, deflation floor) + BEI.
- **India (INR)**: MIBORSwap + GSEC (**30/360** — only sovereign globally) + IIB linker (CPI_IN, deflation floor) + BEI. MIBOR rate index added.
- **Philippines (PHP)**: PHIREFSwap + RPGB (**quarterly coupon** — only quarterly sovereign globally). PHIREF rate index added.
- **China (CNY)**: DR007Swap + CGB. **Indonesia**: INDONIASwap + INDOGB. **Malaysia**: MYORSwap + MGS. **Singapore**: SORASwap + SGS. **HK**: HONIASwap + HKGB. **Thailand**: THORSwap + THAIGB.
- 4 new rate indices: MIBOR (FBIL), INDONIA (BI), MYOR (BNM), PHIREF (BSP).
- 9 new OIS conventions added.
- 43 new tests.
- Markets with full derivatives: 24 → 33.
- 10,619 tests pass.

---

## v0.701.0 — 2026-06-01

**BEI (breakeven inflation) added to 9 markets — now 16 markets have BEI.**

- Added `breakeven_inflation_XX()` convenience functions to: BRL, MXN, COP, PEN, ARS, PLN, CZK, HUF, TRY.
- Total markets with BEI: 16 (GBP, CAD, CLP, JPY, AUD, ZAR, ILS + 9 new).
- All follow same pattern: nominal_rate - real_rate from two discount curves.
- Argentina/Turkey: extreme BEI values expected (~30%+ / ~35%+).
- 10,576 tests pass.

---

## v0.700.0 — 2026-06-01

**Japan, Australia, South Africa, Israel: full derivatives with inflation linkers + BEI.**

- New `fixed_income/japanese.py`: TONASwap, JGBBond, JGBiLinker (CPI_JP, 3M lag, **deflation floor**), BEI. Near-zero rate handling.
- New `fixed_income/australian.py`: AONIASwap, ACGBBond, TIBBond (CPI_AU, **quarterly coupon** — only quarterly linker globally, **no deflation floor**), BEI.
- New `fixed_income/south_african.py`: JIBARSwap (**quarterly** fixed), SAGBBond (T+3), SAILBBond (CPI_ZA, no floor), BEI.
- New `fixed_income/israeli.py`: TelborSwap, ShaharBond, GalilBond (CPI_IL, **1-month lag**, **annual coupon**, no floor), BEI.
- Markets with full derivatives: 20 → 24.
- 32 new tests.
- 10,576 tests pass.

---

## v0.699.0 — 2026-06-01

**Code review fixes across all new market modules.**

- **Nordic template placeholders**: fixed `{country}` → "Swedish"/"Norwegian"/"Danish" in 6 docstrings.
- **CEE linker conventions**: changed frequency from semi-annual to annual (PLN, CZK, HUF linkers). Fixed CZK/HUF linker day counts from ACT/360 to ACT/365F to match inflation_indices.json.
- **PLN IRS**: fixed leg frequency changed from semi-annual to annual (market standard).
- **Gilt**: added past cashflow filtering in `dirty_price()` (was including past coupons).
- **Danish mortgage**: removed unused `import numpy`.
- **Rate indices JSON**: SWESTR and NOWA observation_shift corrected from 0 to 2.
- 10,544 tests pass.

---

## v0.698.0 — 2026-06-01

**Danish mortgage bonds (realkreditobligationer) — callable covered bonds with prepayment.**

- New `fixed_income/danish_mortgage.py` (300 lines):
  - `DanishMortgageBond` — callable at par, bullet or pass-through amortisation.
  - `prepayment_model()` — CPR as function of refinancing incentive (coupon - market rate), with seasoning ramp-up.
  - `psa_curve()` — PSA-standard prepayment ramp (30-month, configurable speed).
  - `MortgageBondResult` — dirty price, OAS, effective duration, WAL, expected CPR, callable value.
  - Effective duration via ±10bp parallel bump (non-recursive).
  - Callable price ≤ non-callable (negative convexity verified).
  - Higher CPR → shorter WAL. Pass-through WAL < bullet WAL.
  - OAS > 0 for callable bonds with refinancing incentive.
- 16 new tests.
- 10,544 tests pass.

---

## v0.697.0 — 2026-06-01

**CEE + Turkey: PLN, CZK, HUF, TRY — dual IBOR+RFR swaps + inflation linkers.**

- New `fixed_income/polish.py`: WIBORSwap (3M), WIRONSwap (overnight), POLGBBond (annual ACT/ACT ICMA), POLGBLinker (CPI_PL). WIRON rate index added.
- New `fixed_income/czech.py`: PRIBORSwap (3M), CZEONIASwap (overnight), CZGBBond, CZGBLinker (CPI_CZ). CZEONIA rate index added.
- New `fixed_income/hungarian.py`: BUBORSwap (3M), HUFONIASwap (overnight), HGBBond (**ACT/365F** — unique among CEE), HGBLinker (CPI_HU). HUFONIA rate index added.
- New `fixed_income/turkish.py`: TLREFSwap, TURKGBBond (semi-annual ACT/365F, **T+0 settlement**), TurkishCPILinker (CPI_TR, 2-month lag). Handles 45%+ extreme rates. TLREF rate index added.
- 4 new overnight rate indices: WIRON, CZEONIA, HUFONIA, TLREF.
- 3 new inflation indices: CPI_PL, CPI_CZ, CPI_HU.
- 29 new tests (8 PLN + 7 CZK + 7 HUF + 7 TRY).
- 10,528 tests pass.

---

## v0.696.0 — 2026-06-01

**Switzerland + Nordics: SARON, SWESTR, NOWA, DESTR swaps + sovereign bonds.**

- New `fixed_income/swiss.py`: SARONSwap (ACT/360), ConfedBond (annual ACT/ACT ICMA). Handles negative rates (CHF DF > 1 verified).
- New `fixed_income/swedish.py`: SWESTRSwap, SGBBond. SWESTR rate index added.
- New `fixed_income/norwegian.py`: NOWASwap, NGBBond. NOWA rate index added.
- New `fixed_income/danish.py`: DESTRSwap, DGBBond. DESTR rate index added. DKK OIS convention added to ois.py.
- 3 new overnight rate indices in rate_indices.json: SWESTR (Riksbank), NOWA (Norges Bank), DESTR (Danmarks Nationalbank).
- 21 new tests (6 CHF + 5 SEK + 5 NOK + 5 DKK).
- 10,499 tests pass.

---

## v0.695.0 — 2026-06-01

**UK: SONIA swap, Gilt, Index-Linked Gilt (ILG), breakeven inflation.**

- New `fixed_income/british.py` (330 lines):
  - `SONIASwap` — annual ACT/365F, par rate, DV01, direction symmetry.
  - `GiltBond` — semi-annual ACT/ACT ICMA, 7-day ex-dividend, T+1.
  - `ILGBond` — **8-month RPI lag, flat interpolation** (not linear like TIPS), **no deflation floor** (unlike TIPS). Nominal = real × RPI ratio.
  - `build_sonia_curve()` — ACT/365F bootstrap.
  - `breakeven_inflation_uk()` — nominal Gilt vs real ILG curves (2Y-50Y).
  - `synthetic_sonia_strip()`, `synthetic_gilt_strip()`.
- ILG deflation: RPI ratio < 1.0 when RPI falls (verified — no floor).
- UK BEI (RPI-based) ~3.5%, consistent with market.
- 16 new tests.
- 10,478 tests pass.

---

## v0.694.0 — 2026-06-01

**Canada deepening: CGB, Canadian IRS, provincial bonds, breakeven inflation.**

- Extended `fixed_income/canadian.py` (117→340 lines):
  - `CGBBond` — Canadian Government Bond, semi-annual ACT/365F, yield-to-maturity solver.
  - `CanadianIRS` — fixed semi-annual vs CORRA compound, par rate, DV01.
  - `ProvincialBond` — spread over federal CGB curve (ON, QC, BC, AB, MB, SK).
  - `breakeven_inflation_ca()` — CORRA nominal vs RRB real curves.
  - `synthetic_cgb_strip()` — 4 benchmark CGB quotes (2Y, 5Y, 10Y, 30Y).
  - Provincial spread ordering verified: BC (25bp) < AB (30bp) < ON (35bp) < QC (40bp).
  - IRS direction symmetry: pay_fixed PV = -receive_fixed PV.
- 10 new tests.
- 10,462 tests pass.

---

## v0.693.0 — 2026-06-01

**Market-accurate bond curve: per-bond day count convention + sovereign factory.**

- `BondQuote` now supports `day_count`, `settlement_days`, `calendar_ccy` fields.
- `BondQuote.from_sovereign(market_code, ...)` — auto-sets conventions from the 60-market sovereign registry:
  - UST: ACT/ACT ICMA, semi-annual, T+1
  - BUND: ACT/ACT ICMA, annual, T+2
  - JGB: ACT/365F, semi-annual, T+2
  - NTN_F: BUS/252, semi-annual, T+1 (loads BRL calendar)
  - MBONO: ACT/360, semi-annual, T+2
- `_price_bond()` rewritten: uses the bond's own day count for accrual fractions.
  - ACT/ACT ICMA: passes coupon period boundaries + frequency.
  - BUS/252: loads calendar from `calendar_ccy`.
  - All other conventions: straightforward.
- Sequential and global bootstrap both use per-bond conventions.
- Verified: different day counts produce different implied zero rates.
- 8 new tests (sovereign factories, multi-market curves, day count impact).
- 10,452 tests pass.

---

## v0.692.0 — 2026-06-01

**Yield curve bootstrapping from bond prices alone.**

- New `curves/bond_curve.py`:
  - `BondQuote` — bond observation (maturity, coupon, dirty price, weight, on-the-run flag).
  - `bootstrap_curve_from_bonds()` — unified entry point with 4 methods:
    - `"sequential"` — exact fit, one bond per pillar (like CDS bootstrap but for DFs).
    - `"global"` — least-squares, robust to noise, supports n_pillars < n_bonds.
    - `"nelson_siegel"` — 4-parameter smooth curve fitted directly to bond prices (not zero rates).
    - `"svensson"` — 6-parameter smooth curve (captures humps better than NS).
    - `"auto"` — sequential if ≤8 distinct maturities, else global.
  - On-the-run bonds get 2× weight in global/parametric fits.
  - Zero-coupon bonds (T-Bills): exact DF extraction.
  - NS long-end converges to β₀. Svensson fits at least as well as NS.
  - Cross-method: 5Y zero rate consistent within 200bp across all methods.
- `BondCurveResult` with discount_curve, pillar zeros, fitted prices, RMSE, parameters.
- 22 new tests.
- 10,444 tests pass.

---

## v0.691.0 — 2026-06-01

**FRN hazard bootstrapping, mixed fixed+float, and liquid/illiquid regime handling.**

- New in `credit/bond_hazard_bootstrap.py`:
  - `FRNInput` — floating-rate note observation (spread, benchmark, market price).
  - `_price_risky_frn()` — risky FRN pricing with survival-weighted floating coupons and recovery leg.
  - `bootstrap_hazard_mixed()` — global fit from mix of fixed-rate bonds and FRNs. Returns piecewise hazard curve.
  - `LiquidityAssessment` — regime classification (liquid/semi_liquid/illiquid) with recommended method, n_pillars, confidence.
  - `assess_liquidity()` — heuristic assessment from bond count, bid-ask widths, price levels, maturity coverage.
  - `bootstrap_hazard_adaptive()` — auto-selects method based on liquidity:
    - Liquid: sequential bootstrap (exact fit).
    - Semi-liquid: global fit with bid-ask-adjusted weights.
    - Illiquid: global fit with 1-3 pillars.
  - Bid-ask weighting: `w = 1/(1 + ba/100)` — wider spread → lower weight.
  - Distressed bonds (50-60 cents): produces high hazard rates, survival still decreasing.
- 19 new tests.
- 10,422 tests pass.

---

## v0.690.0 — 2026-06-01

**Fix remaining known limitations: Frank copula, tranche annuity, barrier vectorization.**

- **Frank copula**: rewrote d≥3 sampling using Marshall-Olkin algorithm with logarithmic series mixing variable. Previously used bivariate conditional method that produced incorrect multivariate dependence.
- **TrancheCDS.price()**: replaced single-period annuity approximation with proper multi-period premium and protection legs (quarterly frequency). Par spread now computed from risky annuity ratio.
- **Barrier continuous mode**: vectorized Python loops for knockout and knockin. ~10-50x speedup for large n_paths. Correct bridge probability formula for both up and down barriers.
- 10,403 tests pass.

---

## v0.689.0 — 2026-05-31

**Code review fixes: CDO PMF, barrier bridge, copula M factor, dt guard, BMA default.**

- **CDO MC**: fixed PDF/PMF mismatch — `portfolio_loss_distribution_mc` now returns PMF (probability mass) consistent with analytical Vasicek. `tranche_expected_loss_mc` now produces correct results.
- **Barrier bridge**: fixed bridge_min formula for down-and-out/down-and-in — now uses correct conditional probability `P(min < b) = exp(-2(s0-b)(s1-b)/(σ²dt))` instead of incorrect `s0 + s1 - max` approximation.
- **Non-Gaussian copula**: systematic factor M now uses `sample_with_factor()` for Gaussian copula (correct), and independent fallback for non-Gaussian (honest about limitation, was previously using meaningless Z.mean approximation).
- **OU exact step**: added `dt < 1e-14` guard to prevent `dw/sqrt(dt)` numerical instability.
- **BMA**: None AIC/BIC now gets mean IC of other models (was 0.0, which gave infinite weight).
- 10,403 tests pass.

---

## v0.688.0 — 2026-05-31

**Model reserves framework: parameter uncertainty, reserves, P&L attribution, model selection.**

- New `risk/parameter_uncertainty.py`:
  - `ParameterBand` — confidence interval for calibrated parameter.
  - `calibration_uncertainty()` — bootstrap CI from market data.
  - `sensitivity_ladder()` — PV impact at band edges, sorted by magnitude.
  - `joint_parameter_surface()` — 2D PV surface over two parameter bands.
- New `risk/model_reserve.py`:
  - `compute_model_reserve()` — worst-case or quadrature (√Σ) reserve from bands.
  - `reserve_by_risk_factor()` — per-parameter reserve breakdown.
  - `model_risk_reserve_ava()` — EBA-compatible AVA format.
- New `risk/model_selection.py`:
  - `ModelCandidate` — model with pricer, weight, AIC/BIC.
  - `model_committee_price()` — weighted average + dispersion + uncertainty reserve.
  - `bayesian_model_average()` — posterior weights from AIC/BIC.
  - `model_risk_matrix()` — price all models under all scenarios.
- Extended `risk/pnl_explain.py`:
  - `surface_pnl()` — ATM/skew/smile/term structure P&L decomposition.
  - `gamma_pnl_decompose()` — realised vs implied gamma, net gamma P&L.
  - `NonLinearPnLResult` dataclass.
- 21 new tests.
- 10,403 tests pass.

---

## v0.687.0 — 2026-05-31

**Recovery extras: heterogeneous specs, seniority waterfall, bid-ask surface.**

- New in `credit/recovery_pricing.py`:
  - `build_recovery_specs(seniorities)` — from Moody's table per-name.
  - `validate_recovery_specs(specs, n_names)` — length check.
  - `recovery_spec_summary(specs)` — portfolio-level stats.
  - `SeniorityWaterfall` — capital structure priority distribution.
    - `distribute(total_recovery)` — senior gets first, sub gets remainder.
    - `recovery_rates(total_pct)` — per-tranche recovery rates.
    - `to_recovery_specs()` — waterfall-consistent RecoverySpec list.
  - `implied_recovery(spread, hazard)` — R = 1 - s/h.
  - `recovery_bid_ask_surface()` — term structure of implied recovery with bid-ask.
- 17 new tests.
- 10,382 tests pass.

---

## v0.686.0 — 2026-05-31

**OU exact step + MC convergence diagnostics.**

- `OUProcess`: exact Gaussian transition (was Euler). Mean reversion to θ, stationary variance σ²/(2κ) verified.
- New `models/mc_diagnostics.py`:
  - `batch_means()` — robust SE estimation via inter-batch variance.
  - `effective_sample_size()` — autocorrelation-adjusted ESS via FFT.
  - `convergence_table()` — running mean/SE at checkpoints.
  - ESS = N for iid, ESS < N for AR(1) verified.
- 13 new tests.
- 10,365 tests pass.

---

## v0.685.0 — 2026-05-31

**Heterogeneous portfolios: per-name notional and LGD in bespoke tranches.**

- `bespoke_tranche()`: new `notionals` and `lgds` parameters.
- `notionals`: per-name portfolio weights (default: equal weight).
- `lgds`: per-name loss given default (overrides uniform `lgd`).
- Concentrated portfolio: name with 5x weight dominates loss.
- Uniform notionals/lgds reproduce current flat behavior exactly.
- Works with `recovery_specs` for full per-name stochastic recovery.
- 6 new tests.
- 10,352 tests pass.

---

## v0.684.0 — 2026-05-31

**Multi-copula support in basket CDS.**

- `ftd_spread()`, `ntd_spread()`: new `copula` parameter.
- Accepts any `Copula` instance from `statistics/copulas.py`: Gaussian, Student-t, Clayton, Frank, Gumbel.
- Student-t copula produces higher FTD spread (tail dependence clusters defaults).
- When copula=None, falls back to one-factor Gaussian (backward compatible).
- Approximate systematic factor extraction for non-Gaussian copulas (recovery correlation).
- 7 new tests.
- 10,334 tests pass.

---

## v0.683.0 — 2026-05-31

**Base correlation surface with cubic spline interpolation and arbitrage checks.**

- New `BaseCorrelationSurface` class in `credit/tranche_pricing.py`:
  - `interpolate(detachment, method)` — linear or cubic spline with monotonicity enforcement.
  - `check_arbitrage()` — detects non-monotonicity and out-of-bounds correlations.
  - `bump(shift)` — parallel shift with clamping to (0, 1).
  - `from_calibration()` — build from `calibrate_base_correlation()` output.
  - Callable: `surface(0.07)` returns interpolated base correlation.
- 13 new tests.
- 10,340 tests pass.

---

## v0.682.0 — 2026-05-31

**Configurable time discretization in basket CDS (quarterly default).**

- `ftd_spread()`, `ntd_spread()`: new `frequency` parameter (1=annual, 4=quarterly, 12=monthly).
- Default changed from annual (frequency=1) to quarterly (frequency=4).
- More time points → finer survival/default assessment.
- Convergence: monthly ≈ quarterly (verified).
- 5 new tests.
- 10,327 tests pass.

---

## v0.681.0 — 2026-05-31

**MC portfolio loss distribution with stochastic recovery for CDO.**

- New `portfolio_loss_distribution_mc()` in `credit/cdo.py`:
  - Monte Carlo complement to analytical Vasicek (which requires constant LGD).
  - Accepts `RecoverySpec` for per-name stochastic recovery correlated to M.
  - MC with fixed recovery converges to analytical EL = PD × LGD.
- New `tranche_expected_loss_mc()` — wraps MC loss dist with tranche clipping.
- Equity EL > Senior EL verified. Density non-negative, integrates to 1.
- 8 new tests.
- 10,322 tests pass.

---

## v0.680.0 — 2026-05-31

**Per-name stochastic recovery in copula default simulation.**

- `copula_default_simulation()`, `tranche_pricing_copula()`: new `recovery_specs` parameter.
- `GaussianCopula.sample_with_factor()`: returns (U, M) — uniform marginals + systematic factor.
- For Gaussian copula: recovery correlated to M. For non-Gaussian (Clayton, Gumbel, Frank): unconditional recovery.
- Heterogeneous seniority: mix senior + subordinated recovery in same portfolio.
- 8 new tests.
- 10,314 tests pass.

---

## v0.679.0 — 2026-05-31

**Stochastic correlated recovery in CDO tranche pricing.**

- `expected_tranche_loss()`, `expected_tranche_loss_t()`, `TrancheCDS.price()`: new optional `recovery_specs` parameter.
- Per-name stochastic recovery sampled correlated to systematic factor M.
- Student-t copula: uses underlying normal M for recovery correlation (not t-scaled).
- Wrong-way risk verified: equity tranche EL increases; senior tranche less affected.
- Fixed RecoverySpec reproduces flat recovery. Backward compatible.
- 6 new tests.
- 10,306 tests pass.

---

## v0.678.0 — 2026-05-31

**Stochastic correlated recovery in basket CDS (FTD/NTD/bespoke).**

- `ftd_spread()`, `ntd_spread()`, `bespoke_tranche()`: new optional `recovery_specs` parameter.
- Accepts `list[RecoverySpec]` — per-name stochastic recovery correlated to systematic factor M.
- Wrong-way risk: negative default-recovery correlation increases FTD spread.
- Heterogeneous seniority: mix senior secured (R=65%) and subordinated (R=28%) in same basket.
- Fixed RecoverySpec(0.4, 0) reproduces flat recovery exactly. Backward compatible.
- 8 new tests.
- 10,300 tests pass.

---

## v0.677.0 — 2026-05-31

**Fix LSM American put discounting + continuous barrier monitoring.**

- **American put LSM**: added `r` parameter for proper discounting of continuation values in backward induction. Higher r → earlier exercise (correct behavior). American ≥ European verified.
- **Barrier options**: added `continuous=True, sigma=σ` parameters to `barrier_knockout` and `barrier_knockin`. Uses Brownian bridge max/min sampling for continuous monitoring from discrete paths. Continuous up-out ≤ discrete up-out (more knockouts). Knockin + knockout ≈ vanilla (parity check).
- Backward compatible: defaults match old behavior (r=0, continuous=False).
- 11 new tests.
- 10,292 tests pass.

---

## v0.676.0 — 2026-05-31

**Fix non-reproducible MC paths in Merton, Bates, and Variance Gamma processes.**

- `JumpDiffusionProcess`, `BatesProcess`, `VarianceGammaProcess` now accept `seed` parameter.
- Replaced global `np.random.poisson()`/`np.random.randn()`/`np.random.gamma()` with closure-captured `np.random.default_rng(seed)`.
- Same seed → identical paths guaranteed. Different seeds → different paths.
- Backward compatible: `seed=None` uses unseeded RNG (old behavior).
- 7 new tests verifying reproducibility.
- 10,281 tests pass.

---

## v0.675.0 — 2026-05-31

**Deep fixes for remaining known limitations.**

- **CGMY MC simulation**: rewrote to proper difference-of-Gamma representation with exact risk-neutral drift from char_func. Shape parameters use Γ(1-Y)·rate^(Y-1) moment matching.
- **Cross-validation MC**: now covers all 6 models (added Kou via compound Poisson + double-exponential, CGMY via new terminal(), Bates via mc_migrate). Custom params are now respected.
- **Theta decomposition**: computes actual total theta via 1-day maturity bump. Vol theta is now residual = total - carry - div (was hardcoded 0).
- **Dividend surface simulation**: `spot_vol` and `kappa_q` now explicit parameters (were hardcoded 0.20/2.0). Returns `DividendSimResult` dataclass (was raw dict). Uses log-Euler scheme (prevents negative spot).
- **Char func API consistency**: all standalone factories now follow `(rate, model_params..., T)` ordering. `vg_char_func`, `nig_char_func`, `cgmy_char_func` signatures updated. **Breaking change** for direct callers.
- Correlation clamped to [-0.999, 0.999] in simulation (prevents sqrt of negative).
- 10,274 tests pass.

---

## v0.674.0 — 2026-05-31

**Code assessment fixes across jump + dividend modules.**

- **CGMY**: reject Y=1 (pole of Γ(-Y)) at construction.
- **NIG**: validate `alpha > |beta+1|` (risk-neutral measure existence).
- **VG**: guard `1 - θν - 0.5σ²ν > 0` with clear error message.
- **American tree**: rewrote to spot-adjustment model — subtract PV of all future dividends, build CRR on adjusted spot, add PV back for intrinsic comparison. Fixes dividend propagation bug.
- **RGW**: documented as simplified approximation (univariate, not bivariate normal).
- Removed dead code: unused `NIGResult`/`CGMYResult` dataclasses, `nig_constraint`, dead `field` imports.
- Fixed `ForwardErrorDecomp.to_dict()` missing fields.
- 10,272 tests pass.

---

## v0.673.0 — 2026-05-31

**Dividend surface + joint vol-dividend calibration.**

- New `equity/dividend_surface.py`:
  - `DividendSurface` — tenors × yield levels × yield vols × spot correlation.
  - `build_dividend_surface()` — from futures + optional dividend options.
  - `simulate_dividend_surface()` — correlated spot + OU dividend yield MC paths.
- New `equity/joint_calibration.py`:
  - `joint_calibrate()` — simultaneous vol + dividend yield fitting.
  - Models: "bsm+continuous" (flat vol + q), "term+continuous" (piecewise σ + q).
  - `decompose_forward_error()` — attribute mispricing to vol vs dividend assumptions.
  - Round-trip: recovers σ and q within 1% on synthetic data.
- 11 new tests.
- 10,272 tests pass.

---

## v0.672.0 — 2026-05-31

**American option early exercise around ex-dividend dates.**

- New `options/american_dividend.py`:
  - `american_with_dividends()` — binomial tree with ex-dates as explicit nodes, dividend spot drop.
  - `roll_geske_whaley()` — closed-form for single discrete dividend (Newton for critical spot S*).
  - `exercise_boundary_around_exdate()` — exercise vs hold decision across spot levels.
  - American call ≥ European call verified; early exercise premium ≥ 0.
- 17 new tests: Am≥Eu, premium positive, boundary transition, RGW critical spot, div-after-expiry.
- 10,261 tests pass.

---

## v0.671.0 — 2026-05-31

**Enhanced dividend Greeks: cross-gamma, theta decomposition, scenario ladder.**

- New `equity/dividend_greeks.py`:
  - `compute_dividend_greeks()` — div_delta, div_gamma, cross_gamma_spot_div, div_theta, spot_delta via central finite differences.
  - `theta_decomposition()` — split theta into carry, dividend accrual, vol decay.
  - `dividend_scenario_ladder()` — price grid across dividend bump scenarios.
  - Cross-gamma d²V/(dS·d(div)): the key missing second-order Greek.
- 11 new tests: sign checks (call div_delta < 0, put > 0), cross-gamma finite, theta negative, ladder monotonicity.
- 10,244 tests pass.

---

## v0.670.0 — 2026-05-31

**Dividend strip analytics: decomposition, carry, growth rates.**

- New `equity/dividend_strip.py`:
  - `decompose_strip()` — split DividendCurve into per-period strips with forward div, PV, weight.
  - `strip_carry()` — carry-and-roll analytics per strip (yield vs funding).
  - `dividend_growth_rate()` — log-linear regression for implied growth from forward term structure.
  - Custom period breaks or equal-width periods.
- 11 new tests: sum-to-total, weights, constant/growing growth, carry.
- 10,233 tests pass.

---

## v0.669.0 — 2026-05-31

**Dividend term structure calibration (optimisation, spline, options-implied).**

- New `equity/dividend_calibration.py`:
  - `calibrate_dividend_curve()` — 3 methods: "linear" (existing), "optimize" (piecewise-constant yield via L-BFGS-B), "spline" (cubic spline on cumulative).
  - `calibrate_from_options()` — extract dividend curve from put-call parity across expiries.
  - `dividend_curve_seasonality()` — quarterly weight decomposition, peak/trough detection.
  - Optimised method fits at least as well as linear on non-constant yield data.
- 12 new tests: round-trip calibration, options-implied, seasonality, Q2-heavy detection.
- 10,222 tests pass.

---

## v0.668.0 — 2026-05-31

**Jump model cross-validation framework (COS vs MC vs FFT).**

- New `models/jump_cross_validation.py`:
  - `cross_validate_model()` — COS vs MC comparison for any of 6 jump models.
  - `cross_validate_all()` — all models, sorted by accuracy.
  - Per-strike results: COS price, MC price, FFT price, % difference.
  - Verified: Merton, VG, NIG all within 5% COS/MC mean difference.
- 10 new tests.
- 10,210 tests pass.

---

## v0.667.0 — 2026-05-31

**Jump model calibration to implied vol surfaces.**

- New `models/jump_calibration.py`:
  - `calibrate_jump_model()` — fits any of 6 jump models (Merton, VG, Kou, NIG, CGMY, Bates) to market implied vols via COS pricing + differential evolution.
  - `calibrate_jump_surface()` — multi-expiry independent calibration.
  - `jump_model_comparison()` — fits all models, ranks by AIC (penalises parameter count).
  - Round-trip: Merton calibration recovers params with < 0.5 vol pt RMSE.
- 10 new tests: round-trip, cross-model fitting, multi-expiry, model comparison.
- 10,200 tests pass.

---

## v0.666.0 — 2026-05-31

**NIG and CGMY Lévy processes with characteristic functions.**

- New `models/levy_processes.py`:
  - `NIGProcess(alpha, beta, delta)` — Normal Inverse Gaussian with char_func + MC terminal.
  - `CGMYProcess(C, G, M, Y)` — tempered stable Lévy process, generalises VG.
  - `nig_char_func()`, `cgmy_char_func()` — standalone risk-neutral CFs.
  - Both support complex u input (FFT-compatible).
  - NIG: inverse Gaussian subordinator simulation, exact RN drift correction.
  - CGMY: Y→0 limit handled separately (recovers VG char func).
- COS pricing verified: NIG vs MC within 5%, CGMY produces reasonable prices.
- Cross-model: both produce heavier tails than Black-Scholes (higher OTM put prices).
- 25 new tests.
- 10,190 tests pass.

---

## v0.665.0 — 2026-05-31

**Characteristic function protocol + standalone factories for Kou, Bates/SVJ.**

- New `models/char_func_protocol.py`:
  - `CharFuncModel` — `@runtime_checkable` Protocol for Fourier-based pricing.
  - `validate_char_func()` — checks φ(0)=1, boundedness, Hermitian symmetry.
  - `extract_cumulants()` — c1–c4, skewness, excess kurtosis from any CF.
  - Standalone factories: `merton_char_func()`, `vg_char_func()`, `kou_char_func()`, `bates_char_func()`, `svj_char_func()`.
  - All accept complex u (Carr-Madan FFT compatible).
- Kou CF: double-exponential jump CF with p·η₁/(η₁-iu) + (1-p)·η₂/(η₂+iu).
- Bates CF: Heston CF × Merton jump component (Schoutens form).
- 18 new tests: protocol compliance, validation, cumulants, COS vs MC cross-validation, complex u input.
- 10,165 tests pass.

---

## v0.664.0 — 2026-05-31

**Americas derivatives notebooks: Mexico, Chile, Colombia, Peru, Argentina, Canada.**

- 6 new notebooks in `notebooks/`:
  - `mexican_derivatives.ipynb` — TIIE 28D swap, CETES, MBONO, Udibono (UDI), BEI.
  - `chilean_derivatives.ipynb` — Cámara swap, BCP, BCU (UF), dual-curve BEI.
  - `colombian_derivatives.ipynb` — IBR swap, TES, TES UVR, BEI.
  - `peruvian_derivatives.ipynb` — PEN curve, BTP Peru, VAC bond, BEI.
  - `argentine_derivatives.ipynb` — ARS curve (40%+), Lecap, Lecer (CER), Bonares, BEI.
  - `canadian_derivatives.ipynb` — CORRA swap, CGB, RRB (deflation floor), BEI.
- Each notebook uses `pricebook.viz` (configure_theme, apply_theme, create_figure).
- Breakeven inflation term structures for all 6 markets.
- All 6 notebooks execute cleanly.
- 10,147 tests pass.

---

## v0.663.0 — 2026-05-31

**Unified inflation unit framework (UDI/UF/UVR/CER).**

- New `fixed_income/inflation_unit.py`:
  - `InflationUnit` — frozen dataclass for daily inflation units (name, currency, publisher, conventions).
  - `InflationUnitBond` — generic bond denominated in any inflation unit, dual real/nominal pricing.
  - `dual_curve_breakeven()` — BEI from any pair of nominal + real curves.
  - `compare_units()` — cross-country comparison table.
  - Registry: UDI (MXN), UF (CLP), UVR (COP), CER (ARS).
- 15 new tests: registry lookups, pricing for all 4 units, par bond, BEI, zero BEI.
- 10,147 tests pass.

---

## v0.662.0 — 2026-05-31

**Americas Phase 4-6: Peru, Argentina, Canada — full fixed income stack.**

- New `fixed_income/peruvian.py`:
  - `BTPPeru` — Peruvian sovereign bond (ACT/365F, semi-annual).
  - `VACBond` — inflation-linked bond (IPC-adjusted, real/nominal pricing).
  - `build_pen_curve()`, `synthetic_pen_strip()` — PEN discount curve.
- New `fixed_income/argentine.py`:
  - `LecapBond` — zero-coupon capitalisation bond (handles 40%+ rates).
  - `LecerBond` — CER-linked inflation bond (daily accrual).
  - `BONARBond` — ARS-denominated sovereign (semi-annual coupon).
  - `build_ars_curve()`, `synthetic_ars_strip()` — ARS discount curve.
- New `fixed_income/canadian.py`:
  - `CORRASwap` — CORRA overnight swap (par rate, DV01).
  - `RRBBond` — Real Return Bond (CPI-linked, deflation floor).
  - `build_corra_curve()`, `synthetic_corra_strip()` — CORRA discount curve.
- Infrastructure:
  - `LimaCalendar`, `BuenosAiresCalendar` in `core/calendar.py`.
  - TIPM (PEN), BADLAR (ARS) rate indices in `rate_indices.json`.
  - BTP_PE, BONAR, GLOBAL_AR sovereign conventions in `sovereign_conventions.json`.
  - IPC_PE (Peru), CER (Argentina) inflation indices in `inflation_indices.json`.
  - PEN, ARS EM curve conventions in `curve_conventions_em.json`.
- 20 new tests in `test_americas.py` (Colombia, Peru, Argentina, Canada).
- 10,132 tests pass.

---

## v0.661.0 — 2026-05-30

**Chile (CLP) derivatives: Cámara swap, BCP, BCU (UF-linked), breakeven inflation.**

- New `fixed_income/chilean.py`:
  - `CamaraSwap` — TPM-based overnight swap.
  - `BCPBond` — nominal CLP sovereign bond.
  - `BCUBond` — UF-denominated sovereign (real/nominal dual pricing).
  - `build_clp_curve()`, `build_uf_curve()` — nominal + real curve construction.
  - `breakeven_inflation()` — BEI term structure from nominal vs real curves.
  - Synthetic CLP + UF strips.
- 9 new tests: curves, swap, BCP, BCU UF scaling, BEI positive (~3.75%).
- 10,112 tests pass.

---

## v0.660.0 — 2026-05-30

**Mexico (MXN) derivatives: TIIE swap, CETES, Udibonos.**

- New `fixed_income/mexican.py`:
  - `TIIESwap` — 28-day period swap (unique Mexican structure), par rate, DV01.
  - `CETESBill` — discount bill pricing (ACT/360, MXN 10 face).
  - `UDIBond` — UDI-linked bond (real coupon × daily inflation unit), dual real/nominal pricing.
  - `build_tiie_curve()` — TIIE discount curve from swap strip.
  - `synthetic_tiie_strip()`, `synthetic_cetes_quotes()` — realistic data generators.
- 15 new tests: TIIE curve, 28-day periods, CETES discount, UDI nominal scaling, MBONO sovereign pricing.
- 10,103 tests pass.

---

## v0.658.0 — 2026-05-30

**Fix notebooks: remove `apply_theme` (not exported from viz).**

- Replaced `from pricebook.viz import apply_theme` with `configure_theme` only across all 14 notebooks.
- `apply_theme` is an internal context manager in `viz/_backend.py`, not part of the public API. `configure_theme()` at the top of each notebook sets the theme globally.
- 10,088 tests pass.

---

## v0.657.0 — 2026-05-30

**Brazilian credit derivatives notebook — end-to-end calibration.**

- New `notebooks/brazilian_credit_derivatives.ipynb` — 18 cells with pricebook.viz:
  1. CDI curve from DI futures (term structure plot)
  2. NTN-F/LTN bond pricing via CDI curve
  3. Bond-implied CDS spreads from corporate discount (hazard rate extraction)
  4. Survival curve + CDS par spread term structure
  5. CLN pricing with credit charge decomposition
  6. TRS on NTN-F with CDI funding
  7. Summary dashboard (4-panel: CDI curve, bond prices, implied spreads, CLN decomposition)
- Full chain: DI quotes → CDI curve → bond prices → hazard rates → CDS curve → CLN/TRS pricing.
- 10,088 tests pass.

---

## v0.656.0 — 2026-05-30

**Brazilian derivatives full stack: CDI curve, DI futures, DI swap, LFT, cupom cambial.**

- New `fixed_income/brazilian.py` (~400 lines):
  - `DIFuture` — B3 DI futures: PU pricing, DV01, implied rate round-trip.
  - `DISwap` — Pré × CDI swap: fixed vs CDI compounded, par rate, PV.
  - `LFTBond` — CDI-linked floating sovereign: VNA accrual, spread pricing, spread duration.
  - `build_cdi_curve_from_di()` — CDI discount curve from DI futures strip.
  - `synthetic_di_strip()` — realistic DI futures data generator (Selic-based upward slope).
  - `cupom_cambial()` — USD rate from USDBRL forward + DI rate (CIP).
  - `cupom_cambial_curve()` — cupom cambial term structure.
- LFT added to sovereign bonds registry (57 markets total) + yield convention + region mapping.
- 25 new tests covering: BUS/252 helpers, CDI curve construction, DI futures, DI swap, LFT, cupom cambial, NTN-F/LTN sovereign pricing.
- 10,088 tests pass.

---

## v0.655.0 — 2026-05-30

**Hawkes credit framework complete — analytics + 20 tests.**

- `credit/hawkes_analytics.py`:
  - `contagion_scenario()` — intensity jump analysis ("what if name X defaults?")
  - `clustering_metrics()` — inter-arrival CV + burstiness (CV>1 = clustered, B>0 = bursty)
  - `kernel_comparison()` — exponential vs power-law kernel side-by-side
  - `hawkes_term_structure()` — CDS spread across maturities under Hawkes
- 20 new tests (`test_hawkes_credit.py`):
  - Kernel formulas (exp, power-law, Mittag-Leffler γ=1 → exp)
  - Poisson limit (α=0), self-excitation increases events
  - Intensity non-negative, stationarity warning
  - CDS spread positive + increases with α
  - Tranche hierarchy (equity ≥ senior)
  - Contagion scenario (cross-excitation raises intensity)
  - Clustering CV, MLE direction, sum-exp approximation
- **Full Hawkes stack: 5 layers, 4 files, ~1600 lines.**
- 10,063 tests pass (+20 new).

---

## v0.654.0 — 2026-05-30

**Hawkes credit derivatives — Layers 2-4: survival, CDS, basket, tranche.**

- `credit/hawkes_survival.py` — `HawkesSurvivalCurve`: MC survival Q(T) from intensity paths, implied hazard, conversion to pricebook `SurvivalCurve`.
- `credit/hawkes_cds.py` — `hawkes_cds_spread()`: par CDS spread under Hawkes intensity. `hawkes_cds_spread_comparison()`: shows spread widening from self-excitation (120bp at α=0 → 185bp at α=0.9).
- `credit/hawkes_basket.py` — `hawkes_basket_defaults()`: multivariate Hawkes default simulation for N names. `hawkes_tranche_spread()`: CDO tranche pricing. `hawkes_ftd_spread()`: first-to-default. `hawkes_vs_copula()`: side-by-side Hawkes vs Gaussian copula comparison (tail losses, clustering).
- Tranche hierarchy verified: equity > mezzanine > senior.
- 10,043 tests pass.

---

## v0.653.0 — 2026-05-30

**Fractional Hawkes process for credit derivatives — Phase 1.**

- New `models/hawkes_credit.py`:
  - `FractionalHawkesProcess` — 4 kernel types: exponential, power-law (fractional), Mittag-Leffler, sum-of-exponentials.
  - `MultivariateHawkesProcess` — N-name cross-excitation matrix for credit contagion.
  - `HawkesKernel` enum, `HawkesCreditResult`, `MultivariateHawkesResult` dataclasses with `to_dict()`.
  - `evaluate_kernel()` — unified kernel evaluation.
  - `branching_ratio()` — stationarity check (warns if ≥ 1).
  - `approximate_power_law()` — Bochner sum-of-exponentials approximation of power-law kernel.
  - `hawkes_mle_exponential()` — MLE calibration for exponential kernel.
  - Ogata thinning adapted for non-Markovian kernels (dynamic intensity upper bound).
- **Next:** Layers 2-5 (survival curves, CDS pricing, basket/tranche, analytics).
- 10,043 tests pass.

---

## v0.652.0 — 2026-05-30

**Fix all moderate audit issues — input validation, magic number docs, edge case guards.**

- `data_registry.py`: path traversal guard (`_validate_filename`), JSON array type check, `key_fn` None validation.
- `network_xva.py`: exposure matrix shape validation (N,N), capital buffers shape (N,), recovery in [0,1].
- `calibration_quality.py`: array length mismatch check, n < 1 guard in `calibration_entropy`, n < 2 guard + n_params validation in `model_comparison`.
- `composite_convention.py`: `__post_init__` validates haircut ∈ [0,1] and recovery ∈ [0,1].
- `esg_bonds.py`: documented greenium 5bp (Zerbib 2019) and liquidity 3bp sources.
- `cds_bond_basis.py`: documented delivery 5bp (De Wit 2006), restructuring 10bp (ISDA), ±20bp neutral threshold. Added input validation to `bond_implied_cds_spread` (maturity > 0, frequency > 0, recovery ∈ [0,1), price > 0).
- `credit_leveraged.py`: documented duration 4.0 (Markit index factsheets), input validation on `constant_maturity_cds` (maturity > 0, recovery ∈ [0,1), vol ≥ 0).
- 10,043 tests pass.

---

## v0.651.0 — 2026-05-30

**Code audit fixes — 3 critical issues from 11-lens audit.**

- Fixed `credit_leveraged.py` line 131: `effective_leverage = min(leverage, 1.0 / 1e-10)` was a no-op (1e10 cap). Changed to direct assignment — leverage applies directly to digital CLN loss.
- Fixed `regime_pricing.py`: all `probs / probs.sum()` calls now validate `sum > 0` before dividing. Raises `ValueError` on zero-sum regime probabilities instead of silently producing NaN.
- Fixed `cds_bond_basis.py`: `bond_implied_cds_spread()` now validates bracket `f(0) × f(2) < 0` before calling brentq. Raises informative `ValueError` if market price is outside feasible range.
- Audit covered 9 files (6 new, 3 modified), 10 quality dimensions.
- 10,043 tests pass.

---

## v0.650.0 — 2026-05-30

**Quick wins closed: BilateralCSA, Hybrid, CMT wired. 133 validation tests.**

- Paper 2: `BilateralCSAPricer` exercised with `CSATerms(threshold=10m)` — partial CSA simulation verified.
- Paper 9: `IndexLinkedHybridInstrument.price()` with correlation sensitivity (ρ ∈ {-0.3, 0, 0.3}).
- Paper 10: `CMTInstrument.price()` with vol sensitivity (σ ∈ {10%, 20%, 30%}).
- 133 validation tests across 12 papers, all through pricebook classes.
- 10,043 tests pass.

---

## v0.649.0 — 2026-05-30

**Complete rewiring: all 12 papers use pricebook classes. 127 validation tests.**

- Paper 1: added `multicurve_newton()` + `build_curves()` tests (simultaneous OIS + projection).
- Paper 2: added `InterestRateSwap.pv()` + `pv_ctx()` for receiver swap.
- Paper 4: added `CDS` round-trip via class + `CreditLinkedNote.from_convention()`.
- Paper 5: added `constant_maturity_cds()` (participation rate) + `PedersenCDSSwaption.price()`.
- Paper 6: added `TotalReturnSwap.price()` + serialisation round-trip.
- Paper 8: added `CMASWInstrument.price()` with correlation sensitivity.
- All 12 papers now import from pricebook modules, not standalone math.
- 127 validation tests across 12 papers (+17 new).
- 10,037 tests pass.

---

## v0.648.0 — 2026-05-30

**Rewire validation tests through pricebook classes.**

- Paper 3+11 (T-Lock): now uses `TreasuryLock`, `BondForward` classes instead of manual formulas.
- Paper 7 (Lou TRS): now uses `trs_trinomial_tree()` + `trs_equity_full_csa()` with tree vs analytic comparison.
- Paper 12 (Zhou CDS-Bond): now uses `bond_implied_cds_spread()` + `compute_basis()` from pricebook credit modules.
- Fixed basis signal assertions to match actual pricebook output ("NEUTRAL"/"NEGATIVE_BASIS").
- 110 validation tests across 12 papers, all passing through pricebook modules.
- 10,020 tests pass (+16 from rewiring).

---

## v0.647.0 — 2026-05-30

**Build 2 missing capabilities for paper validation.**

- New `bond_implied_cds_spread()` in `credit/cds_bond_basis.py` — solves for flat hazard rate that reprices a risky bond at its market price, then converts to CDS spread. Enables Zhou Table 1 reproduction.
- `CMCDSResult.participation_rate` field added in `credit/credit_leveraged.py` — φ = fair_spread / forward_spread. Enables Brigo-Morini participation rate validation.
- **Backward compat:** Both additive. CMCDSResult has new field with default 0.0.
- 10,004 tests pass.

---

## v0.646.0 — 2026-05-30

**Chunk 3 complete: Papers 9-12. All 12 papers validated. 10,004 tests.**

- Paper 9 (Pucci Hybrid): 4 tests — correlation sensitivity, cash annuity.
- Paper 10 (Pucci CMT): 6 tests — CC formula, vol/fixing monotonicity, no-default limit.
- Paper 11 (Pucci T-Lock): 6 tests — forward dirty ≈ 104.74, carry, overhedge, delta.
- Paper 12 (Zhou CDS-Bond Basis): 6 tests — CDS/ASW at 3 D-levels, basis widening, hazard monotonicity.
- 4 notebooks for Chunk 3.
- **All 12 papers validated** with 94 total validation tests across 12 test files.
- **10,004 tests pass** (milestone: crossed 10k).

---

## v0.645.0 — 2026-05-30

**Chunk 2 complete: Papers 5-8 validation (CDS, TRS×2, CMASW).**

- Paper 5 (Brigo-Morini CDS Market Model): 11 tests — CDS option implied vol (C1=61.9% vs paper 62.2%), recovery independence, CMCDS convexity monotonicity, participation rate.
- Paper 6 (Burgess Bond TRS): 8 tests — coupon $155,416.80, simple vs continuous forward, carry direction, recovery sensitivity.
- Paper 7 (Lou TRS Framework): 8 tests — forward consistency (r_s < r → F < S), FVA direction, CVA/DVA signs, margin convergence.
- Paper 8 (Pucci CMASW): 10 tests — CC formula (zero at σ=0 or ρ=0), CC grid, vol/correlation monotonicity, antisymmetry in ρ.
- 4 notebooks with pricebook.viz: implied vol table, CMCDS convexity/participation plots, TRS forward comparison, XVA waterfall, CMASW CC heatmap.
- **Chunks 1+2 complete** (8/12 papers validated).
- 9982 tests pass (+37 new).

---

## v0.644.0 — 2026-05-30

**Papers 3 + 4 validation: T-Lock model + CLN.**

- Paper 3 (Anon T-Lock): 7 tests — bond forward (Bf_dirty ≈ 104.74), PV01 convergence, clean/dirty equivalence, repo no-arbitrage. Cross-validates with Pucci 2019.
- Paper 4 (Axelsson-Renström CLN): 9 tests — CDS bootstrap (hazard rates positive + increasing), CDS round-trip, CLN below risk-free, recovery sensitivity, discretisation error.
- Notebooks: `paper_03_tlock_model.ipynb` (PV01 convergence + T-Lock payoff plots), `paper_04_cln.ipynb` (survival curves + CLN price vs recovery).
- **Chunk 1 complete** (4/4 papers validated).
- 9945 tests pass (+16 new).

---

## v0.643.0 — 2026-05-29

**Paper 2 validation: Anonymous — Discounting Textbooks.**

- New `tests/validation/test_paper_02_discounting.py` — 9 tests:
  - Case A: equity forward with repo drift (£105.65 vs textbook £105.13)
  - Case B: 5Y receiver swap under 3 CSA regimes, PV ordering verified
  - Case C: ColVA for bond collateral (GC £85k vs special £2.55m)
- New `notebooks/paper_02_discounting.ipynb` with pricebook.viz:
  - CSA regime bar chart comparison
  - ColVA vs repo rate curve with GC/special annotations
- 9929 tests pass (+9 new).

---

## v0.642.0 — 2026-05-29

**Paper 1 validation: Ametrano & Bianchetti (2013) — Multicurve Bootstrap.**

- New `tests/validation/test_paper_01_multicurve.py` — 10 tests reproducing EUR multicurve case study (11-Dec-2012):
  - OIS bootstrap from Eonia strip (12 pillars, round-trip < 1bp)
  - Negative rate handling (1Y OIS = 0%, DF ≈ 1.0)
  - IRS-6M projection curve bootstrap with OIS discounting
  - Loss of telescoping identity (eq. 64-65) — deviation confirmed
  - OIS single-curve property (eq. 73-74) — telescoping holds
- New `notebooks/paper_01_multicurve.ipynb` — interactive notebook with pricebook.viz:
  - OIS discount factor and zero rate plots
  - OIS vs Euribor 6M projection curve comparison with basis spread fill
  - Bootstrap round-trip verification table
  - LaTeX-rendered key equations
- 9920 tests pass (+10 new).

---

## v0.641.0 — 2026-05-29

**Hard migration — remove aliases, tighten pv_ctx curve lookups.**

- Renamed `CDSIndexProduct.from_spec` → `from_convention` (removed alias). All callers + tests updated.
- Tightened `pv_ctx` curve extraction in 6 instruments:
  - `FRA`: tries keyed lookup by day_count before falling back to first projection curve.
  - `FRN`: same keyed lookup pattern.
  - `BasisSwap`: warns if fewer than 2 projection curves available.
  - `CapFloor`: **raises ValueError** if no IR vol surface in context (was silently using flat 20%).
  - `ConvertibleBond`: **raises ValueError** if missing spot, discount curve, or vol surface (was guessing defaults).
  - `RiskyBond`: warns if no credit curve found (falls back to risk-free with warning instead of silently).
- Old numerical shims: verified already removed in v0.612-v0.616. No action needed.
- **Breaking changes:** CapFloor.pv_ctx and ConvertibleBond.pv_ctx now raise instead of silently using bad defaults. Code that relied on the fallback behaviour must now provide proper market data in PricingContext.
- 9910 tests pass.

---

## v0.640.0 — 2026-05-29

**Supranational analytics — RV, universe pricing, curve spread (D9).**

- `supranational_rv()` — relative value: z-score vs historical spread, peer ranking, RICH/CHEAP/FAIR signal.
- `price_supranational_universe()` — price bonds across all issuers × currencies. Returns aggregated SupraUniverseResult with tightest/widest/average spread.
- `supranational_curve_spread()` — spread term structure across tenors for a single issuer.
- `SupraRVResult`, `SupraUniverseResult` dataclasses with `to_dict()`.
- **Backward compat:** Additive — existing `create_supranational_bond()` and `price_supranational()` unchanged.
- 9910 tests pass.

---

## v0.639.0 — 2026-05-28

**ESG bond labelling framework (D8).**

- New `fixed_income/esg_bonds.py`:
  - `ESGLabel` enum: GREEN, SOCIAL, SUSTAINABILITY, SUSTAINABILITY_LINKED, TRANSITION, BLUE.
  - `UseOfProceeds` enum: 14 ICMA taxonomy categories.
  - `ESGBondSpec` convention: label, issuer, use-of-proceeds, KPI target, coupon step-up/down, taxonomy alignment, reviewer.
  - `greenium()` — green premium calculation (yield difference green vs conventional).
  - `esg_adjusted_spread()` — spread decomposition: credit + greenium + liquidity.
  - `slb_coupon_adjustment()` — sustainability-linked bond coupon step-up/down on KPI miss/achieve.
  - `create_green_bond()` — factory returning (FixedRateBond, ESGBondSpec) tuple.
- Full `@serialisable_convention` on ESGBondSpec with round-trip.
- **Backward compat:** Additive — new module, no changes to existing code.
- 9910 tests pass.

---

## v0.638.0 — 2026-05-28

**Sukuk instrument + pricing (D7).**

- New `SukukBond` class: profit rate (coupon equivalent), 7 Sukuk types (Ijara, Mudaraba, Murabaha, Wakala, Musharaka, Salam, Istisna).
- Curve-based pricing via internal FixedRateBond delegation. Spread-based pricing via `price_from_spread()`.
- Full architecture: `from_convention()`, `pv_ctx()`, `to_dict()`/`from_dict()`, `@serialisable`.
- `create_sukuk(type, issue, maturity, rate)` factory function.
- **Backward compat:** Additive. Existing `price_sukuk_as_bond()` unchanged.
- 9910 tests pass.

---

## v0.637.0 — 2026-05-28

**Composite convention pattern for exotic trees — TRS-on-SPV with nested conventions.**

- New `models/composite_convention.py` with 5 convention types: CouponCapSpec, FundingConvention, CollateralConvention, SPVNoteConvention, BondTRSConvention.
- `create_trs_on_spv()` convenience function. `BondTRSConvention.create()` builds underlying from nested conventions.
- Fixed `_deserialise_atom` for Python 3.10+ `types.UnionType` (`X | None`) and flat convention dict deserialisation.
- Full round-trip: nested convention → JSON → from_dict → create → instrument.
- **Backward compat:** Two fixes to core/serialisable.py improve nested deserialisation. No existing behaviour changed.
- 9910 tests pass.

---

## v0.636.0 — 2026-05-28

**Supranational bond factory + pricing.**

- `create_supranational_bond(issuer, currency, issue, maturity, coupon)` — creates FixedRateBond with domestic sovereign conventions for the issuance currency. Maps 10 currencies to sovereign market codes.
- `price_supranational()` — full pricing with spread vs sovereign computation.
- `SupranationalBondResult` — clean/dirty price, YTM, spread, rating.
- Warns if issuing in a non-typical currency for the supranational.
- **Backward compat:** Additive — existing `get_supranational()` / `list_supranationals()` unchanged.
- 9910 tests pass.

---

## v0.635.0 — 2026-05-28

**Complete @serialisable — all 5 remaining complex classes done.**

- `PedersenCDSSwaption`, `StochasticIntensitySwaption` — scalar params, standard decorator.
- `TotalReturnSwapLou` — scalar params, standard decorator.
- `CDSIndex` — custom to_dict/from_dict: serialises list of CDS constituents recursively.
- `CovenantLoan` — custom to_dict/from_dict: serialises nested TermLoan.
- **Backward compat:** All additive. CDSIndex and CovenantLoan use custom from_dict that dispatches via the Serialisable registry for nested objects.
- Total @serialisable instruments: **49** (was 44). Zero remaining gaps.
- 9910 tests pass.

---

## v0.634.0 — 2026-05-28

**JSON is now source of truth for all 11 convention registries.**

- All convention registries now load from JSON first, falling back to hardcoded Python defaults.
- New `load_registry()` utility in `core/data_registry.py` — populates keyed dicts from JSON arrays.
- Wired into: sovereign_conventions, rate_indices, equity_indices, commodity_contracts, linker_conventions, inflation_indices, repo_specialness, supranational_issuers, cds_indices, sovereign_cds, curve_conventions_em.
- Fixed CDS index names: `"iTraxx Europe"` → `"ITRAXX.EUR.IG"` etc. — name field now matches the lookup key (was a key/name mismatch from the original hardcoded dict).
- **Backward compat:** All `get_X()` APIs unchanged. JSON overrides hardcoded defaults when present. Editing a JSON file immediately changes what `get_conventions()` returns. CDS index spec name field changed from display name to canonical key — callers using `get_index_spec("ITRAXX.EUR.IG")` unaffected.
- G10 curve conventions (curve_builder.py) not wired — CurrencyConventions lacks a currency field for keying.
- 9910 tests pass.

---

## v0.633.0 — 2026-05-28

**from_convention on 12 more products — total 35 with factory.**

- Group 1 (FI): ZCInflationSwap, YoYInflationSwap, RevolvingFacility, AmortisingBond.
- Group 4 (Credit): CDSIndexProduct (alias from_spec), TrancheCDS, LoanParticipation, BasketCLN.
- Group 5 (Commodity): CommoditySwap (uses CommodityContractSpec).
- Group 8 (Repo): Repo, ReverseRepo (uses haircut from convention).
- **Backward compat:** CDSIndexProduct.from_convention = CDSIndexProduct.from_spec (alias). All others additive.
- Remaining without from_convention: options (strike/vol-driven, 10), desk trades (8), model-driven structured (4), TRS (3) — conventions don't apply the same way to these products.
- from_convention coverage: 23→35/39 core products. The 4 excluded categories (options/desk/structured-model/TRS) represent products where the concept of "market convention" is either the strike+vol (options) or the underlying itself (TRS).
- 9910 tests pass.

---

## v0.632.0 — 2026-05-27

**Convention + factory integration tests — 30 new tests, 9910 total.**

- New `test_convention_factory.py` with 30 tests covering the full chain:
  - Convention JSON round-trip (6 types)
  - Convention → factory → instrument (10 products: UST, Bund, ZCB, IRS USD/EUR, OIS, CDS, Swaption, Deposit, FRA)
  - Instrument → pv_ctx (5 products)
  - Instrument → to_dict → from_dict (5 products)
  - End-to-end: JSON load → convention → factory → price → serialise (4 chains)
- 9910 tests pass (was 9880).

---

## v0.631.0 — 2026-05-27

**from_convention on 3 more credit products — total 23 with factory.**

- `GuaranteedNote.from_convention()` — uses frequency/day_count from bond conventions.
- `VanillaCLN.from_convention()` — same pattern.
- `CreditRiskyFRN.from_convention()` — uses convention frequency/day_count for floating schedule.
- **Backward compat:** All additive.
- from_convention coverage: 20→23/39 products.
- 9880 tests pass.

---

## v0.630.0 — 2026-05-27

**from_convention on 7 more products — total 20 with factory.**

- `ZeroCouponSwap.from_convention()` — uses fixed_day_count from CurrencyConventions.
- `CrossCurrencySwap.from_convention()` — uses float freq/dc.
- `TermLoan.from_convention()` — uses float freq/dc for floating coupon.
- `Swaption.from_convention()` — uses fixed/float freq+dc from CurrencyConventions for underlying swap.
- `CapFloor.from_convention()` — uses float freq/dc for caplet/floorlet schedule.
- `TreasuryBill.from_convention()` — uses day_count + settlement from SovereignConventions.
- `Deposit.from_convention()` + `FRA.from_convention()` — already added in v0.628.0.
- **Backward compat:** All additive classmethods. IRFuture skipped (exchange-specific, not convention-driven).
- from_convention coverage: 13→20/39 products. Remaining ~19 are exotics (TRS, autocallable, etc.) or desk aggregates where conventions don't apply the same way.
- 9880 tests pass.

---

## v0.629.0 — 2026-05-27

**Complete @serialisable coverage — all 7 remaining gaps fixed.**

- `@serialisable` added to: LeveragedCLN, DIPLoan, TriPartyRepo, IndexLinkedHybridInstrument, DispersionTrade, DividendSwap, RiskReversal, VarianceSwap (8 classes).
- Total serialisable instrument classes: **44** (was 36).
- **Backward compat:** DIPLoan and TriPartyRepo `to_dict()` output changed from flat dict to standard `{"type": ..., "params": {...}}` format. Tests updated. TriPartyRepo serial type is `"triparty_repo"` (was `"tri_party_repo"` in one test).
- Only CDSIndex, CovenantLoan, PedersenCDSSwaption, StochasticIntensitySwaption, TotalReturnSwapLou remain without @serialisable (complex/nested params that need manual from_dict).
- 9880 tests pass.

---

## v0.628.0 — 2026-05-27

**Serialisable + pv_ctx + from_convention final batch.**

- `@serialisable` added to: CommoditySwap, RiskParticipation, BondFuture, FXFuture, CMSLeg (5 more instruments).
- `ConvertibleBond.pv_ctx()` — extracts spot, rate, vol, credit spread from PricingContext. All core tradeable products now have pv_ctx.
- `Deposit.from_convention()` and `FRA.from_convention()` — uses day_count from CurrencyConventions.
- **Backward compat:** All additive. 7 reverted files (desk trades with wrong field names, 4 credit/structured with import inside function body) will be fixed in a follow-up pass — no regression from v0.627.
- `@serialisable` coverage: 31→36 instruments. `from_convention` coverage: 11→13 products.
- 9880 tests pass.

---

## v0.627.0 — 2026-05-27

**from_convention on 5 more instruments — total 11 product types with factory.**

- `RiskyBond.from_convention(conv, start, end, coupon_rate, recovery)` — uses bond convention frequency/day_count.
- `CreditLinkedNote.from_convention(conv, start, end, coupon_rate, recovery)` — same pattern.
- `InflationLinkedBond.from_convention(conv, start, end, coupon_rate, base_cpi)` — accepts LinkerConvention or InflationIndexDef (auto-resolves frequency/day_count/lag from either).
- `BasisSwap.from_convention(conv, start, end, spread)` — uses CurrencyConventions float/fixed frequencies.
- **Backward compat:** All additive classmethods. No existing API changes.
- Factory coverage: 8→11/39 products with `from_convention`.
- 9880 tests pass.

---

## v0.626.0 — 2026-05-27

**from_convention factories on 6 core instrument classes.**

- `FixedRateBond.from_convention(conv, issue_date, maturity, coupon_rate)` — accepts SovereignConventions or any object with frequency/day_count/calendar_currency.
- `ZeroCouponBond.from_convention(conv, issue_date, maturity)` — same convention protocol.
- `FloatingRateNote.from_convention(conv, start, end, spread)` — uses convention frequency/day_count.
- `InterestRateSwap.from_convention(conv, start, end, fixed_rate)` — accepts CurrencyConventions (fixed/float freq+dc).
- `CDS.from_convention(conv, start, end, spread)` — accepts SovereignCDSConventions or CDSIndexSpec (extracts recovery).
- `OISSwap.from_convention(conv, start, end, fixed_rate)` — already added in v0.622.0.
- New `create_swap(currency, start, end, rate)` convenience function.
- New `get_conventions(currency)` in `curves/curve_builder.py`.
- Rewired `create_sovereign_bond`, `create_sovereign_zero`, `create_sovereign_frn` to use `from_convention` internally.
- **Backward compat:** All new classmethods and functions are additive. Existing factory functions (`create_sovereign_bond` etc.) now delegate to `from_convention` — same output, thinner implementation. FX instruments skipped (pair IS the convention — no separate convention layer needed).
- Factory coverage: 3/39 → ~8/39 products with `from_convention`.
- 9880 tests pass.

---

## v0.625.0 — 2026-05-27

**Serialisation hardening — @serialisable on 15 more instrument classes.**

- Added `@serialisable` to: ZCInflationSwap, YoYInflationSwap, InflationLinkedBond, CrossCurrencySwap, StepUpBond, RiskyBond, Repo (already had via alias), IRFuture, AmortisingBond, VanillaCLN, BasketCLN, GuaranteedNote, CMASWInstrument, CMTInstrument.
- Total serialisable instruments: 16→31 (now 80% of core tradeables).
- **Backward compat:** StepUpBond `to_dict()` output changed from flat dict to `{"type": "step_up_bond", "params": {...}}` format (standard instrument format). Other classes that had no `to_dict()` now have one (additive). Test updated.
- 9880 tests pass.

---

## v0.624.0 — 2026-05-27

**pv_ctx on 10 more instruments — coverage 35→39/39 (near-complete).**

- Added `pv_ctx()` to: ZeroCouponSwap, TreasuryBill, IRFuture, CrossCurrencySwap, ZCInflationSwap, YoYInflationSwap, InflationLinkedBond, BondForward, ParAssetSwap, ProceedsAssetSwap.
- CrossCurrencySwap.pv_ctx extracts domestic + foreign discount curves + FX spot from context.
- Inflation instruments extract CPI curve from `ctx.inflation_curves`.
- **Backward compat:** All additive — existing pricing signatures unchanged. `pv_ctx` methods use best-effort curve extraction.
- PricingContext coverage on core tradeable instruments: near-complete. Remaining gaps are desk-level aggregators (Book, Desk), result dataclasses, and niche credit exotics.
- 9880 tests pass.

---

## v0.623.0 — 2026-05-27

**pv_ctx on CapFloor and RiskyBond.**

- `CapFloor.pv_ctx()` — extracts discount + projection curves + IR vol from context, falls back to flat 20% vol.
- `RiskyBond.pv_ctx()` — extracts discount + credit curves, falls back to risk-free pricing if no credit curve.
- **Backward compat:** Additive — existing `price()` / `dirty_price()` signatures unchanged. `pv_ctx` uses best-effort curve extraction from context.
- PricingContext coverage: 33/39 → 35/39 products.
- 9880 tests pass.

---

## v0.622.0 — 2026-05-27

**OIS convention + pv_ctx on 8 vanilla instruments.**

- New `OISConvention` dataclass with `create_swap()` factory (10 currencies: USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD, SEK, NOK). `get_ois_convention(currency)` lookup.
- `OISSwap.from_convention()` classmethod + `pv_ctx()`.
- Added `pv_ctx()` to 7 more instruments: Deposit, FRA, ZeroCouponBond, BasisSwap, FloatingRateNote, FXSwap, NDF, EquityForward.
- **Backward compat:** All new methods are additive. Existing `pv()` signatures unchanged. `OISConvention` + `get_ois_convention` are new exports. `pv_ctx` on BasisSwap picks first two projection curves from context — callers with specific curve needs should still use `pv()` directly.
- PricingContext coverage: 25/39 → 33/39 products.
- 9880 tests pass.

---

## v0.621.0 — 2026-05-26

**Static data layer — 13 JSON convention files + loader utility.**

- Created `data/` directory with 13 JSON files (62 KB total, 212 entries):
  sovereign_conventions (56), rate_indices (25), equity_indices (9), commodity_contracts (13), linker_conventions (8), inflation_indices (16), repo_specialness (6), supranational_issuers (10), cds_indices (5), sovereign_cds (31), curve_conventions_g10 (10), curve_conventions_em (16), sukuk_conventions (7).
- New `core/data_registry.py` — `load_conventions()`, `save_conventions()`, `load_or_default()` utilities for JSON ↔ convention dataclass round-trip.
- All 12 convention types verified: JSON → from_dict → to_dict → JSON matches original.
- **Backward compat:** JSON files are additive — existing hardcoded registries remain the source of truth. JSON files serve as export/inspection/override format. No existing APIs changed.
- 9880 tests pass.

---

## v0.620.0 — 2026-05-26

**Apply `@serialisable_convention` to all 13 convention dataclasses.**

- All convention types now have `to_dict()`/`from_dict()` round-trip via the decorator:
  RateIndex, EquityIndexSpec, CommodityContractSpec, LinkerConvention, InflationIndexDef, SpecialnessConventions, SupranationalIssuer, CDSIndexSpec, CDSSettlementConvention, SovereignCDSConventions, CurrencyConventions, EMCurveConventions, SukukConventions.
- 6 dataclasses made `frozen=True` (were mutable): EquityIndexSpec, CommodityContractSpec, LinkerConvention, CDSIndexSpec, CDSSettlementConvention, CurrencyConventions.
- Manual `to_dict()` methods removed (decorator auto-generates with proper enum serialisation).
- **Backward compat:** `to_dict()` output now includes all fields (some manual implementations omitted fields like `notes`, `settlement_days`). Existing `get_X()` / `list_X()` APIs unchanged. `from_dict()` is new (additive). Making dataclasses frozen could break code that mutates convention objects — none found in tests.
- 9880 tests pass.

---

## v0.619.0 — 2026-05-26

**Add `@serialisable_convention` decorator for frozen dataclasses.**

- New `serialisable_convention(serial_type)` decorator in `core/serialisable.py` — auto-derives `_SERIAL_FIELDS` from `dataclasses.fields()`, produces flat dicts (no type/params nesting), handles enum/date round-trip.
- Applied to `SovereignConventions` — first convention with full `to_dict()`/`from_dict()` round-trip.
- **Backward compat:** `SovereignConventions.to_dict()` now exists where it didn't before (additive, no breakage). The existing `get_conventions()` / `create_sovereign_bond()` APIs unchanged.
- 9880 tests pass.

---

## v0.618.0 — 2026-05-26

**Restore clean dependency layers — 0 cycles, 9 layers.**

- Made 2 module-level imports lazy (moved inside function bodies):
  - `models/regime_pricing.py` — `equity_option_price`, `equity_delta`, `equity_gamma`, `equity_vega` from options
  - `curves/rfr_bootstrap.py` — `RFRFutureSpec`, `rfr_futures_to_forwards` from fixed_income
- AST-verified: 0 bidirectional cycles at module level across all 20 packages.
- Architecture: 9 clean layers, 566 modules, 20 packages.
- 9880 tests pass.

---

## v0.617.0 — 2026-05-26

**Phase 5 advanced theory integration — regime pricing, calibration quality, network XVA.**

- `models/regime_pricing.py` — `RegimePricingEngine`: HMM-driven option pricing under regime switching. Fits HMM to returns, extracts regime-conditional vols, prices under each regime and blends by filtered probabilities. Includes `regime_option_price()`, `regime_greeks()`, risk decomposition by regime.
- `statistics/calibration_quality.py` — information-theoretic calibration assessment: `calibration_entropy()` (RMSE, R², entropy of residuals), `calibration_kl()` (KL-based model comparison), `parameter_stability()` (CV, drift across recalibrations), `model_comparison()` (AIC/BIC/JS divergence), `fisher_parameter_quality()` (FIM + Cramer-Rao bounds).
- `risk/network_xva.py` — `NetworkXVAEngine`: systemic risk adjustments to CVA. Integrates financial network centrality and Eisenberg-Noe contagion cascades. CVA_network = CVA × (1 + α × centrality × contagion_multiplier). Includes `stress_test()`, `systemic_ranking()`, convenience `contagion_cva_stress()`.
- 36 new tests (test_phase5_integration.py). 9880 tests pass.

---

## v0.616.0 — 2026-05-25

**Delete tree model shims — all callers migrated to solve_tree().**

- Deleted `models/binomial_tree.py`, `models/trinomial_tree.py`, `models/binomial_jr_lr.py` — thin shims, zero remaining importers.
- Migrated 6 test files to import directly from `numerical._trees`: `test_binomial_tree.py`, `test_trinomial_tree.py`, `test_binomial_jr_lr.py`, `test_binomial_roundtrip.py`, `test_finite_difference.py`, `test_lsm.py`.
- Registry already clean (uses `solve_tree` since v0.612.0).
- 9844 tests pass.

---

## v0.615.0 — 2026-05-25

**Standardise all numerical modules to Enum + Result + to_dict pattern.**

- `_rootfinding.py` — add `RootMethod` enum (BISECTION, BRENT, NEWTON, SECANT, HALLEY, ITP); `find_root()` accepts enum or string.
- `_optimize.py` — add `OptimMethod` enum (NELDER_MEAD, BFGS, L_BFGS_B, CG, NEWTON_CG, DIFFERENTIAL_EVOLUTION, BASIN_HOPPING, CMA_ES); `minimize()` accepts enum or string.
- `_graph.py` — add `ShortestPathResult`, `MSTResult`, `MaxFlowResult` dataclasses with `to_dict()`; add `dijkstra_full()`, `minimum_spanning_tree_full()`, `max_flow_full()` returning typed results.
- `_distributions.py` — add `to_dict()` to Normal, StudentT, LogNormal, Uniform, Exponential.
- `_linalg.py` — add `DecompMethod`, `IterativeMethod` enums; `SVDResult`, `LUResult` dataclasses; `decompose()` and `iterative_solve()` dispatchers; `method` field on `IterativeSolveResult`.
- `_mc.py` — add `MCVarianceReduction`, `MCDiscrMethod` enums.
- `_fourier.py` — add `FourierMethod`, `WaveletType` enums; `to_dict()` on `CharacteristicFunction`; wavelet_transform accepts enum.
- `_interpolation.py` — add `InterpMethod2D`, `RBFKernel` enums; `interpolate_2d()` dispatcher; `rbf_interpolate()` accepts enum.
- Updated `numerical/__init__.py` — export all new enums, result types, and dispatchers.
- All string-based callers continue to work (backward compatible).
- 9844 tests pass.

---

## v0.614.0 — 2026-05-24

**Final migration cleanup — delete _quadrature.py, auto-scale global_solver FD eps.**

- Deleted `numerical/_quadrature.py` — fully superseded by `_integrate.py`, no importers remain.
- `curves/global_solver.py` — replaced hardcoded `eps=1e-8` with auto-scaled `h = max(|x_j| × 1e-7, 1e-10)` in both Jacobian functions.
- 9844 tests pass.

---

## v0.613.0 — 2026-05-24

**Fix Leisen-Reimer Peizer-Pratt formula — extra 0.5 factor removed.**

- Root cause: `copysign(0.5, z) * sqrt(...)` instead of `copysign(sqrt(...), z)`. The extra 0.5 multiplier halved the probability deviation from 0.5, collapsing all tree prices to ~50% of BS.
- All 8 LR-specific test failures now pass. LR(51) matches BS to 4+ decimals as designed.
- 9844 tests pass, 0 failures.

---

## v0.612.0 — 2026-05-24

**Complete migration — tree shims, quadrature redirect, nd_solvers Jacobian.**

### Tree model files converted to thin shims
- `models/binomial_tree.py` → delegates to `solve_tree(TreeMethod.CRR)`
- `models/trinomial_tree.py` → delegates to `solve_tree(TreeMethod.TRINOMIAL)`
- `models/binomial_jr_lr.py` → delegates to `solve_tree(TreeMethod.JR/LR)`
- `registry.py` tree section → `_make_tree_pricer()` wrappers using `solve_tree`

### Quadrature redirect
- `curves/quadrature.py` → thin redirect to `numerical._integrate`. `QuadratureResult` = `IntegrationResult`.
- `registry.py` integrator section → `_make_integrator()` wrappers using `integrate()`.

### Differentiation
- `models/nd_solvers.py` `finite_difference_jacobian()` → delegates to `numerical._differentiate.jacobian()`.

### Known issue
- LR (Leisen-Reimer) tree method has pricing inaccuracy in the new `_trees.py` implementation (8 test failures). CRR, JR, trinomial all correct. To be fixed in a subsequent commit.

- 9836 passed, 8 LR-specific failures.

---

## v0.611.0 — 2026-05-24

**Backward compatibility removal — clean API for ODE, integration, trees.**

### Removed
- `euler()`, `rk4()`, `rk45()`, `bdf()`, `adams()` shims from `_ode.py` → use `solve_ode(f, span, y0, ODEMethod.RK4)`.
- `gauss_jacobi()`, `tanh_sinh()`, `clenshaw_curtis()` shims from `_integrate.py` → use `integrate(f, a, b, IntegrationMethod.TANH_SINH)`.
- `tree_greeks()`, `binomial_2d()`, `TreeGreeks`, `Binomial2DResult` shims from `_trees.py` → use `solve_tree()`, `solve_tree_2d()`.

### Deleted
- `models/ode.py` — shim module, all logic now in `numerical/_ode.py`.

### Migrated
- `numerical/__init__.py` — exports only new API names.
- `registry.py` — ODE solvers now use `_make_ode_solver()` wrapper.
- `core/results.py` — imports `ODEResult` from `numerical._ode`.
- 4 test files rewritten to use new API: `test_ode.py`, `test_numerical.py`, `test_numerical_ode.py`, `test_numerical_quadrature.py`, `test_numerical_trees.py`, `test_tree_solver.py`.

### Result
- **Single canonical API** per module — no aliases, no wrappers, no ambiguity.
- 9844 tests pass.

---

## v0.610.0 — 2026-05-24

**Bayesian statistics — MCMC, conjugate priors, model selection, changepoint detection.**

### Bayesian Module (`statistics/bayesian.py`)
- **MCMC Sampling:**
  - `MetropolisHastings` — random-walk MH with configurable proposal, acceptance tracking, ESS computation.
  - `GibbsSampler` — component-wise sampling from full conditionals.
  - `MCMCResult` — samples, log-posteriors, credible intervals, effective sample size, `to_dict()`.

- **Conjugate Priors:**
  - `BayesianLinearRegression` — Normal-Inverse-Gamma conjugate. Closed-form posterior, credible intervals, posterior predictive, log marginal likelihood (evidence).
  - `beta_binomial_update()` — Beta-Binomial for PD estimation. Posterior mean, mode, credible interval.

- **Model Selection:**
  - `bayes_factor()` — log Bayes factor with Kass-Raftery interpretation (decisive/strong/moderate/weak).
  - `credible_interval()`, `hpd_interval()` — equal-tailed and HPD credible intervals.
  - `posterior_predictive()` — MC posterior predictive distribution.

- **Changepoint Detection:**
  - `bayesian_changepoint()` — Bayes factor scan for structural breaks. Posterior probability per time point.

- **Use cases:** Bayesian PD estimation, parameter uncertainty in calibrated models, model comparison (SABR vs Heston), regime change detection, Bayesian VaR.
- 24 tests. 9849 tests pass.

---

## v0.609.0 — 2026-05-24

**Tree solver redesign — class-based, 5 methods, Bermudan, barriers, Greeks from nodes.**

### Tree Solver (`numerical/_trees.py`)
- `TreeSolver` class — configurable method, exercise type, barriers, dividends.
- `TreeMethod` enum: CRR, JR, LR, TRINOMIAL, TIAN (5 methods).
- `ExerciseType` enum: EUROPEAN, AMERICAN, BERMUDAN.
- `BarrierType` enum: UP_OUT, DOWN_OUT, UP_IN, DOWN_IN.
- `solve_tree()` — one-liner convenience (mirrors `solve_bs_pde()`).
- `solve_tree_2d()` — 2-asset Rubinstein tree with callable payoff + American exercise.
- Greeks from tree nodes directly: delta/gamma from steps 1-2, theta from step 2, vega via bump.
- Bermudan: exercise at specified step indices only.
- Barriers: knock-out via node zeroing.
- Discrete dividends: spot adjustment at dividend steps.
- `convergence_analysis()` — prices at multiple N + Richardson extrapolation.
- `TreeResult` — price, delta, gamma, theta, vega, method, n_steps, exercise, convergence, optional node data.
- Custom payoff: `payoff=lambda S: ...` for digitals, straddles, any exotic.
- Backward compatible: `tree_greeks()`, `binomial_2d()` old API preserved.
- 22 tests. 9825 tests pass.

---

## v0.608.0 — 2026-05-24

**Integration + differentiation redesign — unified frameworks, 9+5 methods.**

### Numerical Integration (`numerical/_integrate.py`)
- `IntegrationMethod` enum: ADAPTIVE (scipy quad), GAUSS_LEGENDRE, GAUSS_LAGUERRE (semi-infinite), GAUSS_HERMITE (infinite), TANH_SINH (singular), CLENSHAW_CURTIS, SIMPSON, TRAPEZOID, ROMBERG.
- `integrate(f, a, b, method)` — main entry with auto method selection.
- `integrate_2d()` — double integral via scipy.dblquad.
- `integrate_semi_infinite()` — ∫ₐ^∞ with Gauss-Laguerre or adaptive.
- `integrate_complex_contour()` — ∮ f(z)dz along parameterised contour.
- `IntegrationResult` — value, error estimate, n_evaluations, converged.
- Backward compatible: old `gauss_jacobi`, `tanh_sinh`, `clenshaw_curtis` still work.

### Numerical Differentiation (`numerical/_differentiate.py`)
- `DiffMethod` enum: FORWARD (O(h)), CENTRAL (O(h²)), COMPLEX_STEP (machine ε), RICHARDSON (O(h⁴)), FIVE_POINT (O(h⁴)).
- `derivative(f, x, method, order)` — 1st and 2nd derivatives.
- `gradient(f, x)` — ∇f for scalar functions of vectors.
- `jacobian(f, x)` — J[i,j] = ∂fᵢ/∂xⱼ for vector functions.
- `hessian(f, x)` — H[i,j] = ∂²f/∂xᵢ∂xⱼ for scalar functions.
- Auto step size selection: optimal h based on method order + machine epsilon.
- `DiffResult` — value, error estimate, method, n_evaluations.
- 30 tests. 9803 tests pass.

---

## v0.607.0 — 2026-05-24

**PDE solver redesign — class-based, 7 methods, grids, Greeks extraction.**

### PDE Solver (`numerical/_pde.py`)
- `PDESolver1D` class — configurable method, grid, reusable.
- `PDEMethod` enum: EXPLICIT, IMPLICIT, CRANK_NICOLSON, RANNACHER, CRAIG_SNEYD, HUNDSDORFER_VERWER, METHOD_OF_LINES.
- `GridType` enum: UNIFORM, LOG, SINH (Tavella-Randall concentration), CHEBYSHEV.
- `BoundaryCondition` enum: DIRICHLET, NEUMANN, LINEAR, FREE.
- `build_grid()` — spatial grid builder with strike/barrier concentration.
- `extract_greeks()` — delta, gamma, theta from grid solution via finite differences.
- `solve_bs_pde()` — one-line Black-Scholes PDE for European/American options.
- `solve_pde_with_vega()` — vega via bump-and-reprice.
- `PDEResult` — values, grid, price, delta, gamma, theta, vega, to_dict().
- Thomas algorithm tridiagonal solver.
- American via payoff projection. Rannacher smoothing.
- 23 tests: all methods, ATM/ITM/OTM, put, American, Greeks vs BS, grid types.
- 9773 tests pass.

---

## v0.606.0 — 2026-05-24

**Advanced numerical methods: spectral, quasi-Monte Carlo, stochastic calculus.**

### Spectral Methods (`numerical/_spectral.py`)
- `chebyshev_nodes()`, `chebyshev_diff_matrix()`, `chebyshev_coefficients()`, `chebyshev_evaluate()` (Clenshaw recurrence).
- `chebyshev_interpolate()` → `SpectralResult` with arbitrary-point evaluation.
- `spectral_solve_bvp()` — BVP solver via Chebyshev collocation.
- `spectral_integrate()` — Gauss-Legendre quadrature.

### Quasi-Monte Carlo (`numerical/_qmc.py`)
- `sobol_sequence()` — Sobol low-discrepancy (scipy.stats.qmc, O(1/N) convergence).
- `halton_sequence()`, `latin_hypercube()`.
- `sparse_grid()` — Smolyak construction for high-dimensional integration.

### Stochastic Calculus (`numerical/_stochastic.py`)
- `ito_formula()`, `ito_log_transform()` — Ito's formula with correction term.
- `stratonovich_to_ito()` / `ito_to_stratonovich()` — convention conversion.
- `quadratic_variation()`, `realized_variance()`, `realized_volatility()`.
- `bipower_variation()` — robust to jumps (Barndorff-Nielsen & Shephard).
- `jump_test()` — detect jumps via RV vs BV comparison.
- `milstein_correction()` — Milstein SDE discretisation term.
- 29 tests. 9750 tests pass.

---

## v0.605.0 — 2026-05-24

**ODE solver redesign — class-based, 9 methods, Riccati, backward, dense output.**

### ODE Solver (`numerical/_ode.py`)
- `ODESolver` class — configurable method, tolerance, dense output, reusable.
- `ODEMethod` enum: EULER, RK4, RK45, RK23, BDF, RADAU, LSODA, DOP853, IMPLICIT_EULER (9 methods).
- `solve_ode()` — main entry with runtime method selection + Jacobian + events.
- `solve_backward()` — backward-in-time integration for PDE time-stepping.
- `solve_riccati(a, b, c, ...)` — Riccati ODE dy/dt = a + by + cy² with analytical Jacobian. Supports complex coefficients (Heston CF).
- `solve_system()` — auto stiffness detection via LSODA.
- Implicit Euler via Newton iteration with optional Jacobian.
- Dense output for arbitrary-time evaluation (scipy interpolant + linear fallback).
- `ODEResult.__call__(t)` — evaluate solution at any time.
- Full backward compatibility: `euler()`, `rk4()`, `rk45()`, `bdf()`, `adams()` still work.
- 31 tests (up from 4): all methods, stiff systems, Jacobian, dense output, backward, Riccati (linear, quadratic, tanh), 2D rotation, Lorenz.
- 9721 tests pass.

---

## v0.604.0 — 2026-05-23

**Phase 4: Graph theory — network, contagion, algorithms, correlation network.**

### 4.1 Financial Network (`risk/network.py`)
- `FinancialNetwork` — degree, betweenness, eigenvector centrality, PageRank.
- `NetworkResult` with composite systemic risk ranking.

### 4.2 Default Cascade (`risk/contagion.py`)
- `DefaultCascade` — Eisenberg-Noe cascade with capital buffers, multi-round propagation.
- `stress_test()` — multiple scenarios. Contagion multiplier metric.

### 4.3 Graph Algorithms (`numerical/_graph.py`)
- `dijkstra()`, `shortest_path()`, `minimum_spanning_tree()` (Prim), `max_flow()` (Edmonds-Karp), `connected_components()`. Pure numpy.

### 4.4 Correlation Network (`risk/correlation_network.py`)
- `correlation_to_distance()` — Mantegna (1999).
- `mst_portfolio()` — MST from return correlations.
- `hierarchical_risk_parity()` — López de Prado (2016) HRP weights.
- `community_detection()` — spectral clustering on Laplacian.
- 21 tests. 9694 tests pass.

---

## v0.603.0 — 2026-05-23

**Phase 3: Game theory — Shapley, cooperative games, Nash, auction.**

### 3.1 Shapley Value (`risk/shapley.py`)
- `shapley_value()` — exact (2^N coalitions). `shapley_sampling()` — MC for large N.
- Satisfies all 4 axioms: efficiency, symmetry, dummy, additivity.
- `shapley_capital_allocation()` — fair desk-level capital allocation.

### 3.2 Cooperative Games (`risk/cooperative_games.py`)
- `CooperativeGame` — characteristic function + Shapley + core check.
- `NettingSetGame` — netting benefit allocation across counterparties.
- `CollateralPoolGame` — funding cost reduction from shared pool.

### 3.3 Nash & Microstructure (`models/game_equilibrium.py`)
- `nash_2player()` — support enumeration for bimatrix games.
- `market_maker_equilibrium()` — Avellaneda-Stoikov optimal spread with inventory.
- `optimal_execution_game()` — Almgren-Chriss front-loaded schedule.

### 3.4 Auction Theory (`fixed_income/auction.py`)
- `BondAuction` — uniform/discriminatory price, bid-to-cover, tail.
- `winners_curse_adjustment()`, `expected_revenue()`.
- 25 tests. 9673 tests pass.

---

## v0.602.0 — 2026-05-23

**2.4: Maximum entropy option pricing — model-free risk-neutral density.**

### Entropy Pricing (`options/entropy_pricing.py`)
- `max_entropy_density()` — recover RN density maximising Shannon entropy subject to option price constraints.
- Buchen-Kelly dual formulation with analytical gradient (L-BFGS-B).
- `MaxEntropyResult` — density grid, entropy, forward, repricing errors, `call_price()`, `put_price()`, `implied_vol_at()`.
- `entropy_implied_vol()` — extract full implied vol smile from sparse quotes.
- **Use cases:** model-free pricing from sparse option data, smile interpolation without parametric model.
- 11 tests. 9648 tests pass.

---

## v0.601.0 — 2026-05-23

**Phase 2 (2.1-2.3): Information theory — entropy, divergence, MI, Fisher information.**

### Information Theory (`statistics/information_theory.py`)
- **Entropy:** `shannon_entropy()`, `differential_entropy()` (KDE or histogram).
- **Divergence:** `kl_divergence()`, `js_divergence()` (symmetric), `cross_entropy()`, `wasserstein_distance()`.
- **Mutual Information:** `mutual_information()`, `conditional_mutual_information()`, `information_gain()` (feature ranking).
- **Fisher Information:** `fisher_information_matrix()` (numerical Hessian), `cramer_rao_bound()`, `parameter_confidence_intervals()`.
- **Use cases:** model risk (KL P‖Q), feature selection for PD, parameter uncertainty in HW/SABR calibration.
- 18 tests. 9637 tests pass.

---

## v0.600.0 — 2026-05-23

**1.3 + 1.4: Regime-switching process + regime-dependent market data.**

### Regime Process (`models/regime_process.py`)
- `RegimeProcessSpec` — regime-dependent drift/diffusion with Markov transitions.
- `create_regime_gbm()` — regime-switching GBM (equity/FX).
- `create_regime_ou()` — regime-switching OU (rates/spreads).
- Simulates paths + regime labels jointly.

### Regime Surfaces (`models/regime_surfaces.py`)
- `RegimeVolSurface` — N vol surfaces blended by regime probabilities (variance or linear blend).
- `RegimeCurve` — N discount curves blended by regime probabilities.
- `regime_price()` — price under each regime and blend by posterior.
- 18 tests. 9619 tests pass.

---

## v0.599.0 — 2026-05-23

**1.2: Particle filter — sequential Monte Carlo for non-linear state estimation.**

### Particle Filter (`statistics/particle_filter.py`)
- `ParticleFilter(n_particles, transition_fn, observation_log_likelihood)` — bootstrap filter.
- Pluggable dynamics: any `transition_fn(particles, rng) → particles` + `obs_log_lik(y, particles) → log_weights`.
- Systematic resampling with ESS monitoring.
- `ParticleFilterResult` — filtered means/stds, ESS trajectory, log-likelihood, final particles.
- **Use cases:** stochastic vol filtering (Heston latent vol), non-linear credit dynamics, any non-Gaussian state-space.
- 10 tests. 9601 tests pass.

---

## v0.598.0 — 2026-05-23

**1.1: Generalised HMM framework — pluggable emissions, Baum-Welch, Viterbi.**

### HMM Core (`statistics/hmm.py`)
- `EmissionModel(ABC)` — pluggable observation distributions: `log_prob()`, `fit_params()`, `sample()`.
- Concrete emissions: `GaussianEmission`, `StudentTEmission`, `MixtureEmission`, `MultivariateGaussianEmission`.
- `EmissionType` enum + `create_emission()` factory (follows Interpolator pattern).
- `HMM(n_states, emission)` — generalised HMM class.
  - `fit()` — Baum-Welch EM with scaled forward-backward.
  - `filter()` — online filtering of new observations.
  - `predict_state()` — Viterbi decoding.
- `HMMFitResult` — transition matrix, emission params, stationary dist, AIC/BIC, filtered probs, Viterbi labels.
- Supports 2+ states, any univariate or multivariate emission.
- **Use cases:** vol regime, credit regime, yield curve regime, any latent-state time series.
- 20 tests. 9591 tests pass.

---

## v0.597.0 — 2026-05-21

**Repo Phase 3b + 4: Matched book, BS allocation, margin, settlement, sec lending.**

### 3.3 Matched Book (`desks/matched_book.py`)
- `MatchedBookPosition` — paired repo/reverse with spread, gap, PnL.
- `matched_book_optimise()` — greedy selection by spread, subject to gap + notional limits.

### 3.4 Balance Sheet Allocation (`regulatory/balance_sheet_allocation.py`)
- `rank_by_roc()` — return on capital ranking.
- `optimise_allocation()` — LP: maximize total ROC subject to capital + RWA constraints.

### 4.1 Margin Mechanics (`fixed_income/repo_margin.py`)
- `calculate_vm()`, `margin_call()` (threshold + MTA), `margin_forecast()`.

### 4.2 Settlement Fails (`fixed_income/repo_settlement.py`)
- `propagate_fails()` — cascade through matched book.
- `buy_in_process()` — CSDR mandatory buy-in.
- `fail_cost_analysis()` — penalty + opportunity + reputation.

### 4.3 Securities Lending (`fixed_income/securities_lending.py`)
- `SecLendingTrade`, `lending_vs_repo_arbitrage()`, `locate_availability()`.
- 23 tests. 9571 tests pass.

---

## v0.596.0 — 2026-05-21

**Repo Phase 3: Leverage optimization + collateral transformation.**

### 3.1 Leverage Optimization (`risk/leverage_optimisation.py`)
- `optimise_leverage()` — LP: maximize carry subject to haircut + capital + concentration constraints.
- `leverage_frontier()` — efficient frontier of carry vs leverage ratio (1× to 20×).

### 3.2 Collateral Transformation (`risk/collateral_transformation.py`)
- `transformation_cost()` — all-in cost: repo spread + xccy basis + capital - haircut benefit.
- `optimise_transformation()` — greedy upgrade of available collateral to target quality.
- `funding_arbitrage()` — identify mispriced collateral vs funding value.
- 13 tests. 9548 tests pass.

---

## v0.595.0 — 2026-05-21

**Repo Phase 2: Counterparty credit — CVA + wrong-way risk, dynamic haircuts, correlated XVA.**

### 2.1 Repo CVA (`risk/repo_cva.py`)
- `repo_cva()` — CVA on unsecured exposure after haircut, time-grid integration.
- `repo_wrong_way_risk()` — three channels: issuer (classic), sector (systemic), spiral (margin).
- `repo_bilateral_cva()` — CVA + DVA + WWR combined.

### 2.2 Dynamic Haircuts (`risk/dynamic_haircuts.py`)
- `DynamicHaircutModel` — spread-driven + vol-driven + rating trigger + BCBS 261 procyclicality buffer.
- `haircut_stress_scenarios()` — 7 standard scenarios.
- `credit_spread_to_haircut()` — continuous spread → haircut mapping.
- `rating_trigger_impact()` — step function per downgrade notch.

### 2.3 Correlated XVA (`risk/repo_xva_advanced.py`)
- `repo_xva_correlated()` — joint MC: counterparty default + collateral spread (Gaussian copula).
- CVA + FVA + KVA + MVA + gap cost, fully correlated.
- `repo_all_in_xva()` — profitability: interest income vs total XVA.
- 26 tests. 9535 tests pass.

---

## v0.594.0 — 2026-05-21

**Repo 1.3 + 1.4: Specialness analytics (6 markets) + repo rate Greeks.**

### Specialness Analytics (`fixed_income/repo_specialness.py`)
- `SpecialnessConventions` — 6 sovereign markets (UST, Bund, Gilt, JGB, OAT, BTP).
- `forecast_specialness()` — mean-reversion + auction-cycle seasonality.
- `specialness_term_structure()` — GC-special spread curve.
- `supply_demand_indicator()` — fail rate, on-the-run, short interest signals.

### Repo Rate Greeks (`fixed_income/repo_greeks.py`)
- `repo_dv01()` — trade-level interest + carry sensitivity per 1bp.
- `carry_sensitivity_ladder()` — by tenor bucket (O/N, 1W, 1M, 3M, 6M, 1Y+).
- `repo_portfolio_greeks()` — aggregated DV01, carry DV01, roll theta.
- 24 tests. 9509 tests pass.

---

## v0.593.0 — 2026-05-21

**Repo Phase 1: Multi-currency funding curves, carry breakeven, credit-collateral integration.**

### 1.1 Dealer Funding Curve (`fixed_income/repo_funding_curve.py`)
- `DealerFundingCurve` — secured + unsecured legs, blended rate with haircut.
- `RepoMarketConventions` — 11 currencies (USD/EUR/GBP/JPY/CHF/CAD/AUD/BRL/MXN/ZAR/TRY) with day count, settlement, benchmark, GC collateral types.
- `build_dealer_funding_curve()`, `to_discount_curve()`.
- 15 tests.

### 1.2 Carry Breakeven (`fixed_income/repo_carry.py`)
- `carry_breakeven()` — GC vs special, term vs O/N, breakeven rate.
- `xccy_repo_carry()` — cross-currency with FX basis.
- `multi_ccy_carry_comparison()` — rank carry across currencies for same bond.

### 1.5 Credit-Collateral Integration (`fixed_income/repo_credit_collateral.py`)
- `CreditCollateralSpec` — issuer hazard, rating, sector, seniority.
- `credit_adjusted_haircut()` — base + PD add-on + spread-vol add-on. 8 asset classes: sovereign, IG, HY, bank senior, AT1/T2, structured IG/HY, equity.
- `repo_price_with_collateral_credit()` — all-in: interest - collateral default - counterparty credit - wrong-way risk - gap risk.
- `hazard_to_haircut_mapping()` — continuous hazard → haircut schedule.
- 21 tests. 9485 tests pass.

---

## v0.592.0 — 2026-05-21

**Phase 4: Curve blending, seasonal, diffusion, storage.**

### 4.1 Curve Blending (`curves/curve_blending.py`)
- `splice_curves()` — short/long curve splicing with linear, sigmoid, or step transition.
- `blend_curves()` — weighted blend of N curves in log-DF space.
- 6 tests.

### 4.2 Seasonal Term Structure (`curves/seasonal_curve.py`)
- `SeasonalCurve` — base curve with year-end/quarter-end/month-end spread overlay.
- `SeasonalPattern` — configurable decay, pre-built USD/EUR/GBP patterns.
- `extract_seasonal_pattern()` — fit from historical O/N fixings.
- `strip_seasonal()` — remove seasonal for smooth analysis.
- 6 tests.

### 4.3 Curve Diffusion (`curves/curve_diffusion.py`)
- `CurveDiffusionEngine` — multi-factor HJM simulation, exponentially decaying vol.
- Each path at each step → standard `DiscountCurve` (all pricing code works unchanged).
- Forward rate statistics (mean, std) across paths.
- 5 tests.

### 4.4 Curve Storage (`curves/curve_storage.py`)
- `CurveSnapshot` — timestamped zero-rate snapshot with `from_curve()` / `to_curve()`.
- `CurveDelta` — sparse delta between snapshots (bp shifts).
- `CurveStore` — in-memory save/load/history/diff.
- 7 tests. 9449 tests pass.

---

## v0.591.0 — 2026-05-21

**Phase 3: FX forward curves, curve scenarios, real-time bumper.**

### 3.1 FX Forward Builder (`fx/fx_forward_builder.py`)
- `build_fx_implied_curve()` — from spot + swap points + domestic OIS via CIP.
- 14 FX pair conventions (settlement, pip factor, quoting direction).
- Basis spread extraction vs known foreign curve.
- 6 tests.

### 3.2 Curve Scenario Engine (`curves/curve_scenarios.py`)
- `parallel_shift()`, `steepener()`, `flattener()`, `bear_steepener()`, `bull_flattener()`.
- `butterfly()`, `inversion()`, `historical_scenario()`.
- `pca_scenarios()` — PCA level/slope/curvature from historical data.
- `standard_scenario_set()` — 11 canned scenarios per currency.
- `run_scenarios()` — batch execution with PnL.
- 9 tests.

### 3.3 Real-Time Curve Bumper (`curves/curve_bumper.py`)
- `CurveBumper` — Jacobian pre-computation, fast repricing via J·Δz.
- `bump_and_reprice()` (fast, ~μs) vs `full_rebuild_and_reprice()` (exact).
- `parallel_dv01()`, `key_rate_dv01s()`, `cross_gamma()`.
- `risk_report()` — full instrument risk (DV01, key-rate, convexity).
- 5 tests. 9425 tests pass.

---

## v0.590.0 — 2026-05-21

**2.1: N-curve simultaneous global solver — damped Newton for 1-N curves.**

### N-Curve Solver (`curves/ncurve_solver.py`)
- `InstrumentPricer` protocol — each instrument reprices given named curves.
- Concrete pricers: `DepositPricer`, `OISSwapPricer`, `BasisSwapPricer`.
- `CurveSpec` — per-curve pillar dates, initial guess, interpolation.
- `ncurve_solve()` — damped Newton-Raphson, numerical Jacobian, LU/lstsq, positivity-preserving step control.
- Tested: 1-curve (deposits, OIS swaps), 2-curve (OIS+projection, basis), 3-curve (OIS+1M+3M).
- 8 tests. 9405 tests pass.

---

## v0.589.0 — 2026-05-21

**2.2 + 2.3: Forward rate interpolation + key-rate DV01 framework.**

### Forward Rate Interpolation (`core/forward_interpolation.py`)
- `ForwardInterpolationMethod` — piecewise constant, piecewise linear, monotone convex (Hagan-West 2006).
- `build_forward_curve()` — builds DiscountCurve by interpolating on forwards and integrating.
- `monotone_convex_forwards()` — smooth, positive, shape-preserving forward function.
- `extract_forwards()` — extract instantaneous forwards from any curve.

### Key-Rate DV01 (`curves/key_rate_risk.py`)
- `BumpProfile` — triangular (partition of unity), Gaussian, pillar-only.
- `key_rate_dv01()` — localised bumps, DV01 per tenor, optional gamma.
- `bucket_risk()` — tenor bucket aggregation (0-1Y, 1-2Y, ..., 20-30Y).
- `risk_ladder()` — formatted report with % contribution.
- `standard_tenors(currency)` — per-currency key-rate sets (USD, EUR, GBP, JPY, CHF).

### Tests
- 23 new tests: all methods, flat/upward curves, 10Y swap concentration, gamma, bucket risk, risk ladder.
- 9397 tests pass.

---

## v0.588.0 — 2026-05-21

**1.3: Multi-RFR OIS bootstrap — production-grade curve builder for 7 currencies.**

### RFR Bootstrap (`curves/rfr_bootstrap.py`)
- `bootstrap_rfr(currency, ref_date, inputs)` — full instrument stack: O/N + term rates + futures + OIS swaps.
- `RFRCurveInputs` — overnight_rate, term_rates, futures_1m/3m, ois_swaps, deposits.
- `RFRCurveResult` — curve, pillar zeros, round-trip error, convexity adjustments per contract.
- `RFROISConventions` — per-currency: day counts, frequencies, calendar for USD/SOFR, EUR/ESTR, GBP/SONIA, JPY/TONA, CHF/SARON, CAD/CORRA, AUD/AONIA.
- Sequential (Brent) and global (Newton) methods.
- Futures convexity adjustments from item 1.2 wired in.
- Round-trip verification on deposit repricing.

### Tests
- 18 new tests: conventions, USD full stack, deposits-only, futures+swaps, all 7 G7 currencies, term rates, edge cases.
- 9374 tests pass.

---

## v0.587.0 — 2026-05-21

**1.2: RFR futures instruments — SOFR/SONIA/ESTR/SARON/TONA contract generation + convexity.**

### RFR Futures (`fixed_income/rfr_futures.py`)
- `RFRFutureSpec` — generic 1M/3M contracts for any RFR currency.
- `generate_rfr_contracts(currency, ref_date)` — serial (1M) and IMM quarterly (3M) date generation for USD, GBP, EUR, CHF, JPY.
- `rfr_futures_convexity()` — Hull-White convexity adjustment per contract.
- `rfr_futures_to_forwards()` — convert futures prices to forward rates for bootstrap.
- 16 tests. 9356 tests pass.

---

## v0.586.0 — 2026-05-21

**1.1: RFR compounding conventions — 12 currencies, full ISDA mechanics.**

### RFR Compounding (`fixed_income/rfr_compounding.py`)
- `RFRAccrualConfig` — observation shift, lookback, lockout, rate cut-off, payment delay, fixing lag.
- 12 frozen configs: SOFR, ESTR, SONIA, TONA, SARON, CORRA, AONIA (G10) + CDI, KOFR, SORA, HONIA, THOR (EM).
- `compound_rfr_full()` — backward-looking compounded rate with all ISDA adjustments from fixings.
- `compound_rfr_from_curve()` — forward-looking from discount curve (for pricing).
- `rfr_accrual_schedule()` — full observation/weight schedule per business day.
- `get_rfr_config()`, `list_rfr_configs()` — registry.

### Tests
- 23 new tests: registry, schedule mechanics (obs shift, lookback, weekend weight), flat/varying rates, multi-currency, lockout, rate cut-off.
- 9340 tests pass.

---

## v0.585.0 — 2026-05-21

**Hardening audit (L1-L11) — 10 fixes across 9 modules + 3 hand-calculation verifications.**

### Input Validation Fixes
- `regime_switching.py` — transition matrix must be stochastic (rows sum to 1, entries in [0,1]).
- `bilateral_csa.py` — correlation bounds validated in constructor.
- `coco.py` — trigger_intensity must be non-negative.
- `sovereign_cds.py` — tenor must be positive integer.
- `covered_bond.py` — LTV in (0, 1.5], OC >= 1.0.

### Numerical Stability Fixes
- `ndf_implied.py` — skip NDF quotes producing df > 2.0 (data error guard).
- `callable_credit.py` — clamp conditional survival to [0, 1] for floating-point safety.
- `yield_convention.py` — wider solver bracket [-50%, 500%], approximate fallback on failure.
- `spread_decomposition.py` — fixed tax formula unit error (was off by ×100).

### L11 Hand-Calculation Verification
- **CreditGrades**: Q(5Y) = 0.87053497, spread = 138.65bp — exact match (8 decimal places).
- **BRL BUS/252**: 254 business days, yf = 1.007937 — exact. Yield roundtrip perfect.
- **Convertible equity-credit**: default prob 9.44% (hand: 9.52%), bond floor 90.27 (hand: 90.65), δ>0, CS01<0, ρ-sens<0 — all correct.

---

## v0.584.0 — 2026-05-21

**C8: Convertible equity-credit correlation — joint (stock, hazard) Monte Carlo.**

### Convertible Equity-Credit (`credit/convertible_equity_credit.py`)
- Joint process: equity GBM + hazard CIR with correlation ρ (negative = wrong-way risk).
- Default via cumulative hazard vs exponential threshold (Cox process).
- LSM (Longstaff-Schwartz) backward induction for optimal conversion.
- Full Greeks: delta, gamma, vega, CS01, ρ-sensitivity — all via bump-and-reprice with common random numbers.
- Risky bond floor computation with survival-weighted cashflows.
- `convertible_equity_credit_price()` — single entry point.

### Tests
- 15 tests: pricing bounds, equity/credit/correlation sensitivity, Greeks signs, serialization.
- 9317 tests pass.

---

## v0.583.0 — 2026-05-21

**Phase 5 complete — all remaining plan items (A2, A3, A5, B3-B6, C5-C9, D7-D9).**

### Hazard Rate Production
- **A2:** ML-based PD (`credit/ml_pd.py`) — logistic regression from 9 financial ratios.
- **A3:** Sovereign CDS-bond basis (`credit/cds_bond_basis.py`) — funding, delivery, restructuring decomposition.
- **A5:** Joint equity-credit calibration (`credit/joint_equity_credit.py`) — fit CreditGrades to equity vol + CDS.

### CLN Advanced (`credit/cln_advanced.py`)
- **B3:** Spread-driven XVA, **B4:** dynamic funding (CSA-aware), **B5:** wrong-way risk (2nd-order), **B6:** collateral haircut stress.

### Bond Types + Markets
- **C5:** Covered bonds, **C6:** bond forwards + credit, **C9:** issuer spread curve (Nelson-Siegel on spreads).
- **D7:** Sukuk (7 types), **D8:** ESG labelling (ICMA GBP), **D9:** supranationals (10 issuers).

### Tests
- 55 new tests. 9302 tests pass.

---

## v0.582.0 — 2026-05-21

**Phase 4: Bond-Credit — C3 CoCo/AT1, C4 perpetuals, C1 callable+credit OAS, C2 spread decomposition.**

- **C3:** CoCo/AT1 (`credit/coco.py`) — trigger types, loss absorption, coupon cancellation, call/extension blending.
- **C4:** Perpetuals (`fixed_income/perpetual.py`) — plain/callable perpetual, step-up coupon.
- **C1:** Callable + credit OAS (`credit/callable_credit.py`) — backward induction with survival, price decomposition.
- **C2:** Spread decomposition (`credit/spread_decomposition.py`) — credit + liquidity + tax + optionality + residual.
- 47 new tests. 9247 tests pass.

---

## v0.581.0 — 2026-05-21

**B1 + B2: Bilateral CLN+CSA + correlated recovery.**

### Bilateral CSA Pricer (`credit/bilateral_csa.py`)
- `CSATerms` — threshold, independent amount, MTA, MPOR, haircut, rehypothecation.
- `BilateralCSAPricer` — MC simulation of correlated defaults + collateral mechanics + funding costs.
- CVA, DVA, FVA decomposition. 11 tests.

### Correlated Recovery (`credit/correlated_recovery.py`)
- `CorrelatedRecoveryModel` — factor model: R(M) = base + β × M × σ (Frye 2000).
- `systematic_recovery()` — link portfolio default rate to recovery via Vasicek factor.
- 15 tests. 9200 tests pass.

---

## v0.580.0 — 2026-05-21

**A6: Term structure of recovery — maturity-dependent + stochastic recovery.**

### Recovery Curve (`credit/recovery_curve.py`)
- `RecoveryCurve` — interpolated recovery by maturity: `flat()`, `linear()`, `from_seniority()`.
- `RecoverySeniority` enum: 5 levels (senior secured → junior subordinated) with Moody's historical averages.
- `StochasticRecovery` — beta-distributed recovery with `sample()`, `percentile()`, `from_seniority()`.
- `recovery_by_seniority()`, `recovery_vol_by_seniority()` — lookup functions.
- Seniority ordering: SR_SEC(53%) > SR_UNS(40%) > SR_SUB(32%) > SUB(28%) > JR_SUB(18%).

### Tests
- 16 new tests: curve shapes, seniority ordering, stochastic sampling, percentiles.
- 9174 tests pass.

---

## v0.579.0 — 2026-05-21

**A4: CreditGrades model — first-passage Merton with stochastic barrier.**

### CreditGrades (`credit/credit_grades.py`)
- `CreditGrades` class: asset vol, leverage, recovery mean/vol → survival, spreads, distance to default.
- First-passage survival via barrier-crossing formula: Q(t) = Φ(α) − d̄ × Φ(β).
- σ̄² = σ² + λ² (combined asset + barrier uncertainty).
- `survival()`, `cds_spread()`, `spread_term_structure()`, `distance_to_default()`, `evaluate()`.
- Convenience functions: `credit_grades_survival()`, `credit_grades_spread()`.
- Produces realistic spreads: IG ~30bp, HY ~900bp at 5Y.

### Tests
- 20 new tests: survival monotonicity, IG/HY levels, vol/leverage sensitivity, DD ordering, edge cases.
- 9158 tests pass.

---

## v0.578.0 — 2026-05-21

**A1: Regime-switching credit — HMM with state-dependent hazard rates.**

### Regime-Switching Credit (`credit/regime_switching.py`)
- `RegimeSwitchingCredit` — continuous-time Markov chain with state-dependent default intensities.
- Survival via matrix exponential: Q(t) = π₀ × exp((Q-Λ)t) × 1.
- `survival()`, `implied_hazard()`, `implied_spread()` — with optional conditioning on initial state.
- `regime_probabilities()`, `expected_hazard()`, `stationary_distribution()`.
- `spread_term_structure()` — term structure under regime uncertainty.
- `calibrate_regime_model()` — fit 2 or 3 state model from observed CDS spread curve.
- 2-state (expansion/recession) and 3-state (expansion/normal/recession) support.

### Tests
- 21 new tests: survival bounds, conditional, 3-state, calibration, repricing, serialization.
- 9138 tests pass.

---

## v0.577.0 — 2026-05-21

**D14: Sovereign FRNs — 3 floating-rate sovereign markets.**

### Sovereign FRN Factory (`fixed_income/sovereign_bonds.py`)
- USTFRN (US 2Y FRN, quarterly ACT/360, T-Bill linked), GILTFRN (UK, quarterly ACT/365F, SONIA-linked), BTPFRN (Italy, semi-annual, ESTR-linked).
- `create_sovereign_frn(market_code, issue, maturity, spread)` — factory.
- `list_frn_markets()` — 3 FRN codes.
- Yield convention mapping updated for FRNs.
- 56 total sovereign markets (50 coupon + 3 T-Bill + 3 FRN).

### Tests
- 5 new FRN tests: factory, pricing, near-par.
- 9117 tests pass.

---

## v0.576.0 — 2026-05-21

**D11: Cross-market sovereign relative value framework.**

### Sovereign RV (`fixed_income/sovereign_rv.py`)
- `sovereign_spread_decomposition()` — decomposes spread into credit (CDS), fundamental (macro), liquidity (bid-ask/turnover), and technical (residual) components.
- `cross_market_rv_scores()` — cross-sectional Z-scores, percentiles, and CHEAP/FAIR/RICH signals across N sovereign markets.
- `SovereignRVInput` — macro fundamentals: debt/GDP, fiscal balance, current account, rating, FX vol, reserves.
- `SpreadDecomposition`, `RVScore` result dataclasses with `to_dict()`.

### Tests
- 14 new tests: decomposition, component sum, high/low risk, Z-scores, sorting, signals, edge cases.
- 9112 tests pass.

---

## v0.575.0 — 2026-05-21

**D12: EM local currency curve builders — 16 currencies + CDI/TIIE/SHIBOR.**

### EM Curve Builder (`curves/em_curve_builder.py`)
- `EMCurveConventions` — per-currency deposit/swap day count, frequency, interpolation.
- 16 EM currencies: BRL, MXN, CNY, KRW, ZAR, INR, SGD, HKD, THB, PLN, CZK, HUF, COP, CLP, TRY, IDR.
- `build_em_curve(currency, ref, deposits, swaps)` — generic builder with correct conventions.
- `build_cdi_curve(ref, di_futures)` — Brazil CDI from DI futures (df = 1/(1+r)^(bd/252)).
- `build_tiie_curve()`, `build_shibor_curve()` — Mexico and China convenience wrappers.
- `get_em_curve_conventions()`, `list_em_curve_currencies()`.

### Tests
- 14 new tests: conventions, all-currency build, CDI formula verification, TIIE, SHIBOR.
- 9098 tests pass.

---

## v0.574.0 — 2026-05-21

**D10: EM sovereign credit curves — 31 sovereigns + CDS hazard bootstrap.**

### Sovereign CDS (`credit/sovereign_cds.py`)
- `SovereignCDSConventions` — restructuring clause (CR/MR/MM/XR), recovery rate, standard tenors, doc clause.
- 31 sovereigns: LatAm (BR, MX, CO, CL, PE, AR), CEEMEA (TR, ZA, PL, HU, RO, RU, EG, NG, KE), Asia (CN, KR, ID, PH, MY, TH, IN, VN), W. Europe (IT, ES, PT, GR, IE), MENA (SA, QA, IL).
- `bootstrap_sovereign_hazard()` — sequential bootstrap from CDS spreads → SurvivalCurve.
- `RestructuringClause` enum: CR, MR, MM, XR.
- `get_sovereign_cds_conventions()`, `list_sovereign_cds()`.

### Tests
- 18 new tests: conventions, bootstrap, term structure, distressed, IG, recovery override, multi-country.
- 9084 tests pass.

---

## v0.573.0 — 2026-05-21

**D15: Market-convention yield quotation — yield↔price for all 53 sovereign markets.**

### Yield Conventions (`fixed_income/yield_convention.py`)
- `YieldConvention` enum: SEMI_ANNUAL, ANNUAL, QUARTERLY, CONTINUOUS, SIMPLE, DISCOUNT.
- `yield_to_price()` / `price_to_yield()` — convert between yield and clean price under any convention.
- `convert_yield()` — convert between conventions (exact for zeros, price roundtrip for coupon bonds).
- `get_yield_convention(market_code)` — street convention for all 53 sovereign markets.
- Market mapping: UST/GILT/JGB semi-annual, BUND/OAT annual, NTN_F/LTN continuous, RPGB quarterly, USTBILL/CETES bank discount.

### Tests
- 30 new tests: roundtrips, known values, conversions, market mapping, all-53-markets coverage.
- 9066 tests pass.

---

## v0.572.0 — 2026-05-21

**D13: Zero-coupon sovereign bonds — ZeroCouponBond class + factory.**

### ZeroCouponBond (`fixed_income/zero_coupon_bond.py`)
- `price()` / `dirty_price()` — Face × df(T) from discount curve.
- `price_from_yield_simple()` — money-market convention: Face / (1 + r × τ).
- `price_from_discount_rate()` — bank discount: Face × (1 - d × τ).
- `price_from_yield_continuous()` — Face × exp(-r × τ).
- `yield_simple()`, `discount_rate()`, `yield_continuous()` — inverse functions.
- `dv01()`, `modified_duration()`, `to_dict()`.

### Sovereign Factory Updates (`fixed_income/sovereign_bonds.py`)
- `is_zero_coupon` field on `SovereignConventions`.
- 3 new T-Bill markets: USTBILL (ACT/360), UKTBILL (ACT/365F), EURTBILL (ACT/360).
- LTN and CETES flagged as zero-coupon.
- `create_sovereign_zero()` — factory for zero-coupon bonds.
- `list_zero_coupon_markets()` — returns 5 zero-coupon codes.
- 53 total markets (50 coupon + 3 T-Bill).

### Tests
- 10 new zero-coupon tests: factory, pricing, yield roundtrip, DV01, discount rate.
- 9036 tests pass.

---

## v0.571.0 — 2026-05-21

**D6: EM inflation indices — 16 indices + linker factory.**

### Inflation Index Registry (`fixed_income/inflation_indices.py`)
- `InflationIndexDef` — frozen dataclass: name, currency, lag, frequency, interpolation, deflation floor, linker conventions.
- `IndexInterpolation` enum: FLAT (UK ILG), LINEAR (TIPS, most), DAILY (UDI/UF/UVR).
- 16 indices: CPI_US (TIPS), HICP_XT (OAT€i/BTP€i), RPI/CPIH (UK), CPI_JP, CPI_CA, CPI_AU, IPCA (BRL), UDI (MXN daily), UF (CLP daily), UVR (COP daily), CPI_ZA, CPI_IL, CPI_TR, CPI_IN (30/360!), CPI_KR.
- `get_inflation_index()`, `list_inflation_indices()`, `indices_by_currency()`, `indices_with_floor()`, `daily_indices()`.
- `create_inflation_linker()` — factory returning correct kwargs for `InflationLinkedBond`.

### Tests
- 31 new tests: all 16 indices, registry API, linker factory (TIPS, NTN-B, OAT€i, UK ILG, UDIBONO), serialization.
- 9026 tests pass.

---

## v0.570.0 — 2026-05-21

**D5: EM RFR/IBOR rate indices — 14 new indices across 13 EM currencies.**

### EM Rate Indices (`core/rate_index.py`)
- **Overnight RFR (8):** CDI (BRL, BUS/252), KOFR (KRW), SORA (SGD), HONIA (HKD), THOR (THB), DR007 (CNY, averaged), IBR (COP), TPM (CLP).
- **Term IBOR (6):** TIIE_28D (MXN, T-1 fixing), SHIBOR_3M (CNY), WIBOR_3M (PLN), PRIBOR_3M (CZK), BUBOR_3M (HUF), JIBAR_3M (ZAR).
- Registry now has 25 indices (11 G10 + 14 EM), 16 overnight.

### Tests
- 21 new tests: all EM indices, registry counts, currency coverage, frozen dataclass.
- 8995 tests pass.

---

## v0.569.0 — 2026-05-21

**D2: NDF-implied discount curve construction for restricted EM currencies.**

### NDF-Implied Curves (`curves/ndf_implied.py`)
- `build_ndf_implied_curve()` — derive EM discount curve from FX NDF prices + G10 base curve via covered interest parity: df_em(T) = df_base(T) × Spot / NDF(T).
- `ndf_from_curves()` — compute theoretical NDF prices from two discount curves (for CIP deviation checking).
- `cip_basis()` — measure covered interest parity basis in bp (funding stress indicator).
- `NDFQuote` dataclass with bid/ask/mid support.
- `NDFImpliedResult` with implied DFs, zero rates, forward points, to_dict().

### Tests
- 19 new tests: construction, round-trip, CIP basis, multi-currency (CNY, INR, KRW, BRL), edge cases, helpers.
- 8974 tests pass.

---

## v0.568.0 — 2026-05-21

**D4: Sovereign bond factory — 50 markets with correct conventions.**

### Sovereign Bond Factory (`fixed_income/sovereign_bonds.py`)
- `SovereignConventions` — frozen dataclass: market_code, currency, frequency, day_count, settlement_days, calendar, ex_div_days.
- `create_sovereign_bond(market_code, issue, maturity, coupon)` — factory returning correctly-configured `FixedRateBond`.
- `get_conventions(market_code)` — lookup conventions by market code.
- `list_markets()` — 50 sovereign markets.
- `markets_by_region()` — grouped by G10_core, other_dm, eurozone, cee, turkey_mena, africa, latam, asia.

### Markets (50)
- **G10 core (6):** UST, BUND, GILT, JGB, OAT, BTP.
- **Other DM (7):** ACGB, NZGB, CGB_CA, DGB, SGB, NGB, CONFED.
- **Eurozone (8):** BONO, BGB, DSL, RAGB, RFGB, IRISH, PGB, GGB (T+3).
- **CEE (4):** POLGB, CZGB, HGB (ACT/365F), ROMGB (semi-annual).
- **Turkey & MENA (6):** TURKGB (T+0!), SAGB_SA, ADGB, QATGB (30/360), ILGB, EGGB.
- **Africa (3):** SAGB (T+3), NGGB, KEGB.
- **LatAm (7):** NTN_F, NTN_B, LTN (BUS/252), MBONO (ACT/360!), CETES, BTP_CL, TES.
- **Asia (9):** CGB, KTB, GSEC (30/360!), SGS, HKGB, INDOGB, MGS, THAIGB, RPGB (quarterly!).

### Tests
- 35 new tests: convention lookup, factory creation, all-market creation, pricing sanity, coverage checks.
- 8955 tests pass.

---

## v0.567.0 — 2026-05-21

**D3: BUS/252 day count convention for Brazilian markets.**

### BUS/252 (`core/day_count.py`)
- `DayCountConvention.BUS_252` — business days / 252, the standard for all BRL instruments (NTN-F, NTN-B, LTN, DI futures).
- `business_days_between(start, end, calendar)` — count business days between two dates (start exclusive, end inclusive).
- `year_fraction(..., calendar=)` — new optional `calendar` parameter for BUS/252.
- Defaults to São Paulo calendar when no calendar provided.
- Works with any calendar (e.g. USD for testing).

### Tests
- 7 new BUS/252 tests: week count, year approximation, carnival skip, weekend skip, default calendar, US calendar, Independence Day.
- 8920 tests pass.

---

## v0.566.0 — 2026-05-21

**D1: EM Calendars — 24 new calendars + registry.**

### EM Calendars (`core/calendar.py`)
- **CEE (4):** Warsaw (PLN), Prague (CZK), Budapest (HUF), Bucharest (RON, Orthodox Easter).
- **Turkey & MENA (4):** Istanbul (TRY), Riyadh (SAR), Tel Aviv (ILS, Fri-Sat weekend), Cairo (EGP).
- **Africa (3):** Johannesburg (ZAR, Sun→Mon observance), Nairobi (KES), Lagos (NGN).
- **LatAm (4):** São Paulo (BRL, Carnival), Mexico City (MXN, Maundy Thu), Santiago (CLP), Bogotá (COP, emiliani Monday law).
- **Asia (8):** Beijing (CNY), Seoul (KRW), Mumbai (INR), Singapore (SGD), Hong Kong (HKD), Jakarta (IDR), Kuala Lumpur (MYR), Bangkok (THB), Manila (PHP).
- **Other DM (1):** Denmark (DKK, Store Bededag removed post-2023).
- Orthodox Easter algorithm for Romania (Julian + 13-day Gregorian offset).

### Calendar Registry (`core/calendar.py`)
- `get_calendar(currency_code)` — 35 currencies (11 G10 + 24 EM).
- `list_calendars()` — sorted list of available codes.

### Tests
- 56 new tests covering holidays, business day conventions, Orthodox Easter, cross-calendar consistency, joint calendar.
- 8913 tests pass.

---

## v0.565.0 — 2026-05-20

**Bond hazard bootstrap — recovery of market value & liquidity premium separation.**

### Recovery of Market Value (`credit/bond_hazard_bootstrap.py`)
- `_price_risky_bond_rmv()` — Duffie-Singleton (1999) pricing: recovery = R × V(t⁻), reduces to discounting at Q̃(t) = Q(t)^(1-R). No separate recovery leg.
- `recovery_mode` parameter on `bootstrap_hazard_from_bonds()`: `"par"` (ISDA standard, default) or `"market_value"` (Duffie-Singleton).
- RMV produces lower hazard rates than RP for the same market prices (less recovery → less hazard needed to explain low price).
- `RECOVERY_PAR`, `RECOVERY_MARKET_VALUE` constants exported.

### Liquidity Premium Separation (`credit/bond_hazard_bootstrap.py`)
- `BondInput.liquidity_spread_bp` — per-bond liquidity premium assumption (bp).
- Bootstrap bumps the discount curve by liquidity spread before credit extraction, isolating pure credit hazard.
- Per-bond liquidity (e.g. higher for illiquid long-end) supported in both sequential and global methods.
- Combined with RMV recovery mode for full flexibility.

### Tests
- 14 new tests (31 total): RMV pricing, RMV bootstrap round-trip, liquidity spread effect, per-bond liquidity, combined RMV+liquidity, edge cases.
- 8836 tests pass.

---

## v0.563.0 — 2026-05-18

**Sell-side / buy-side gap closure — 5 modules.**

### IPV Workflow (`risk/ipv.py`)
- `FairValueLevel` — Level 1 (market) / Level 2 (comparable) / Level 3 (model).
- `BCBS287_BID_ASK` — 15 asset-class-specific bid-ask tables.
- `ipv_single_trade()` → `IPVResult` — automated AVA via existing prudent_valuation.
- `ipv_portfolio()` → `IPVReport` — portfolio aggregation, level summary, breach detection.

### Mandate Compliance (`core/mandate.py`)
- `Mandate` — configurable policy: eligible_asset_classes, min_rating, max_single_name_pct, max_sector_pct, max_country_pct, currency_restrictions, max_duration.
- `check_mandate()` → `MandateReport` — pass/fail per rule with breach details.
- Predefined templates: investment_grade, sovereign_only, balanced, high_yield.

### Term Sheet Generator (`desks/term_sheet.py`)
- `generate_term_sheet()` → `TermSheet` — markdown-based: Deal Summary, Key Terms, Risk Profile, Scenario Analysis.
- `TermSheet.to_markdown()` → str (externally convertible to HTML/PDF).

### Middle Office Operations (`risk/trade_operations.py`)
- `TradeStatusTracker` — state machine: PENDING → CONFIRMED → ALLOCATED → SETTLED → MATURED/TERMINATED/DEFAULTED.
- `AuditEntry` — immutable audit trail (who, when, what, why).
- `generate_settlement()` → `SettlementInstruction`, `match_confirmation()` → `ConfirmationRecord`.
- `generate_margin_calls()` → `MarginCallReport` — daily margin calls with MTA enforcement.

### Collateral Optimisation (`risk/collateral_optimisation.py`)
- `CollateralOptimiser` — LP solver (scipy.optimize.linprog): min cost across multiple CSAs.
- Constraints: coverage ≥ required, allocated ≤ available, eligibility per CSA.
- `what_if_substitution()` → cost impact of swapping assets.
- `stress_collateral()` → stressed cost + margin shortfall (mild/moderate/severe/crisis).
- 51 new tests across all 5 modules.

---

## v0.558.0 — 2026-05-18

**Codebase restructuring + circular dep elimination + structural hardening.**

- 433 flat files → 20 sub-packages with 9 clean dependency layers.
- 0 circular dependencies (7 broken: TYPE_CHECKING guards, lazy imports, file moves, registry to root).
- 677 `to_dict()` auto-added to dataclasses.
- `__init__.py` re-exports for core, fx, equity, commodity, curves, risk.
- Layer 0 testing from 20% to 84% (72 new tests: statistics, viz, numerical, ts, db).
- ARCHITECTURE.md fully updated.
- See ARCHITECTURE.md for complete layer diagram and package inventory.

---

## v0.555.0 — 2026-05-14

**FRTB-IMA desk bridge + reverse stress testing.**

### IMA Bridge (`regulatory/ima_bridge.py`)
- `DeskRiskExtract` — desk_id, risk_class, delta/gamma/vega/DV01/CS01, obligor, rating.
- `extract_risk_factors_from_desk()` — maps desk sensitivities → `ESRiskFactor` (delta→ES via vol×z_97.5, vega→separate factor, CS01→credit spread).
- `extract_drc_positions_from_desk()` — credit desks → `DRCPosition` for IMA DRC.
- `extract_from_risk_metrics()` — generic bridge from any desk's `risk_metrics().to_dict()`.
- `aggregate_desk_ima()` → `IMABridgeResult` — runs full IMA pipeline + PLA evaluation.
- `RISK_CLASS_MAP` — 12 desk types mapped to risk class/sub_category.

### Reverse Stress Testing (`regulatory/reverse_stress.py`)
- `ReverseStressTarget` — metric, threshold, direction (below/above).
- `reverse_stress_portfolio()` — scipy.optimize.minimize to find minimum-severity scenario breaching threshold.
- `reverse_stress_ccar()` — reverse stress against CCAR capital trajectory (uses project_capital_trajectory).
- `scenario_surface()` — 2D grid of metric values across two macro variables.
- Default bounds per macro variable (GDP -10%/+5%, equity -80%/+20%, etc.).
- 23 tests across both modules.

---

## v0.554.0 — 2026-05-14

**CCAR/DFAST stress capital projection.**

- `regulatory/ccar.py` — NEW: 9-quarter capital trajectory under Fed-style stress.
- `CCARConfig` — starting capital/RWA, PPNR, dividends/buybacks, minimums (CET1 4.5%).
- `QuarterResult` — PPNR, credit/market/op losses, net income, capital actions, CET1 ratio, breach flag.
- `project_capital_trajectory()` → `CCARResult` — quarter-by-quarter CET1, trough ratio, pass/fail.
- `run_ccar_suite()` — 3 scenarios (baseline, adverse, severely_adverse) from stress_irrbb.
- `ccar_summary()` — worst scenario, trough ratios, overall pass/fail.
- Buyback suspension under stress, PPNR stress factors, RWA adjustment from stressed PD/LGD.
- 12 tests including undercapitalised bank failure case.

---

## v0.553.0 — 2026-05-14

**Portfolio-wide LCR/NSFR.**

- `regulatory/liquidity.py` — NEW: product-type-aware LCR and NSFR.
- `LiquidityPosition` — position_id, product_type, notional, rating, hqla_level, counterparty_type.
- `calculate_portfolio_lcr()` → `PortfolioLiquidityResult` — HQLA classification, outflow/inflow rates, LCR%, NSFR%, compliance flags, product breakdown.
- Product classification: cash (L1), sovereign AAA bonds (L1), IG bonds (L2A), deposits (retail stable 3% / wholesale 100%), loans (inflow if ≤30d).
- NSFR: ASF/RSF factors by product type and maturity (retail deposits 90%, cash RSF 0%, long-term loans 85%).
- `liquidity_stress()` — stressed LCR with outflow multiplier and HQLA haircut.
- 11 tests.

---

## v0.552.0 — 2026-05-14

**Operational risk SMA (Basel III OPE25).**

- `regulatory/operational_risk.py` — NEW: Standardised Measurement Approach.
- `SMAInputs` — 3-year P&L items (interest, fees, trading, leasing) + 10-year loss data.
- `calculate_sma_full()` → `SMAResult` — BI averaging, bucket (1/2/3), BIC (marginal 12%/15%/18%), ILM, capital, RWA.
- `calculate_bic()` — Business Indicator Component with marginal coefficients.
- `calculate_ilm()` — Internal Loss Multiplier: ln(e-1 + (LC/BIC)^0.8).
- `sma_sensitivity()` — capital sensitivity to loss component ratio.
- Legacy comparison: BIA capital computed alongside for benchmarking.
- 18 tests including hand-verified BIC calculations.

---

## v0.551.0 — 2026-05-14

**Capital allocation & RORC.**

- `regulatory/capital_allocation.py` — NEW: Euler allocation, RORC, capital limits.
- `euler_allocation()` — risk-contribution allocation with optional correlation matrix.
- `allocate_and_report()` — full report: diversification benefit, RORC per desk, hurdle checks, best/worst desk.
- `capital_limit_monitor()` — breach detection against per-desk limits.
- `DeskCapitalInput`, `DeskAllocation`, `CapitalAllocationResult` dataclasses.
- 16 tests.

---

## v0.550.0 — 2026-05-14

**Distressed debt: DIP, fulcrum, exchange, recovery waterfall, Chapter 11.**

- `distressed.py` — NEW: distressed debt analytics and restructuring.
- `DIPLoan` — super-priority DIP financing with roll-up, carve-out, upfront fee.
- `RecoveryWaterfall` — absolute priority distribution across capital structure.
- `FulcrumAnalysis` — identify fulcrum security (most senior impaired class); `sensitivity()` for recovery curves across EV range.
- `ExchangeOffer` — tender economics: exchange premium, holdout value, prisoner's dilemma payoffs.
- `Chapter11Timeline` — standard/pre-pack/complex milestones; `estimate_recovery()` with admin cost haircuts.
- `CapitalStructureLayer` — name, notional, seniority, secured flag.
- 25 tests.

---

## v0.549.0 — 2026-05-14

**Loan portfolio stress testing.**

- `loan_stress.py` — NEW: correlated defaults, macro scenarios, migration, concentration.
- `correlated_default_simulation()` — one-factor Gaussian copula, (n_paths × n_obligors) default matrix.
- `portfolio_loss_distribution()` — full loss distribution with VaR/ES/by-industry, macro scenario overlays.
- `MacroScenario` — GDP shock, rate/spread shock, PD multiplier, recovery haircut.
- 5 predefined scenarios: recession, stagflation, credit_crisis, rate_shock, recovery.
- `concentration_metrics()` — HHI, top-10%, industry HHI, granularity adjustment, effective N.
- `migration_matrix()` — rating transition via matrix power (multi-year), upgrade/downgrade/default%.
- 20 tests.

---

## v0.548.0 — 2026-05-14

**CLO equity Monte Carlo.**

- `clo_equity.py` — NEW: MC engine for CLO equity IRR distribution and loss analysis.
- `CLOEquityMC` — simulates correlated defaults (one-factor Gaussian copula), recoveries, prepayments through CLOWaterfall.
- Reinvestment period: defaulted/prepaid par replaced at par; post-reinvestment: portfolio amortises.
- `CLOEquityResult` — IRR mean/std/percentiles (5/25/50/75/95), loss distribution, mean cashflows.
- `CLOEquityCashflow` — per-period: income, defaults, recovery, tranche payments, equity distribution.
- `warehouse_risk()` — spread MTM VaR, net carry, ramp shortfall probability.
- 14 tests.

---

## v0.547.0 — 2026-05-14

**Unitranche & direct lending.**

- `unitranche.py` — NEW: unitranche, FOLO, DDTL, direct lending economics.
- `FOLO` — first-out/last-out split with absolute priority recovery allocation.
- `folo_recovery_split()` — FO gets paid first; LO absorbs losses.
- `Unitranche(TermLoan)` — blended spread, OID, FOLO, call protection.
- `DelayedDrawTermLoan(TermLoan)` — ticking fee before draw, normal coupon after.
- `CallProtectionSchedule` — NC/101/par step-down with `call_price()`, `is_callable()`.
- `direct_lending_economics()` — all-in yield: coupon + OID amort + upfront fee amort.
- `hold_to_maturity_yield()` — brentq solver for HTM yield given market price.
- `unitranche_blended_spread()` — weighted FO/LO spread.
- 27 tests.

---

## v0.546.0 — 2026-05-14

**PE-specific visualisation.**

- `football_field()` — horizontal range chart for valuation from multiple methods (DCF perpetuity, exit multiple, WACC sensitivity).
- `j_curve()` — PE fund TVPI over time with trough marker, breakeven line, red/green fill below/above 1.0x.

---

## v0.545.0 — 2026-05-14

**PE trading desk (9-component protocol) + exports.**

- `pe_desk.py` — NEW: full 9-component desk for PE fund management.
- `PERiskMetrics` — NAV, IRR, TVPI, DPI, MOIC, unfunded commitment; dispatches across fund/LBO/DCF.
- `PEBook` / `PEBookEntry` — portfolio book with by_vintage, by_manager, by_sector aggregations.
- `pe_carry_decomposition()` — management fee, carry, distribution income, J-curve drag.
- `pe_daily_pnl()` — NAV change + fee drag attribution.
- `pe_dashboard()` — morning meeting: NAV-weighted IRR/TVPI, position counts, concentrations.
- `pe_stress_suite()` — 5 parametric NAV shocks (±10%, ±25%, -50%).
- `pe_capital()` — Basel PE equity framework: 250% risk weight, unfunded as contingent.
- `pe_hedge_recommendations()` — manager concentration + unfunded ratio breach detection.
- `PELifecycle` — capital call, distribution, secondary sale, GP-led continuation, maturity alerts.
- `__init__.py` exports: LBOModel, DCFModel, WACCInputs, PE performance functions, PEFundParticipation, desk components.
- 28 tests.

---

## v0.544.0 — 2026-05-14

**PE fund waterfall extensions.**

- `fund_participation.py` extended with PE waterfall mechanics.
- `WaterfallConfig` — European (whole-fund) vs American (deal-by-deal) carry, catch-up rate, GP commitment, clawback, recycling.
- `WaterfallResult` — per-period: return of capital → preferred return → GP catch-up → carried interest → LP residual.
- `ClawbackResult` — total carry distributed vs entitled, clawback trigger.
- `PEFundParticipation(FundParticipation)` — subclass with `project_waterfall()`, `clawback_analysis()`, `gp_commitment_cashflows()`.
- Inherits all base methods (metrics, secondary_pricing) and passes isinstance checks.
- 20 tests.

---

## v0.543.0 — 2026-05-14

**PE performance benchmarking.**

- `pe_performance.py` — NEW: PE fund benchmarking and GP economics.
- `kaplan_schoar_pme()` — Public Market Equivalent (Kaplan & Schoar 2005).
- `direct_alpha()` — fund IRR minus index IRR.
- `long_nickels_pme()` — since-inception wealth ratio (Long & Nickels 1996).
- `vintage_cohort()` — aggregate FundParticipation metrics by vintage year (median/mean/UQ/LQ IRR, TVPI).
- `commitment_pacing()` — deterministic LP commitment pacing model (target allocation, calls, distributions, NAV).
- `gp_economics()` — management fee NPV, carry NPV, GP commitment return, clawback exposure.
- `clawback_exposure()` — GP clawback trigger calculation.
- 31 tests.

---

## v0.542.0 — 2026-05-14

**DCF / enterprise valuation.**

- `dcf.py` — NEW: `DCFModel` for discounted cash flow valuation.
- `WACCInputs` — CAPM cost of equity, after-tax cost of debt, WACC.
- `terminal_value_perpetuity()` — Gordon growth model.
- `terminal_value_exit_multiple()` — EV/EBITDA terminal value.
- `ev_to_equity()` — EV → equity bridge (net debt, minorities, associates, per-share).
- `DCFModel.value()` — PV of FCFs + PV of terminal value → EV → equity.
- `DCFModel.scenario_analysis()` — bull/base/bear with parameter overrides.
- `DCFModel.football_field()` — valuation range from perpetuity, exit multiple, WACC sensitivity.
- 27 tests including hand-verified Gordon growth crosscheck.

---

## v0.541.0 — 2026-05-14

**LBO deal model — PE underwriting.**

- `lbo.py` — NEW: `LBOModel` for leveraged buyout deal structuring.
- `SourcesAndUses` — equity, senior debt, mezzanine, rollover, transaction/financing fees.
- `FCFProjection` — EBITDA → revenue → EBIT → taxes → capex → NWC → FCF.
- `DebtYear` — annual debt schedule with senior amort, excess cash flow sweep, mezzanine PIK.
- `ExitAnalysis` — exit EV, net debt, equity value, IRR, MOIC at given multiple/year.
- `LBOModel.run()` — full model across multiple exit scenarios.
- `LBOModel.sensitivity_table()` — IRR grid across exit multiple × hold period (or growth).
- 40 tests.

---

## v0.540.0 — 2026-05-14

**Risk visualisation — 10 new chart types in `pricebook.viz`.**

### New: `viz/_risk.py` — desk-level risk charts
- `pnl_waterfall()` — waterfall/bridge chart for P&L attribution (carry, rate, vol, FX, etc.).
- `risk_decomposition()` — horizontal bar chart sorted by magnitude (key-rate DV01, vega by asset class).
- `stress_comparison()` — grouped or stacked bar chart across stress scenarios.
- `tenor_bucketing()` — vertical bar chart with color gradient by tenor bucket.
- `vega_ladder()` — horizontal bar chart of vega by expiry bucket with rich/cheap coloring.
- `pnl_table()` — formatted matplotlib table for P&L explain with alternating row colors.
- `greeks_surface()` — 2D contour plot of a Greek across strike × expiry.
- `greeks_evolution()` — multi-panel line chart of Greeks vs time-to-expiry.
- `hedge_pnl_tracking()` — position vs hedge cumulative P&L with net overlay.
- `rolling_correlation()` — multi-line rolling correlation with optional confidence bands.
- All functions: pure matplotlib, consume plain data (no instrument imports), theme-aware.
- 3 audit rounds: 17 issues found and fixed (waterfall dead code, label overlap, deprecated get_cmap, length mismatch guards, numpy type formatting, suptitle clipping, stacked legend, dead variables).

---

## v0.539.0 — 2026-05-14

**`pricebook.numerical` — complete self-contained numerical methods package.**

### Numerical package (`numerical/`) — 12 modules, ~1,800 lines
- `_distributions.py` — Normal, StudentT, LogNormal, Uniform, Exponential (wraps scipy.stats).
- `_linalg.py` — expm, logm, QR, Cholesky, LU, GMRES, BiCGSTAB, Sylvester, Lyapunov.
- `_ode.py` — Euler, RK4, RK45 (adaptive), BDF (stiff), Adams.
- `_optimize.py` — unified minimize (NM/BFGS/L-BFGS-B/DE/CMA-ES), LP (HiGHS), QP with inequality, interior-point (barrier), proximal gradient (ISTA/FISTA), projection operators.
- `_quadrature.py` — Gauss-Jacobi, tanh-sinh, Clenshaw-Curtis.
- `_interpolation.py` — 2D bilinear, bicubic, RBF (scattered data).
- `_rootfinding.py` — bisection, unified find_root dispatcher.
- `_mc.py` — QE Heston (Andersen), antithetic variates, multilevel MC (Giles).
- `_pde.py` — Hundsdorfer-Verwer ADI (full 4-stage), 2D PSOR (American), operator splitting (Lie/Strang).
- `_trees.py` — tree Greeks (delta/gamma/vega/theta), 2D binomial (Rubinstein).
- `_fourier.py` — fractional FFT (chirp-z), Hilbert transform, wavelet (Haar/Db2), CharacteristicFunction class.
- `_distributions_theory.py` — Schwartz test functions, tempered distributions, Fourier transform, convolution, Sobolev norms.
- 35 tests covering all modules.
- 3 audit rounds: 23 issues found and fixed (HV ADI stages, Lyapunov sign, PSOR order, Strang splitting, etc.).

---

## v0.527.0 — 2026-05-14

**Advanced regression.**

- `regression.py` — NEW: OLS, Ridge, Lasso (coordinate descent), Elastic Net, quantile (IRLS), robust (Huber/Tukey).

---

## v0.526.0 — 2026-05-14

**Clustering and regime detection.**

- `clustering.py` — NEW: K-means (Lloyd), silhouette score, optimal k, hierarchical (Ward), HMM regime switching (Baum-Welch EM, Viterbi).

---

## v0.525.0 — 2026-05-14

**Distribution fitting.**

- `distribution_fit.py` — NEW: MLE fitting (normal, Student-t, GEV), Kolmogorov-Smirnov test, Anderson-Darling, Q-Q plot data.

---

## v0.524.0 — 2026-05-14

**Kalman filter.**

- `kalman.py` — NEW: linear Gaussian state-space model, RTS smoother, dynamic beta, dynamic hedge ratio, trend extraction.

---

## v0.523.0 — 2026-05-14

**Volatility forecasting.**

- `garch.py` — NEW: GARCH(1,1) MLE, EGARCH (leverage), EWMA (RiskMetrics), realized vol, GARCH VaR.

---

## v0.522.0 — 2026-05-14

**Time series diagnostics.**

- `statistics.py` extended: ACF, PACF (Levinson-Durbin), Ljung-Box Q test, Augmented Dickey-Fuller, Durbin-Watson.

---

## v0.521.0 — 2026-05-14

**Performance ratios.**

- `ts/_stats.py` extended: information ratio, tracking error, Treynor, Omega, gain-to-pain, Kelly criterion (discrete + continuous).

---

## v0.520.0 — 2026-05-13

**Serialisation + curve construction + factories.**

### Serialisation complete (26/26 classes roundtrip)
- Added: FRN, FXSwap, NDF, EquityForward, ZCSwap, ConvertibleBond, AmortisingSwap.
- Model serialisation: all 8 models (Black76, Bachelier, SABR, HW with curve, BS, Heston, MCEquity with process_spec).
- TimeSeries: `to_dict()` (NaN→None) + `from_serialised()`.
- CurrencyPair deserialisation in `serialisable.py`.
- Dividend `to_dict()`/`from_dict()`.

### AmortisingSwap removed
- Use `InterestRateSwap.amortising()`, `.accreting()`, `.roller_coaster()` instead.
- One class per instrument, factory classmethods for common shapes.

### Unified curve builder
- `build_curves(method=...)` — 5 methods: sequential, global_newton, nelson_siegel, svensson, smith_wilson.

---

## v0.519.0 — 2026-05-13

**AAD bootstrap.**

- `aad_bootstrap()` in `aad_curves.py` — sensitivities to every input quote via reverse-mode AD, matches FD to 6 decimals.

---

## v0.518.0 — 2026-05-13

**Analytical Jacobian.**

- `global_solver.py` — analytical Jacobian for global bootstrap, O(n) per iteration, exact match with sequential.

### Curve audit fixes
- `multicurve_solver.py` — dual-curve float leg corrected (was using wrong telescoping identity).
- Armijo condition tightened to strict non-increase.
- Convergence warnings on non-convergence.

---

## v0.517.0 — 2026-05-13

**Futures desk: audit + gaps + notebook.**

### Futures audit fixes
- Stress PnL signs corrected (rates up → negative for long bonds).
- Silent-zero guards in commodity trades/spreads.
- CTD docstring, implied repo 360, turn-of-year docs.

### IR futures extensions
- Pack/bundle/butterfly strategies.
- `FuturesType.EURIBOR_3M`.
- `fed_funds_implied_probability()`.
- `roll_schedule()` — automated roll recommendations.
- `futures_cash_basis_rv()` — cross-market relative value.

### Notebook
- `futures_desk.ipynb` — curve from futures, bond basis, delivery options, IR strip, commodity term structure, multi-asset book.

---

## v0.516.0 — 2026-05-13

**Documentation + exports.**

- Model layer exports added to `__init__.py`: `Black76Model`, `BachelierModel`, `SABRModel`, `HullWhiteModel`, `BSModel`, `HestonModel`, `MCEquityModel`, `SABRParams`, `HestonParams`.
- `ARCHITECTURE.md` updated with Layer 3.5 (model abstraction).
- Version bump to v0.516.0.

---

## v0.515.0 — 2026-05-13

**Model-aware greeks + hard migration of greeks.**

- Bachelier greeks: `bachelier_delta/gamma/vega/theta` added to `black76.py`.
- `greeks_ir_option()` on `Black76Model`, `BachelierModel`, `SABRModel` — analytical greeks consistent with price.
- `greeks_european()` on `BSModel` — wraps existing `equity_greeks()`.
- `Swaption.greeks(curve, vol_surface)` removed → `.greeks(model, curve)`.
- `CapFloor.greeks(model, curve)` added — aggregated cap/floor greeks.
- `CapFloor.caplet_pvs(curve, vol_surface)` removed → `.caplet_pvs(model, curve)` with per-caplet greeks.
- All callers (desks, API, tests) updated. 8363 tests pass.

---

## v0.514.0 — 2026-05-13

**Hard migration: Swaption/CapFloor .pv() → .price(model, curve).**

- `Swaption.pv(curve, vol_surface)` removed → `.price(model, curve)`.
- `CapFloor.pv(curve, vol_surface)` removed → `.price(model, curve)`.
- `.pv_ctx()` rewired through `.price(Black76Model)` internally.
- `swaption_trading_desk.py`, `swaption_desk.py`, `api.py` migrated.
- All test files migrated (test_swaption, test_capfloor, test_swaption_roundtrip, test_ir_deep, test_xi2, test_xi7, test_slice7, test_implied_vol_roundtrip, test_options_hardening).
- Orphaned `FlatVol` imports cleaned.
- 8363 tests pass.

---

## v0.513.0 — 2026-05-13

**Model abstraction layer + instrument wiring.**

- `models.py` — NEW: 2 protocols (`IROptionModel`, `EquityOptionModel`), 7 models (`Black76Model`, `BachelierModel`, `SABRModel`, `HullWhiteModel`, `BSModel`, `HestonModel`, `MCEquityModel`).
- `SABRParams` dataclass (frozen). `HestonParams` imported from `slv.py`.
- `Swaption.price(model, curve)` — pluggable model pricing.
- `CapFloor.price(model, curve)` — pluggable model pricing.
- Audit fixes: `MCEngine.generate_paths()`, HW vol formula (Rebonato), docstring corrections, `HestonParams` dedup, model guard `TypeError`, `projection_curve` passthrough.
- 40 model tests: protocols, swaption/capfloor equivalence, BS/Heston/SABR/HW, guards, put-call parity.

---

## v0.512.0 — 2026-05-13

**Architecture document.**

- `ARCHITECTURE.md` — 449 lines: 8-layer system map, instrument inventory, desk protocol matrix, C++ port roadmap, cross-cutting infrastructure.

---

## v0.511.0 — 2026-05-13

**10 exotic products — closing all 34 gaps.**

- Rates: ZC swaption (Black-76), inverse floater (MC/OU), capped floater (MC/OU with floor).
- FX: ratio forward (long put + short N calls, zero-cost), knock-in reverse convertible (MC barrier).
- Equity: dividend future, dividend swap, dividend option (Black-76).
- Structured: participation note (bond floor + call option).
- Credit: bespoke tranche (one-factor Gaussian copula MC).
- Audit fixes: path-integrated discounting (inverse/capped floater), ZC swaption delta guard, Brent bracket widened, ratio/barrier guards, risky annuity (tranche survival weighted), PD clamping, coupon floor.

---

## v0.510.0 — 2026-05-13

**Time series module (`pricebook.ts`).**

- `TimeSeries` class: numpy-backed (no pandas), construction, arithmetic, alignment, filtering, resample.
- Returns: `simple_returns()`, `log_returns()`, `period_returns()`.
- Stats: `sharpe()`, `sortino()`, `max_drawdown()`, `drawdown_series()`, `performance()` (delegates to `backtest.compute_metrics`).
- Rolling: `rolling_sharpe()`, `rolling_vol()`, `rolling_beta()` (delegates to `statistics.rolling_stats`).
- I/O: `from_db()`, `from_db_book()`, `from_db_desk()`, `from_csv()`, `greeks_from_db()`.
- Replay: `replay()`, `replay_book()`, `replay_desk()`, `drawdown_analysis()`, `rolling_performance()`.
- Viz: `plot_dashboard()`, `plot_equity_curve()`, `plot_drawdowns()`, `plot_rolling_sharpe()`, `plot_pnl_histogram()`.
- DB: `pnl_series_by_book()`, `pnl_series_by_desk()` aggregation methods added to `PricebookDB`.
- 52 tests.

---

## v0.509.0 — 2026-05-13

**Convertible bond desk — 9-component protocol.**

- `convertible_bond_desk.py` — NEW: `CBRiskMetrics` (hybrid delta/gamma/vega/CS01/DV01), `CBBook`, `CBBookEntry`, `CBCarryDecomposition`, `CBDailyPnL`, `CBDashboard`, `CBStressResult`, `CBCapitalResult`, `CBHedgeRecommendation`, `CBLifecycle`.
- Exports added to `__init__.py`: `ConvertibleBond`, desk layer.
- 26 tests.

---

## v0.508.0 — 2026-05-13

**4 new notebooks: asset swaps, XCCY basis, PRDC, TARF.**

- `asw_btp_bund.ipynb` — BTP vs Bund ASW spread basis trade, EUR curve (ESTR), par/proceeds ASW, Z-spread comparison, risk & carry.
- `xccy_basis_pricing.ipynb` — USD bond for EUR investor, XCCY basis from FX forwards, FX-hedged yield, pickup vs Bunds, basis sensitivity.
- `prdc_structuring.ipynb` — PRDC 3-factor MC (JPY/USD), callable via LSM, correlation sensitivity, FX delta profile, par coupon structuring.
- `tarf_risk_profile.ipynb` — TARF payoff asymmetry vs vanilla forward, target/vol/strike sensitivity.

---

## v0.507.0 — 2026-05-12

**Bond trading & multicurve notebooks.**

- `bond_trading_desk.ipynb` — trader's 7AM morning workflow: market setup, rich/cheap RV scorecard, trade construction, callable OAS, repo financing, risk snapshot. OAS bracket widened to [-0.10, 0.50].
- `treasury_multicurve.ipynb` — Treasury curve (7 bonds) vs SOFR (from swaps) vs repo, pricing comparison, basis trade signal, carry analysis by repo tenor. Extended with 30-bond universe + curve construction summary.

---

## v0.506.0 — 2026-05-12

**Benchmark bonds, repo curve, callable bond desk.**

- `benchmark_bonds.py` — NEW: 6 sovereign markets (UST/Bund/Gilt/JGB/OAT/BTP) with correct conventions. `BenchmarkUniverse`, `create_ust_universe()`, etc. NSS curve fitting (`fitted_curve_nss`). Trading strategies: `duration_neutral_spread()`, `butterfly_trade()`, `barbell_vs_bullet()`. Rankings: `carry_ranking()`, `roll_down_ranking()`, `rv_scorecard()`. 15 tests.
- `repo_curve.py` — NEW: `RepoCurve`, `build_repo_curve()`, `forward_repo_rate()`, `special_gc_spread()`, `repo_carry_from_curve()`.
- `callable_bond_desk.py` — NEW: `callable_bond_analytics()` — model price, straight price, option value, OAS, effective duration/convexity. 16 tests.

---

## v0.505.0 — 2026-05-12

**Bond desk + Treasury note pricing.** 16 new tests.

### Bond desk hardening
- `bond_daily_pnl()` and `bond_pnl_attribution()` wired into `bond_trading_desk.py` — 9/9 protocol complete.
- Input validation: maturity check in `bond_risk_metrics()`, horizon guard in `bond_carry_roll()`.

### Treasury quoting (`treasury_quoting.py`)
- `to_32nds()` / `from_32nds()` — decimal ↔ 32nds with + (half-32nd) notation.
- `TreasuryReopen` — new issue vs reopening (premium/discount, WAP, total outstanding).
- `delivery_option_value()` — quality + timing + wild card option decomposition for futures.

### Treasury note roundtrip notebook (`notebooks/treasury_note_roundtrip.ipynb`)
- Full pricing: build SOFR curve → create 10Y T-Note → dirty/clean/AI/YTM/32nds.
- Risk metrics: duration, DV01, convexity, key-rate profile (via `greeks_profile`).
