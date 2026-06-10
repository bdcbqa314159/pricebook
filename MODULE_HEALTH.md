# Pricebook Module Health Audit — v0.865.0

**Adversarial dual-lens audit of 35 foundation + Top-6-instrument modules. RAW critic output, NOT a verified bug list — see confidence note below.**

Generated: 2026-06-10  •  Scope: core / numerical / models / curves + 6 instruments  •  697 findings

## Quick stats

| Metric | Count |
|---|---:|
| Modules audited | 35 |
| Critic verdicts | 70 (35 × 2 lenses) |
| Total findings | 697 |
| `critical` | 56 |
| `high` | 150 |
| `medium` | 257 |
| `low` | 217 |
| `nit` | 17 |
| **Tier-1 double-confirmed criticals** (both critics → critical) | **13** |
| **Tier-2 double-confirmed near-criticals** (critical + high pairing) | **18** |
| Single-critic criticals (one critic only) | 19 |
| Modules with no findings at all | 0 |

## How to read this report

Adversarial critics over-report by design. A `critical` tag means *one critic believes this would produce a wrong price/Greek*. Findings fall into roughly three buckets:

- **True bugs** — what we want to find. Tier-1 (both critics flagged critical) is the highest confidence.
- **Accepted approximations** — already documented in `reference_approximations.md` (CDS convexity stencil, deep-ITM bounds, etc.).
- **False positives** — the critic missed validation upstream, or the cited code path is unreachable.

**Triage workflow per finding:**
1. Read the cited location + the critic's reasoning.
2. Write a failing test that would catch the alleged bug. If you can write it AND it fails → real bug, fix it.
3. If you can't reproduce, or it's an accepted approximation → document why and dismiss.
4. Each confirmed real bug becomes one slice (failing test + fix + commit).

---

## Tier 1 — both critics flagged CRITICAL (13)

Both the numerical critic AND the code-correctness critic independently called the same issue `critical`. These are the **highest confidence** real bugs and the first to verify.

### T1.1 `numerical/_mc.py` — multilevel_mc telescoping collapses to zero for European payoffs (coarse path = sub-sampled fine path)

**Numerical critic:**
- Location: `_mc.py:171-184 (multilevel_mc fine/coarse coupling)`
- Lines 175-181 compute paths_coarse = paths_fine[:, ::2] and then evaluate the payoff on it. For any payoff that depends only on the terminal value (vanilla European calls/puts), paths_fine[:, -1] equals paths_coarse[:, -1] by construction, so P_fine - P_coarse is identically 0 across all paths. The telescoping sum E[P_L] = E[P_0] + sum_l E[P_l - P_{l-1}] therefore collapses to E[P_0], i.e. the estimator returned is just the coarsest Euler price with base_steps=4 — exactly the level whose bias MLMC is supposed to remove. The reported variance contribution from levels >= 1 is also identically 0, which will look 'good' but is meaningless. Even for path-dependent payoffs the coupling is wrong: in Giles MLMC the level-(l-1) path must be simulated with the aggregated Brownian increments (sum of 
- **Fix:** Re-architect process_fn to expose the underlying normals so the coarse path can be re-simulated by summing pairs of fine increments (standard Giles coupling). Concretely: process_fn(n_paths, n_steps, seed) should return both the path and the normals it consumed; in the MLMC level loop, draw normals of shape (n_paths, n_fine), build paths_fine from them, then build paths_coarse from normals_coarse 

**Code-correctness critic:**
- Location: `multilevel_mc, lines 158-184`
- In `multilevel_mc`, level 0 generates paths via `process_fn(n_paths, base_steps, seed)`. At level l>=1, the fine path uses `n_fine = base_steps * 2**l` and the coarse path is `paths_fine[:, ::2]` (every other sample of the fine path). The MLMC telescoping identity E[P_L] = E[P_0] + sum_l E[P_l - P_{l-1}] requires that at level 1 the 'coarse' path inside the correction has the SAME distribution as the standalone level-0 estimator. For Euler/Milstein/QE schemes, stride-2 subsampling of a fine simulation does NOT produce the same distribution as `process_fn` run with `base_steps` (the discretisation error is different). Therefore E[P_coarse_in_correction] != E[P_0], and the estimator has a bias equal to the difference in discretisation error between coarse-from-fine and fresh-coarse at every 
- **Fix:** Run `process_fn` twice per fine level with coupled Brownian increments: simulate fine, then aggregate increments pairwise (sum) to drive the same SDE at the coarse step size. Concretely, expose a coupled-simulation interface where the coarse path is produced step-by-step using the SAME (summed) increments as the fine path, not by subsampling fine path values. Alternatively, document that this rout

---

### T1.2 `numerical/_fourier.py` — density() calls np.trapz, removed in NumPy 2.x — function crashes

**Numerical critic:**
- Location: `python/pricebook/numerical/_fourier.py:233`
- Line 233 uses np.trapz, which was deprecated in NumPy 1.20 and removed in NumPy 2.0. On the installed environment (numpy 2.4.3) any call to CharacteristicFunction.density(...) raises AttributeError immediately. There are no tests exercising density(), which is why this slipped through. Aside from that, u_max=50.0 and n_quad=200 are hard-coded magic constants — du=0.25 is too coarse for narrow densities (e.g. small sigma, short T) and u_max=50 truncates slowly-decaying CFs (jump models, low vol). Result clamping via np.maximum(density, 0) silently hides the negative-density artefacts that signal under-resolution rather than reporting them.
- **Fix:** Replace np.trapz with np.trapezoid (NumPy 2.x name) or a Simpson's rule. Derive u_max from CF tail decay (e.g. require |phi(u_max)| < tol) and choose du from x-grid spacing (Nyquist). When clamping at zero, also raise/log a warning if the L1 mass of the negative tail exceeds a small fraction (e.g. 1e-3) so that under-resolution does not pass silently.

**Code-correctness critic:**
- Location: `_fourier.py:233`
- `density[k] = np.trapz(integrand, dx=du) / math.pi` at line 233 uses np.trapz, which was removed from the public namespace in NumPy 2.0 (renamed to np.trapezoid). The repo's installed numpy is 2.4.3, where `np.trapz` raises AttributeError on access. Verified by direct repro: calling CharacteristicFunction(...).density(np.array([0.0,0.1,0.2])) raises `AttributeError: module 'numpy' has no attribute 'trapz'. Did you mean: 'trace'?`. Any caller of `CharacteristicFunction.density` gets a hard crash; the function is effectively dead.
- **Fix:** Replace `np.trapz(integrand, dx=du)` with `np.trapezoid(integrand, dx=du)` (NumPy >=1.22 has this alias; required in NumPy >=2.0).

---

### T1.3 `numerical/_integrate.py` — integrate_2d swaps x and y vs docstring

**Numerical critic:**
- Location: `python/pricebook/numerical/_integrate.py:103-129`
- scipy.integrate.dblquad requires func(y, x) where y is the inner variable, but the docstring says 'f(x, y) -> float'. The code passes the user-supplied f directly to dblquad with no argument swap. Empirical check: integrate_2d(lambda x, y: x, (0, 3), (0, 1)) returns 1.5; with the documented convention the answer is 4.5. The same swap also affects the y_range(x) callable path because the inner-variable role is reversed. Any non-symmetric 2D integrand in pricebook will price wrong (e.g., spread / basket integrals where x and y play distinct roles such as one being a stock and the other a rate).
- **Fix:** Either change the docstring/contract to 'f(y, x)' and document inner/outer variables, or wrap with `g = lambda y, x: f(x, y)` before calling dblquad. The same convention must be applied consistently when y_range is a callable of x.

**Code-correctness critic:**
- Location: `python/pricebook/numerical/_integrate.py:103-129 (integrate_2d)`
- Docstring (line 110) advertises `f(x, y)`. The implementation passes `f` straight to `scipy.integrate.dblquad`, whose documented contract is `func(y, x)` — y is the first argument, x the second. For any integrand that is not symmetric in its two args (e.g. `f(x, y) = x` on `[0,1] x [0,2]`) the result is wrong: caller expects 1.0, dblquad returns 2.0 (verified by direct invocation). Also note y_lo/y_hi take `x` per dblquad — but because the *first* argument fed to gfun/hfun is dblquad's outer integration variable (x), that part is consistent; only the integrand axis is flipped.
- **Fix:** Wrap: `wrapped = lambda y, x: f(x, y)` and pass `wrapped` to `dblquad`. Verify with a non-symmetric test such as `f(x, y) = x` on `[0,1] x [0,2]` returning 1.0.

---

### T1.4 `numerical/_optimize.py` — interior_point silently drops equality constraints

**Numerical critic:**
- Location: `python/pricebook/numerical/_optimize.py:250-256`
- Lines 250-253 build a `constraints` list from `equality_constraints` but the subsequent `_minimize` call on line 255 uses `method="BFGS"`, which is an unconstrained method that ignores the `constraints` keyword (and `constraints` is in fact not passed). So any user calling `interior_point(..., equality_constraints=[h])` gets an unconstrained barrier solve and the equality constraints are silently ignored. The output `x` will not satisfy h(x)=0.
- **Fix:** Either (a) raise NotImplementedError if equality_constraints is non-empty, (b) switch the inner solve to SLSQP/trust-constr and pass `constraints=constraints`, or (c) handle equalities via a separate Lagrangian / projection step. Do not silently accept and ignore them.

**Code-correctness critic:**
- Location: `_optimize.py:218-256 (interior_point)`
- Lines 250-256 build a `constraints` list from `equality_constraints` but never pass it to `_minimize`. Additionally, the solver is `method="BFGS"`, which is unconstrained and does not accept a `constraints` argument. Any caller passing equality_constraints gets a result that does NOT satisfy those constraints — the equality constraints are completely silently dropped. This is a wrong-result bug for a documented public API.
- **Fix:** Either (a) reject equality_constraints with NotImplementedError, (b) switch to method="SLSQP" and pass `constraints=constraints`, or (c) add equality constraints to the barrier via a penalty / augmented-Lagrangian term. Document the choice.

---

### T1.5 `numerical/_trees.py` — Tian V formula is wrong; method silently degrades to CRR

**Numerical critic:**
- Location: `_trees.py:110-121 (_tian_params)`
- In `_tian_params`, V is defined as `M**2 * (math.exp(vol**2 * dt) - 1)`. Tian (1993) defines V = exp(sigma^2 * dt) (not M^2*(V-1)). With the correct V, V>1 always so V^2+2V-3>0 and the moment-matching tree is well-defined. With the code's definition, V ~ M^2*sigma^2*dt is tiny for typical inputs (e.g. sigma=0.25, T=1, N=200 -> V ~ 6e-4), so `disc_arg = V^2+2V-3` ~ -3 < 0 on every step, and the `if disc_arg < 0` branch silently falls back to CRR. The user's choice of `TreeMethod.TIAN` produces CRR numbers without warning.
- **Fix:** Use V = math.exp(vol**2 * dt). Then u = 0.5*M*V*(V + 1 + sqrt(V^2+2V-3)), d = 0.5*M*V*(V + 1 - sqrt(V^2+2V-3)), p = (M-d)/(u-d). Remove the silent CRR fallback or replace with a hard error if disc_arg < 0 (which should not occur under the corrected formula). Add a unit test that solves a European call with Tian and CRR at large N and checks they agree to ~1e-4, not identically.

**Code-correctness critic:**
- Location: `_trees.py:372-377 (_apply_barrier)`
- Lines 372-377: knock-in branches contain only `pass` with comment 'complex — for now, only knock-out supported'. A user calls `TreeSolver(barrier_type=BarrierType.DOWN_IN, barrier_level=80)` and gets the vanilla European price back with no warning. There is no way for the caller to detect that the requested barrier was ignored — the result has no error field. This is a silent wrong-result for a publicly documented feature.
- **Fix:** Raise NotImplementedError("knock-in barriers not yet supported; use in-out parity: knock_in = vanilla - knock_out") instead of silently returning vanilla. Or implement via in-out parity automatically.

---

### T1.6 `numerical/_trees.py` — Discrete dividends applied to terminal spot grid, ignoring step timing

**Numerical critic:**
- Location: `_trees.py:241-244 (and absence in _solve_trinomial)`
- In `_solve_binomial`, the loop `for step, amount in self.dividends.items(): if step <= n: S = np.maximum(S - amount, 0.01)` subtracts every dividend from the *terminal* spot vector S, regardless of the ex-dividend step. This is not a valid escrowed-dividend treatment (which would subtract PV(dividends) from S0 at t=0 and rebuild the tree on the deflated spot) nor a node-by-node dividend reduction during backward induction. It is closer to subtracting the sum of nominal dividend amounts from S(T), which mis-prices both Europeans and Americans (early exercise around ex-dates is completely missed). The floor of 0.01 also silently masks the case where the dividend exceeds the spot. The trinomial path doesn't apply dividends at all.
- **Fix:** Either (a) implement escrowed-dividend Black-Scholes-style adjustment: replace spot with spot - sum(PV(div_i)) for div steps <= n and tree on that, accurate only for Europeans; or (b) apply per-step shifts during backward induction by subtracting amount from S_step at the dividend step before evaluating exercise. Option (b) is required for American/Bermudan. Add a test against an analytical Europe

**Code-correctness critic:**
- Location: `_trees.py:241-244 (_solve_binomial)`
- Lines 241-244: `for step, amount in self.dividends.items(): if step <= n: S = np.maximum(S - amount, 0.01)`. The terminal spot tree is uniformly reduced by EVERY dividend amount whose step <= n, regardless of when in time that dividend is supposed to occur. A dividend at step=1 is treated identically to a dividend at step=n. Worse, no dividend logic appears anywhere in the backward induction; dividends are never applied at the actual ex-date. The variable `step` is unused except as a flag of inclusion. With dividends=[(1, 2.0), (50, 3.0)] on a 100-step tree, the user gets terminal payoffs computed on (spot * d^(n-j) * u^j) - 5.0 clamped at 0.01, which is mathematically meaningless. Any caller using `dividends=` gets a wrong price.
- **Fix:** Apply dividends inside backward induction: at each step that matches a dividend step, shift the spot tree by the dividend amount before computing intrinsic / continuation. Document the dividend convention (cash vs proportional) explicitly. Trinomial path should mirror this.

---

### T1.7 `models/mc_engine.py` — scheme="milstein" silently runs Euler (copy-paste bug)

**Numerical critic:**
- Location: `python/pricebook/models/mc_engine.py:212`
- Line 212: `step_fn = euler_step if self.scheme == "euler" else euler_step`. Both branches return euler_step. The milstein_step_1d function defined above is never invoked by the engine regardless of what the user passes for `scheme`. Any user relying on Milstein for higher-order weak convergence (e.g., CIR-like 1D processes without an exact_step) gets standard Euler discretisation error O(sqrt(dt)) instead of O(dt), with no warning. The docstring still advertises milstein as a supported scheme.
- **Fix:** Replace with `step_fn = euler_step if self.scheme == "euler" else milstein_step_1d`, and either thread a diffusion_deriv through the engine or raise NotImplementedError if scheme=="milstein" without one. Add a unit test that compares Milstein vs Euler convergence rates on GBM (Milstein should converge weakly at O(dt), not O(sqrt(dt))).

**Code-correctness critic:**
- Location: `mc_engine.py:212`
- Line 212: `step_fn = euler_step if self.scheme == "euler" else euler_step`. Both branches return `euler_step`, so requesting `scheme="milstein"` silently runs Euler with no warning. The Milstein step function exists (`milstein_step_1d`, line 125) but is never wired in. Users specifying Milstein will get convergence/bias matching Euler and no indication anything is wrong. This is a wrong-result bug on any realistic input where the user chose Milstein for its higher weak/strong order.
- **Fix:** Use `step_fn = milstein_step_1d if self.scheme == "milstein" else euler_step` and either accept a `diffusion_deriv` argument on MCEngine or raise on unknown scheme strings.

---

### T1.8 `models/cos_method.py` — V_k integration bounds wrong when a > 0 (call) or b < 0 (put) — deep ITM mispricing

**Numerical critic:**
- Location: `python/pricebook/models/cos_method.py:92-99`
- Lines 93-99 use chi(0, b) - psi(0, b) for calls and -chi(a, 0) + psi(a, 0) for puts unconditionally. The Fang-Oosterlee derivation gives these formulas only under the implicit assumption a ≤ 0 ≤ b (i.e., the payoff kink at y=0 lies inside the truncation interval). For deep ITM options the truncation [a,b] = [x + c1 - L*sqrt(c2), x + c1 + L*sqrt(c2)] is shifted by x = log(S/K), and when |x| > L*sqrt(c2) the kink at 0 falls outside [a,b]. The correct V_k integrates the payoff over [a, b] (since the entire interval is in-the-money). Empirical evidence: BS call S/K=50, vol=20%, T=1: COS=4910.67 vs BS=4904.88, rel err 1.18e-3, does NOT converge as N → ∞ (same error at N=64, 256, 1024). At S/K=10000 the error is 1.85%. Deep ITM put S=10, K=1000: COS=1346.60 vs BS=941.23, 43% error. Replacing the
- **Fix:** For call: lo = max(a, 0.0); V_k = 2/(b-a) * (_chi(k, a, b, lo, b) - _psi(k, a, b, lo, b)). For put: hi = min(b, 0.0); V_k = 2/(b-a) * (-_chi(k, a, b, a, hi) + _psi(k, a, b, a, hi)). Add a regression test at S/K ∈ {0.01, 100} that fails on current code and passes after the fix.

**Code-correctness critic:**
- Location: `cos_method.py:92-99 (V_k construction inside `cos_price`)`
- For the CALL branch `_chi(k,a,b,0,b) - _psi(k,a,b,0,b)` integrates the call payoff over [0,b] regardless of whether `a > 0` (in which case the call is in-the-money throughout [a,b] and the lower limit should be `a`, not 0) or `b < 0` (call is OTM throughout and the value should be 0). Symmetric problem for the PUT branch on [a,0]. The Fang & Oosterlee formula assumes 0 lies within [a,b]; when the cumulant-based truncation interval drifts off zero (high |log(S/K)|, short T, or low vol), the formula evaluates an integral with reversed/wrong limits and yields nonsense. Verified empirically: spot=100, K=200, T=0.01, vol=20%, rate=0, L=10 yields a CALL of 11.78 vs. true BS=0; spot=1, K=1000, T=0.01, vol=10% yields 8811 vs. true ~0; spot=1000, K=1 PUT yields 1121 vs. ~0. These are realistic stre
- **Fix:** Clamp the V_k integration limits to [a,b]. For CALL use `_chi(k,a,b, max(a,0.0), b) - _psi(k,a,b, max(a,0.0), b)` when `b>0` else 0; for PUT use `-_chi(k,a,b, a, min(b,0.0)) + _psi(k,a,b, a, min(b,0.0))` when `a<0` else 0. (Equivalently, special-case `b<=0` for CALL and `a>=0` for PUT to return 0 immediately.)

---

### T1.9 `models/hull_white.py` — Swaption uses r0 instead of alpha(T_expiry) for short rate at expiry nodes

**Numerical critic:**
- Location: `python/pricebook/models/hull_white.py:164-175`
- In `tree_european_swaption`, line 173: `r_j = r0 + j * dr` where `r0 = f(0,0)` is the initial instantaneous forward. The model's short rate at expiry node j is r(T_expiry) = alpha(T_expiry) + j*dr, NOT r(0) + j*dr. alpha is the curve-consistent drift accumulated across all tree steps and absorbs the entire forward-rate slope from 0 to T_expiry. By using r0 you are evaluating the analytical bond prices P(T_expiry, t_pay, r_j) at a short rate that systematically lags the model's true short rate at expiry whenever the curve is not flat. For an upward-sloping curve this depresses r_j by ~ (f(T_expiry) - f(0)), pushing bond prices up and skewing the (1 - p_end - K*annuity) payoff. For a 5Y expiry under a 100bp/yr slope, the error in r_j is ~5%, which translates into bond-price errors of B*5% on
- **Fix:** Return the final `alpha` from `_evolve_state_prices` (along with Q, dr, j_max) and use `r_j = alpha + j * dr` in the swaption. Equivalently, persist an alpha array over all steps. Cross-check against Jamshidian's closed-form HW swaption price; the two must agree to MC tolerance.

**Code-correctness critic:**
- Location: `hull_white.py:164-173`
- Line 173: `r_j = r0 + j * dr` where `r0 = self._forward_rate(0.0)`. The actual short rate at node j at time `expiry_T` is `alpha(expiry_T) + j*dr`, where `alpha` is the per-step calibration parameter computed inside `_evolve_state_prices`. But `_evolve_state_prices` only returns `r0`, not the final `alpha`. Using r0 means the rate axis at expiry is centred on the initial short rate rather than on the forward rate at expiry, so the analytical bond prices `p_end` and `annuity` computed via `zcb_price(expiry_T, ..., r_j)` are evaluated at the wrong short-rate locations. For any non-flat curve and any non-trivial expiry_T this yields a wrong swaption price.
- **Fix:** Return alpha (or the alpha trajectory) from `_evolve_state_prices` and use `r_j = alpha(expiry_T) + j*dr` at expiry. Alternatively, evolve the swaption payoff backward on the tree from expiry to 0 using the actual node rates (avoiding the issue entirely).

---

### T1.10 `curves/global_solver.py` — Residual index collision when two instruments share a maturity

**Numerical critic:**
- Location: `python/pricebook/curves/global_solver.py:46-92`
- global_bootstrap (line 80, 92) and coupled_bootstrap (line 245, 256, 277) both write residuals via res[pillar_idx[mat]] = ... (assignment). pillar_dates is the deduplicated union of deposit and swap maturities, and the swap-collection code at line 55 explicitly anticipates a swap maturity already being a deposit pillar ('if mat not in pillar_dates'). When such a collision occurs, the swap residual overwrites the deposit residual (loops iterate in deposit-then-swap order), so the deposit price constraint is silently dropped. The Newton solve then has n equations but only n-k independent constraints (k = collisions), producing a singular or under-determined Jacobian and a wrong curve at the colliding pillar. Conversely, if two swaps share a maturity (e.g., one calendrical pillar reached by t
- **Fix:** Either (a) raise on collisions explicitly, or (b) build a residual vector indexed by instrument number (not by pillar), and a separate map from instruments to which pillar(s) they constrain. Then the Jacobian must be square only after explicit collision detection.

**Code-correctness critic:**
- Location: `python/pricebook/curves/global_solver.py:74-92 (single curve), 242-256 (coupled)`
- In `global_bootstrap._residuals`, every instrument writes to `res[pillar_idx[mat]]`. If two instruments share a maturity (e.g. a 1Y deposit and a 1Y swap, or two swaps at the same node, or duplicate deposits), only the last write survives — the earlier equation is silently dropped. Because the deposit loop appends to `all_instruments` before swaps, a deposit residual is replaced by the swap residual at that pillar. The same overwrite happens in `_jacobian_analytical` (`J[row, ...]` writes), turning what should be an over-determined or contradictory system into an under-determined one. The Jacobian's affected row will also tend to be singular (missing the deposit's direct df_i term), so Newton silently converges to a curve that does NOT reprice all input instruments. The same pattern affect
- **Fix:** Either (a) detect duplicates up front and raise ValueError, or (b) assign each instrument its own residual row independent of pillar identity, and add an explicit pillar->row mapping (extra rows pinned by a regularisation/interpolation constraint when there are more instruments than pillars).

---

### T1.11 `curves/multicurve_solver.py` — Projection swap PV_float drops the first period entirely

**Numerical critic:**
- Location: `multicurve_solver.py:123-131`
- In _reprice_errors for projection instruments, dates_up_to = [pillars <= maturity] and the floating-leg PV loop is `for j in range(1, len(dates_up_to))`. When the projection swap is a single-period instrument (e.g. the shortest pillar = first swap maturity, the standard bootstrap shape), len(dates_up_to)==1 and the loop body never executes, so pv_float stays at 0. The implied model_rate is therefore 0, the residual is approximately equal to the input rate (~4% for a 4% swap), and the Newton iteration cannot drive it to zero by adjusting any DFs because pv_float has no functional dependence on x. For multi-period projection swaps, the very first period (from reference_date to the first pillar) is still dropped — typically the highest-weight cash flow — biasing the implied forward downward b
- **Fix:** Either prepend reference_date to dates_up_to before the loop (so j=1 covers ref->first_pillar), or start with prev=reference_date and iterate over all of dates_up_to as a forward-period generator analogous to _compute_annuity. The PV_float schedule must exactly match the annuity schedule.

**Code-correctness critic:**
- Location: `multicurve_solver.py:125-131 (the `for j in range(1, len(dates_up_to))` loop in `_reprice_errors`, projection branch)`
- In `_reprice_errors`, the float-leg PV is built as `for j in range(1, len(dates_up_to)): d_start = dates_up_to[j-1]; d_end = dates_up_to[j]; ...`. The first segment of the schedule — from `reference_date` to `dates_up_to[0]` — is never included in `pv_float`. However, `_compute_annuity(ois, dates_up_to, day_count)` *does* include that first period (it initialises `prev = reference_date`). So for every projection swap whose `dates_up_to` has length >= 2, `model_rate = pv_float / annuity` uses N-1 forward periods over N annuity periods. For typical pillar layouts this is a systematic bias that makes the solver converge to wrong DFs. For a single-pillar instrument (`len(dates_up_to)==1`) the forward loop does nothing at all, `pv_float==0`, and `model_rate==0`, so the error is just `-inst['rat
- **Fix:** Either include the first period explicitly (`d_start = reference_date` for j=0, using a synthetic fixing on the proj curve from ref_date to dates_up_to[0]) or change the loop to iterate over (prev, d) pairs starting from `prev = reference_date`, mirroring `_compute_annuity`.

---

### T1.12 `curves/multicurve_solver.py` — Projection swap PV_float and annuity use inconsistent schedules

**Numerical critic:**
- Location: `multicurve_solver.py:123-135`
- _compute_annuity(ois, dates_up_to, ...) uses prev=reference_date as the starting accrual point, so it accumulates one term per element in dates_up_to (including the first ref->pillar period). The PV_float loop on lines 125-131 starts at j=1 and only emits len(dates_up_to)-1 terms. The two legs of the swap therefore disagree on the number of periods and on the schedule by exactly one period, regardless of tenor. Even after fixing finding #1, this asymmetry must be eliminated or the par rate will be wrong by the contribution of the missing/extra coupon.
- **Fix:** Use a single helper that yields (d_start, d_end, df_start_proj, df_end_proj, df_end_ois) and have both PV_float and annuity be built from the same iterator.

**Code-correctness critic:**
- Location: `multicurve_solver.py:125-131 (the `for j in range(1, len(dates_up_to))` loop in `_reprice_errors`, projection branch)`
- In `_reprice_errors`, the float-leg PV is built as `for j in range(1, len(dates_up_to)): d_start = dates_up_to[j-1]; d_end = dates_up_to[j]; ...`. The first segment of the schedule — from `reference_date` to `dates_up_to[0]` — is never included in `pv_float`. However, `_compute_annuity(ois, dates_up_to, day_count)` *does* include that first period (it initialises `prev = reference_date`). So for every projection swap whose `dates_up_to` has length >= 2, `model_rate = pv_float / annuity` uses N-1 forward periods over N annuity periods. For typical pillar layouts this is a systematic bias that makes the solver converge to wrong DFs. For a single-pillar instrument (`len(dates_up_to)==1`) the forward loop does nothing at all, `pv_float==0`, and `model_rate==0`, so the error is just `-inst['rat
- **Fix:** Either include the first period explicitly (`d_start = reference_date` for j=0, using a synthetic fixing on the proj curve from ref_date to dates_up_to[0]) or change the loop to iterate over (prev, d) pairs starting from `prev = reference_date`, mirroring `_compute_annuity`.

---

### T1.13 `curves/aad_curves.py` — Swap bootstrap silently flat-extrapolates intermediate coupon DFs

**Numerical critic:**
- Location: `aad_curves.py:186-195 (with aad_interp.py:58-59 as the silent extrapolation source)`
- In `aad_bootstrap`, the temporary curve `temp_curve` is built from already-bootstrapped pillars only (deposits + earlier swaps). For each subsequent swap, intermediate coupon DFs are read via `temp_curve.df(schedule[k])`. But `aad_log_linear_interp` clamps to `ys[-1] * 1.0` when `x >= xs[-1]` (aad_interp.py:58-59), so every coupon date beyond the last existing pillar receives the last pillar's DF as a constant. In a realistic deposit (≤1Y) → 5Y swap bootstrap, coupons 2..10 (i.e. t = 1.0 .. 4.5) all use the same 1Y deposit DF. The bootstrap is therefore not a real par-swap solve; it is a one-shot algebraic plug with grossly wrong intermediate annuity weights, and the bias grows with maturity. For a 5Y swap at 4% par on a 1Y-deposit base, this typically gives ~tens of bp error in the inferr
- **Fix:** Replace the single-pass solve with a Newton / fixed-point iteration: build a candidate curve including a tentative df_mat, evaluate the par-swap residual, and iterate until convergence (typically 3-5 Newton steps). Alternatively, use a direct solve for the par-rate equation on the log-DF at maturity. As a minimum guard, raise loudly when any `schedule[k]` for k in range(1, len-1) lies past `pillar

**Code-correctness critic:**
- Location: `aad_curves.py:178-189`
- In `aad_bootstrap`, when `deposit_quotes` is empty, `pillar_dates` is empty after Phase 1, so `temp_curve` is set to `None`. In the swap loop, every interior coupon DF is then taken as `Number(1.0)` via `df_k = temp_curve.df(schedule[k]) if temp_curve else Number(1.0)`. For a swap longer than one schedule period (i.e. anything semi-annual longer than ~6 months), the annuity is computed with df=1 at every interior coupon, which is not the correct discount factor and produces an incorrect bootstrapped `df_mat`. This is silently wrong on a realistic input (the canonical 'bootstrap from swaps only' scenario, common in shorter-tenor curves or when deposits are bypassed).
- **Fix:** After computing the first swap pillar (which is unambiguous from its own formula or by treating it as the deposit-equivalent), construct a `temp_curve` from the pillars built so far for every subsequent swap. Alternatively, raise when `temp_curve is None` and the schedule has more than two dates, instead of silently substituting 1.

---

## Tier 2 — critical + high pairing (18)

One critic called `critical`, the other called the same area `high`. Probably real but slightly lower confidence than Tier 1.

| # | Module | Numerical (sev) | Code-correctness (sev) | Title |
|---|---|---|---|---|
| 1 | `core/discount_curve.py` | critical | high | roll_down forgets to renormalize DFs by P(0, new_ref) |
| 2 | `numerical/_pde.py` | critical | high | LOG grid produces ~17% overprice on ATM call |
| 3 | `numerical/_pde.py` | high | critical | Implicit Dirichlet boundary not properly enforced in tridiagonal solve |
| 4 | `numerical/_pde.py` | high | critical | American option boundary uses European discounted strike |
| 5 | `numerical/_fourier.py` | high | critical | wavelet_transform crashes on non-power-of-2 input lengths |
| 6 | `numerical/_integrate.py` | high | critical | _romberg is dead — scipy.integrate.romberg removed in SciPy 1.15 |
| 7 | `numerical/_optimize.py` | high | critical | interior_point: equality-only problem never converges |
| 8 | `numerical/_trees.py` | high | critical | Knock-in barriers silently priced as vanilla |
| 9 | `numerical/_trees.py` | high | critical | Trinomial probability clamp does not re-normalise; risk-neutral measure broken u |
| 10 | `models/mc_engine.py` | high | critical | greek() ignores param_name and bumps every state variable jointly |
| 11 | `models/g2pp_calibration.py` | high | critical | Bare `except Exception: return 0.0` in swaption pricer silently masks numerical  |
| 12 | `models/g2pp_calibration.py` | high | critical | Silent fallback `y_star = 0.0` when bracketing fails produces wrong strikes down |
| 13 | `curves/aad_curves.py` | high | critical | First swap with no deposits assumes DF=1 over the entire pre-maturity period |
| 14 | `fixed_income/bond.py` | high | critical | `_price_from_ytm` uses first-period notional as redemption; disagrees with curve |
| 15 | `options/swaption.py` | critical | high | SABR-HW blender returns 0.0 at T=0 even with positive intrinsic |
| 16 | `credit/cds.py` | critical | high | Convexity PnL multiplies by \|PV\| instead of notional |
| 17 | `credit/cds.py` | high | critical | Variable-notional schedule silently dropped in isda_upfront / roll_down / theta  |
| 18 | `credit/cds.py` | high | critical | protection_leg_pv silently uses only notional[0] when schedule_dates omitted |

Detail: see full JSON; or grep the raw output by title.

## Single-critic criticals — 19

Flagged by only ONE critic, not corroborated. These deserve spot-check but the base-rate of real-bug-vs-false-positive is genuinely uncertain. Skim through, mark suspicious ones for verification.

| # | Module | Lens | Title | Location |
|---|---|---|---|---|
| 1 | `core/survival_curve.py` | code-correctness | Round-trip serialization drops user-supplied pillar at reference_date | `survival_curve.py:215 (_sc_to_dict) interacting with __init_` |
| 2 | `core/pricing_context.py` | code-correctness | from_dict passes None for empty container fields, breaking subsequent accessors | `python/pricebook/core/pricing_context.py:242, 247, 253-254` |
| 3 | `core/pricing_context.py` | code-correctness | to_dict / from_dict silently drop multi-currency and many other fields | `python/pricebook/core/pricing_context.py:216-234, 237-254` |
| 4 | `numerical/_pde.py` | numerical | SINH grid goes negative when concentration_point ≠ midpoint | `_pde.py:107-112` |
| 5 | `numerical/_fourier.py` | numerical | Skewness has wrong sign (c3 stencil) | `python/pricebook/numerical/_fourier.py:211` |
| 6 | `numerical/_fourier.py` | numerical | Excess-kurtosis stencil is catastrophically unstable (h=1e-4 -> h^4 at machine epsilon) | `python/pricebook/numerical/_fourier.py:202,215` |
| 7 | `numerical/_integrate.py` | numerical | Clenshaw-Curtis weights are wrong for odd n | `python/pricebook/numerical/_integrate.py:237-254` |
| 8 | `numerical/_trees.py` | code-correctness | solve_tree_2d always returns zero Greeks | `_trees.py:557-558 (solve_tree_2d)` |
| 9 | `numerical/_trees.py` | code-correctness | solve_tree_2d American exercise only handles 'spread_call' | `_trees.py:548-554 (solve_tree_2d)` |
| 10 | `numerical/_trees.py` | code-correctness | solve_tree convenience wrapper drops exercise_dates → Bermudan silently becomes European | `_trees.py:470-491 (solve_tree)` |
| 11 | `models/mc_engine.py` | code-correctness | greek() ignores param_name argument | `mc_engine.py:289-326` |
| 12 | `models/cos_method.py` | numerical | c2 cumulant floor of 0.001 destroys low-variance pricing | `python/pricebook/models/cos_method.py:79` |
| 13 | `models/hull_white.py` | numerical | Swaption hard-codes annual payments and integer-year tenor | `python/pricebook/models/hull_white.py:176-183` |
| 14 | `models/g2pp_calibration.py` | numerical | Swaption pricer does not implement Brigo-Mercurio eq. 4.31 — inner term is the uncondition | `python/pricebook/models/g2pp_calibration.py:215-273` |
| 15 | `models/g2pp_calibration.py` | numerical | Wrong measure for x integration — risk-neutral N(0, var_x) instead of T-forward N(M_x(t),  | `python/pricebook/models/g2pp_calibration.py:207-217` |
| 16 | `models/lmm.py` | numerical | Rebonato swaption vol uses uncorrelated (rho=0) sum instead of standard rho=1 approximatio | `lmm.py:228 (LMM.rebonato_swaption_vol)` |
| 17 | `curves/bootstrap.py` | numerical | Hull-White futures convexity formula is wrong — missing T1 factor | `python/pricebook/curves/bootstrap.py:110-116 and :274-276` |
| 18 | `curves/bootstrap.py` | code-correctness | FRA/future with no deposits silently uses df_start = 1.0 | `bootstrap.py:87-94 and 125-132` |
| 19 | `curves/multicurve_solver.py` | code-correctness | Annuity truncates at last pillar <= maturity while df_T is interpolated to maturity | `multicurve_solver.py:111-115 and 123,133` |

## Per-module summary

Ranked by total finding count (`crit/high` weighted x10/x3 to surface risky modules).

| Module | Crit | High | Med | Low | Nit | Risk score |
|---|---:|---:|---:|---:|---:|---:|
| `numerical/_trees.py` | 8 | 9 | 10 | 7 | 0 | 117.7 |
| `numerical/_pde.py` | 3 | 10 | 12 | 7 | 0 | 72.7 |
| `models/g2pp_calibration.py` | 4 | 6 | 12 | 8 | 0 | 70.8 |
| `models/mc_engine.py` | 4 | 6 | 10 | 7 | 0 | 68.7 |
| `numerical/_optimize.py` | 2 | 11 | 12 | 5 | 0 | 65.5 |
| `curves/multicurve_solver.py` | 4 | 5 | 8 | 5 | 1 | 63.5 |
| `numerical/_integrate.py` | 4 | 4 | 5 | 5 | 0 | 57.5 |
| `models/hull_white.py` | 3 | 6 | 8 | 7 | 0 | 56.7 |
| `numerical/_fourier.py` | 4 | 2 | 6 | 4 | 0 | 52.4 |
| `curves/global_solver.py` | 2 | 6 | 8 | 6 | 0 | 46.6 |
| `credit/cds.py` | 2 | 5 | 9 | 7 | 0 | 44.7 |
| `curves/aad_curves.py` | 2 | 6 | 6 | 6 | 0 | 44.6 |
| `options/swaption.py` | 1 | 8 | 8 | 5 | 0 | 42.5 |
| `curves/bootstrap.py` | 2 | 4 | 9 | 7 | 1 | 41.7 |
| `models/cos_method.py` | 3 | 2 | 5 | 5 | 0 | 41.5 |
| `numerical/_mc.py` | 2 | 3 | 6 | 7 | 0 | 35.7 |
| `numerical/auto_diff.py` | 0 | 9 | 8 | 7 | 1 | 35.7 |
| `core/pricing_context.py` | 2 | 3 | 5 | 8 | 1 | 34.8 |
| `models/lmm.py` | 1 | 4 | 6 | 5 | 1 | 28.5 |
| `core/survival_curve.py` | 1 | 3 | 7 | 7 | 0 | 26.7 |
| `fixed_income/bond.py` | 1 | 3 | 6 | 7 | 0 | 25.7 |
| `fixed_income/swap.py` | 0 | 5 | 8 | 7 | 0 | 23.7 |
| `core/serialisable.py` | 0 | 4 | 10 | 8 | 0 | 22.8 |
| `core/discount_curve.py` | 1 | 1 | 7 | 7 | 1 | 20.7 |
| `curves/smith_wilson.py` | 0 | 3 | 9 | 5 | 1 | 18.5 |
| `options/equity_option.py` | 0 | 5 | 2 | 4 | 1 | 17.4 |
| `core/calendar.py` | 0 | 3 | 6 | 9 | 1 | 15.9 |
| `numerical/_distributions.py` | 0 | 2 | 9 | 8 | 0 | 15.8 |
| `core/schedule.py` | 0 | 3 | 5 | 4 | 3 | 14.4 |
| `curves/nelson_siegel.py` | 0 | 2 | 8 | 3 | 1 | 14.3 |
| `fixed_income/fra.py` | 0 | 2 | 6 | 7 | 0 | 12.7 |
| `core/day_count.py` | 0 | 2 | 5 | 6 | 0 | 11.6 |
| `models/feynman_kac.py` | 0 | 1 | 7 | 7 | 1 | 10.7 |
| `core/trade.py` | 0 | 1 | 6 | 7 | 2 | 9.7 |
| `models/black76.py` | 0 | 1 | 3 | 3 | 1 | 6.3 |

## Per-module narrative verdicts

Both critics produced a paragraph-length overall verdict per module. Reproduced verbatim.

### `core/calendar.py`
**Numerical critic:**

> This is a calendar / business-day module — no numerical PDE/MC/closed-form formulas, so most of the 11 lenses (convergence, vol=0 limit, RNG seed, hand-calc on an ATM option) do not apply. The applicable lenses are correctness of date math, boundary handling at year/month ends, calibration to official statutes, and consistency between calendars. Most of the implementation is solid: Anonymous Gregorian Easter is verified against known dates (2020-2030), Orthodox Easter (Meeus Julian + 13-day offset) matches Wikipedia for 2020-2030, Victoria Day "Monday before May 25" is correct, Bogotá emiliani Mondayisation is correct, the JointCalendar year-boundary spill-back through cache[year]+cache[year+1] works. The main concrete bug is that LondonCalendar, AUDCalendar, NZDCalendar use the base-class _observe (Saturday→Friday, Sunday→Monday), but the UK Banking and Financial Dealings Act 1971, the AU Public Holidays Act, and the NZ Holidays Act 2003 all specify that a Christmas Day falling on Saturday is substituted to the FOLLOWING Monday, not the preceding Friday. This produces a wrong observed-holiday date once every ~6-7 years (e.g. 2021, 2027) and a knock-on missing-substitute date for Boxing Day. There are also several documented-as-approximate issues (Japan vernal/autumnal equinox hardcoded, Tel Aviv Hebrew calendar approximated by fixed Gregorian dates, Islamic / Chinese lunar holidays omitted) — these are openly disclaimed in the docstrings and are acceptable approximations rather than bugs. One minor robustness issue: _nth_weekday(year, m, w, 5) silently spills into the next month when the 5th occurrence doesn't exist.

**Code-correctness critic:**

> The module is well-structured and most boundary conditions are handled correctly (year-boundary spill in is_holiday, Boxing-Day collision when Dec 25 is Sunday, Victoria Day exclusivity, leap-year-safe nth-weekday). However, several real correctness issues exist: (1) `Calendar._observe` hard-codes US-style observation (Sat -> Fri, Sun -> Mon), which is then applied to LondonCalendar, AUDCalendar, NZDCalendar and CADCalendar where the actual convention is "next available working day" -- this produces wrong holiday dates whenever Jan 1, Boxing Day or Australia Day falls on a Saturday; (2) `is_holiday` only consults `d.year` and `d.year + 1`, which fails to find a holiday that was *originally* in year Y but observed forward to year Y+1 -- this case doesn't fire today because no calendar produces such a forward spill, but is fragile; (3) `JohannesburgCalendar._observe` overrides the static base, but every helper inside `_compute_holidays` is bound by self/method-resolution -- looks correct, however `JointCalendar` exposes a non-trivial race / aliasing on the shared `_holiday_cache`; (4) `_nth_weekday` has no validation for n=0, negative n, or n exceeding the number of weekdays in a month. Mathematical/observance correctness of EM calendars (Hebrew/Islamic/lunar) is explicitly out-of-scope per docstrings. Findings below.

### `core/day_count.py`
**Numerical critic:**

> The module is generally well-structured and most conventions (ACT/360, ACT/365F, 30E/360, ACT/ACT ISDA) check out against hand-calcs. However, there are several concerns: (1) the function labelled "30/360 ISDA 2006" actually implements the "30/360 US" / SIA Bond Basis variant — pure ISDA 2006 4.16(f) "Bond Basis" does NOT include the end-of-Feb adjustment; this is a convention/citation mismatch that could mispriced bonds depending on which convention the caller expected. (2) `business_days_between` uses (start, end] (exclusive start, inclusive end) which is the OPPOSITE of the standard Anbima/B3 convention for BRL instruments where DU(t,T) counts [t, T) — this directly affects NTN-F/LTN/DI futures pricing when start and end are not both business days. (3) `_act_act_icma` silently falls back to ACT/365F when ref dates are missing — a silent approximation that masks a misuse bug. (4) `_act_act_icma` does not handle irregular (long/short stub) periods per ICMA Rule 251.2, which requires splitting into notional regular periods. (5) `date_from_year_fraction` uses a fixed 365.25 regardless of convention, so it does not round-trip with any of the actual day-count conventions exposed.

**Code-correctness critic:**

> The module is generally sound. Day-count formulas handle leap years, ACT/ACT ISDA year-boundary splits, and 30/360 ISDA adjustment rules correctly. Found one real boundary bug (ZeroDivisionError on frequency=0 in ACT/ACT ICMA), and a handful of medium/low robustness issues: silent fallback to ACT/365 when ICMA reference fields are missing, silent clamping of negative year fractions to the reference date, unchecked calendar None for BUS/252 once start==end is bypassed (handled), and unhandled NaN/Inf in date_from_year_fraction. No off-by-one errors in business-day counting or year-boundary splitting were found.

### `core/discount_curve.py`
**Numerical critic:**

> The module is mostly sound for its core operations (df, zero_rate, forward_rate, flat construction, bumped) and the log-linear interpolation path produces correct results that match hand-calculation on a flat 5% curve. However, `roll_down` has a critical correctness bug: it builds the rolled curve using the original P(0, T_i) discount factors but anchors them to the new reference date, effectively forgetting to divide by P(0, new_ref). Under an unchanged 5% flat yield curve, a 1-day rolldown produces a zero rate of 5.014% instead of 5%, which is roughly 1.4bp per day of error and will misstate rolldown P&L. Secondary issues include misleading "numerically stable" docstring on `forward_rate`, serialization losing the interpolation method, silent zero-fallbacks masking bad inputs, missing bounds check in `bumped_at`, and minor inaccuracies in `zero_rate(t=0)` / `instantaneous_forward` for non-log-linear interpolation methods.

**Code-correctness critic:**

> The module is mostly well-defended (positive-DF check, strictly-increasing pillar check, copies of input lists, no mutable default args, t<=0 short-circuits in df/zero_rate). The main correctness issues I found are localised to roll_down/bumped paths and a couple of input-validation gaps: roll_down's "all pillars in the past" fallback silently drops the curve's day_count and interpolation by calling DiscountCurve.flat with defaults; bumped_at lacks pillar_idx range validation (and silently honours negative indices); instantaneous_forward can be called with arbitrarily large t and will raise OverflowError near date.max; and a couple of silent zero-returns mask pathological interpolator output. None of these are obviously catastrophic on normal pricing inputs, but the roll_down fallback in particular will produce a numerically wrong rolled curve whenever a non-default day count is in use.

### `core/pricing_context.py`
**Code-correctness critic:**

> The data class itself is straightforward, but the module has multiple confirmed correctness bugs concentrated in the serialisation layer (lines 210-258) and the `replace()` helper. After a round-trip through `to_dict`/`from_dict`, (a) empty container fields are returned to the constructor as `None` instead of empty dicts, which makes any subsequent accessor raise `TypeError` instead of the documented `KeyError`; (b) several fields declared on the dataclass (`discount_curves`, `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`) are silently dropped — multi-currency contexts therefore lose data. `replace()` shares mutable dicts with the parent context, breaking the "Immutable snapshot" contract. Additional smaller robustness issues exist around `fx_translate("")` and `fx_rate` with zero spot.

### `core/schedule.py`
**Code-correctness critic:**

> The schedule generator handles most boundaries (empty, single period, divides-evenly, short stubs) correctly, but it has a real EOM anchoring bug for front-stub generation: the docstring states EOM is anchored to `start`, but for SHORT_FRONT/LONG_FRONT the backward generator passes the current rolling date (initialised to `end`) into `_add_months`, so EOM is effectively anchored to `end` instead. This produces wrong roll dates when `start` is EOM but `end` is not (a common case for amortising/quarterly trades). There is also a documented-but-fragile `months * 30` heuristic that can misclassify stubs, an unused/unenforced `eom` for WEEKLY, and post-adjustment dates that are not deduped/sorted (adjacent rolls in dense stubs can collide or invert). The forward (back-stub) path is correct.

### `core/serialisable.py`
**Code-correctness critic:**

> The module has several correctness bugs around type dispatch order and the deserialisation path. Most impactful: `_deserialise_atom` only handles `list[date]`, silently returning raw dicts for `list[SomeSerialisable]` or any other parameterised list, so polymorphic list fields will not be reconstructed. Optional-unwrapping treats `Union[A, B, None]` (two non-None types) by returning the value as-is, bypassing reconstruction entirely. The registry's `if key not in _REGISTRY` makes re-imports a silent no-op, so a stale class survives module reload; this also has a small TOCTOU race in the absence of a lock. The numpy `.item()` duck-test runs before the `to_dict` check and can prematurely collapse any object with a callable `.item()` attribute. CurrencyPair string parsing assumes exactly one `/` and raises a confusing unpacking error on malformed data. `Serialisable.from_dict` does a bare `d['params']` lookup that will KeyError on malformed input. Enum deserialisation may fail for integer-valued enums when JSON delivers string ints. These are the real correctness concerns; the rest of the module is solid.

### `core/survival_curve.py`
**Numerical critic:**

> The core survival-curve algebra is sound: log-linear interpolation of Q(t) is mathematically equivalent to piecewise-constant hazard between pillars, and hazard_rate / forward_hazard / default_prob / marginal_default_density all reduce to the right closed forms (I hand-checked the flat 200bp case). However the per-pillar bump used to compute key-rate CS01 (`_bump_survival_curve_at`, surfaced as `SurvivalCurve.bumped_at`) is wrong: it shifts only Q_i and leaves Q_{i+1}, Q_{i+2}, ... unchanged, which contaminates the *next* segment's hazard by an off-target amount of opposite sign — this is a high-severity issue for any per-pillar credit risk report. Several smaller robustness issues exist: non-monotone survival inputs are silently accepted and only clipped to 0 inside `pillar_hazards`; the `len(dates) >= 1` constructor guard can lead to a single-point interpolator error when the lone pillar sits at the reference date; `term_structure` uses `timedelta(days=365)` rather than a day-count-consistent 1Y bump; and NaN survival probabilities pass the validator unchecked. No issue affects the flat-curve hand-calculation, but the bump bug is real and reproducible.

**Code-correctness critic:**

> The module has several real correctness bugs that fire on realistic inputs. The most serious is a round-trip serialization bug when a user provides a pillar at the reference date: to_dict silently drops that pillar's survival probability while keeping all dates, so from_dict raises a length-mismatch error. Related to that, the constructor allows a user-supplied pillar at t=0 with survival != 1, which produces a discontinuity at t=0 (survival(ref)=1.0 by early-return but survival(ref + 1 day) interpolates from the user's value). The hazard_rate / marginal_default_density methods compute hazards from raw pillars assuming log-linear interpolation, but survival() respects whatever interpolation method was selected — so when interpolation != LOG_LINEAR the two are mutually inconsistent. There is also no validation that input dates are strictly increasing or that supplied survival_probs are non-increasing; sorting violations later cause a confusing ValueError from the interpolator rather than from this class. forward_survival accepts d1 > d2 silently and returns a > 1 result, and default_prob behaviour across the reference_date silently truncates the interval to [ref, d2]. Several robustness issues but the t=0 round-trip bug and the hazard/interpolation inconsistency are the wrong-result bugs to fix first.

### `core/trade.py`
**Code-correctness critic:**

> The Trade/Portfolio module is small and mostly correct, but contains a real aliasing bug in Portfolio.__init__ (the caller's trades list is shared with the portfolio and mutated by add()), a falsy-replacement quirk (empty list passed in is silently replaced with a fresh one, breaking aliasing only in that path — inconsistent behaviour), a dead/broken to_dict on Trade that would crash if ever invoked before module-load completes (vars(self) returns a non-JSON-serialisable instrument), missing validation of `direction` (any int silently scales the PV), missing to_dict guard on the instrument during serialisation, and a pv() return-type inconsistency for an empty Portfolio. No critical wrong-result bugs in the pricing math itself.

### `credit/cds.py`
**Numerical critic:**

> The core pricing of protection_leg_pv, premium_leg_pv, risky_annuity, par_spread, and bootstrap is solid and matches standard ISDA/O'Kane conventions (ACT/360 premium leg, ACT/365 protection integration, mid-period accrued-on-default discounted to d_mid, weighted RPV01 in par_spread). The bootstrap brentq bracket is sensible and the round-trip check is appropriate. However there is a real wrong-result bug in cds_pnl_attribution where the convexity term is multiplied by |pv_t0| instead of notional even though spread_convexity is documented as "normalised by notional" — the two scales differ for any non-par trade, so the convexity attribution is wrong outside the par limit. There is also a systemic pattern of dropping the per-period notional schedule in derived methods: isda_upfront, roll_down, theta, rec01, and the aged-CDS construction inside cds_pnl_attribution all rebuild a CDS with the scalar self.notional (= first period notional), which silently misprices any amortising/accreting trade. protection_leg_pv has a hidden contract-violation path: passing a list notional without schedule_dates falls through to the scalar branch and uses only notional[0]. A few smaller concerns: pv_ctx silently swaps credit curves on KeyError, carry uses self.start (not today) as the accrual anchor, and the docstring sign on spread PnL (≈-CS01×Δs) contradicts the (correct) code.

**Code-correctness critic:**

> The core pricing functions (protection_leg_pv, premium_leg_pv) and the par CDS solver look defensively coded, but the CDS class drops the variable-notional schedule whenever it constructs a "child" CDS for greeks/PnL (theta, roll_down, rec01, isda_upfront, cds_pnl_attribution, spread_convexity divisor) — it consistently passes the scalar self.notional (= notional_schedule[0]) instead of the full schedule, so amortising/accreting CDS produce silently-wrong greeks and P&L. There is also a silent data-loss path in protection_leg_pv when a list notional is passed without schedule_dates (only notional[0] is used). pv_ctx can raise on bare except, average_notional/upfront have no runtime guard against zero (relying on constructor invariants), and the serialisation registration only persists the first notional. Reported below ordered by severity.

### `curves/aad_curves.py`
**Numerical critic:**

> The curve classes themselves (AADDiscountCurve / AADSurvivalCurve) are thin and reasonable: they prepend (0, 1.0) when needed, log-linear interpolate, and clamp short-dated queries. The serious problems live in `aad_bootstrap`. The swap bootstrap is non-iterative and queries a temp curve built only from previously-bootstrapped pillars to obtain the discount factors of intermediate coupon dates — but for any realistic deposit/swap term mix those intermediate coupons lie strictly beyond the last existing pillar, so the underlying `aad_log_linear_interp` flat-extrapolates and silently substitutes the last pillar DF. The result is a biased curve (not just imprecise — wrong in a way that scales with maturity) and the bias is invisible because no test exercises the bootstrap. Additional concerns: no domain guard on the resulting df_mat (it can go negative or zero, which the downstream `.log()` in log-linear interp will then blow up on, breaking the AAD tape), no spot-lag/settlement handling on deposits, swap_dc reused inconsistently across all pillars regardless of how they were quoted, no test coverage on the bootstrap at all. The interpolation/curve glue is solid; the bootstrap is not safe to use as written.

**Code-correctness critic:**

> The module has a clear correctness bug in the swap bootstrap: when there are no deposits, the temporary curve used to discount interior coupons is `None`, and every interior DF is replaced by `Number(1.0)`, silently producing a wrong bootstrapped curve for any swap longer than the first frequency period. There are several edge-case crashes and silent wrong-result paths: a single pillar at t=0 makes `df()` crash on interp validation, unsorted pillar dates are not rejected and break `bisect`, pillar dates before `reference_date` produce a non-monotone time axis without any guard, and duplicate maturities (deposit + swap on same date) cause a division by zero in the interpolator. The frequency for swap bootstrap is hard-coded SEMI_ANNUAL, which silently produces a wrong-rate bootstrap for any market that quotes annual swaps. A few robustness issues round out the list.

### `curves/bootstrap.py`
**Numerical critic:**

> The bootstrap structure is sound (deposit→FRA→futures→swap chaining, brentq solving, log-linear interpolation, round-trip verification), but the inlined Hull-White convexity adjustment for IR futures is mathematically wrong and disagrees with the canonical hw_convexity_adjustment that already exists in pricebook.fixed_income.ir_futures. The error scales like 1/T1 — it severely understates the spread for medium- and long-dated futures (5Y, 10Y) where convexity matters most. A second confirmed issue: in single-curve bootstrap, the float leg telescopes to (1 − df(T)) so float_day_count and float_frequency are silent no-ops despite the API advertising them. A handful of medium/low robustness gaps (no ordering checks on FRAs/futures; TOY filter rejects negative spreads; brentq bracket too narrow for steep negative rates at long tenors). The dual-curve variant is more honest because the float side does not telescope. Recommend: replace the inlined convexity block with a call to the canonical helper, document the float-convention no-op in single-curve mode, and tighten input validation.

**Code-correctness critic:**

> The module has several real correctness issues. The most damaging is silent `df_start = 1.0` fallback for FRAs/futures when no deposits are provided — this produces wrong discount factors for any FRA/future not starting at the reference date. Other significant issues: no validation that FRA/future start < end (already partial for FRAs but missing for futures), no validation that FRA/future end_dates strictly exceed last pillar (the DiscountCurve constructor will raise a cryptic error rather than the bootstrap reporting a clear ordering problem), turn-of-year spread silently ignored if negative (`> 0` guard), deposits with maturity == reference_date produce zero year fraction and a degenerate pillar that triggers ValueError in DiscountCurve, swap schedules pinned to `reference_date` (no separate effective date — wrong for T+2 spot-starting swaps), and brentq bracket `[1e-6, 3.0]` can fail to bracket the root for very high rates/long tenors with no recovery. The `_verify_forward_curve_round_trip` uses a different numerical form of the forward rate than `bootstrap_forward_curve`'s objective — same math, but the verify form is more sensitive to df near-equality and could spuriously trigger the 1e-6 warning.

### `curves/global_solver.py`
**Numerical critic:**

> global_solver.py implements a global Newton bootstrap (single-curve) and a coupled OIS+projection bootstrap via finite-difference Jacobian. The mathematical structure is right (deposit residual df = 1/(1+rτ); par swap residual fixed_pv - float_pv = 0), but several real bugs and robustness gaps exist. Most notably, the residual layout assumes one instrument per pillar — if a deposit and a swap (or any two instruments) share a maturity, the swap residual overwrites the deposit residual and one constraint is silently lost. The coupled solver also has hard-coded day-count conventions inconsistent with typical OIS fixed legs and a silent fwd=0 fallback when df2≤0 that hides curve failures. Newton iteration has no damping/line search and clamps DFs to 1e-10 silently, which can mask divergence. The analytical Jacobian's "between-pillar" branch is O(n²) per non-pillar coupon despite docstring claiming O(n).

**Code-correctness critic:**

> The solver has a real correctness defect when two instruments share a maturity: residuals and Jacobian rows are addressed by `pillar_idx[mat]` so a deposit and a swap (or any two instruments) with the same maturity stomp on one another's row — one equation is silently dropped and the system becomes under-determined / singular. The same pattern exists in `coupled_bootstrap`. Beyond that, there are several robustness gaps: empty input lists cause crashes inside numpy/DiscountCurve, the `fwd=0` fallback silently masks degenerate inputs, `dates >= reference_date` is not pre-validated, and the `LinAlgError` path silently exits without raising. Math/numerical concerns are deferred to the numerical critic.

### `curves/multicurve_solver.py`
**Numerical critic:**

> The module bundles a multi-curve Newton solver, a curve validator, and an "analytical" Jacobian. Auditing against the 11 lenses surfaces several correctness defects, the most serious being that the projection-swap PV_float is computed by a loop that drops the first period entirely — making the model rate badly wrong for every projection swap whose maturity is the first projection pillar (the typical bootstrap input), and partially wrong for longer swaps. End-to-end sanity testing confirms the solver does NOT converge on textbook inputs (residual stuck at ~4% regardless of pillar layout). Additional issues: damped Newton commits the step even when no improvement was achieved, the "analytical" Jacobian emits a phantom all-zero column for the t=0 pillar (and is in fact a finite-difference Jacobian despite its name), instruments exceeding the pillar count are silently dropped, and the projection-swap PV_float and annuity use inconsistent date ranges (PV drops the first period; annuity includes it). The validation routine and discount-factor monotonicity check are essentially correct. This module is not production-safe in its current state.

**Code-correctness critic:**

> The solver has several real correctness bugs. The most serious is in `_reprice_errors` for projection instruments: the floating-leg PV loop starts at `j=1`, which means the very first coupon period (reference_date → first pillar) is silently dropped from `pv_float`, but it IS included in the annuity — so model par rate is systematically biased low for every projection swap whose maturity covers more than one pillar. A second class of bugs comes from the assumption that there is exactly one instrument per pillar and that instrument maturities equal pillar dates: when those assumptions are violated (a) some rows of the error vector stay at zero, letting Newton "converge" on a rank-deficient system with arbitrary DFs, and (b) the annuity stops at the last pillar ≤ maturity while `df_T` is interpolated to `maturity`, producing inconsistent fixed-/discount-leg accounting. There are also smaller robustness issues: stale residual on non-convergence, unchecked NaN from `lstsq`, jacobian sized over the t=0 pillar (always-zero column), and a divide guard pattern that uses `max(annuity, 1e-10)` after an `if annuity > 0` test (the floor is dead code).

### `curves/nelson_siegel.py`
**Numerical critic:**

> The Nelson-Siegel / Svensson formulas themselves are implemented correctly and match the cited 1987/1994 references, including the correct limit at t=0 (y(0)=β0+β1) and the correct sign convention for β1 (short-minus-long). However, there is a real day-count mismatch when materialising the curve as a DiscountCurve: tenors-in-years are converted to dates with 365.25 days/yr but the resulting DiscountCurve uses ACT/365 Fixed by default, so the recovered zero rate at each pillar differs from the NS yield by roughly t/(t+small) — about 4 bp at 10y for a 5% rate. Several robustness issues are present in calibration: no degeneracy guard for Svensson τ1≈τ2, no underdetermination check (4/6 params with fewer pillars), a discontinuous 1e10 barrier that Nelder-Mead does not handle well, and no fallback / multi-start for the well-known non-convex NS objective. _ns_factor2 also suffers catastrophic cancellation in the band 1e-10 < x < ~1e-6. The yield functions themselves have no τ>0 guard.

**Code-correctness critic:**

> The Nelson-Siegel module is straightforward and mostly safe. Core yield functions handle the t<=0 edge case. The main correctness gaps are at the calibration boundary (empty market data → IndexError/ZeroDivisionError with no validation), silent length-mismatch via zip, no convergence signal returned to caller (a non-converged optimizer still produces a "result" dict that looks legitimate), and direct yield/curve functions blindly accept invalid tau (zero or negative) which either raises deep inside or silently returns wrong values via the small-x branch. None of these affect the default happy path, but they are real failure modes on user-supplied inputs.

### `curves/smith_wilson.py`
**Numerical critic:**

> The Wilson kernel implementation matches the EIOPA Smith-Wilson formula and produces correct DFs at pillars to machine precision, with forward rates converging cleanly to UFR at long maturities. The core math is sound. The real issues are at the wrapper layer and input validation: `smith_wilson_curve` resamples the analytical SW curve at a sparse set of default tenors (gaps of 10-20 years in the extrapolation zone) and hands them to a log-linear `DiscountCurve`, introducing ~1e-3 DF / multi-bp rate errors at non-pillar intermediate maturities — a silent quality loss versus the analytical formula. There is also no input validation: alpha<=0, duplicate/zero maturities, mismatched list lengths, and arbitrageable market DFs all either crash with an unhelpful LinAlgError or silently produce nonsense (e.g. tiny alpha yields zeta ~ 1e15; arbitrageable inputs give DF > 1 at intermediate t). The `smith_wilson_forward` silent fallback to UFR on non-positive DF masks bugs rather than diagnoses them. No citation to EIOPA technical docs in the module.

**Code-correctness critic:**

> The module is small and the happy path is straightforward, but it has multiple unguarded boundary cases that can fail or silently produce wrong results on realistic inputs: Smith-Wilson can produce non-positive discount factors that `DiscountCurve` rejects, `smith_wilson_forward` silently returns `ufr` when this happens (masking failures), the `extrapolation_tenors` default of `[30,40,...,120]` is added to the curve unconditionally and can collide with calibration maturities at integer year boundaries via `date_from_year_fraction` rounding (raising 'strictly increasing' errors from `DiscountCurve`), and there is no validation of inputs (empty lists, duplicate/unsorted maturities, alpha=0, mismatched lengths, t=0 in extrapolation). Most failure modes raise rather than corrupt, but the forward-rate fallback to `ufr` is a true silent-wrong-result bug.

### `fixed_income/bond.py`
**Numerical critic:**

> The bond module's structural logic is correct (no principal double-count, sensible accrued/ex-div flow, correct Newton direction for IRR), but it has a real day-count bug that propagates everywhere ACT/ACT ICMA is used (i.e., every US Treasury note priced via FixedRateBond.treasury_note). Because `accrued_interest` and the underlying `FixedLeg` constructor call `year_fraction(...)` without passing `ref_start`, `ref_end`, and `frequency`, ICMA silently falls back to ACT/365 Fixed. This causes (a) semi-annual UST coupons to be 1.9836 or 2.0164 per 100 instead of the canonical exact 2.0000 per period, (b) accrued interest to be wrong by ~0.8% mid-period, and (c) `_price_from_ytm` at par yield to return 99.9998 (5y) or 99.9995 (30y) instead of exactly 100. For UST quoted in 32nds (i.e. ~3.1bp resolution), this is observable mispricing. Two other issues: `_price_from_ytm` uses `self.face_value` (first-period notional) as the redemption amount, while `dirty_price` uses `notional_schedule[-1]` — these disagree for sinking-fund schedules, so YTM solved from market price will be inconsistent with the curve-based price. And `treasury_note()` requests T+1 settlement but installs no calendar, so `settlement_date()` can land on a Saturday. The BEY/Street-convention difference vs Bloomberg YTM is on the documented "accepted approximations" list and is not flagged.

**Code-correctness critic:**

> The module has several real correctness bugs. The most serious is inconsistent use of the principal-redemption notional between dirty_price (uses last-period notional) and the YTM/duration/convexity/dv01_yield/_price_from_ytm helpers (use first-period notional via self.face_value). For sinking-fund or amortising bonds this guarantees the YTM solved by brentq does not invert the dirty-price function, and downstream risk metrics are computed against the wrong redemption amount. A second user-facing bug is that ex-dividend pricing is inconsistent: accrued_interest correctly returns a negative value in the ex-div window, but dirty_price still PVs the upcoming coupon, so clean_price = dirty - (negative accrued) double-counts the coupon the buyer will not receive. Other issues include duration/convexity not guarding maturity vs settlement, an under-wide brentq bracket for YTM, and the treasury_note factory using calendar days for T+1 because no calendar is attached.

### `fixed_income/fra.py`
**Numerical critic:**

> The FRA pricing formula is mathematically correct (cash-settled-at-start convention is equivalent to discounted accrual: `N·(f-K)·τ·df2`, verified algebraically using `1+f·τ = df1/df2`). Par rate is the projection-curve forward, independent of discount curve, which is the right answer for a linear product. The dual-curve plumbing is sensible. The module is solid for the happy path. The findings below are robustness gaps: missing div-by-zero guards (df2=0 in forward_rate, 1+f·τ=0 in pv for sufficiently negative rates), no past-start-date handling, silent first-curve fallback in `pv_ctx` that can mask a wrong-curve selection, and a structural oddity where `from_convention` is defined at module scope rather than inside the class (works via the descriptor protocol but is fragile to refactors). No critical bug in the priced value for normal market inputs.

**Code-correctness critic:**

> The core FRA pricing logic (forward_rate, pv, par_rate) is small and largely correct: the constructor rejects degenerate date order and non-positive notional, and aliasing/mutation are not concerns. The real correctness risks are concentrated in (1) pv_ctx silently selecting an arbitrary projection curve when the day-count keyed lookup misses, which can yield a wrong PV in a multi-curve context; (2) the constructor unconditionally calling year_fraction without the auxiliary args required by BUS/252 (calendar) and ACT/ACT ICMA (frequency, refs), which will raise for those conventions even though they are listed as valid DayCountConvention members; and (3) unguarded divisions in forward_rate and pv that blow up in plausible degenerate inputs. The module-level @classmethod attached after class definition is unusual but functionally works.

### `fixed_income/swap.py`
**Code-correctness critic:**

> The module is mostly thin orchestration over FixedLeg/FloatingLeg, so most heavy correctness work happens elsewhere. However the factory classmethods have several edge-case bugs: `accreting()` silently ignores `final_notional` for single-period swaps, `amortising()`'s "linearly to zero" claim doesn't match the implementation (last period is `initial/n`, not zero), and `_map_notional_to_schedule` has an empty-list IndexError and a silent fallback that can mask schedule-misalignment between legs. `par_rate` swallows degenerate cases by returning 0.0, and the floating row in `cashflow_schedule` reports forward rate but the corresponding `amount` includes spread — a column-level inconsistency that misleads downstream consumers. `average_notional` is unguarded against empty schedules. No race conditions or mutation aliasing bugs were found — the orchestration is otherwise clean.

### `models/black76.py`
**Numerical critic:**

> The module implements Black-76 and Bachelier with standard formulas, correct put-call parity, sensible guards on T<=0 / vol<=0 / non-positive F,K (for Black-76), and natural support for negative forwards in Bachelier. Hand-calcs reproduce: ATM Black-76 with F=K=100, vol=0.2, T=1, df=1 gives 7.965567 (matches reference); Bachelier ATM matches the closed-form sigma*sqrt(T/(2*pi)) identity; put-call parity holds for both engines including negative forwards. Two genuine concerns: (1) the docstring for theta says it is the partial derivative w.r.t. time_to_expiry, but the formula returned is the classical "time-decay" theta which has the OPPOSITE sign of dPrice/dT_expiry — the docstring contradicts the code (and the Bachelier theta docstring "per year" similarly conflates calendar-time decay with derivative w.r.t. T). The formula is the standard market convention; the docstring is wrong. (2) Bachelier delta at exact F==K with T<=0 or vol<=0 returns 0 instead of the correct ATM limit of +/-0.5*df — minor edge-case asymmetry compared to Black-76 which does handle the ATM-at-expiry case explicitly. The rest of the code is clean.

**Code-correctness critic:**

> The module is a thin, mostly side-effect-free set of closed-form pricing/Greek functions. Most boundary handling (T<=0, vol<=0, non-positive forward/strike for Black-76) is present. I found a few real correctness issues: (1) Black-76 put delta at expiry with forward==strike returns -0.5*df but is gated by `vol<=0 or T<=0` — the ATM-at-expiry branch returns 0.0 for puts (asymmetric vs call), actually wait — re-read: for puts at ATM it returns -0.5*df (symmetric, OK). The actual issues are: bachelier_delta's ATM-at-expiry case returns 0.0 for both call and put (should be ±0.5*df for symmetry with Black-76, but at-expiry ATM delta is technically not unique — still asymmetric handling vs Black-76 is a robustness issue). (2) No NaN guards on inputs — passing NaN propagates silently. (3) Greeks for Black-76 don't guard against forward<=0 / strike<=0 (will throw on log/division), unlike the price function. (4) `option_type == OptionType.CALL` followed by an unconditional else assumes binary enum — if a user accidentally passes a string like "call" (Enum has string value), comparison silently falls through to put branch. Listing the concrete bugs below.

### `models/cos_method.py`
**Numerical critic:**

> The COS implementation is broadly faithful to Fang & Oosterlee (2008) — the chi/psi closed-form coefficients are correct, the Σ' (k=0 half-weight) is correct, the strike-factored payoff coefficients are correct, and the BS char-func wrapper is correct. The pricer matches BS to machine precision and Heston to ~1e-7 for moderate-moneyness, moderate-vol parameter regimes. However, two distinct bugs make the module unsafe outside the regime covered by the tests: (1) the call/put V_k formulas use the integration bounds [0, b] / [a, 0] unconditionally, which is only correct when a ≤ 0 ≤ b. For deep ITM options where a > 0 (call) or b < 0 (put), the V_k coefficients integrate over a wrong region and the price fails to converge — by 1.2% at S/K=50, 1.9% at S/K=10000, and 43% for a deep ITM put (S=10, K=1000). (2) The c2 cumulant is floored at 0.001, which silently corrupts low-variance pricing: an ATM 1-year call with vol=1% returns 0.07 vs the correct 4.88 (98% error) even at N=512. Both bugs are masked by the existing tests because they only probe moderate moneyness (S/K ∈ [0.8, 1.2]) and moderate vol (20%). Put-call parity continues to hold despite the V_k bug because the call/put errors cancel under linearity, which means parity is NOT a useful regression test for this defect.

**Code-correctness critic:**

> The COS price function has one critical correctness bug at extreme moneyness: the integration limits in the V_k payoff coefficients (`_chi`/`_psi` over [0,b] for calls and [a,0] for puts) are not clamped to [a,b]. When the truncation interval lies entirely on the wrong side of zero (deep OTM, short T, low vol), this produces wildly wrong prices instead of ~0. Confirmed empirically: spot=100, K=200, T=0.01, vol=20% call returns 11.78 instead of 0. Several robustness gaps as well: T=0 returns a spurious nonzero price due to a hard 0.001 floor on c2, L=0 crashes with ZeroDivisionError, spot<=0/strike<=0 raise ValueError without a clear message, and `div_yield` is declared but never used (silent contract mismatch with callers that expect the dividend yield to be honored).

### `models/feynman_kac.py`
**Numerical critic:**

> The module is a thin cross-validation harness rather than a pricer in its own right, so most numerical correctness is delegated to the underlying MC and PDE engines (which test out OK for European GBM). The main issues are (a) a silent 4% default discount rate when rate_fn=None, (b) an implicit convention in sde_to_pde that sigma_fn must be in dollar-volatility units (sigma*S), with no guard or check, (c) a "consistent" flag that compares MC-vs-PDE difference to MC stderr only, ignoring PDE discretisation bias which can dominate at the default 200x200 grid, (d) a silent max(., 0) clamp in pde_to_sde that masks bad inputs, and (e) hardcoded MC time-grid size (100) that ignores user-suppliable n_time. No outright wrong-price bug in the cross-check itself for a vanilla European call with default settings, but the helpers are easy to misuse and the consistency test can produce false negatives or false positives at the margins.

**Code-correctness critic:**

> The module is small and mostly delegates to other components, but it has a few real correctness issues. The most notable: `verify_feynman_kac` ignores the user-supplied `n_time` for the MC grid (hardcoded to 100), and the consistency check uses a 3-sigma bound that is meaningless when the PDE-MC discretization bias dominates stderr at small T. There is also a callable detection bug in `sde_to_pde` when `rate_fn` is a non-callable non-numeric value, and a division-by-zero subtlety when `mc_result.price` is a small negative float. Other lenses (lifetime/ownership, threading, mutation of defaults) do not apply or are clean.

### `models/g2pp_calibration.py`
**Numerical critic:**

> The G2++ swaption pricer is structurally wrong. It claims to implement Brigo-Mercurio's 2-factor swaption formula via "Jamshidian on y conditional on x, then integrate over x". The integration scaffolding is in place (Gauss-Hermite over x, brentq solve for y*) but the inner term invokes the unconditional 2-factor ZCB option formula `_g2pp_zcb_option(...)` — which already integrates over both factors and does NOT take x_val or y_star as inputs. The strikes K_k(x_val, y*) are passed in, but the option formula is evaluated under the unconditional bond price distribution. This is not Brigo-Mercurio eq. 4.31; it does not converge to a valid swaption price. Compounding this: (i) the x-integration is done under the risk-neutral measure (mean 0) rather than the T-forward measure (the BM eq. 4.31 measure shift M_x is omitted entirely); (ii) Jamshidian's trick is not literally applicable to G2++ ZCBs because their dependence on (x,y) is two-dimensional — BM resolves this by an analytic-in-y inner integral, not a 1-D root for y*; (iii) several silent fallbacks (`y_star = 0.0`, blanket `except Exception: return 0.0`) mask whatever the code does compute; (iv) calibration tolerances in tests (RMSE < 500bp) are loose enough that the bug is undetected. The variance helper `_g2pp_V` and ZCB reconstruction `_g2pp_zcb` look algebraically correct against BM 4.2, but the swaption pricer that uses them is broken.

**Code-correctness critic:**

> The G2++ calibration module has several correctness issues. The most serious is a wide `except Exception: return 0.0` in `g2pp_swaption_price` that silently converts any pricing failure into a price of zero — this directly corrupts the calibration objective by treating broken evaluations as a feasible point with bounded loss. Several other findings concern boundary cases (zero expiry, very small mean-reversion `a`/`b` near the optimizer bounds, brentq root-bracket fallback to `y_star=0`), inconsistent treatment of small `a` (`a > 1e-12` vs `a > 0`), and a sign/units bug where the per-swaption error stored as `error_bp` is in vol-units × 10_000 but is named "bp" while the user may interpret it as basis points of bp-vol — only a labeling/units concern but worth flagging given the rmse code uses it directly. There is also a `polish=False` DE with a subsequent L-BFGS-B polish, but L-BFGS-B has no protection against the objective returning the sentinel `1e6` on infeasible candidates, which yields a flat zero-gradient region — the local polish may stall and `converged` may report True misleadingly. Lifetime/None/race lenses do not apply (pure functional code, no threading, no mutable defaults).

### `models/hull_white.py`
**Numerical critic:**

> The analytical HW bond pricing (`zcb_price`, `_log_A`) matches Brigo-Mercurio (3.39) and correctly recovers the initial curve at t=0. The trinomial tree's state-price evolution is curve-consistent (Q sums to target_df each step). However, there are two serious bugs in the European swaption pricer: (1) the short rate at each expiry node is taken as `r0 + j*dr` instead of `alpha(T_expiry) + j*dr`, so the analytical bond prices used to compute the swap value are evaluated at the wrong short rate — this biases swaption prices materially for non-flat curves or long expiries; (2) the underlying swap is hard-coded to annual payments with integer-year tenors and unit year-fraction, so any non-annual / non-integer-tenor swap is mispriced. The tree branching probabilities also use the interior-node Hull formula for the top/bottom branch geometry, which is the wrong probability set (Hull's "Procedure 1" requires distinct probabilities at the boundary). Several silent fallbacks (`alpha = r0`, `return 0.0` from `_log_A`) mask numerical pathologies. `B(t,T)` loses precision for small `a*tau` (use `expm1`). No convergence test, no put-call/parity check, no cross-validation against Jamshidian closed-form swaption.

**Code-correctness critic:**

> The module has several real correctness bugs. The most severe: `tree_european_swaption` uses `r0` (initial forward) instead of the calibrated `alpha(expiry_T)` for the short rate at expiry nodes, since `_evolve_state_prices` doesn't return the alpha trajectory. This produces wrong swaption prices for any non-trivial maturity. The trinomial probability clamping can violate sum-to-one. Float-truncation in payment counting drops payments. `date_from_year_fraction` with `round(t*365.25)` causes small-`t` discount-factor lookups to collapse to `df=1.0` in `_log_A`, contaminating analytical bond prices. Silent fallbacks (return 0.0 on bad DFs, `alpha = r0` on bad calibration) hide failures.

### `models/lmm.py`
**Numerical critic:**

> Multi-factor HJM is broadly sound (ZCB-reproduction matches the flat curve to ~1e-4 absolute on a 5y horizon). The LMM piece has several real problems. The Rebonato swaption-vol formula is incorrect: it sums only diagonal terms (effectively ρ_ij = 0 off-diagonal), so a flat 20% forward-vol structure yields a 10% swaption vol instead of the textbook ~20%, and the docstring's "ρ=δ_ij" comment further muddies the convention; the T_expiry argument is also a no-op. The LMM simulation/caplet pricer is a measure mongrel: simulate() advances `numeraire_idx=period` each period (intended spot-LIBOR measure) yet caplet_price() discounts the terminal payoff with a deterministic df, which is only correct under the T_{j+1}-forward measure. For ATM/near-ATM caplets the drift contributions are tiny so MC matches Black, but the construction is not the spot-measure caplet pricer the docstring claims, and would mis-price away from ATM or for non-caplet payoffs. The drift index also starts at `numeraire_idx+1` rather than the standard η(t) for spot measure. Other issues: dead `mc_migrate` imports in the "via engine" functions (they don't use the engine at all), the L>=0 floor is decorative (cannot prevent NaN propagation), and the rebonato weight formula silently assumes T_0=0.

**Code-correctness critic:**

> The module contains one high-severity silent units bug (LMM simulation uses tau as the per-period time, ignoring the actual tenor spacing the user passes in), two medium robustness issues (length mismatches between L0/tenors/vols are unvalidated and can either crash or silently broadcast, and the "via_engine" wrapper functions import engine helpers they never call so the docstring promise is unfulfilled), plus a couple of low-severity oddities (dead T_start/T_end variables with an apparent off-by-one, redundant T_expiry > 0 guard). The MultiFactorHJM path code is well-shaped, and rebonato_swaption_vol guards both the swap-rate and annuity denominators. No critical wrong-result bugs were found within the in-scope correctness lenses (math correctness, e.g. drift restriction / Brownian correlation / measure choice, is explicitly out of scope per the prompt).

### `models/mc_engine.py`
**Numerical critic:**

> The engine is well-structured around a clean SDE/payoff abstraction, the Cholesky-based correlation via einsum is correct, the antithetic concatenation produces the right shapes, and seed reuse across Greek bumps correctly enables common random numbers. However there are several real numerical problems. The most important is a copy-paste bug at line 212 that silently downgrades scheme="milstein" to Euler, so any user relying on Milstein accuracy gets first-order weak convergence without warning. The greek() API documents a param_name parameter that is ignored, and the implementation bumps process.x0 uniformly via array broadcasting, which is wrong for multi-factor processes (e.g., bumping S0 by epsilon also bumps V0 in a Heston-style spec). There is no validation that x0.shape matches n_factors, so a scalar x0 with n_factors=2 silently broadcasts to both factors with no error. n_paths=1 produces NaN stderr without a guard. discount_factor is forced to be scalar, which is silently wrong for stochastic-rate processes. TimeGrid does not validate t0=0 or monotonicity. None of these are mathematical-formula errors per se — Euler/Milstein/Cholesky math is right — but the silent scheme fallback and the misleading greek() API will produce wrong results in plausible usage patterns.

**Code-correctness critic:**

> Module has several real correctness bugs. The most serious are: (1) the "milstein" scheme is silently aliased to euler (the ternary is `euler_step if scheme=="euler" else euler_step`), so users requesting Milstein get Euler with no warning — a wrong-result bug. (2) `greek()` mutates `self.process.x0` and on exception leaves the process in a bumped state — exception unsafety with caller-visible state corruption. (3) The Cholesky correlation einsum `'...j,kj->...k'` multiplies by `L^T` rather than `L`, which produces wrong correlation structure for non-diagonal correlations. (4) `antithetic=True` with `n_paths=1` silently produces zero paths (n_half = 1//2 = 0), and the downstream `np.std(..., ddof=1)` then divides by zero. (5) `greek()` adds a scalar `bump` to a vector `x0` for multi-factor processes — bumps all factors at once rather than just the named one, despite the `param_name` argument suggesting otherwise (in fact `param_name` is completely ignored). Lesser issues: `from_dates` discards timezone info, division by zero in `relative_error` is guarded but `confidence_95` is not, multi-factor `exact_step` signature inconsistent with 1D's (no `process` argument), `TimeGrid.__init__` crashes on empty `times`.

### `numerical/_distributions.py`
**Numerical critic:**

> Thin wrapper over scipy.stats / math.erf for common distributions used across the codebase. The core formulas are correct and hand-calc reproducible (Normal.cdf(1.96)=0.97500, Normal.pdf(0)=1/sqrt(2pi), LogNormal(0,1).mean()=exp(0.5), Exponential mean/rate inversion). The substantive concerns are (a) missing domain guards on sigma>0 for Normal/LogNormal/StudentT and on p in [0,1] for ppf, (b) np.where masking of unsafe log/exp evaluations that fire RuntimeWarnings on LogNormal.pdf(0), LogNormal.cdf(<=0), and Exponential.pdf/cdf at very negative x, and (c) a clearly mis-named StudentT.tail_dependence that lacks the correlation parameter needed for any bivariate copula tail-dependence formula — as written it computes a univariate quantity and labels it as a copula property. The Normal helpers route arrays through scipy.special.erf and scalars through math.erf, which is internally consistent. Return-type leak: scalar inputs to Normal.cdf actually return 0-d ndarray, not float, due to np.asarray upcasting.

**Code-correctness critic:**

> Module is mostly solid for the happy path but has a cluster of robustness issues around input validation (sigma/scale not checked at construction; NaN slips past Uniform's guard) and around np.where masking patterns that evaluate the invalid branch (division by zero at x=0 in LogNormal.pdf, log of non-positive in LogNormal.cdf) producing runtime warnings even though the masked result is correct. StudentT.tail_dependence has a suspicious 1e-10 fudge to avoid divide-by-zero at df=1 that silently produces a wildly wrong result rather than failing loudly for df<=1. There is also a return-type inconsistency in Normal.ppf driven by the np.isscalar pitfall (0-d ndarray and similar are not "scalar" to numpy). No outright wrong-result bugs in normal use, but several edge-case correctness traps.

### `numerical/_fourier.py`
**Numerical critic:**

> Fractional FFT (Bluestein) and Hilbert transform are mathematically correct and reproduce numpy / scipy / direct DFT references to machine precision. Haar and DB2 wavelets are orthonormal (energy-preserving). The CharacteristicFunction class is the weak link: (i) c3 stencil returns the wrong sign so skewness comes out negated for any non-symmetric distribution; (ii) the 5-point c4 stencil with h=1e-4 is catastrophically unstable (h^4 = 1e-16 is at machine epsilon) — on a pure Gaussian it returns excess kurtosis ~ -1391 instead of 0, and on Poisson(2) it is off by ~75 percent; (iii) density() calls np.trapz which was removed in NumPy 2.x so it crashes on import-level numpy 2.4.3; (iv) wavelet_transform crashes on non-power-of-2 inputs (the docstring describes this as a quality preference, but it is a hard requirement); (v) several silent guards mask divergences (max(c2, 0), max(c2, 1e-20)). These are wrong-result bugs for cumulant / density / pricing paths even though the lower-level FFT primitives are sound.

**Code-correctness critic:**

> One critical bug: CharacteristicFunction.density() calls np.trapz which has been REMOVED in NumPy 2.x (this environment runs NumPy 2.4.3). The method crashes with AttributeError on any call, full stop. One high-severity bug: wavelet_transform with Haar wavelet silently produces wrong results for odd-length inputs of length 3 (broadcasting masks the size mismatch) and crashes with ValueError for odd-length inputs >= 5. Several medium-severity edge-case crashes: density(n_quad=1) hits IndexError on linspace, density(scalar) hits TypeError on len(), hilbert_transform of empty array raises in fft. WaveletResult.levels can disagree with actual decomposition depth when the loop breaks early.

### `numerical/_integrate.py`
**Numerical critic:**

> The module is a thin wrapper over scipy/numpy quadrature plus a handful of hand-rolled rules. The fixed-order Gauss family (Legendre, Hermite, Laguerre) and tanh-sinh all reproduce known integrals to machine precision. However there are three real, demonstrable bugs and several robustness concerns: (1) `integrate_2d` silently swaps argument order versus its docstring because `scipy.integrate.dblquad` requires `f(y, x)`; (2) `_romberg` calls `scipy.integrate.romberg`, which has been removed in current SciPy (1.15+); the entire ROMBERG method is dead and raises ImportError; (3) `_clenshaw_curtis` weights are derived from a formula that is only valid for even `n` — for odd `n` it returns the wrong endpoint normalisation and miscounts the topmost interior cosine term, producing visibly wrong integrals (e.g. ~0.5% relative error on x^14 at n=15, ~0.2% at n=21). On top of that, `converged=True` is hard-coded for every method even when the rule is fixed-order on an unverified integrand, `error_estimate=0.0` is reported for all non-adaptive methods, `tanh_sinh` has a non-refinable step `h=0.1` (the `n` parameter only widens truncation, not refinement), and `integrate_semi_infinite` silently falls back to adaptive for any non-Laguerre method. None of these affect Gauss-Legendre / tanh-sinh / Simpson / Trapezoid / adaptive paths — those are correct — but anything currently calling Clenshaw-Curtis with odd `n`, calling ROMBERG, or relying on the documented `f(x,y)` ordering of `integrate_2d` is buggy.

**Code-correctness critic:**

> Two confirmed correctness bugs: (1) `_romberg` calls `scipy.integrate.romberg`, which was removed in SciPy >= 1.15 — every call raises ImportError on the installed environment (SciPy 1.17). (2) `integrate_2d` passes `f` directly to `scipy.integrate.dblquad`, but dblquad's contract is `func(y, x)` (y first); the public docstring promises `f(x, y)`, so for any non-symmetric integrand the computed value is wrong. Beyond those, several robustness issues: `_simpson` divides by zero when `n==0`, `_clenshaw_curtis` produces NaN when `n==0`, and `_adaptive` / fixed-order routines always set `converged=True` even when the underlying solver issued an `IntegrationWarning` or `n` is too small to converge — i.e. the API lies about convergence. Method dispatch in `integrate_semi_infinite` silently downgrades any non-Laguerre choice to adaptive on `[a, inf)`, ignoring the user's method.

### `numerical/_mc.py`
**Numerical critic:**

> The QE-Heston step itself is a faithful and largely correct implementation of Andersen (2008) — conditional moments, the psi=s^2/m^2 split, the quadratic-regime b^2/a, and the exponential-regime inverse CDF all match the paper. However the surrounding helpers are broken in ways that will silently produce wrong results. multilevel_mc as written degenerates to the coarsest-level estimator for any path-independent (European) payoff because the coarse path is obtained by sub-sampling the fine path rather than by re-simulating with aggregated Brownian increments — the Giles telescoping correction is identically zero, so the function returns the biased level-0 price while pretending to be MLMC. antithetic_paths is either a no-op that just negates the normals it was already given, or, when called with only terminal values, performs a mirror-around-sample-mean that is not a valid antithetic. Tests in the repo exercise only positivity and a loose mean-reversion bound; there is no convergence test, no comparison against Heston Fourier/closed-form, and no MLMC bias decay measurement. The unused MCVarianceReduction / MCDiscrMethod enums signal that the public API was sketched in but not wired up.

**Code-correctness critic:**

> The QE Heston step has a few minor robustness issues but is largely sound. The MLMC implementation has a substantive correctness bug: the level-l "coarse" path is constructed by stride-2 subsampling of the level-l fine path, while the level-0 path is generated by calling `process_fn` directly with `base_steps`. For most discretisation schemes (Euler, Milstein) the subsampled fine path does not have the same distribution as a coarse-step simulation, so the MLMC telescoping identity is biased. The antithetic_paths function has two interface inconsistencies and silently produces NaN on non-positive terminal values. Lifetime/race/exception lenses do not apply (pure numpy, no shared state, no resources).

### `numerical/_optimize.py`
**Numerical critic:**

> This is a thin wrapper around scipy plus a couple of hand-rolled routines (interior_point, proximal_gradient, simplex/L1 projections). The scipy wrappers and projection algorithms (Michelot for simplex, sign-restore for L1 ball) are correct. The two hand-rolled iterative routines are the problem: interior_point silently drops equality constraints (builds the constraint list but passes BFGS which is unconstrained), and proximal_gradient lies about its result (fun=0.0 always, converged=True even at maxiter). Several silent fallbacks (DE seed=42 default, basin_hopping converged=True, linprog returning zeros on infeasible) can mask calibration failures upstream. None of this affects pricers that don't actually call interior_point or proximal_gradient, but anything calling them gets wrong/misleading answers.

**Code-correctness critic:**

> Several real correctness bugs. The most serious is interior_point silently ignoring equality_constraints — the constraints list is built but never passed to scipy.minimize, and the chosen "BFGS" solver does not accept constraints anyway, so any caller relying on equality constraints gets an unconstrained (wrong) result. Other confirmed bugs: aliasing in projection_l1_ball (returns input unchanged), basin_hopping always reports converged=True, proximal_gradient always reports fun=0.0 and converged=True regardless of actual convergence, and several functions reference loop variables (outer, k) after the loop that are undefined when maxiter <= 0 (NameError). minimize cannot parse hyphenated method strings like "Nelder-Mead" despite the enum name suggesting otherwise. interior_point has no feasibility check on x0, so a single infeasible starting point silently corrupts the entire optimisation via the 1e15 sentinel.

### `numerical/_pde.py`
**Numerical critic:**

> The 1D BS PDE solver gets the ATM European call within 1.5e-4 of Black-Scholes at 400×400 with the default UNIFORM grid, so the core θ-scheme operator is essentially correct. However, the solver has several serious bugs that produce wrong prices in non-default configurations: (a) the LOG grid is unstable and overprices an ATM call by ~17% (13.86 vs 11.84); (b) the SINH grid silently goes negative when concentration_point ≠ midpoint (e.g. concentration at strike=100 produces a grid spanning [-149.5, 349.5]); (c) the upper-boundary discount factor is reversed in step index — at step 0 it uses exp(-rT) instead of exp(-r·dt); (d) the implicit Dirichlet boundary is enforced incorrectly in the tridiagonal system (rhs[0]=0, then V_new[0] is overwritten to V_old[0] after the solve, dropping the implicit boundary contribution at i=1); (e) for American options the boundary V[0] uses the European discounted strike K·exp(-rτ) − S[0] instead of the intrinsic K − S[0]; (f) the grid s_max = 5·spot is independent of σ√T and clips the lognormal tail catastrophically for high-vol / long-T cases (σ=0.8, T=2 underprices an ATM call by ~30%, 31.58 vs BS 45.13); (g) the convergence is non-monotonic (N=800 has larger error than N=400, suggesting the scheme is dominated by boundary error not interior O(Δt²+Δx²)). The vol→0, deep-ITM, and ATM cases on uniform grids look correct.

**Code-correctness critic:**

> The module has several correctness bugs around boundary handling, theta sign, the Rannacher state machine, the tridiagonal solver guard, the searchsorted indexing for Greeks, vega computation under American exercise, the LOG grid built from a spot-dependent s_min, and the n_time=0 division-by-zero path. Most fire on realistic inputs (call/put with rate ≠ 0, deep-in/out spots, default usage of PDESolver1D with spot below grid). The Rannacher scheme is also implemented incorrectly (only changes theta after step 2 but never restores it for subsequent steps, which actually works — but for explicit/CN it's also broken in a subtler way).

### `numerical/_trees.py`
**Numerical critic:**

> The CRR/JR/LR binomial cores and Kamrad-Ritchken trinomial recurrence look broadly correct in shape, but several issues degrade correctness in practice. The most serious is in `_tian_params`, where V is implemented as `M^2 * (exp(vol^2 * dt) - 1)` rather than Tian's `V = exp(vol^2 * dt)`; under realistic inputs this makes the discriminant always negative and silently routes "tian" through CRR. Discrete dividends are subtracted directly from the terminal spot grid for every dividend regardless of `step`, which is not a valid escrowed-dividend treatment and produces wrong prices whenever `dividends` is set. Knock-in barriers fall through with a silent `pass`, returning a vanilla price labelled as a barrier option. Richardson extrapolation in `convergence_analysis` assumes O(1/N^2) error, which is wrong for CRR/JR (oscillatory O(1/N)), and assumes a doubling schedule that is not enforced. Several numerical guards are missing (vol=0 in CRR/JR/Tian gives 0/0; LR has no guard for vol=0 or T=0 or spot/strike=0). Probabilities in the 2D Rubinstein tree are clamped to 0 then renormalised, silently masking invalid risk-neutral measures near |rho|->1. Detailed findings below.

**Code-correctness critic:**

> Multiple silent-wrong-result bugs in this module. The dividend handling is broken (every dividend is blanket-subtracted from terminal spots regardless of its step index, instead of being applied at the ex-date inside backward induction). Knock-in barriers fall through silently and return vanilla prices. The 2D solver returns hard-coded zero Greeks, silently degrades non-"spread_call" American payoffs to European, and the convenience wrapper `solve_tree` provides no way to pass `exercise_dates` so Bermudan reduces to European. Trinomial ignores `dividends` and `store_tree`. Exception safety in `_compute_vega` and `convergence_analysis` is broken (instance state mutated without try/finally). Several smaller boundary/edge bugs (binomial theta uses V_ud at step 2 assuming u*d=1, which is wrong for JR/Tian/LR; ZeroDivisionError if `n_steps=0`; 2D probability normalisation collapses to all-zero prices when every clipped probability is zero).

### `numerical/auto_diff.py`
**Numerical critic:**

> The forward-mode dual-number AD core is mathematically correct on its happy path: the addition, multiplication, division, and chain-rule paths reproduce hand-calculations (x^2+2x+1, x^x, x^y, exp/log/sin/cos) to machine precision, and the gradient/Jacobian drivers seed and read derivatives correctly. The module is, however, riddled with silent numerical fallbacks that mask bugs rather than report them: `sqrt(0)` returns `Dual(0,0)` instead of raising on the genuinely infinite derivative; `__pow__` with a Dual exponent uses `log(|self.val|)` so negative bases produce a finite but mathematically meaningless derivative; `__rpow__` with a non-positive base returns derivative 0 silently; the `grad`/`jacobian_ad`/`derivative` drivers return zeros when the user's function fails to thread Duals through (the most common bug in forward AD code). The `hessian_ad` entry promised in the module docstring is not implemented. Comparison operators dispatched against NaN never raise. `max_dual` ties pick the first argument as a Dual but a constant if the first arg is a float, so identical inputs give order-dependent derivatives. Overriding `__eq__` while keeping `__slots__` silently makes `Dual` unhashable, which will surface as a confusing error if anyone tries to use it as a dict key. None of these are wrong-price bugs on a vanilla call delta, but several are correctness traps in active use — especially the silent zero-derivative fallbacks.

**Code-correctness critic:**

> The module has several genuine correctness bugs that will produce wrong results on realistic inputs. The most serious is `__eq__` being defined without `__hash__`, breaking hashability while comparing only values (which itself violates the hash/eq contract). The `__pow__` path silently returns 0 derivative when base is zero (rather than the correct n*0^(n-1)*der for integer n), `__rpow__` silently swallows non-positive bases, `sqrt` returns (0,0) at zero rather than propagating the singular derivative correctly or raising, and several divisions can divide by zero without guards. `jacobian_ad` calls `f` an extra time just to get output dimension (caller-visible side effects). `derivative` returns 0.0 derivative for constants without distinguishing from f returning Dual with 0 derivative — but more importantly, when `f` returns a non-Dual result that includes operations on Dual inputs (e.g. via np.array), the `isinstance` check fails silently and returns 0.

### `options/equity_option.py`
**Numerical critic:**

> The core Black-Scholes math (price, delta, gamma, vega, theta, rho) is correct: closed-form Greeks match bump-and-revalue to 1e-6, and put-call parity holds at machine precision for both prices and Greeks. The theta formula matches Hull's full continuous-dividend expression. Issues live in the limit/edge-case handling: the `T<=0 or vol<=0` guard collapses two different limits (T=0 expiry, vs vol=0 with T>0) into the same branch and uses a spot-vs-strike intrinsic check that gives the wrong answer when the forward differs from spot. Several Greeks return 0 in the vol=0, T>0 case when the deterministic-payoff limit is non-trivial. Spot<=0 and strike=0 are not guarded in any Greek and raise raw math domain errors. There is also a redundant in-function scipy import in theta. None of these affect the standard interior pricing path; severity is high for the vol=0/T>0 bug because it's a recognisable edge case, medium for the others.

**Code-correctness critic:**

> The module is a thin wrapper over Black-76 plus dividend yield adjustments. The vanilla pricing call is fine, and the per-Greek formulae for T>0, vol>0 look consistent with Hull. The main correctness issue is in the degenerate-input branches (T<=0 or vol<=0): equity_delta compares `spot` to `strike` instead of `forward` to `strike` and returns a unit-magnitude value instead of the exp(-qT) discount, equity_delta gives the wrong limit ATM (0 instead of ±0.5×exp(-qT)) which is also inconsistent with black76_delta, and equity_theta/equity_rho ignore the dividend/rate terms that survive when vol=0 but T>0. These are edge-case bugs and will not fire on a normal user call (vol>0), but they are real wrong-result paths.

### `options/swaption.py`
**Numerical critic:**

> The core Swaption.price path is correct: it computes the forward swap rate and annuity from the underlying, applies the standard Black-76 (or Bachelier/SABR) ann*Black(F,K,sigma,T) formula via the model, and multiplies by notional. The expired-payoff branch and the duck-typed tree path (price_swaption) are also reasonable. The principal problems live in (a) the SABR-HW blender, which has a critical T=0 zero-price bug, a magic 1% vol fallback that masks failures, a docstring/formula mismatch on what blend_half_life means, and a theoretically dubious vol-level blend; (b) inconsistency between analytical Greeks (swap-rate sensitivities) and the bump fallback (curve-bump sensitivities) being labelled identically; (c) lossy serialisation registration and a from_convention factory that silently drop calendar/convention/stub/eom; (d) tree-model path bypasses the expired guard. None of the issues invalidate vanilla single-curve Black-76 swaption pricing on a fresh trade, but the blender and Greeks inconsistencies are user-facing wrong-result risks.

**Code-correctness critic:**

> Several real correctness bugs in the auxiliary code paths. The main `price` flow for Black76/Bachelier/SABR-style models is sound. The major issues are: (1) `price_swaption_sabr_hw` returns 0.0 at T==0 instead of intrinsic value, dropping legitimate ITM payout; (2) the same function silently substitutes a hard-coded 1% vol when both SABR and HW vols fail, producing arbitrary prices; (3) the serialisation field list omits `calendar`, `convention`, `stub`, `eom`, so to_dict→from_dict round-trips silently lose these and produce different prices; (4) `from_convention` also drops `calendar`/`convention`/`stub`/`eom`; (5) bump-and-reprice greeks only bump the discount curve, not the projection curve, under-counting delta in a dual-curve setup; (6) bump-and-reprice gamma with bump=1e-4 and notional-scaled prices is numerically catastrophic; (7) `blend_half_life=0` causes division by zero with no guard. There are also smaller robustness gaps around theta/vega missing from the bump fallback and Greeks at expiry returning zero delta.

---

## Recommended next steps

### 1. Verify Tier 1 (13 items)
For each Tier 1 entry: read the code, attempt to write a failing test. If the test fails, you've confirmed the bug. Fix it as one slice.

### 2. Triage Tier 2 (18 items)
Skim each row. Anything that reads as a clear concrete bug → promote to a fix-slice. Anything ambiguous → defer.

### 3. Spot-check single-critic criticals (19 items)
Quick triage pass. Most can be dispatched as 'accepted approximation' or 'context-missing-from-critic' without deep investigation.

### 4. The 150 'high' findings
Cluster around: numerical edge cases (T=0, vol=0, ATM par), serialisation round-trip drops, schedule alignment across curves. Worth a follow-up audit slice focused on each cluster.

### 5. Update accepted-approximations memory
Some 'critical' findings are likely already-accepted approximations. Updating `reference_approximations.md` with each dismissed finding builds institutional knowledge so the next audit doesn't re-surface them.

## Raw findings JSON

Full structured output (697 findings, ~720 KB):
`/private/tmp/claude-501/.../tasks/www7hfs2m.output`

Each finding has `severity`, `title`, `detail`, `location`, `fix`. Filter by package/severity/lens as needed.
---

# Wave 2 (partial) — risk + top instruments + remaining models

**Strategy retired after this wave.** Going forward: methodical, conversational, single-module audits (no big workflow fan-outs). Wave 2 ran on 96 modules but the strict JSON output schema caused 71 of 192 agent calls to fail. Salvageable data below.

## Coverage

| Tier | Count |
|---|---:|
| Modules requested | 96 |
| Fully audited (both critics) | 58 |
| Partial (one critic only) | 3 |
| Not audited (both critics failed) | 35 |
| Findings collected | 1048 |
| → `critical` | 46 |
| → `high` | 231 |
| → `medium` | 354 |
| → `low` | 367 |
| → `nit` | 50 |
| Double-confirmed criticals (both critics → critical, fuzzy-matched) | 0 |

## Wave 2 double-confirmed criticals

## All Wave 2 critical findings (46)

| # | Module | Lens | Title | Location |
|---|---|---|---|---|
| 1 | `risk/xva.py` | numerical | CFA/DFA double-counted in total_xva_decomposition | `python/pricebook/risk/xva.py:518-555` |
| 2 | `risk/scenario.py` | numerical | Every scenario constructor drops half of PricingContext fields | `python/pricebook/risk/scenario.py:55-62, 72-79, 94-101, 113-` |
| 3 | `risk/scenario.py` | numerical | `parallel_shift` does not bump the multi-currency `discount_curves` dict | `python/pricebook/risk/scenario.py:49-62` |
| 4 | `risk/simm.py` | numerical | Vega and curvature inputs silently ignored | `python/pricebook/risk/simm.py:192-204` |
| 5 | `risk/simm.py` | numerical | Bucket margin loses sign needed for across-bucket aggregation | `python/pricebook/risk/simm.py:181-218` |
| 6 | `risk/simm.py` | numerical | Across-risk-class aggregation drops the SIMM correlation matrix | `python/pricebook/risk/simm.py:149-150` |
| 7 | `risk/backtest.py` | numerical | walk_forward leaks test data into signal generation (lookahead bias) | `python/pricebook/risk/backtest.py:289-296` |
| 8 | `risk/factor_model.py` | numerical | Shrinkage intensity is not Ledoit-Wolf and is scale-dependent | `python/pricebook/risk/factor_model.py:255-261` |
| 9 | `risk/factor_model.py` | numerical | factor_timing signal direction is opposite of the docstring | `python/pricebook/risk/factor_model.py:300-317` |
| 10 | `risk/cvar_optimisation.py` | numerical | LP-failure fallback returns VaR mislabelled as CVaR | `python/pricebook/risk/cvar_optimisation.py:130-134` |
| 11 | `risk/vol_stress.py` | numerical | correlations argument silently ignored in cross_asset_vol_stress | `python/pricebook/risk/vol_stress.py:99-116` |
| 12 | `risk/dynamic_allocation.py` | numerical | multi_period_mv passes unsupported kwarg to mean_variance — raises TypeError on first call | `python/pricebook/risk/dynamic_allocation.py:201` |
| 13 | `risk/portfolio_margin.py` | numerical | SPAN PnL: gamma term has wrong units — explodes by O(notional) | `python/pricebook/risk/portfolio_margin.py:86-98` |
| 14 | `risk/portfolio_margin.py` | numerical | 'Covered call' branch is actually a naked short call | `python/pricebook/risk/portfolio_margin.py:255-259` |
| 15 | `risk/ipv.py` | numerical | concentration_ava called with mid-price where base_spread_bp is expected | `python/pricebook/risk/ipv.py:210` |
| 16 | `risk/ipv.py` | numerical | future_admin_cost_ava called with complexity_score and maturity_years swapped | `python/pricebook/risk/ipv.py:223` |
| 17 | `fixed_income/callable_bond.py` | numerical | Tree never applies alpha(t) drift — does not reprice the initial curve | `python/pricebook/fixed_income/callable_bond.py:84, 94` |
| 18 | `fixed_income/callable_bond.py` | numerical | Transition probabilities divide by 6 instead of 2 — drift is 1/3 of correct value | `python/pricebook/fixed_income/callable_bond.py:98-100` |
| 19 | `fixed_income/callable_bond.py` | code-correctness | Last coupon double-counted at maturity | `python/pricebook/fixed_income/callable_bond.py:87, 121-122` |
| 20 | `fixed_income/frn.py` | numerical | discount_margin shifts the coupon spread, not the discount curve — wrong sign and magnitude vs canon | `python/pricebook/fixed_income/frn.py:124-152` |
| 21 | `fixed_income/risky_bond.py` | code-correctness | dirty_price raises ValueError when coupon period straddles the curve reference date | `python/pricebook/fixed_income/risky_bond.py:80-82` |
| 22 | `fixed_income/inflation.py` | numerical | IE01 destroys curve term structure instead of parallel-shifting it | `python/pricebook/fixed_income/inflation.py:443-450` |
| 23 | `fixed_income/inflation.py` | numerical | dirty_price ignores settlement when discounting | `python/pricebook/fixed_income/inflation.py:380-400` |
| 24 | `fixed_income/treasury_benchmark.py` | numerical | ctd_switch_analysis implied-repo formula is non-standard and missing accrued interest | `treasury_benchmark.py:325-336` |
| 25 | `fixed_income/ir_futures.py` | numerical | HW convexity discontinuous at a→0 (2x jump) and disagrees with sibling formula in curves/bootstrap.p | `python/pricebook/fixed_income/ir_futures.py:162-170 and pyth` |
| 26 | `fixed_income/callable_floater.py` | numerical | Transition probabilities do not match Hull-White (1994); mean-reversion is 1/3 of canonical | `python/pricebook/fixed_income/callable_floater.py:121-123, 2` |
| 27 | `fixed_income/callable_floater.py` | numerical | Tree never solves for alpha(t); does not reproduce initial discount curve | `python/pricebook/fixed_income/callable_floater.py:117-119, 1` |
| 28 | `fixed_income/callable_floater.py` | code-correctness | brentq called with unsupported kwarg `xtol` — OAS is always NaN | `python/pricebook/fixed_income/callable_floater.py:451` |
| 29 | `options/bermudan_swaption.py` | numerical | Trinomial transition probabilities use wrong denominator (/6 instead of /2) | `python/pricebook/options/bermudan_swaption.py:73-75` |
| 30 | `options/bermudan_swaption.py` | numerical | Tree is not calibrated to the input curve (no alpha(t) drift) | `python/pricebook/options/bermudan_swaption.py:58,68,102` |
| 31 | `options/bermudan_swaption.py` | numerical | Off-by-one in exercise step / time-mismatched max against exercise value | `python/pricebook/options/bermudan_swaption.py:98-117` |
| 32 | `options/capfloor.py` | numerical | strip_caplet_vols_from_quotes does not strip anything — returns flat vol | `python/pricebook/options/capfloor.py:296-397 (esp. 360-392)` |
| 33 | `options/capfloor.py` | numerical | calibrate_capfloor_sabr fits SABR to a hand-fabricated synthetic smile, not to market data | `python/pricebook/options/capfloor.py:440-441 and 400-459` |
| 34 | `options/autocallable.py` | numerical | coupon_barrier parameter is completely ignored — coupons paid unconditionally | `python/pricebook/options/autocallable.py:142-167` |
| 35 | `options/bermudan_barrier.py` | numerical | Double-discounting in LSM continuation regression | `python/pricebook/options/bermudan_barrier.py:247-251` |
| 36 | `options/bermudan_lmm.py` | numerical | Discounting inconsistent with simulated measure — uses F_0 as a continuously-compounded short rate | `python/pricebook/options/bermudan_lmm.py:237-238, 266-267, 3` |
| 37 | `options/bermudan_lmm.py` | numerical | Andersen-Broadie upper bound: sub-simulation does not implement the AB recursion | `python/pricebook/options/bermudan_lmm.py:491-572` |
| 38 | `options/bermudan_lmm.py` | numerical | Identity correlation in `*_via_engine` variant produces a different model (independent forwards) | `python/pricebook/options/bermudan_lmm.py:82, 608` |
| 39 | `options/slv.py` | numerical | slv_mc and slv_mc_via_engine use different leverage/mixing formulas — not drop-in replacements | `python/pricebook/options/slv.py:56-73 (SLVModel.leverage) vs` |
| 40 | `options/asian_option.py` | numerical | Turnbull-Wakeman partial-fixings: index mismatch when known fixings are not a prefix | `python/pricebook/options/asian_option.py:428-440 and :188-21` |
| 41 | `options/asian_option.py` | numerical | MC methods ignore the averaging schedule — fixings forced uniform on [0, T] | `python/pricebook/options/asian_option.py:458-526 and :544-57` |
| 42 | `options/convertible_bond.py` | numerical | Terminal coupon dropped in CB.price() — inconsistent with bond_floor and deep-OTM limit | `python/pricebook/options/convertible_bond.py:110-115, 136-13` |
| 43 | `options/convertible_bond.py` | code-correctness | ZeroDivisionError for maturities below ~1 month | `convertible_bond.py:89-93, 294-304, 387-391, 605-616` |
| 44 | `options/convertible_bond.py` | code-correctness | Integer spot silently truncates paths in soft_call and CoCo | `convertible_bond.py:314, 399` |
| 45 | `credit/cds_swaption.py` | numerical | cds_swaption_black double-counts survival_to_expiry against the risky annuity | `python/pricebook/credit/cds_swaption.py:155, 165, 347, 526, ` |
| 46 | `credit/cln_xva.py` | numerical | SIMM IM under-counts by ~10_000x: CS01/DV01 are per-bp but SIMM weights expect per-unit deltas | `python/pricebook/credit/cln_xva.py:44-53` |

## Modules NOT audited in Wave 2 (35) — deferred to methodical pass

- `credit/exotic_loan.py`
- `credit/loan.py`
- `credit/distressed.py`
- `credit/tranche_pricing.py`
- `credit/recovery_locked_cds.py`
- `credit/cds_index.py`
- `credit/cds_strategies.py`
- `fx/fx_forward.py`
- `fx/fx_option.py`
- `fx/fx_barrier.py`
- `fx/fx_swap.py`
- `fx/ndf.py`
- `fx/fx_smile_cube.py`
- `fx/prdc.py`
- `fx/fx_basis.py`
- `equity/equity_forward.py`
- `equity/dividend_model.py`
- `equity/variance_swap.py`
- `equity/equity_index_futures.py`
- `equity/trs.py`
- `models/mc_processes.py`
- `models/mc_advanced.py`
- `models/hjm.py`
- `models/lmm_calibration.py`
- `models/fft_pricing.py`
- `models/fft_2d.py`
- `models/fokker_planck.py`
- `models/density_evolution.py`
- `models/sde_adaptive.py`
- `models/hundsdorfer_verwer.py`
- `models/fourier_greeks.py`
- `models/cos_bermudan.py`
- `models/levy_processes.py`
- `models/jump_process.py`
- `models/exact_simulation.py`

These will be picked up one at a time in the methodical bottom-up audit (post-notebook).

Raw Wave 2 JSON (1MB+) at `/private/tmp/claude-501/.../tasks/wly66lns5.output`. Will not be re-run in this strategy.
