"""Monte-Carlo expected-exposure engine (L5 risk & capital).

Produces the `ExposurePair` that CVA and DVA consume: expected positive exposure
`EPE(t_j) = E[(V(t_j))^+]` and expected negative exposure `ENE(t_j) = E[(-V(t_j))^+]`,
both from one simulation. For each grid date it draws the Hull-White short rate
under that date's t_j-forward measure — one exact Gaussian (`model.forward_short_rate`,
the same simulation the MC swaption uses) — and reprices the *remaining* swap
analytically from the model's `zero_bond` (the swap value is `notional - couponbond`
for a payer, its negative for a receiver, exactly as in the Jamshidian decomposition).

Two consequences make the oracle exact-able:
- `sigma = 0` removes all randomness, so `EE(t_j)` is the deterministic forward
  swap value's positive part.
- Under the t_j-forward measure, `P(0,t_j) * EE(t_j)` is the co-terminal swaption
  expiring at t_j — so the discounted exposure IS a swaption strip, and feeding
  `EE(t_j)` to `cva` (which multiplies by `DF(t_j)`) yields the correct discounted
  expected exposure `Sum_j swaption(t_j) * dQ_j`.

Scope (like the MC swaption): a vanilla swap under a flat-curve HW with `a > 0`;
the exposure grid is the fixed-coupon dates. Path-dependent exposure, netting sets,
collateral (PFE), and non-swap products are later slices.

Provenance:
  quarry: python/pricebook/risk/ (exposure / xva)
  source: Brigo & Mercurio s.3.3 (T-forward simulation); Gregory, The xVA Challenge
  oracle: sigma=0 deterministic exposure (exact) + discounted EE == co-terminal swaptions;
          PFE quantile == V at the r-quantile; collateral caps at H; OU path moments;
          measure consistency — joint paths reproduce forward-measure EE/PFE (A6.1)
  slice:  mc-exposure; bcva; pfe-quantile; margined-exposure; mpor-paths; measure-consistency
"""

from __future__ import annotations

import math
import random
from datetime import timedelta

from pricebook_ng.engine.swaption import coupon_bond_cashflows
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import VanillaSwap
from pricebook_ng.risk.xva import ExposurePair, ExposureProfile

_CURVE_DC = DayCountConvention.ACT_365_FIXED


def _simulate_netting_set(
    swaps: list[VanillaSwap], model: HullWhite, numerics: NumericalConfig
) -> tuple[list[date], list[list[float]]]:
    """Per exposure date, the netting set's PORTFOLIO value `Σ_i V_i(t_j)` across paths:
    each swap repriced on the SAME drawn rate and summed, so offsetting trades net. The grid
    is the union of the swaps' coupon dates strictly inside their lives (valuation first). A
    swap with no remaining coupons at `t_j` (already matured) contributes 0."""
    valuation = model.market.valuation_date
    specs = []  # (dates, amounts, notional, is_payer) per swap
    coupon_dates: set[date] = set()
    for swap in swaps:
        dates, amounts, notional = coupon_bond_cashflows(swap)
        specs.append((dates, amounts, notional, swap.pay_fixed))
        coupon_dates.update(d for d in dates if valuation < d < dates[-1])
    grid = [valuation, *sorted(coupon_dates)]

    rng = random.Random(numerics.mc_seed)
    values_by_date: list[list[float]] = []
    for t_j in grid:
        t = year_fraction(valuation, t_j, _CURVE_DC)
        remaining = [[(d, amt) for d, amt in zip(dts, amts) if d > t_j] for dts, amts, _, _ in specs]
        values = []
        for _ in range(numerics.mc_paths):
            r = model.forward_short_rate(t, rng.gauss(0.0, 1.0))
            total = 0.0
            for (_dts, _amts, notional, is_payer), rem in zip(specs, remaining):
                if not rem:
                    continue
                coupon_bond = sum(amt * model.zero_bond(t_j, d, r) for d, amt in rem)
                total += (notional - coupon_bond) if is_payer else (coupon_bond - notional)
            values.append(total)
        values_by_date.append(values)
    return grid, values_by_date


def _simulate_swap_values(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig
) -> tuple[list[date], list[list[float]]]:
    """Single-swap case of `_simulate_netting_set` (a netting set of one). Shared by the EE
    means and the PFE quantiles."""
    return _simulate_netting_set([swap], model, numerics)


def _quantile(sorted_values: list[float], q: float) -> float:
    """The `q`-quantile of an ascending-sorted sample (linear interpolation, type 7)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(pos)
    if lo + 1 >= n:
        return sorted_values[-1]
    return sorted_values[lo] + (pos - lo) * (sorted_values[lo + 1] - sorted_values[lo])


def exposure_profiles(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig
) -> ExposurePair:
    """MC expected exposure of `swap` under `model`, on a grid of the fixed coupon
    dates — both sides from one simulation. Per path the remaining swap value is
    `V = +-(notional - couponbond(t_j; r))` with `r` drawn under the t_j-forward
    measure; `EPE = mean max(V, 0)` (feeds CVA), `ENE = mean max(-V, 0)` (feeds DVA)."""
    grid, values_by_date = _simulate_swap_values(swap, model, numerics)
    epe = [sum(max(v, 0.0) for v in values) / len(values) for values in values_by_date]
    ene = [sum(max(-v, 0.0) for v in values) / len(values) for values in values_by_date]

    g = tuple(grid)
    return ExposurePair(ExposureProfile(g, tuple(epe)), ExposureProfile(g, tuple(ene)))


def collateralized_exposure(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig, threshold: float
) -> ExposurePair:
    """Exposure under a two-way CSA with variation margin and an uncollateralised
    `threshold` H: collateral posts the mark-to-market beyond H, so exposure is capped —
    `E_coll = min(max(+-V, 0), H)`. `threshold = 0` is fully collateralised (exposure 0);
    a huge threshold recovers the uncollateralised `exposure_profiles`.

    Marginal (per-date) model: it does NOT capture the margin-period-of-risk close-out gap
    that leaves residual exposure under full collateralisation — that needs joint-path
    simulation of V over the MPOR, a later slice."""
    grid, values_by_date = _simulate_swap_values(swap, model, numerics)
    n = numerics.mc_paths
    epe = [sum(min(max(v, 0.0), threshold) for v in values) / n for values in values_by_date]
    ene = [sum(min(max(-v, 0.0), threshold) for v in values) / n for values in values_by_date]
    g = tuple(grid)
    return ExposurePair(ExposureProfile(g, tuple(epe)), ExposureProfile(g, tuple(ene)))


def pfe_profile(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig, quantile: float
) -> ExposureProfile:
    """Potential future exposure at confidence `quantile`: `PFE_q(t_j)` is the q-quantile
    of positive exposure `max(V(t_j), 0)` across paths — the tail of the exposure the EPE
    averages over. A high-quantile PFE (e.g. 99%) also serves as a dynamic initial-margin
    proxy that feeds `mva`. (A margin-period-of-risk IM on the change in V is a refinement.)"""
    grid, values_by_date = _simulate_swap_values(swap, model, numerics)
    pfe = [
        _quantile(sorted(max(v, 0.0) for v in values), quantile) for values in values_by_date
    ]
    return ExposureProfile(tuple(grid), tuple(pfe))


def _simulate_rate_paths(
    model: HullWhite, times: list[float], numerics: NumericalConfig
) -> list[list[float]]:
    """Joint short-rate paths `r(t)` at the ascending year-fractions `times`, under the
    RISK-NEUTRAL measure: exact Ornstein-Uhlenbeck steps on the HW state `x` (mean-zero,
    `x(0)=0`), with `r(t) = x(t) + alpha(t)`, `alpha(t) = r0 + (sigma^2/2a^2)(1-e^{-a t})^2`
    (flat curve). Unlike the per-date forward-measure draws behind the EE profiles, this
    keeps the cross-date correlation the margin-period-of-risk gap needs.

    Measure (Amendment A6.1): the EE/PFE profiles use the per-date forward measure; this uses
    risk-neutral paths. They are ONE model under a change of numeraire —
    `E^Q[D(0,t)·max(V,0)] = P(0,t)·E^{T_t}[max(V,0)]` — not rival measures, and the two may not
    silently diverge: `test_measure_consistency` binds them (the joint-path marginal shifted by
    the forward-measure drift reproduces the forward-measure EE/PFE per date). Target is a single
    risk-neutral path engine once a path-based EE oracle exists; the swaption-strip identity
    survives as that marginal check."""
    a, sigma, r0 = model.a, model.sigma, model.curve.rate
    alpha = [r0 + (sigma**2 / (2.0 * a**2)) * (1.0 - math.exp(-a * t)) ** 2 for t in times]
    step_coeffs = []  # (decay, vol) for the OU step from the previous time
    prev = 0.0
    for t in times:
        dt = t - prev
        decay = math.exp(-a * dt)
        vol = sigma * math.sqrt((1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a)) if dt > 0 else 0.0
        step_coeffs.append((decay, vol))
        prev = t

    rng = random.Random(numerics.mc_seed)
    paths = []
    for _ in range(numerics.mc_paths):
        x = 0.0
        rates = []
        for (decay, vol), alpha_t in zip(step_coeffs, alpha):
            x = x * decay + vol * rng.gauss(0.0, 1.0)
            rates.append(x + alpha_t)
        paths.append(rates)
    return paths


def mpor_exposure(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig, mpor_days: int
) -> ExposureProfile:
    """Collateralised exposure over the margin period of risk: even fully margined, at
    close-out the collateral reflects the value `mpor_days` earlier, so the residual
    exposure is `E(t_j) = mean max(V(t_j) - V(t_j - MPOR), 0)`. Both marks value the same
    go-forward coupons (> t_j), so the difference is the value move over the close-out gap,
    driven by correlated rate paths. `mpor_days = 0` gives exactly zero.

    Scope: single netting set of one swap, zero threshold. The float leg is valued via the
    `notional - couponbond` form at the pre-gap date too (float ~ par; the error is O(MPOR),
    second-order over ~10 days)."""
    valuation = model.market.valuation_date
    dates, amounts, notional = coupon_bond_cashflows(swap)
    is_payer = swap.pay_fixed
    maturity = dates[-1]
    gap = timedelta(days=mpor_days)

    entries = []  # (d_now, d_pre, remaining coupons)
    time_set = set()
    for d_j in (d for d in dates if valuation < d < maturity):
        d_pre = d_j - gap
        remaining = [(d, amt) for d, amt in zip(dates, amounts) if d > d_j]
        t_now = year_fraction(valuation, d_j, _CURVE_DC)
        t_pre = year_fraction(valuation, d_pre, _CURVE_DC)
        entries.append((d_j, d_pre, remaining))
        time_set.update((t_now, t_pre))

    times = sorted(time_set)
    index = {t: i for i, t in enumerate(times)}
    paths = _simulate_rate_paths(model, times, numerics)

    def value(reprice_date, coupons, short_rate):
        v = notional - sum(amt * model.zero_bond(reprice_date, d, short_rate) for d, amt in coupons)
        return v if is_payer else -v

    grid = [valuation]
    ee = [0.0]                                          # no close-out exposure at valuation
    for d_j, d_pre, remaining in entries:
        i_now = index[year_fraction(valuation, d_j, _CURVE_DC)]
        i_pre = index[year_fraction(valuation, d_pre, _CURVE_DC)]
        total = 0.0
        for rates in paths:
            change = value(d_j, remaining, rates[i_now]) - value(d_pre, remaining, rates[i_pre])
            total += max(change, 0.0)
        grid.append(d_j)
        ee.append(total / numerics.mc_paths)
    return ExposureProfile(tuple(grid), tuple(ee))


def netting_set_exposure(
    swaps: list[VanillaSwap], model: HullWhite, numerics: NumericalConfig, pfe_quantile: float
) -> tuple[ExposurePair, ExposureProfile]:
    """Portfolio EPE/ENE and PFE for a netting set from ONE simulation (offsetting trades net
    on shared paths). The single entry point behind the L6 `xva_report`, so its six
    adjustments come from one exposure pass rather than six."""
    grid, values_by_date = _simulate_netting_set(swaps, model, numerics)
    n = numerics.mc_paths
    g = tuple(grid)
    epe = ExposureProfile(g, tuple(sum(max(v, 0.0) for v in vals) / n for vals in values_by_date))
    ene = ExposureProfile(g, tuple(sum(max(-v, 0.0) for v in vals) / n for vals in values_by_date))
    pfe = ExposureProfile(
        g, tuple(_quantile(sorted(max(v, 0.0) for v in vals), pfe_quantile) for vals in values_by_date)
    )
    return ExposurePair(epe, ene), pfe
