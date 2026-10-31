"""Jarrow-Yildirim: the convexity correction a year-on-year inflation swap needs (L3).

JY models the nominal and real forward curves in an HJM framework related by the
foreign-currency analogy, in a Hull-White specification. For the YoY swap, all of that reduces
to **one scalar correction** per period: the expected future real bond price under the nominal
forward measure is the current forward price times `e^C` (eq 2.5.9).

    beta(t,T) = (1 - e^{-k(T-t)}) / k                                    (2.4.3, 2.4.16)

    C(t,T0,T1) = sr * beta_r(T0,T1) * [ beta_r(t,T0)*(rho_rI*sI - 0.5*beta_r(t,T0))
                                      + (rho_nr*sn/(kn+kr))*(1 + kr*beta_n(t,T0))
                                      - (rho_nr*sn/(kn+kr))*beta_n(t,T0) ]   (2.5.11)

**`sigma_r` is the overall factor, and that is the tranche's anchor.** The thesis states it
directly after 2.5.11 -- *"This term accounts for the stochasticity of real rates. Indeed it
vanishes for sigma_r = 0."* So `sigma_r -> 0` recovers the deterministic-real-rate limit
**exactly, by construction**, not as a numerical limit.

**`rho -> 0` does NOT remove it**, and that correction is why this module reads the source rather
than the stock. The oracle plan originally specified `rho -> 0` as the anchor, taken from
`_stock/.../inflation_advanced.py::yoy_convexity_adjustment`, which computes
`sigma_I * sigma_n * rho * tenor * year` -- a first-order approximation in which `rho` *is* a
factor, which drops the real-rate convexity entirely, and which takes a `real_vol` argument it
never uses. At `rho = 0` the surviving JY term is `sr*beta_r(T0,T1)*beta_r(t,T0)*(-0.5*beta_r(t,T0))`:
a pure real-rate convexity with no correlation in it. A test pins that so nobody simplifies 2.5.11
back toward the approximation.

**`rho_nI` is absent on purpose.** It is a real JY parameter but appears only in the cap/floor
log-price variance, never in `C` -- so it lands with the cap/floor, its first consumer.

  crosscheck: quantlib YoYInflationSwap / JY convexity -- deferred to the optional QuantLib tier
  (thread 0013/0014).

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py (the `JYParams` grouping and
          the beta helpers; its ZC/YoY structure mined for shape only. Its sibling
          `inflation_advanced.yoy_convexity_adjustment` is deliberately NOT carried -- see above)
  source: BERHAN-2012 (docs/references.md) ch 2, eqs 2.4.3, 2.4.16, 2.5.9, 2.5.11
  oracle: vanishes exactly at sigma_r=0 (the anchor); does NOT vanish at rho=0, asserted against
          the surviving term; matches an independent transcription of 2.5.11 over a 108-point
          parameter sweep; beta's closed form and its k->0 limit; monotone in sigma_r and span
  slice:  tranche 4 -- the year-on-year swap and Jarrow-Yildirim
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class HullWhiteParams:
    """One Hull-White factor: mean reversion `kappa` and volatility `sigma`.

    A bundle rather than four loose floats on the model, because JY has *two* such factors and
    naming them `kappa_n, sigma_n, kappa_r, sigma_r` invites passing the nominal pair where the
    real belongs. The grouping is the model's own structure -- the thesis derives it as two
    factors plus the index."""

    kappa: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError(f"mean reversion must be positive, got {self.kappa}")
        if self.sigma < 0.0:
            raise ValueError(f"volatility cannot be negative, got {self.sigma}")


@dataclass(frozen=True)
class JYCorrelations:
    """JY's three correlations: nominal-real, real-index, nominal-index.

    **`nominal_index` arrived in tranche 5, not tranche 4.** The YoY convexity `C` (2.5.11) reads
    only the first two; `rho_nI` enters solely the cap/floor log-ratio variance `V^2` (2.6.4, its
    eighth term). So it was deliberately absent while it had no consumer and is required now that
    it has one -- which is the pull discipline, and the reason it is **required rather than
    defaulted**: a default would be a hidden knob on a real model parameter."""

    nominal_real: float
    real_index: float
    nominal_index: float

    def __post_init__(self) -> None:
        for name, rho in (
            ("nominal_real", self.nominal_real),
            ("real_index", self.real_index),
            ("nominal_index", self.nominal_index),
        ):
            if not -1.0 <= rho <= 1.0:
                raise ValueError(f"{name} correlation must lie in [-1, 1], got {rho}")


def hw_beta(kappa: float, t: float, maturity: float) -> float:
    """`beta(t,T) = (1 - e^{-kappa(T-t)})/kappa` -- eqs 2.4.3 / 2.4.16.

    Defined once and shared by both factors, so the nominal and real betas cannot drift. As
    `kappa -> 0` it tends to `T - t`, which is the limit that catches a misplaced `kappa`: every
    wrong-denominator variant still looks plausible at a realistic `kappa`."""
    return (1.0 - math.exp(-kappa * (maturity - t))) / kappa


@dataclass(frozen=True)
class JYParams:
    """The JY parameter set `C` depends on: two Hull-White factors, the index vol, the correlations.

    Bundled because `yoy_convexity` otherwise takes seven arguments, two over the limit, and the
    rule's answer is always to group the cohesive ones. These four *are* the model's parameters —
    the same grouping `JarrowYildirim` carries alongside its snapshot — so the bundle is the
    model's own structure rather than an arbitrary split to satisfy a count."""

    nominal: HullWhiteParams
    real: HullWhiteParams
    index_vol: float
    correlations: JYCorrelations

    def __post_init__(self) -> None:
        if self.index_vol < 0.0:
            raise ValueError(f"index volatility cannot be negative, got {self.index_vol}")


def yoy_convexity(t: float, period_start: float, period_end: float, params: JYParams) -> float:
    """`C(t, T_{i-1}, T_i)` -- eq 2.5.11. Zero exactly when `params.real.sigma` is zero."""
    nominal, real = params.nominal, params.real
    index_vol, correlations = params.index_vol, params.correlations
    beta_r_period = hw_beta(real.kappa, period_start, period_end)
    beta_r_t = hw_beta(real.kappa, t, period_start)
    beta_n_t = hw_beta(nominal.kappa, t, period_start)
    cross = correlations.nominal_real * nominal.sigma / (nominal.kappa + real.kappa)
    bracket = (
        beta_r_t * (correlations.real_index * index_vol - 0.5 * beta_r_t)
        + cross * (1.0 + real.kappa * beta_n_t)
        - cross * beta_n_t
    )
    return real.sigma * beta_r_period * bracket


@dataclass(frozen=True)
class JarrowYildirim:
    """The JY model: its snapshot plus two Hull-White factors, the index vol and the correlations.

    Two fields: the snapshot and `JYParams`. Nine loose parameters would be four over the limit,
    and naming them `kappa_n, sigma_n, kappa_r, sigma_r, ...` invites passing the nominal pair
    where the real belongs — so the grouping is the model's own structure (two HW factors related
    by the foreign-currency analogy, plus the index), not an arbitrary split to satisfy a count.

    Carries `market` like every model here, so the engine reaches market data only through the
    model and a model/market mismatch cannot be expressed."""

    market: MarketSnapshot
    params: JYParams

    def convexity(self, t: float, period_start: float, period_end: float) -> float:
        """`C(t, T_{i-1}, T_i)` for this parameter set — the model's own capability, so a pricer
        composes it rather than reassembling 2.5.11 from loose parameters."""
        return yoy_convexity(t, period_start, period_end, self.params)

    def log_ratio_variance(self, t: float, period_start: float, period_end: float) -> float:
        """`V^2(t, T_{i-1}, T_i)` (2.6.4) — the cap/floor's variance, same reasoning as above."""
        return yoy_variance(t, period_start, period_end, self.params)


def yoy_variance(t: float, period_start: float, period_end: float, params: JYParams) -> float:
    """`V^2(t, T_{i-1}, T_i)` -- the variance of `ln(I(T_i)/I(T_{i-1}))` under the nominal
    `T_i`-forward measure. **Eq 2.6.4**, eight terms.

    Reduces **exactly** to `sigma_I^2 * (T_i - T_{i-1})` when both rate vols are zero: six terms
    carry `sigma_n` or `sigma_r` outright and the cross terms carry both. That reduction is
    reduction-*sanity* and not the correctness guarantee -- it kills six of the eight, so only the
    independently-transcribed sweep checks their signs and factors (thread 0031).

    A variance cannot be negative, and an eight-term expression with three signed correlations is
    exactly where a negative one would hide; a test sweeps adverse correlations for that reason,
    because the symptom would otherwise surface far from here as a domain error inside a sqrt."""
    kn, sn = params.nominal.kappa, params.nominal.sigma
    kr, sr = params.real.kappa, params.real.sigma
    si = params.index_vol
    rho_nr = params.correlations.nominal_real
    rho_ri = params.correlations.real_index
    rho_ni = params.correlations.nominal_index

    span = period_end - period_start          # T_i - T_{i-1}
    lead = period_start - t                   # T_{i-1} - t
    beta_n = hw_beta(kn, period_start, period_end)
    beta_r = hw_beta(kr, period_start, period_end)

    total = sn * sn / (2.0 * kn) * beta_n**2 * (1.0 - math.exp(-2.0 * kn * lead))
    total += si * si * span
    total += sr * sr / (2.0 * kr) * beta_r**2 * (1.0 - math.exp(-2.0 * kr * lead))
    total -= (
        2.0 * rho_nr * (sn * sr / (kn + kr)) * beta_n * beta_r
        * (1.0 - math.exp(-(kn + kr) * lead))
    )
    total += (sn * sn / kn**2) * (
        span + (2.0 / kn) * math.exp(-kn * span)
        - (1.0 / (2.0 * kn)) * math.exp(-2.0 * kn * span) - 3.0 / (2.0 * kn)
    )
    total += (sr * sr / kr**2) * (
        span + (2.0 / kr) * math.exp(-kr * span)
        - (1.0 / (2.0 * kr)) * math.exp(-2.0 * kr * span) - 3.0 / (2.0 * kr)
    )
    total -= 2.0 * rho_nr * (sn * sr / (kn * kr)) * (
        span - beta_n - beta_r + (1.0 - math.exp(-(kn + kr) * span)) / (kn + kr)
    )
    total += 2.0 * rho_ni * (sn * si / kn) * (span - beta_n)
    total -= 2.0 * rho_ri * (sr * si / kr) * (span - beta_r)
    return total
