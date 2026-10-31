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
    """The correlations `C` depends on: nominal-real and real-index.

    **Only these two.** `rho_nominal_index` is a JY parameter but enters only the cap/floor
    variance, so carrying it here would be a field with no consumer -- the smell this project has
    now caught four times, every one of them in stock code."""

    nominal_real: float
    real_index: float

    def __post_init__(self) -> None:
        for name, rho in (("nominal_real", self.nominal_real), ("real_index", self.real_index)):
            if not -1.0 <= rho <= 1.0:
                raise ValueError(f"{name} correlation must lie in [-1, 1], got {rho}")


def hw_beta(kappa: float, t: float, maturity: float) -> float:
    """`beta(t,T) = (1 - e^{-kappa(T-t)})/kappa` -- eqs 2.4.3 / 2.4.16.

    Defined once and shared by both factors, so the nominal and real betas cannot drift. As
    `kappa -> 0` it tends to `T - t`, which is the limit that catches a misplaced `kappa`: every
    wrong-denominator variant still looks plausible at a realistic `kappa`."""
    return (1.0 - math.exp(-kappa * (maturity - t))) / kappa


def yoy_convexity(
    t: float,
    period_start: float,
    period_end: float,
    nominal: HullWhiteParams,
    real: HullWhiteParams,
    index_vol: float,
    correlations: JYCorrelations,
) -> float:
    """`C(t, T_{i-1}, T_i)` -- eq 2.5.11. Zero exactly when `real.sigma` is zero."""
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
