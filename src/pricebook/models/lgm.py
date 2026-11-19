"""The Linear Gauss-Markov (LGM) one-factor short-rate model (L3). `LICHTERS-2015` ch 11.1.

LGM is Hull-White in the parameterisation whose Monte-Carlo is **exact**. The state `z(t)` is
**driftless** with **deterministic** variance `zeta(t)`, so a simulation draws `z ~ N(0, zeta(t))`
in a single step - there is no SDE to discretise and no stepping error - and reads bond prices off
it through a deterministic `H(t)`. That is what an exposure engine wants, and it is why the book
builds exposure on LGM rather than on the mean-reverting Hull-White SDE.

**The source, transcribed from both renderings and reconciled (§4):**

    N(t)     = exp(H_t z_t + 1/2 H_t^2 zeta_t) / P(0,t)                             (11.1)
    P(t,T,z) = (P(0,T)/P(0,t)) * exp[ -(H_T - H_t) z_t - 1/2 (H_T^2 - H_t^2) zeta_t ]  (11.4)

**The convexity term is `(H_T^2 - H_t^2)`, not `(H_T - H_t)^2`** - the two agree only where
`H_t = 0`, which is `t = 0`, the one date at which nothing has been simulated and therefore the one
date an LGM is never used at. Tranche 57's spec carried the second form; the source has the first
(confirmed in `-raw` and `-layout`), and both of the oracles the spec itself named catch the
difference: the numéraire martingale by 7.5 SE and the Monte-Carlo companion by 20.8 SE. Measured on
this tranche's parameters the two terms are identical at `t = 0` and **2.27x apart at `t = 3`**.

**This model is BUILT FROM a `HullWhiteParams` pair, so "LGM is Hull-White" is true by construction
rather than by coincidence** - `H(t) = hw_beta(kappa, 0, t)` and
`zeta(t) = sigma^2 (e^{2 kappa t} - 1) / (2 kappa)`, which reproduces Hull-White's own
`bond_option_variance` to machine precision. That is the strongest possible form of the reduction
anchor **and its blind spot**: being exact in closed form, it cannot see the simulation, which is
the entire point of the model. Hence the mandatory MC companion (§4, T39). `kappa != 0` needs no
assertion - `HullWhiteParams` already refuses a non-positive mean reversion.

**No new parameter type.** The LGM's numbers *are* Hull-White's two; what differs is the
parameterisation - `H` and `zeta` are functions of them, not data - so inventing an `LGMParams`
would be a second spelling of one fact, and a chance for the two to disagree. (The tranche's flagged
§6 question, answered by what the type would hold.)

**The era invariant holds:** the curve is passed in, so the model evolves rates off whatever the
snapshot carries - a single discount curve or a `CurveSet`'s OIS curve. Nothing here makes a
`CurveSet` mandatory.

Provenance:
  stock:  none - the archived predecessor had no LGM
  source: LICHTERS-2015 (docs/references.md) ch 11.1 - eq (11.1) the numéraire, (11.2) the
          transition density, (11.3) the reduced zero bond and (11.4) the zero bond price;
          transcribed in both renderings and reconciled
  oracle: tests/L3/test_lgm.py - the exact reduction to Hull-White (the state reproduces
          `bond_option_variance` to machine precision, and `P(0,T,z0) = P(0,T)`), the MANDATORY MC
          companion pricing a zero-bond option against the analytic Hull-White price within 4*SE,
          the numéraire martingale `E[P(t,T)/N(t)] = P(0,T)` with its resolution stated against the
          eq-(11.4) convexity term AND against the eq-(11.1) half it CANNOT resolve (1.60 SE inside
          a 4*SE band), the POINTWISE transcription of eq (11.1) that covers that gap exactly
          (re-audit #6, F-6.1 -- the numéraire had only ever appeared as a divisor, so §4's
          transcribe-every-load-bearing-formula-once was unmet), and `zeta`'s monotonicity and
          integral form
  crosscheck: quantlib Gsr / LgmImpliedYieldTermStructure - deferred to the optional QuantLib tier
  slice:  tranche 57 - the LGM short-rate model
"""

from __future__ import annotations

import math

from pricebook.foundation.variates import standard_normals
from pricebook.foundation.rng import UniformSource
from pricebook.market.curve import CurveHandle
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.hull_white import HullWhiteParams, hw_beta


def lgm_H(params: HullWhiteParams, years: float) -> float:
    """`H(t)` - the deterministic scaling, which is Hull-White's beta measured from today.

    Written through `hw_beta` rather than as its own closed form, for the reason that function
    already gives: the two are the same object, and the shared one cannot drift."""
    return hw_beta(params.kappa, 0.0, years)


def lgm_zeta(params: HullWhiteParams, years: float) -> float:
    """`zeta(t) = integral_0^t alpha_s^2 ds` with `alpha_s = sigma e^{kappa s}`, so
    `zeta(t) = sigma^2 (e^{2 kappa t} - 1) / (2 kappa)`.

    The state's variance, hence non-negative and non-decreasing - both asserted, because they are
    requirements the finance imposes rather than consequences of this transcription."""
    if years < 0.0:
        raise ValueError(f"the state variance needs a non-negative horizon, got {years}")
    return params.sigma**2 * math.expm1(2.0 * params.kappa * years) / (2.0 * params.kappa)


def lgm_numeraire(
    params: HullWhiteParams, years: float, state: float, curve: CurveHandle
) -> float:
    """`N(t) = exp(H_t z_t + 1/2 H_t^2 zeta_t) / P(0,t)` - eq (11.1)."""
    h = lgm_H(params, years)
    return math.exp(h * state + 0.5 * h * h * lgm_zeta(params, years)) / curve.df_at(years)


def lgm_zero_bond(
    params: HullWhiteParams, years: float, maturity: float, state: float, curve: CurveHandle
) -> float:
    """`P(t,T,z)` - eq (11.4), the stochastic zero bond.

    The convexity is `1/2 (H_T^2 - H_t^2) zeta_t`, which is **not** `1/2 (H_T - H_t)^2 zeta_t`; see
    the module docstring for why the difference is invisible at `t = 0` and material everywhere
    else. At `t = 0` this returns `P(0,T)` for `z = 0`, which is the source's "by construction
    calibrated to today's yield term structure".
    """
    if maturity < years:
        raise ValueError(
            f"a zero bond cannot mature ({maturity}) before it is observed ({years})"
        )
    h_t = lgm_H(params, years)
    h_T = lgm_H(params, maturity)
    zeta = lgm_zeta(params, years)
    exponent = -(h_T - h_t) * state - 0.5 * (h_T * h_T - h_t * h_t) * zeta
    return curve.df_at(maturity) / curve.df_at(years) * math.exp(exponent)


def simulate_lgm_states(
    params: HullWhiteParams, years: float, config: MonteCarloConfig
) -> tuple[float, ...]:
    """`z(t) ~ N(0, zeta(t))` - one exact draw per path, and the reason this model was pulled.

    There is no time stepping: the state is Gaussian with a deterministic variance, so a horizon is
    reached in a single draw and carries **no discretisation error** to band against. A Heston or
    Hull-White SDE simulation has to step; this does not, which is what makes it the clean engine
    for an exposure grid.

    `zeta(0) = 0`, so every path is exactly `0.0` at the anchor rather than a small random number -
    the degenerate case falling out of the arithmetic instead of needing a branch.
    """
    deviation = math.sqrt(lgm_zeta(params, years))
    count = config.paths + (config.paths % 2)  # Box-Muller consumes uniforms in pairs
    source = UniformSource(config.seed)
    normals = standard_normals(source.uniforms(count))
    return tuple(deviation * z for z in normals[: config.paths])
