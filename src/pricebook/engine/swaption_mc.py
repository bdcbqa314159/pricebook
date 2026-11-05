"""The European swaption under the LMM by Monte Carlo (L4). MCINERNEY-ZASTAWNIAK §5.9-5.10.

Under the `P_N`-forward measure, with `N` the bond the swap matures on,

    V(0) = notional · B(0,T_N) · E[ (A/B_N)(T_first) · (ω·(S(T_first) − K))⁺ ]

where `A` is the swap annuity (5.32) and `S` the forward swap rate (5.31), both evaluated at expiry
from the simulated forwards. Dividing by `B(T_first,T_N)` is the change of numéraire: it is the asset
the simulation is a martingale measure for, so no stochastic discount factor is needed beyond the
forwards themselves.

**Terminal forwards suffice, and that is tranche 17's ruling rather than an accident.** `A`, `B_N`
and `S` at expiry are functions of the forwards at expiry alone, so tranche 16's
`simulate_terminal_forwards` serves this product unchanged. The Bermudan (tranche 18) needs the
forwards at *each* exercise date; widening the engine's return from an endpoint to a path lands
there, pulled by that real second consumer, rather than speculatively here (§6).

**The measure is checked, not assumed.** `A/B_N` is the annuity only when `N` is the span's own end
bond. Under any other numéraire the same arithmetic returns a plausible wrong number, so a mismatch
is a `PricingFailure` — the one failure mode §3.4 exists for.

**The one-period span is the exact anchor.** There `A/B_N = τ_i` and `S = f_i` hold identically, per
path, so the payoff *is* `τ_i(f_i − K)⁺` — a caplet. This engine and tranche 16's caplet engine, on
one seed, therefore agree to machine precision after a deterministic scaling: a cross-product oracle
with different code on each side, and no Monte Carlo error in the comparison (§4).

Provenance:
  stock:  parked/ng/.../swaption_mc.py — read for SHAPE and edge cases only via
          docs/stock/mc_base_inventory.md; never imported (§5)
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §5.9 eqs 5.31/5.32 (the swap rate and annuity),
          §5.10 (the payer/receiver payoff and the annuity as numéraire), §5.12 eq 5.40 via
          models/swap_rate_vol.py for the bounded cross-check; JAECKEL-2002 §12.3 eqs 12.15/12.16
          (the coterminal span) and §13.2 eq 13.2 via engine/mc_path_engine.py for the evolution
  oracle: tests/L4/test_swaption_mc.py — the one-period reduction to tranche 16's caplet engine at
          1e-12 and to our Black within 4 SE; the price rising with the correlation by >5%; the
          bounded agreement with Black(Rebonato) inside the freezing slack, with that band PROVEN
          able to fail; the correlation sensitivity under common random numbers; payer/receiver
          parity at the forward swap rate; determinism; and the measure/span refusals
  crosscheck: quantlib LfmSwaptionEngine / Gaussian1dSwaptionEngine — deferred to the optional
          QuantLib tier
  slice:  tranche 17 — the European swaption under the LMM
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.engine.mc_path_engine import PathGrid, simulate_terminal_forwards
from pricebook.foundation.mc_estimator import estimate
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel
from pricebook.models.swap_rate_vol import tenor_discount_factors

#: The grid's horizon must be the swaption's expiry. Compared with a relative tolerance rather than
#: `==` because a caller computing `T_first` by summing accruals will land a few ULP away from our
#: own sum, and refusing that would be refusing a correct input.
_HORIZON_TOLERANCE = 1e-12


def price_european_swaption(
    swaption, model: LiborMarketModel, grid: PathGrid, config: MonteCarloConfig
) -> PricingResult | PricingFailure:
    """Simulate the LMM to the swaption's expiry and average the discounted payoff."""
    tenor = model.tenor
    span = swaption.span
    if span.end > len(tenor):
        return PricingFailure(
            f"the swap span [{span.first}, {span.end}) does not fit the tenor's "
            f"{len(tenor)} forwards"
        )
    if model.numeraire_index != span.end:
        return PricingFailure(
            f"the model's numeraire index is {model.numeraire_index}, but pricing this swaption "
            f"needs the span's end bond {span.end} as numeraire"
        )

    expiry = math.fsum(tenor.accruals[: span.first])
    if expiry <= 0.0:
        return PricingFailure(
            f"the swap span starts at index {span.first}, so the swaption expires at t = 0 and has "
            f"no optionality to value"
        )
    if abs(grid.horizon - expiry) > _HORIZON_TOLERANCE * max(1.0, expiry):
        return PricingFailure(
            f"the grid's horizon is {grid.horizon} but the swaption expires at {expiry}; the "
            f"simulation must end where the payoff is observed"
        )

    payoffs = _payoffs(
        simulate_terminal_forwards(model, grid, config),
        tenor.accruals,
        span,
        swaption.strike,
        swaption.kind.omega,
    )

    discount = tenor_discount_factors(tenor)[span.end]
    pv = estimate(payoffs).mean * discount * swaption.notional.amount
    return PricingResult(pv=Money(pv, swaption.notional.currency))


def _payoffs(paths, accruals, span, strike: float, omega: float) -> list[float]:
    """`(A/B_end) · (ω(S − K))⁺` for every path — eqs 5.31/5.32 at expiry, vectorised.

    **Vectorised over paths because the payoff, not the simulation, was the cost.** The first version
    walked each path in Python to build its discount factors; the simulation it consumes was already
    vectorised, so the oracle file spent most of its time here — the exact failure the `slow` gate's
    own docstring names (*"a per-path Python loop made a file 60-70x its vectorised cost"*). The
    arithmetic is unchanged: `B(T_first,T_j)` is the running product of `1/(1 + f_k τ_k)` across the
    span, `A = Σ_m τ_m B_m`, `S = (1 − B_end)/A`, and the annuity is taken relative to the numéraire
    bond `B_end`.

    `A/B_end` collapses to exactly `τ_i` for a one-period span, and `S` to exactly `f_i`, which is
    the identity the exact caplet anchor rests on — so the span columns are sliced rather than the
    whole curve, keeping that reduction exact rather than approximately recovered."""
    first, count = span.first, span.count
    forwards = np.asarray(paths, dtype=float)[:, first : first + count]
    tau = np.asarray(accruals[first : first + count], dtype=float)

    # B(T_first, T_{first+1..end}) — the running product along the span, one column per payment date.
    bonds = np.cumprod(1.0 / (1.0 + forwards * tau), axis=1)
    annuity = bonds @ tau
    numeraire_bond = bonds[:, -1]
    swap_rate = (1.0 - numeraire_bond) / annuity

    intrinsic = np.maximum(omega * (swap_rate - strike), 0.0)
    return (annuity / numeraire_bond * intrinsic).tolist()
