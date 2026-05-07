"""MC instrument adapters: wire existing instruments through the unified engine.

Each adapter function replaces a specific instrument's MC with a call
to MCEngine. The adapter defines the ProcessSpec and payoff, then
delegates to the engine.

    from pricebook.mc_instrument_adapters import (
        autocallable_mc, cliquet_mc, basket_mc, tarf_mc,
        heston_european_mc, sabr_european_mc,
        xva_exposure_mc,
    )

These are drop-in alternatives to the per-instrument MC code.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.mc_engine import MCEngine, TimeGrid, MCResult
from pricebook.mc_processes import (
    BlackScholesProcess, HestonProcess, SABRProcess, CorrelatedGBMProcess,
)
from pricebook.mc_payoffs import (
    european_call, european_put, autocall_payoff, cliquet_payoff,
    basket_call, worst_of_put,
)
from pricebook.mc_exposure import ExposureEngine


# ---------------------------------------------------------------------------
# Batch 2: Autocallable
# ---------------------------------------------------------------------------

def autocallable_mc(
    spot: float,
    rate: float,
    vol: float,
    T: float,
    autocall_barrier: float = 1.0,
    coupon: float = 0.08,
    put_barrier: float = 0.7,
    observation_freq: int = 4,
    n_paths: int = 100_000,
    n_steps: int | None = None,
    seed: int = 42,
) -> MCResult:
    """Autocallable note via MCEngine.

    Args:
        spot: initial spot.
        autocall_barrier: as fraction of spot (1.0 = ATM).
        coupon: annual coupon if autocalled.
        put_barrier: downside barrier as fraction of spot.
        observation_freq: observations per year.
    """
    barrier_abs = spot * autocall_barrier
    put_abs = spot * put_barrier
    steps = n_steps or max(int(T * observation_freq * 3), 12)

    engine = MCEngine(
        BlackScholesProcess(spot, rate, vol),
        TimeGrid.uniform(T, steps),
        n_paths, seed, antithetic=True,
    )
    payoff = autocall_payoff(
        autocall_barrier=barrier_abs, autocall_coupon=coupon,
        put_barrier=put_abs, put_strike=spot,
        observation_freq=observation_freq,
    )
    return engine.price(payoff, math.exp(-rate * T))


# ---------------------------------------------------------------------------
# Batch 2: Cliquet
# ---------------------------------------------------------------------------

def cliquet_mc(
    spot: float,
    rate: float,
    vol: float,
    T: float,
    cap: float = 0.05,
    floor: float = -0.03,
    global_floor: float = 0.0,
    n_periods: int = 12,
    n_paths: int = 100_000,
    seed: int = 42,
) -> MCResult:
    """Cliquet (ratchet) option via MCEngine."""
    engine = MCEngine(
        BlackScholesProcess(spot, rate, vol),
        TimeGrid.uniform(T, n_periods),
        n_paths, seed, antithetic=True,
    )
    payoff = cliquet_payoff(cap=cap, floor=floor, global_floor=global_floor)
    return engine.price(payoff, math.exp(-rate * T))


# ---------------------------------------------------------------------------
# Batch 3: Basket / Multi-asset
# ---------------------------------------------------------------------------

def basket_mc(
    spots: list[float],
    rate: float,
    vols: list[float],
    correlation: np.ndarray,
    strike: float,
    T: float,
    weights: list[float] | None = None,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> MCResult:
    """Basket option via MCEngine with CorrelatedGBM."""
    n = len(spots)
    mus = [rate] * n
    process = CorrelatedGBMProcess(spots, mus, vols, correlation)
    engine = MCEngine(
        process, TimeGrid.uniform(T, 1),
        n_paths, seed, antithetic=True,
    )
    if option_type == "call":
        payoff = basket_call(strike, weights)
    else:
        payoff = worst_of_put(strike)
    return engine.price(payoff, math.exp(-rate * T))


# ---------------------------------------------------------------------------
# Batch 4: Stochastic vol (Heston, SABR)
# ---------------------------------------------------------------------------

def heston_european_mc(
    spot: float,
    strike: float,
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
    use_conditional: bool = False,
) -> MCResult:
    """Heston European via MCEngine or conditional MC.

    Args:
        use_conditional: if True, uses conditional MC (10-50x variance reduction).
    """
    if use_conditional:
        from pricebook.mc_conditional import conditional_mc_heston
        return conditional_mc_heston(
            spot, v0, kappa, theta, xi, rho,
            strike, T, rate, option_type, n_paths, n_steps, seed,
        )

    process = HestonProcess(spot, v0, rate, kappa, theta, xi, rho)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed, antithetic=True)

    payoff = european_call(strike) if option_type == "call" else european_put(strike)
    return engine.price(payoff, math.exp(-rate * T))


def sabr_european_mc(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
) -> MCResult:
    """SABR European via MCEngine."""
    process = SABRProcess(forward, alpha, beta, rho, nu)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)

    # SABR payoff: on forward (first factor), NOT log-space
    def call_payoff(paths, times):
        f_terminal = paths[:, -1, 0]
        return np.maximum(f_terminal - strike, 0.0)

    return engine.price(call_payoff, 1.0)  # no discounting (forward measure)


# ---------------------------------------------------------------------------
# Batch 5: TARF (Target Redemption Forward)
# ---------------------------------------------------------------------------

def tarf_mc(
    spot: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    strike: float,
    target: float,
    T: float,
    n_fixings: int = 12,
    leverage: float = 2.0,
    n_paths: int = 100_000,
    seed: int = 42,
) -> MCResult:
    """TARF via MCEngine.

    Target Redemption Forward: accumulates gains until target reached,
    then knocks out. Losses are leveraged.
    """
    drift = rate_dom - rate_for
    process = BlackScholesProcess(spot, drift, vol)
    engine = MCEngine(process, TimeGrid.uniform(T, n_fixings), n_paths, seed)

    def tarf_payoff(paths, times):
        p = paths[:, :, 0] if paths.ndim == 3 else paths
        spots = np.exp(p)
        n_p, n_fix = spots.shape[0], spots.shape[1] - 1

        accumulated = np.zeros(n_p)
        pv = np.zeros(n_p)
        knocked = np.zeros(n_p, dtype=bool)

        for i in range(1, n_fix + 1):
            s = spots[:, i]
            active = ~knocked

            # Gain (S > K): accumulate toward target
            gain = np.where(s > strike, (s - strike) / spot, 0.0)
            # Loss (S < K): leveraged
            loss = np.where(s < strike, leverage * (strike - s) / spot, 0.0)

            pv[active] += gain[active] - loss[active]
            accumulated[active] += gain[active]

            # Knockout: target reached
            knockout = active & (accumulated >= target)
            knocked |= knockout

        return pv

    return engine.price(tarf_payoff, math.exp(-rate_dom * T))


# ---------------------------------------------------------------------------
# Batch 6: XVA exposure via engine
# ---------------------------------------------------------------------------

def equity_xva_mc(
    spot: float,
    vol: float,
    rate: float,
    notional: float,
    T: float,
    counterparty_spread: float = 0.01,
    n_paths: int = 10_000,
    seed: int = 42,
) -> dict:
    """Equity exposure + CVA/DVA/FVA via ExposureEngine.

    Replaces desk-specific equity_mc_xva().
    """
    process = BlackScholesProcess(spot, rate, vol)

    def revalue(paths, step):
        p = paths[:, step] if paths.ndim == 2 else paths[:, step, 0]
        return notional * (np.exp(p) / spot - 1)

    engine = ExposureEngine(
        process, TimeGrid.uniform(T, max(int(T * 4), 4)),
        n_paths, revalue=revalue,
        counterparty_spread=counterparty_spread,
        seed=seed,
    )
    return engine.compute().to_dict()


def rates_xva_mc(
    r0: float,
    fixed_rate: float,
    notional: float,
    T: float,
    kappa: float = 0.5,
    sigma: float = 0.01,
    counterparty_spread: float = 0.01,
    n_paths: int = 10_000,
    seed: int = 42,
) -> dict:
    """IRS exposure + CVA via ExposureEngine with OU/HW rate model.

    Replaces desk-specific swap_mc_xva().
    """
    from pricebook.mc_processes import OUProcess

    process = OUProcess(r0, kappa, r0, sigma)

    def revalue(paths, step):
        rate = paths[:, step] if paths.ndim == 2 else paths[:, step, 0]
        remaining_T = max(T - step * T / 20, 0.01)
        return notional * (rate - fixed_rate) * remaining_T

    engine = ExposureEngine(
        process, TimeGrid.uniform(T, 20),
        n_paths, revalue=revalue,
        counterparty_spread=counterparty_spread,
        seed=seed,
    )
    return engine.compute().to_dict()
