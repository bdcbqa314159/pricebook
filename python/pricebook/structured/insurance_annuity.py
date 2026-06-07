"""Insurance annuity products with guarantees (variable annuities).

* :func:`gmab`         — Guaranteed Minimum Accumulation Benefit.
* :func:`gmdb`         — Guaranteed Minimum Death Benefit.
* :func:`gmwb`         — Guaranteed Minimum Withdrawal Benefit.
* :func:`ratchet_gmab` — GMAB with periodic ratchet reset.

References:
    Hardy (2003). *Investment Guarantees: Modeling and Risk Management for
        Variable Annuities and Equity-Linked Insurance*.  Wiley.
    Bauer, Kling & Russ (2008). A Universal Pricing Framework for Guaranteed
        Minimum Benefits in Variable Annuities.  ASTIN Bulletin 38(2).
    Milevsky & Salisbury (2006). Financial Valuation of Guaranteed Minimum
        Withdrawal Benefits.  Insurance: Mathematics and Economics 38(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(r: float, t: float) -> float:
    return math.exp(-r * t)


def _gbm_paths(
    S0: float,
    mu: float,
    vol: float,
    T: float,
    steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return shape (n_paths, steps+1) GBM paths."""
    dt = T / steps
    z = rng.standard_normal((n_paths, steps))
    log_returns = (mu - 0.5 * vol ** 2) * dt + vol * math.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# GMAB
# ---------------------------------------------------------------------------

@dataclass
class GMABResult:
    """Output of :func:`gmab`."""
    guarantee_value: float
    account_value: float
    guarantee_cost: float
    fee_pv: float
    net_cost: float
    prob_in_the_money: float

    def to_dict(self) -> dict:
        return vars(self)


def gmab(
    initial_investment: float,
    guarantee_rate: float,
    fee_rate: float,
    vol: float,
    risk_free_rate: float,
    T: float,
    n_paths: int = 50_000,
    seed: int = 42,
) -> GMABResult:
    """Guaranteed Minimum Accumulation Benefit (GMAB).

    At maturity the policyholder receives max(account_value, guaranteed_amount).
    The guarantee cost equals the present value of max(G - A_T, 0) — a put on the
    account value.  The account grows at risk_free_rate less the continuous fee drag.

    Args:
        initial_investment: Premium paid at inception.
        guarantee_rate:     Guaranteed annual growth rate applied to the initial
                            premium (e.g. 0.0 means return-of-premium).
        fee_rate:           Annual continuously-deducted management fee (e.g. 0.015).
        vol:                Annual volatility of the underlying sub-account.
        risk_free_rate:     Risk-neutral drift / discount rate.
        T:                  Policy term in years.
        n_paths:            Monte Carlo paths.
        seed:               Random seed.

    Returns:
        :class:`GMABResult`.
    """
    rng = np.random.default_rng(seed)
    G = initial_investment * math.exp(guarantee_rate * T)
    # Account drifts at (r - fee) under risk-neutral measure
    mu = risk_free_rate - fee_rate
    steps = max(int(T * 52), 1)
    paths = _gbm_paths(initial_investment, mu, vol, T, steps, n_paths, rng)
    A_T = paths[:, -1]

    # Put payoff: policyholder receives G when A_T < G
    payoffs = np.maximum(G - A_T, 0.0)
    guarantee_cost = _df(risk_free_rate, T) * float(np.mean(payoffs))

    # PV of fee stream — discount each step's fee at its own time
    dt = T / steps
    step_times = np.arange(1, steps + 1) * dt  # shape (steps,)
    disc_factors = np.exp(-risk_free_rate * step_times)  # shape (steps,)
    fee_pv = float(np.mean(
        np.sum(fee_rate * paths[:, 1:] * dt * disc_factors, axis=1)
    ))

    account_value = float(np.mean(A_T))
    prob_itm = float(np.mean(A_T < G))
    net_cost = guarantee_cost - fee_pv  # net burden after fees collected

    return GMABResult(
        guarantee_value=G,
        account_value=account_value,
        guarantee_cost=guarantee_cost,
        fee_pv=fee_pv,
        net_cost=net_cost,
        prob_in_the_money=prob_itm,
    )


# ---------------------------------------------------------------------------
# GMDB
# ---------------------------------------------------------------------------

@dataclass
class GMDBResult:
    """Output of :func:`gmdb`."""
    guarantee_value: float
    expected_payout: float
    cost: float
    mortality_weighted_cost: float

    def to_dict(self) -> dict:
        return vars(self)


def gmdb(
    initial_investment: float,
    fee_rate: float,
    vol: float,
    risk_free_rate: float,
    mortality_rates: list[float],
    max_age: int = 100,
    n_paths: int = 20_000,
    seed: int = 42,
) -> GMDBResult:
    """Guaranteed Minimum Death Benefit (GMDB).

    On death in year t the beneficiary receives max(account_value_t, initial_investment).
    The cost is the mortality-weighted PV of the put option at each year.

    Args:
        initial_investment: Single premium.
        fee_rate:           Continuous annual fee charged against the account.
        vol:                Sub-account volatility.
        risk_free_rate:     Risk-neutral discount rate.
        mortality_rates:    Annual probability of death q_x for each year
                            (length = projection horizon in years).
        max_age:            Maximum projection age (cap on horizon).
        n_paths:            Monte Carlo paths per cohort year.
        seed:               Random seed.

    Returns:
        :class:`GMDBResult`.
    """
    rng = np.random.default_rng(seed)
    horizon = min(len(mortality_rates), max_age)
    mu = risk_free_rate - fee_rate

    total_cost = 0.0
    total_payout = 0.0
    survival = 1.0  # probability of being alive at start of year t

    for t in range(1, horizon + 1):
        q_t = float(mortality_rates[t - 1])
        prob_death = survival * q_t
        survival *= (1.0 - q_t)

        # Account value at year t
        steps = max(t * 12, 1)
        paths = _gbm_paths(initial_investment, mu, vol, t, steps, n_paths, rng)
        A_t = paths[:, -1]

        # Put payoff: insurer pays max(G - A_t, 0)
        put_payoffs = np.maximum(initial_investment - A_t, 0.0)
        pv_put = _df(risk_free_rate, t) * float(np.mean(put_payoffs))
        pv_payout = _df(risk_free_rate, t) * float(np.mean(np.maximum(A_t, initial_investment)))

        total_cost += prob_death * pv_put
        total_payout += prob_death * pv_payout

        if survival < 1e-6:
            break

    return GMDBResult(
        guarantee_value=initial_investment,
        expected_payout=total_payout,
        cost=total_cost,
        mortality_weighted_cost=total_cost,
    )


# ---------------------------------------------------------------------------
# GMWB
# ---------------------------------------------------------------------------

@dataclass
class GMWBResult:
    """Output of :func:`gmwb`."""
    guarantee_cost: float
    expected_withdrawals: float
    ruin_probability: float
    optimal_withdrawal: float

    def to_dict(self) -> dict:
        return vars(self)


def gmwb(
    initial_investment: float,
    withdrawal_rate: float,
    fee_rate: float,
    vol: float,
    risk_free_rate: float,
    T: float,
    n_paths: int = 30_000,
    seed: int = 42,
) -> GMWBResult:
    """Guaranteed Minimum Withdrawal Benefit (GMWB).

    The policyholder withdraws ``withdrawal_rate * initial_investment`` per year
    regardless of sub-account performance.  If the account is depleted before
    maturity the insurer covers the shortfall.  The guarantee cost is the PV of
    insurer-covered shortfalls.

    Args:
        initial_investment: Single premium.
        withdrawal_rate:    Annual withdrawal as fraction of initial premium (e.g. 0.07).
        fee_rate:           Continuous annual fee charged before withdrawals.
        vol:                Sub-account volatility.
        risk_free_rate:     Risk-neutral discount rate.
        T:                  Policy term in years.
        n_paths:            Monte Carlo paths.
        seed:               Random seed.

    Returns:
        :class:`GMWBResult`.
    """
    rng = np.random.default_rng(seed)
    annual_withdrawal = withdrawal_rate * initial_investment
    n_years = int(math.ceil(T))
    dt = 1.0
    mu = risk_free_rate - fee_rate

    # Simulate year-by-year account evolution
    log_factor_mean = (mu - 0.5 * vol ** 2) * dt
    log_factor_std = vol * math.sqrt(dt)

    account = np.full(n_paths, float(initial_investment))
    total_shortfall_pv = np.zeros(n_paths)
    total_withdrawal_pv = np.zeros(n_paths)
    ruin_mask = np.zeros(n_paths, dtype=bool)

    for t in range(1, n_years + 1):
        z = rng.standard_normal(n_paths)
        account *= np.exp(log_factor_mean + log_factor_std * z)
        account -= annual_withdrawal
        shortfall = np.maximum(-account, 0.0)
        account = np.maximum(account, 0.0)
        df_t = _df(risk_free_rate, t)
        total_shortfall_pv += shortfall * df_t
        total_withdrawal_pv += annual_withdrawal * df_t
        ruin_mask |= shortfall > 0.0

    guarantee_cost = float(np.mean(total_shortfall_pv))
    expected_withdrawals = float(np.mean(total_withdrawal_pv))
    ruin_probability = float(np.mean(ruin_mask))

    return GMWBResult(
        guarantee_cost=guarantee_cost,
        expected_withdrawals=expected_withdrawals,
        ruin_probability=ruin_probability,
        optimal_withdrawal=annual_withdrawal,
    )


# ---------------------------------------------------------------------------
# Ratchet GMAB
# ---------------------------------------------------------------------------

def ratchet_gmab(
    initial_investment: float,
    fee_rate: float,
    vol: float,
    risk_free_rate: float,
    T: float,
    reset_frequency: float = 1.0,
    n_paths: int = 30_000,
    seed: int = 42,
) -> GMABResult:
    """GMAB with periodic ratchet (high-water-mark) guarantee resets.

    At each reset date the guarantee floor is updated to
    ``max(current_guarantee, account_value)``.  Because the floor can only
    rise, this is more expensive than a plain GMAB.

    Args:
        initial_investment: Single premium.
        fee_rate:           Continuous annual fee.
        vol:                Sub-account volatility.
        risk_free_rate:     Risk-neutral discount rate.
        T:                  Policy term in years.
        reset_frequency:    Years between ratchet resets (default: annual).
        n_paths:            Monte Carlo paths.
        seed:               Random seed.

    Returns:
        :class:`GMABResult` with the ratcheted guarantee cost.
    """
    rng = np.random.default_rng(seed)
    mu = risk_free_rate - fee_rate
    steps_per_reset = max(int(reset_frequency * 52), 1)
    n_resets = int(math.ceil(T / reset_frequency))
    dt = reset_frequency / steps_per_reset

    log_mean = (mu - 0.5 * vol ** 2) * dt
    log_std = vol * math.sqrt(dt)

    account = np.full(n_paths, float(initial_investment))
    guarantee = np.full(n_paths, initial_investment)

    for _ in range(n_resets):
        for _step in range(steps_per_reset):
            z = rng.standard_normal(n_paths)
            account *= np.exp(log_mean + log_std * z)
        # Ratchet: guarantee steps up to current account value if higher
        guarantee = np.maximum(guarantee, account)

    # Final payoff: max(account, guarantee)
    payoffs = np.maximum(guarantee - account, 0.0)
    guarantee_cost = _df(risk_free_rate, T) * float(np.mean(payoffs))

    # PV of fees (approximate: treat as continuous on mean path)
    fee_pv = fee_rate * initial_investment * (
        (1.0 - math.exp(-risk_free_rate * T)) / risk_free_rate if risk_free_rate > 1e-10
        else T
    )

    account_value = float(np.mean(account))
    guarantee_value = float(np.mean(guarantee))
    prob_itm = float(np.mean(account < guarantee))
    net_cost = guarantee_cost - fee_pv

    return GMABResult(
        guarantee_value=guarantee_value,
        account_value=account_value,
        guarantee_cost=guarantee_cost,
        fee_pv=fee_pv,
        net_cost=net_cost,
        prob_in_the_money=prob_itm,
    )
