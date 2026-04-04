"""AAD end-to-end pricing: all Greeks in one backward pass.

Provides AAD-aware pricing functions that accept Number inputs and
return Number outputs, enabling automatic differentiation through
the full pricing chain.
"""

from __future__ import annotations

import math

from pricebook.aad import Number, norm_cdf, log, exp


# ---------------------------------------------------------------------------
# Black-Scholes with Number inputs
# ---------------------------------------------------------------------------


def aad_black_scholes(
    S: Number, K: float, r: Number, sigma: Number, T: float, is_call: bool = True,
) -> Number:
    """Black-Scholes European option price with AAD.

    All Greeks (delta, rho, vega, etc.) come from one backward pass.
    """
    sqrt_T = math.sqrt(T)
    d1 = (log(S / K) + (r + sigma * sigma * 0.5) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = exp(r * (-T))

    if is_call:
        return S * norm_cdf(d1) - K * df * norm_cdf(d2)
    else:
        return K * df * norm_cdf(-d2) - S * norm_cdf(-d1)


# ---------------------------------------------------------------------------
# Swap PV with Number inputs
# ---------------------------------------------------------------------------


def aad_swap_pv(
    notional: float,
    fixed_rate: float,
    payment_times: list[float],
    pillar_dfs: list[Number],
    pillar_times: list[float],
) -> Number:
    """Swap PV (receiver fixed) with AAD discount factors.

    Simplified: each payment period has uniform accrual.
    PV = notional * sum_i (fixed_rate * tau_i * df_i) - notional * (1 - df_N)
    """
    from pricebook.aad_interp import aad_log_linear_interp

    pv = Number(0.0)
    for i, t in enumerate(payment_times):
        tau = t if i == 0 else t - payment_times[i - 1]
        df = aad_log_linear_interp(t, pillar_times, pillar_dfs)
        pv = pv + notional * fixed_rate * tau * df

    df_last = aad_log_linear_interp(payment_times[-1], pillar_times, pillar_dfs)
    floating_pv = notional * (Number(1.0) - df_last)

    return pv - floating_pv


# ---------------------------------------------------------------------------
# CDS PV with Number inputs
# ---------------------------------------------------------------------------


def aad_cds_pv(
    notional: float,
    spread: float,
    payment_times: list[float],
    pillar_dfs: list[Number],
    pillar_times: list[float],
    pillar_survs: list[Number],
    surv_times: list[float],
    recovery: float = 0.4,
) -> Number:
    """Protection buyer CDS PV with AAD.

    Single loop computes both premium and protection legs, reusing
    interpolated values.
    """
    from pricebook.aad_interp import aad_log_linear_interp

    premium = Number(0.0)
    protection = Number(0.0)
    lgd = 1.0 - recovery
    surv_prev = Number(1.0)

    for i, t in enumerate(payment_times):
        tau = t if i == 0 else t - payment_times[i - 1]
        df = aad_log_linear_interp(t, pillar_times, pillar_dfs)
        surv = aad_log_linear_interp(t, surv_times, pillar_survs)

        premium = premium + notional * spread * tau * df * surv
        default_prob = surv_prev - surv
        protection = protection + notional * lgd * df * default_prob
        surv_prev = surv

    return protection - premium
