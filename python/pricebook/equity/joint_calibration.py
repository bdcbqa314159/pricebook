"""Joint vol surface + dividend calibration.

Simultaneously fits volatility parameters and dividend assumptions
to match both option prices and forward prices.

    from pricebook.equity.joint_calibration import (
        joint_calibrate, decompose_forward_error,
    )

References:
    Bos, Kragt & Bovenberg (2017). Pricing and Hedging Dividend Derivatives.
    Buehler (2015). Consistent Dividend and Vol Surface Modelling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from pricebook.models.black76 import OptionType, black76_price
from pricebook.options.implied_vol import implied_vol_black76


@dataclass
class JointCalibResult:
    """Result of joint vol + dividend calibration."""
    vol_params: dict           # fitted volatility parameters
    div_params: dict           # fitted dividend parameters
    rmse_vol: float            # RMSE in implied vol space
    rmse_fwd: float            # RMSE in forward price space
    fitted_vols: list[float]
    market_vols: list[float]
    fitted_fwds: list[float]
    market_fwds: list[float]
    model: str

    def to_dict(self) -> dict:
        return {
            "model": self.model, "rmse_vol": self.rmse_vol,
            "rmse_fwd": self.rmse_fwd,
            "vol_params": self.vol_params, "div_params": self.div_params,
        }


@dataclass
class ForwardErrorDecomp:
    """Attribution of forward mispricing to vol vs dividend assumptions."""
    total_error: float
    vol_component: float       # error from vol assumptions
    div_component: float       # error from dividend assumptions
    residual: float
    per_tenor: list[dict]

    def to_dict(self) -> dict:
        return {
            "total_error": self.total_error,
            "vol_component": self.vol_component,
            "div_component": self.div_component,
        }


def joint_calibrate(
    spot: float,
    vol_surface_data: list[dict],
    div_data: list[dict],
    rate: float,
    model: str = "bsm+continuous",
    vol_weight: float = 1.0,
    fwd_weight: float = 1.0,
) -> JointCalibResult:
    """Jointly calibrate volatility and dividend assumptions.

    Args:
        vol_surface_data: list of {"T": float, "strikes": [...], "vols": [...]}.
        div_data: list of {"T": float, "fwd": float} (market forward prices).
        rate: risk-free rate.
        model:
            "bsm+continuous" — flat vol + continuous dividend yield.
            "term+continuous" — term vol + continuous yield (piecewise).
        vol_weight: weight on vol fitting in objective.
        fwd_weight: weight on forward fitting in objective.

    Returns:
        JointCalibResult with fitted params, RMSE for vols and forwards.
    """
    if model == "bsm+continuous":
        return _calibrate_bsm_continuous(spot, vol_surface_data, div_data, rate,
                                          vol_weight, fwd_weight)
    elif model == "term+continuous":
        return _calibrate_term_continuous(spot, vol_surface_data, div_data, rate,
                                           vol_weight, fwd_weight)
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'bsm+continuous' or 'term+continuous'.")


def _calibrate_bsm_continuous(
    spot, vol_data, div_data, rate, vol_weight, fwd_weight,
):
    """Flat vol σ + continuous yield q → fit both vol surface and forwards."""
    all_vols_mkt = []
    all_strikes = []
    all_Ts = []
    for vd in vol_data:
        for k, v in zip(vd["strikes"], vd["vols"]):
            all_strikes.append(k)
            all_vols_mkt.append(v)
            all_Ts.append(vd["T"])

    fwd_Ts = [d["T"] for d in div_data]
    fwd_mkt = [d["fwd"] for d in div_data]

    def objective(params):
        sigma, q = params[0], params[1]
        if sigma <= 0 or q < 0:
            return 1e6

        # Vol errors
        vol_sse = 0.0
        for i in range(len(all_strikes)):
            T = all_Ts[i]
            K = all_strikes[i]
            fwd = spot * math.exp((rate - q) * T)
            df = math.exp(-rate * T)
            try:
                price = black76_price(fwd, K, sigma, T, df, OptionType.CALL)
                model_vol = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
            except Exception:
                model_vol = sigma
            vol_sse += (model_vol - all_vols_mkt[i])**2

        # Forward errors
        fwd_sse = 0.0
        for j in range(len(fwd_Ts)):
            model_fwd = spot * math.exp((rate - q) * fwd_Ts[j])
            fwd_sse += ((model_fwd - fwd_mkt[j]) / spot)**2  # normalised

        return vol_weight * vol_sse + fwd_weight * fwd_sse

    result = scipy_minimize(objective, [0.20, 0.02],
                             bounds=[(0.05, 0.80), (0.0, 0.10)],
                             method="L-BFGS-B")

    sigma_opt, q_opt = result.x

    # Compute fitted values
    fitted_vols = []
    for i in range(len(all_strikes)):
        T = all_Ts[i]
        K = all_strikes[i]
        fwd = spot * math.exp((rate - q_opt) * T)
        df = math.exp(-rate * T)
        try:
            price = black76_price(fwd, K, sigma_opt, T, df, OptionType.CALL)
            mv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
        except Exception:
            mv = sigma_opt
        fitted_vols.append(mv)

    fitted_fwds = [spot * math.exp((rate - q_opt) * T) for T in fwd_Ts]

    rmse_vol = math.sqrt(np.mean([(f - m)**2 for f, m in zip(fitted_vols, all_vols_mkt)]))
    rmse_fwd = math.sqrt(np.mean([((f - m) / spot)**2 for f, m in zip(fitted_fwds, fwd_mkt)]))

    return JointCalibResult(
        vol_params={"sigma": sigma_opt},
        div_params={"q": q_opt},
        rmse_vol=rmse_vol,
        rmse_fwd=rmse_fwd,
        fitted_vols=fitted_vols,
        market_vols=all_vols_mkt,
        fitted_fwds=fitted_fwds,
        market_fwds=fwd_mkt,
        model="bsm+continuous",
    )


def _calibrate_term_continuous(
    spot, vol_data, div_data, rate, vol_weight, fwd_weight,
):
    """Piecewise term vol + continuous yield → fit vol surface and forwards."""
    n_expiries = len(vol_data)

    # Parameters: [σ_1, ..., σ_n, q]
    n_params = n_expiries + 1

    all_vols_mkt = []
    all_strikes = []
    all_Ts = []
    all_expiry_idx = []  # which σ to use

    for idx, vd in enumerate(vol_data):
        for k, v in zip(vd["strikes"], vd["vols"]):
            all_strikes.append(k)
            all_vols_mkt.append(v)
            all_Ts.append(vd["T"])
            all_expiry_idx.append(idx)

    fwd_Ts = [d["T"] for d in div_data]
    fwd_mkt = [d["fwd"] for d in div_data]

    def objective(params):
        sigmas = params[:n_expiries]
        q = params[n_expiries]
        if q < 0 or any(s <= 0 for s in sigmas):
            return 1e6

        vol_sse = 0.0
        for i in range(len(all_strikes)):
            T = all_Ts[i]
            K = all_strikes[i]
            sigma = sigmas[all_expiry_idx[i]]
            fwd = spot * math.exp((rate - q) * T)
            df = math.exp(-rate * T)
            try:
                price = black76_price(fwd, K, sigma, T, df, OptionType.CALL)
                mv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
            except Exception:
                mv = sigma
            vol_sse += (mv - all_vols_mkt[i])**2

        fwd_sse = 0.0
        for j in range(len(fwd_Ts)):
            mf = spot * math.exp((rate - q) * fwd_Ts[j])
            fwd_sse += ((mf - fwd_mkt[j]) / spot)**2

        return vol_weight * vol_sse + fwd_weight * fwd_sse

    x0 = [0.20] * n_expiries + [0.02]
    bounds = [(0.05, 0.80)] * n_expiries + [(0.0, 0.10)]

    result = scipy_minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

    sigmas_opt = result.x[:n_expiries]
    q_opt = result.x[n_expiries]

    fitted_vols = []
    for i in range(len(all_strikes)):
        T = all_Ts[i]
        K = all_strikes[i]
        sigma = sigmas_opt[all_expiry_idx[i]]
        fwd = spot * math.exp((rate - q_opt) * T)
        df = math.exp(-rate * T)
        try:
            price = black76_price(fwd, K, sigma, T, df, OptionType.CALL)
            mv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
        except Exception:
            mv = sigma
        fitted_vols.append(mv)

    fitted_fwds = [spot * math.exp((rate - q_opt) * T) for T in fwd_Ts]

    rmse_vol = math.sqrt(np.mean([(f - m)**2 for f, m in zip(fitted_vols, all_vols_mkt)]))
    rmse_fwd = math.sqrt(np.mean([((f - m) / spot)**2 for f, m in zip(fitted_fwds, fwd_mkt)]))

    vol_params = {f"sigma_{i+1}": float(s) for i, s in enumerate(sigmas_opt)}

    return JointCalibResult(
        vol_params=vol_params,
        div_params={"q": float(q_opt)},
        rmse_vol=rmse_vol,
        rmse_fwd=rmse_fwd,
        fitted_vols=fitted_vols,
        market_vols=all_vols_mkt,
        fitted_fwds=fitted_fwds,
        market_fwds=fwd_mkt,
        model="term+continuous",
    )


def decompose_forward_error(
    spot: float,
    vol_params: dict,
    div_params: dict,
    market_fwds: list[dict],
    rate: float,
    div_bump: float = 0.001,
    vol_bump: float = 0.01,
) -> ForwardErrorDecomp:
    """Attribute forward mispricing to vol vs dividend assumptions.

    Args:
        market_fwds: list of {"T": float, "fwd": float}.
        div_bump: bump to dividend yield.
        vol_bump: bump to vol (for vol → forward sensitivity).

    Returns:
        ForwardErrorDecomp with per-component attribution.
    """
    q = div_params.get("q", 0.0)

    per_tenor = []
    total_err = 0.0
    vol_comp = 0.0
    div_comp = 0.0

    for mf in market_fwds:
        T = mf["T"]
        fwd_mkt = mf["fwd"]

        # Model forward
        fwd_model = spot * math.exp((rate - q) * T)
        err = fwd_model - fwd_mkt

        # Dividend sensitivity: dF/dq = -S·T·exp((r-q)T)
        fwd_q_up = spot * math.exp((rate - q - div_bump) * T)
        div_sens = (fwd_q_up - fwd_model) / div_bump

        # Vol has no direct effect on forward in BSM, but via smile...
        # In simple model, vol doesn't affect forward → vol component = 0
        vol_contribution = 0.0
        div_contribution = abs(err)  # all forward error is dividend-related in BSM

        total_err += err**2
        div_comp += div_contribution**2

        per_tenor.append({
            "T": T, "error": err, "vol_comp": vol_contribution,
            "div_comp": div_contribution,
        })

    total = math.sqrt(total_err / len(market_fwds)) if market_fwds else 0
    vc = math.sqrt(vol_comp / len(market_fwds)) if market_fwds else 0
    dc = math.sqrt(div_comp / len(market_fwds)) if market_fwds else 0

    return ForwardErrorDecomp(total, vc, dc, max(total - vc - dc, 0), per_tenor)
