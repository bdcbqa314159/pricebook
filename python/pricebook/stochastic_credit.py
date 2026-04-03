"""
Stochastic credit intensity models.

CIR intensity: dλ = kappa*(theta - λ)*dt + xi*sqrt(λ)*dW
    Analytical survival probability via affine model (Riccati ODE).

Cox process: default at first jump of Poisson with stochastic intensity.
    P(tau > T | λ-path) = exp(-∫λ ds)

Joint (rate, hazard) simulation for wrong-way risk.

    from pricebook.stochastic_credit import CIRIntensity, cox_default_mc

    model = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
    surv = model.survival_analytical(lam0=0.02, T=5.0)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.special_process import CIRProcess
from pricebook.brownian import CorrelatedBM
from pricebook.optimization import minimize


class CIRIntensity:
    """CIR model for default intensity (hazard rate).

    dλ = kappa*(theta - λ)*dt + xi*sqrt(λ)*dW

    Affine model: survival probability has closed-form.
    """

    def __init__(self, kappa: float, theta: float, xi: float):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi

    def survival_analytical(self, lam0: float, T: float) -> float:
        """Analytical survival probability P(tau > T) = E[exp(-∫λ ds)].

        Via Riccati ODE for affine models:
            P = A(T) * exp(-B(T) * lam0)
        """
        k, th, xi = self.kappa, self.theta, self.xi

        gamma = math.sqrt(k**2 + 2 * xi**2)

        exp_gT = math.exp(gamma * T)
        denom = (gamma + k) * (exp_gT - 1) + 2 * gamma

        B = 2 * (exp_gT - 1) / denom

        log_A = (2 * k * th / xi**2) * math.log(2 * gamma * math.exp((gamma + k) * T / 2) / denom)

        return math.exp(log_A - B * lam0)

    def simulate_intensity(
        self, lam0: float, T: float, n_steps: int, n_paths: int, seed: int = 42,
    ) -> np.ndarray:
        """Simulate intensity paths. Shape: (n_paths, n_steps+1)."""
        cir = CIRProcess(self.kappa, self.theta, self.xi, seed=seed)
        return cir.sample(x0=lam0, T=T, n_steps=n_steps, n_paths=n_paths)

    def survival_mc(
        self, lam0: float, T: float, n_steps: int = 100, n_paths: int = 50_000,
        seed: int = 42,
    ) -> float:
        """MC survival probability: E[exp(-∫λ ds)]."""
        paths = self.simulate_intensity(lam0, T, n_steps, n_paths, seed)
        dt = T / n_steps
        integral = paths[:, :-1].sum(axis=1) * dt
        return float(np.exp(-integral).mean())


def cox_default_mc(
    intensity_paths: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Simulate default indicators from intensity paths (Cox process).

    For each path, default occurs if a uniform < 1 - exp(-∫λ ds).

    Args:
        intensity_paths: shape (n_paths, n_steps+1).
        dt: time step size.

    Returns:
        Boolean array (n_paths,): True = defaulted.
    """
    integral = intensity_paths[:, :-1].sum(axis=1) * dt
    survival = np.exp(-integral)
    rng = np.random.default_rng(12345)
    U = rng.uniform(size=len(survival))
    return U > survival


class JointRateHazard:
    """Joint simulation of (r(t), λ(t)) with correlation for wrong-way risk.

    r follows OU: dr = a_r*(mu_r - r)*dt + sigma_r*dW1
    λ follows CIR: dλ = kappa*(theta - λ)*dt + xi*sqrt(λ)*dW2
    corr(dW1, dW2) = rho
    """

    def __init__(
        self,
        a_r: float, mu_r: float, sigma_r: float,
        kappa: float, theta: float, xi: float,
        rho: float,
    ):
        self.a_r = a_r
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def simulate(
        self, r0: float, lam0: float, T: float,
        n_steps: int = 100, n_paths: int = 50_000, seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate correlated (r, λ) paths.

        Returns: (r_paths, lam_paths), each (n_paths, n_steps+1).
        """
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        cbm = CorrelatedBM([[1, self.rho], [self.rho, 1]], seed=seed)
        dW = cbm.increments(T, n_steps, n_paths)

        r = np.zeros((n_paths, n_steps + 1))
        lam = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = r0
        lam[:, 0] = lam0

        for i in range(n_steps):
            # OU for r (exact step)
            e_ar = math.exp(-self.a_r * dt)
            r[:, i+1] = self.mu_r + (r[:, i] - self.mu_r) * e_ar + \
                self.sigma_r * math.sqrt((1 - e_ar**2) / (2 * self.a_r)) * \
                dW[:, i, 0] / sqrt_dt

            # CIR for λ (Euler)
            lam_pos = np.maximum(lam[:, i], 0.0)
            lam[:, i+1] = np.maximum(
                lam_pos + self.kappa * (self.theta - lam_pos) * dt +
                self.xi * np.sqrt(lam_pos) * dW[:, i, 1],
                0.0,
            )

        return r, lam

    def survival_mc(
        self, r0: float, lam0: float, T: float,
        n_steps: int = 100, n_paths: int = 50_000, seed: int = 42,
    ) -> float:
        """MC survival under correlated model."""
        _, lam = self.simulate(r0, lam0, T, n_steps, n_paths, seed)
        dt = T / n_steps
        integral = lam[:, :-1].sum(axis=1) * dt
        return float(np.exp(-integral).mean())


def calibrate_cir_intensity(
    par_spreads: list[tuple[float, float]],
    recovery: float = 0.4,
    lam0_guess: float = 0.02,
) -> dict[str, float]:
    """Calibrate CIR intensity to CDS par spreads.

    Simplified: fit kappa, theta, xi so that CIR survival matches
    the survival implied by par spreads.

    Args:
        par_spreads: list of (T, spread) pairs.
        recovery: recovery rate.
        lam0_guess: initial intensity.

    Returns:
        dict with kappa, theta, xi, lam0.
    """
    # Implied survival from spread: surv ≈ exp(-spread/(1-R) * T)
    target_survivals = [
        (T, math.exp(-spread / (1 - recovery) * T))
        for T, spread in par_spreads
    ]

    def objective(params):
        kappa, theta, xi = params
        if kappa <= 0 or theta <= 0 or xi <= 0:
            return 1e10
        model = CIRIntensity(kappa, theta, xi)
        total = 0.0
        for T, target in target_survivals:
            try:
                model_surv = model.survival_analytical(lam0_guess, T)
                total += (model_surv - target) ** 2
            except (ValueError, OverflowError):
                return 1e10
        return total

    result = minimize(objective, x0=[1.0, lam0_guess, 0.1], method="nelder_mead")

    kappa, theta, xi = result.x
    return {
        "kappa": kappa, "theta": theta, "xi": xi,
        "lam0": lam0_guess,
    }
