"""Regression for L2 Wave-2 audit — `adaptive_euler` Brownian-bridge split.

Pre-fix the two-half-step error-estimate split was:

    dW1 = dW * sqrt(0.5)        # ≈ 0.707·dW
    dW2 = dW - dW1              # ≈ 0.293·dW

This sums to dW (so the full and two-half final values are reconciled in
the deterministic case), but the variance is asymmetric:

    Var(dW1) = 0.5·dt      (correct)
    Var(dW2) = (1-√0.5)²·dt  ≈ 0.086·dt   (WRONG — should be 0.5·dt)

The proper Brownian bridge gives, given the full increment dW:

    dW1 = dW/2 + sqrt(dt/4)·Z,  Z ~ N(0,1)   (independent)
    dW2 = dW - dW1

This ensures Var(dW1) = Var(dW2) = dt/2 (correct).

Effect: pre-fix the half-step path artificially closely tracked the full-step
path (because the second half had nearly-deterministic diffusion), so the
local error |x_full − x_two| was systematically UNDER-estimated and the
adaptive step controller failed to refine in stiff regions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.models.sde_adaptive import adaptive_euler


class TestAdaptiveEulerBrownianBridge:
    def test_gbm_terminal_mean_unbiased(self):
        """For GBM under adaptive Euler, terminal mean E[S_T] = S₀·exp(μT)
        should match analytically.  Pre-fix the broken bridge gave wrong
        local error → larger steps accepted → larger Euler bias in mean."""
        S0, mu, sigma, T = 100.0, 0.05, 0.20, 1.0
        # GBM in log-space: x = log S, dx = (μ - σ²/2)·dt + σ·dW
        log_drift = mu - 0.5 * sigma**2
        result = adaptive_euler(
            x0=math.log(S0),
            mu_fn=lambda x, t: log_drift,
            sigma_fn=lambda x, t: sigma,
            T=T,
            n_paths=5_000,
            dt_init=0.01,
            tol=1e-4,
            seed=42,
        )
        log_ST = result.terminal_values
        ST = np.exp(log_ST)
        mean_emp = float(np.mean(ST))
        exact = S0 * math.exp(mu * T)
        rel = abs(mean_emp - exact) / exact
        assert rel < 0.03, (
            f"Empirical mean {mean_emp:.4f} vs exact {exact:.4f} (rel {rel:.3%})"
        )

    def test_step_count_responds_to_tolerance(self):
        """Tighter tolerance → more steps.  Pre-fix the under-estimated
        local error caused poor responsiveness to tol."""
        # Use a slightly stiff process (mean-reverting OU).
        def mu_fn(x, t):
            return -2.0 * (x - 1.0)

        def sigma_fn(x, t):
            return 0.3

        r_loose = adaptive_euler(
            x0=0.0, mu_fn=mu_fn, sigma_fn=sigma_fn, T=1.0,
            n_paths=200, dt_init=0.05, tol=1e-2, seed=42,
        )
        r_tight = adaptive_euler(
            x0=0.0, mu_fn=mu_fn, sigma_fn=sigma_fn, T=1.0,
            n_paths=200, dt_init=0.05, tol=1e-5, seed=42,
        )
        # Tighter tolerance must use more steps on average.
        assert r_tight.n_steps_avg > r_loose.n_steps_avg, (
            f"Loose tol n_steps_avg={r_loose.n_steps_avg:.1f}, "
            f"tight n_steps_avg={r_tight.n_steps_avg:.1f} — the controller "
            f"isn't responding to tolerance."
        )
