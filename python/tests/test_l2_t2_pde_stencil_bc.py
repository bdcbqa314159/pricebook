"""Regression for L2 Tier-2 T2.2 / T2.3 / T2.4 (+ time-index fix) — `PDESolver1D`
non-uniform stencil, Dirichlet BC enforcement, American BC.

Pre-fix `numerical/_pde.py::PDESolver1D` had four interacting bugs:

* T2.2 — discretization for ∂²V/∂S² and ∂V/∂S used the UNIFORM-grid stencil
  (`d2 = σ²S² / (ds_m · ds_p)`, evenly split into sub/super diagonals).
  On a non-uniform grid (LOG, SINH), this is the wrong stencil and biased the
  ATM call by ~13 % on the LOG grid for the canonical
  (S=100, K=100, r=5%, σ=20%, T=1y) test.

* T2.3 — the implicit tridiagonal solve set `diag[0]=1, rhs[0]=0` (zero-init),
  producing `V_new[0] = 0` from the solve, then the code overwrote
  `V_new[0] = V[0]` (the OLD value).  The boundary value used in the implicit
  equation at the boundary-adjacent rows (i=1, i=N−2) was therefore 0 — wrong
  for puts (where the lower BC is non-zero) and wrong for the upper BC of
  calls.

* T2.B — the boundary used `(self.n_time − step) · dt` as the time-to-
  maturity, but after step `k` the time-to-maturity is `(k+1)·dt`, NOT
  `(n_time − k)·dt`.  The expression is only correct at the middle step.

* T2.4 — American boundaries used the European discounted strike, missing
  the early-exercise dominance at S→0 (puts) and S→S_max (calls).

All four are fixed in one combined patch — they touch the same `_theta_step`
and outer time-loop.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.numerical._pde import GridType, PDEMethod, PDESolver1D


def _bs_call(S, K, r, sigma, T, q=0.0):
    from pricebook.models.black76 import _norm_cdf
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return (S * math.exp(-q * T) * _norm_cdf(d1)
            - K * math.exp(-r * T) * _norm_cdf(d2))


def _bs_put(S, K, r, sigma, T, q=0.0):
    return (_bs_call(S, K, r, sigma, T, q)
            - S * math.exp(-q * T) + K * math.exp(-r * T))


class TestNonUniformStencil:
    """T2.2 — LOG grid must agree with Black-Scholes."""

    def test_log_grid_atm_call_matches_bs(self):
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        bs = _bs_call(S, K, r, sigma, T)

        sol = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                          n_space=200, n_time=200, grid_type=GridType.LOG)
        res = sol.solve(S, K, T, sigma, r, is_call=True)

        rel = abs(res.price - bs) / bs
        # Pre-fix: ~13 % overshoot on LOG grid.  Post-fix: < 1 %.
        assert rel < 0.01, (
            f"LOG-grid ATM call: PDE={res.price:.4f}, BS={bs:.4f}, "
            f"rel={rel:.3%}"
        )

    def test_uniform_grid_atm_call_matches_bs(self):
        """Uniform grid was already correct pre-fix; confirm it stays correct."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        bs = _bs_call(S, K, r, sigma, T)
        sol = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                          n_space=200, n_time=200, grid_type=GridType.UNIFORM)
        res = sol.solve(S, K, T, sigma, r, is_call=True)
        rel = abs(res.price - bs) / bs
        assert rel < 0.005


class TestDirichletBC:
    """T2.3 — implicit BC must propagate into the interior solve.
    The easiest way to test: a put on a uniform grid.  Pre-fix, the implicit
    equation at i=1 used V_new[0]=0 (wrong; should be K·e^{−rτ}−S[0]), biasing
    the put price upward (V at S[0] is the highest).
    """

    def test_uniform_grid_atm_put_matches_bs(self):
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        bs = _bs_put(S, K, r, sigma, T)
        sol = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                          n_space=200, n_time=200, grid_type=GridType.UNIFORM)
        res = sol.solve(S, K, T, sigma, r, is_call=False)
        rel = abs(res.price - bs) / bs
        assert rel < 0.01, (
            f"Uniform ATM put: PDE={res.price:.4f}, BS={bs:.4f}, rel={rel:.3%}"
        )

    def test_log_grid_atm_put_matches_bs(self):
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        bs = _bs_put(S, K, r, sigma, T)
        sol = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                          n_space=200, n_time=200, grid_type=GridType.LOG)
        res = sol.solve(S, K, T, sigma, r, is_call=False)
        rel = abs(res.price - bs) / bs
        assert rel < 0.02, (
            f"LOG-grid ATM put: PDE={res.price:.4f}, BS={bs:.4f}, rel={rel:.3%}"
        )


class TestAmericanBoundary:
    """T2.4 — American boundaries must be intrinsic, not European discounted."""

    def test_american_put_dominates_european(self):
        """Deep-ITM American put should be intrinsic-bounded: V ≥ K − S₀.
        Pre-fix the boundary used the European discounted strike, so deep ITM
        could undershoot intrinsic."""
        S, K, r, sigma, T = 80.0, 100.0, 0.05, 0.20, 1.0
        eur = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                          n_space=200, n_time=200, grid_type=GridType.UNIFORM
                          ).solve(S, K, T, sigma, r, is_call=False, is_american=False)
        amer = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                           n_space=200, n_time=200, grid_type=GridType.UNIFORM
                           ).solve(S, K, T, sigma, r, is_call=False, is_american=True)
        intrinsic = K - S
        # American >= European (always) and >= intrinsic.
        assert amer.price >= eur.price - 1e-6
        assert amer.price >= intrinsic - 1e-6, (
            f"American put undershoots intrinsic: PDE={amer.price:.4f}, "
            f"intrinsic={intrinsic:.4f}"
        )


class TestTimeIndex:
    """T2.B — the upper boundary's time-to-maturity at step k is (k+1)·dt,
    not (n_time − k)·dt.  Test: a long-dated deep-OTM call's price is small
    but positive — pre-fix the upper-boundary mistime created a measurable
    bias even after the interior diffusion smeared it out."""

    def test_long_dated_call_no_explosive_bias(self):
        """A 5-year ATM call: pre-fix the inverted time-index polluted the
        whole grid via the upper boundary.  Post-fix should agree with BS."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 5.0
        bs = _bs_call(S, K, r, sigma, T)
        sol = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                          n_space=200, n_time=200, grid_type=GridType.UNIFORM)
        res = sol.solve(S, K, T, sigma, r, is_call=True)
        rel = abs(res.price - bs) / bs
        assert rel < 0.02, (
            f"5y ATM call: PDE={res.price:.4f}, BS={bs:.4f}, rel={rel:.3%}"
        )
