"""Tests for AAD end-to-end integration: all Greeks in one backward pass."""

import math
import time
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.aad import Number, Tape
from pricebook.aad_curves import AADDiscountCurve, AADSurvivalCurve
from pricebook.aad_pricing import aad_black_scholes, aad_swap_pv, aad_cds_pv


REF = date(2024, 1, 15)
RATE = 0.05
HAZARD = 0.02


# ---------------------------------------------------------------------------
# Step 1 — AAD pricing functions
# ---------------------------------------------------------------------------


class TestAADBlackScholes:
    def test_call_value(self):
        with Tape():
            S = Number(100.0)
            r = Number(0.05)
            sigma = Number(0.20)
            pv = aad_black_scholes(S, 100.0, r, sigma, 1.0, is_call=True)
            # ATM call ≈ 10.45
            assert 8.0 < pv.value < 13.0

    def test_put_value(self):
        with Tape():
            S = Number(100.0)
            r = Number(0.05)
            sigma = Number(0.20)
            pv = aad_black_scholes(S, 100.0, r, sigma, 1.0, is_call=False)
            assert pv.value > 0

    def test_all_greeks_one_pass(self):
        """Delta, vega, rho all from one backward pass."""
        with Tape():
            S = Number(100.0)
            r = Number(0.05)
            sigma = Number(0.20)
            pv = aad_black_scholes(S, 100.0, r, sigma, 1.0, is_call=True)
            pv.propagate_to_start()

            delta = S.adjoint
            vega = sigma.adjoint
            rho = r.adjoint

            assert 0.0 < delta < 1.0  # call delta
            assert vega > 0  # long vega
            assert rho > 0   # call rho positive

    def test_delta_fd_check(self):
        eps = 1e-5
        K, r_val, sig_val, T = 100.0, 0.05, 0.20, 1.0

        with Tape():
            S = Number(100.0)
            r = Number(r_val)
            sigma = Number(sig_val)
            pv = aad_black_scholes(S, K, r, sigma, T)
            pv.propagate_to_start()
            aad_delta = S.adjoint

        with Tape():
            pv_up = aad_black_scholes(Number(100.0 + eps), K, Number(r_val), Number(sig_val), T).value
        with Tape():
            pv_dn = aad_black_scholes(Number(100.0 - eps), K, Number(r_val), Number(sig_val), T).value

        fd_delta = (pv_up - pv_dn) / (2 * eps)
        assert aad_delta == pytest.approx(fd_delta, rel=1e-4)

    def test_vega_fd_check(self):
        eps = 1e-5
        S_val, K, r_val, T = 100.0, 100.0, 0.05, 1.0

        with Tape():
            S = Number(S_val)
            r = Number(r_val)
            sigma = Number(0.20)
            pv = aad_black_scholes(S, K, r, sigma, T)
            pv.propagate_to_start()
            aad_vega = sigma.adjoint

        with Tape():
            pv_up = aad_black_scholes(Number(S_val), K, Number(r_val), Number(0.20 + eps), T).value
        with Tape():
            pv_dn = aad_black_scholes(Number(S_val), K, Number(r_val), Number(0.20 - eps), T).value

        fd_vega = (pv_up - pv_dn) / (2 * eps)
        assert aad_vega == pytest.approx(fd_vega, rel=1e-3)

    def test_put_call_parity(self):
        with Tape():
            S = Number(100.0)
            r = Number(0.05)
            sigma = Number(0.20)
            call = aad_black_scholes(S, 100.0, r, sigma, 1.0, is_call=True)
        with Tape():
            S2 = Number(100.0)
            r2 = Number(0.05)
            sigma2 = Number(0.20)
            put = aad_black_scholes(S2, 100.0, r2, sigma2, 1.0, is_call=False)

        # C - P = S - K * e^{-rT}
        parity = call.value - put.value
        expected = 100.0 - 100.0 * math.exp(-0.05)
        assert parity == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# Step 2 — AAD portfolio risk
# ---------------------------------------------------------------------------


class TestAADSwap:
    def _pillar_times(self):
        return [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    def _pillar_dfs_values(self):
        return [1.0, math.exp(-0.05), math.exp(-0.10), math.exp(-0.15),
                math.exp(-0.25), math.exp(-0.35), math.exp(-0.50)]

    def test_atm_swap_near_zero(self):
        """ATM swap PV should be near zero."""
        with Tape():
            dfs = [Number(v) for v in self._pillar_dfs_values()]
            pv = aad_swap_pv(1_000_000, 0.05, [1.0, 2.0, 3.0, 4.0, 5.0],
                             dfs, self._pillar_times())
            # Near ATM for flat 5% curve
            assert abs(pv.value) < 50_000  # within 5% of notional

    def test_ir01_per_pillar(self):
        """All pillar sensitivities from one backward pass."""
        with Tape():
            dfs = [Number(v) for v in self._pillar_dfs_values()]
            pv = aad_swap_pv(1_000_000, 0.05, [1.0, 2.0, 3.0, 4.0, 5.0],
                             dfs, self._pillar_times())
            pv.propagate_to_start()

            adjoints = [df.adjoint for df in dfs]
            # At least some pillars should have non-zero sensitivity
            assert any(a != 0 for a in adjoints)

    def test_ir01_fd_check(self):
        """AAD swap sensitivities match bump per pillar."""
        base_dfs = self._pillar_dfs_values()
        ptimes = self._pillar_times()
        pay_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        eps = 1e-7

        with Tape():
            dfs = [Number(v) for v in base_dfs]
            pv = aad_swap_pv(1_000_000, 0.05, pay_times, dfs, ptimes)
            pv.propagate_to_start()
            aad_derivs = [df.adjoint for df in dfs]

        for idx in range(len(base_dfs)):
            up = list(base_dfs)
            up[idx] += eps
            dn = list(base_dfs)
            dn[idx] -= eps

            with Tape():
                pv_up = aad_swap_pv(1_000_000, 0.05, pay_times, [Number(v) for v in up], ptimes).value
            with Tape():
                pv_dn = aad_swap_pv(1_000_000, 0.05, pay_times, [Number(v) for v in dn], ptimes).value

            fd = (pv_up - pv_dn) / (2 * eps)
            if abs(fd) > 1e-6:
                assert aad_derivs[idx] == pytest.approx(fd, rel=1e-3)


class TestAADCDS:
    def _setup(self):
        ptimes = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        df_vals = [1.0] + [math.exp(-0.05 * t) for t in [1, 2, 3, 5, 7, 10]]
        stimes = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        s_vals = [1.0] + [math.exp(-0.02 * t) for t in [1, 2, 3, 5, 7, 10]]
        pay_times = [1.0, 2.0, 3.0, 5.0]
        return ptimes, df_vals, stimes, s_vals, pay_times

    def test_cds_pv(self):
        ptimes, df_vals, stimes, s_vals, pay_times = self._setup()
        with Tape():
            dfs = [Number(v) for v in df_vals]
            survs = [Number(v) for v in s_vals]
            pv = aad_cds_pv(1_000_000, 0.02, pay_times, dfs, ptimes, survs, stimes)
            # At par spread, PV should be near zero
            assert abs(pv.value) < 50_000

    def test_cds_all_sensitivities(self):
        """IR01 and CS01 per pillar in one pass."""
        ptimes, df_vals, stimes, s_vals, pay_times = self._setup()
        with Tape():
            dfs = [Number(v) for v in df_vals]
            survs = [Number(v) for v in s_vals]
            pv = aad_cds_pv(1_000_000, 0.02, pay_times, dfs, ptimes, survs, stimes)
            pv.propagate_to_start()

            ir_sens = [df.adjoint for df in dfs]
            cs_sens = [s.adjoint for s in survs]

            assert any(a != 0 for a in ir_sens)
            assert any(a != 0 for a in cs_sens)


# ---------------------------------------------------------------------------
# Step 3 — Portfolio: sum of individual AAD
# ---------------------------------------------------------------------------


class TestAADPortfolio:
    def test_portfolio_sensitivities(self):
        """Portfolio = sum of trades, AAD = sum of individual AADs."""
        ptimes = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        df_vals = [1.0] + [math.exp(-0.05 * t) for t in [1, 2, 3, 5, 7, 10]]
        pay1 = [1.0, 2.0, 3.0]
        pay2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Individual
        with Tape():
            dfs = [Number(v) for v in df_vals]
            pv1 = aad_swap_pv(1_000_000, 0.04, pay1, dfs, ptimes)
            pv1.propagate_to_start()
            adj1 = [df.adjoint for df in dfs]

        with Tape():
            dfs = [Number(v) for v in df_vals]
            pv2 = aad_swap_pv(500_000, 0.06, pay2, dfs, ptimes)
            pv2.propagate_to_start()
            adj2 = [df.adjoint for df in dfs]

        # Portfolio (single pass)
        with Tape():
            dfs = [Number(v) for v in df_vals]
            port_pv = aad_swap_pv(1_000_000, 0.04, pay1, dfs, ptimes) + \
                      aad_swap_pv(500_000, 0.06, pay2, dfs, ptimes)
            port_pv.propagate_to_start()
            adj_port = [df.adjoint for df in dfs]

        # Portfolio adjoints should equal sum of individual
        for i in range(len(df_vals)):
            assert adj_port[i] == pytest.approx(adj1[i] + adj2[i], abs=1e-6)


# ---------------------------------------------------------------------------
# Step 4 — Performance benchmark
# ---------------------------------------------------------------------------


class TestAADPerformance:
    def test_aad_faster_than_bumps(self):
        """AAD should compute all pillar Greeks faster than N bump-and-reprices."""
        ptimes = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        df_vals = [1.0] + [math.exp(-0.05 * t) for t in ptimes[1:]]
        pay_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        n_pillars = len(df_vals)
        eps = 1e-7

        # AAD: one forward + one backward
        t0 = time.perf_counter()
        for _ in range(100):
            with Tape():
                dfs = [Number(v) for v in df_vals]
                pv = aad_swap_pv(1_000_000, 0.05, pay_times, dfs, ptimes)
                pv.propagate_to_start()
                _ = [df.adjoint for df in dfs]
        aad_time = time.perf_counter() - t0

        # Bump: N forward passes (2 per pillar for central diff)
        t0 = time.perf_counter()
        for _ in range(100):
            for idx in range(n_pillars):
                up = list(df_vals)
                up[idx] += eps
                dn = list(df_vals)
                dn[idx] -= eps
                with Tape():
                    pv_up = aad_swap_pv(1_000_000, 0.05, pay_times,
                                        [Number(v) for v in up], ptimes).value
                with Tape():
                    pv_dn = aad_swap_pv(1_000_000, 0.05, pay_times,
                                        [Number(v) for v in dn], ptimes).value
        bump_time = time.perf_counter() - t0

        # AAD should be faster (at least 2x for 10 pillars)
        speedup = bump_time / aad_time
        assert speedup > 2.0, f"AAD speedup only {speedup:.1f}x (expected >2x)"


# ---------------------------------------------------------------------------
# Step 5 — Round-trip: AAD = bump for all types
# ---------------------------------------------------------------------------


class TestAADRoundTrip:
    def test_bs_all_greeks_match_fd(self):
        """All BS Greeks from AAD match finite differences."""
        S0, K, r0, sig0, T = 100.0, 105.0, 0.05, 0.25, 0.5
        eps = 1e-5

        with Tape():
            S = Number(S0)
            r = Number(r0)
            sigma = Number(sig0)
            pv = aad_black_scholes(S, K, r, sigma, T)
            pv.propagate_to_start()
            aad_greeks = {"delta": S.adjoint, "rho": r.adjoint, "vega": sigma.adjoint}

        # FD for each
        for param_name, base_val in [("delta", S0), ("rho", r0), ("vega", sig0)]:
            def _price(**overrides):
                with Tape():
                    s = Number(overrides.get("S", S0))
                    ri = Number(overrides.get("r", r0))
                    si = Number(overrides.get("sigma", sig0))
                    return aad_black_scholes(s, K, ri, si, T).value

            if param_name == "delta":
                fd = (_price(S=S0 + eps) - _price(S=S0 - eps)) / (2 * eps)
            elif param_name == "rho":
                fd = (_price(r=r0 + eps) - _price(r=r0 - eps)) / (2 * eps)
            else:
                fd = (_price(sigma=sig0 + eps) - _price(sigma=sig0 - eps)) / (2 * eps)

            assert aad_greeks[param_name] == pytest.approx(fd, rel=1e-3)
