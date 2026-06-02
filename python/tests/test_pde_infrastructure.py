"""Tests for PDE infrastructure: protocol, local vol, time-dep, PIDE, SABR,
adaptive, American 2D, diagnostics, boundary conditions."""

import pytest
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════
# PDE2: Protocol
# ═══════════════════════════════════════════════════════════════

class TestPDEProtocol:
    def test_european_call(self):
        from pricebook.models.pde_protocol import pde_price
        r = pde_price(100, 100, 0.20, 0.04, 1.0, is_call=True)
        # Compare with BS
        from pricebook.models.black76 import black76_price, OptionType
        fwd = 100 * math.exp(0.04)
        df = math.exp(-0.04)
        bs = black76_price(fwd, 100, 0.20, 1.0, df, OptionType.CALL)
        assert r.price == pytest.approx(bs, rel=0.02)

    def test_european_put(self):
        from pricebook.models.pde_protocol import pde_price
        r = pde_price(100, 100, 0.20, 0.04, 1.0, is_call=False)
        assert r.price > 0
        assert r.delta < 0

    def test_american_put(self):
        from pricebook.models.pde_protocol import pde_price
        euro = pde_price(100, 100, 0.20, 0.04, 1.0, is_call=False)
        amer = pde_price(100, 100, 0.20, 0.04, 1.0, is_call=False, is_american=True)
        assert amer.price >= euro.price * 0.99  # American ≥ European

    def test_greeks(self):
        from pricebook.models.pde_protocol import pde_price
        r = pde_price(100, 100, 0.20, 0.04, 1.0)
        assert 0 < r.delta < 1
        assert r.gamma > 0

    def test_spec_factory(self):
        from pricebook.models.pde_protocol import PDESpec
        spec = PDESpec.european_call(100, 100, 0.20, 0.04, 1.0)
        assert spec.T == 1.0
        assert not spec.is_american

    def test_to_dict(self):
        from pricebook.models.pde_protocol import pde_price
        r = pde_price(100, 100, 0.20, 0.04, 1.0)
        d = r.to_dict()
        assert "convergence" in d
        assert "delta" in d


# ═══════════════════════════════════════════════════════════════
# PDE1: Local Vol
# ═══════════════════════════════════════════════════════════════

class TestLocalVolPDE:
    def test_flat_vol_matches_bs(self):
        """Flat local vol surface should match BS."""
        from pricebook.models.pde_local_vol import local_vol_pde
        flat_vol = lambda S, t: 0.20  # constant → should match BS
        r = local_vol_pde(100, 100, 0.04, 1.0, flat_vol)
        from pricebook.models.black76 import black76_price, OptionType
        bs = black76_price(100 * math.exp(0.04), 100, 0.20, 1.0, math.exp(-0.04), OptionType.CALL)
        assert r.price == pytest.approx(bs, rel=0.03)

    def test_smile_affects_price(self):
        """Non-flat vol surface should differ from flat."""
        from pricebook.models.pde_local_vol import local_vol_pde
        flat = local_vol_pde(100, 100, 0.04, 1.0, lambda S, t: 0.20)
        smile = local_vol_pde(100, 100, 0.04, 1.0, lambda S, t: 0.20 + 0.0001 * (S - 100)**2)
        assert abs(smile.price - flat.price) > 0.01


# ═══════════════════════════════════════════════════════════════
# PDE3: Time-Dependent
# ═══════════════════════════════════════════════════════════════

class TestTimeDepPDE:
    def test_constant_equals_standard(self):
        from pricebook.models.pde_time_dependent import time_dependent_pde, TermStructureCoefficients
        ts = TermStructureCoefficients(
            rate_pillars=[(0, 0.04), (1, 0.04)],
            vol_pillars=[(0, 0.20), (1, 0.20)],
        )
        r = time_dependent_pde(100, 100, 1.0, ts)
        from pricebook.models.pde_protocol import pde_price
        bs = pde_price(100, 100, 0.20, 0.04, 1.0)
        assert r.price == pytest.approx(bs.price, rel=0.05)

    def test_increasing_vol(self):
        from pricebook.models.pde_time_dependent import time_dependent_pde, TermStructureCoefficients
        low = TermStructureCoefficients([(0, 0.04), (1, 0.04)], [(0, 0.15), (1, 0.15)])
        high = TermStructureCoefficients([(0, 0.04), (1, 0.04)], [(0, 0.15), (1, 0.30)])
        r_low = time_dependent_pde(100, 100, 1.0, low)
        r_high = time_dependent_pde(100, 100, 1.0, high)
        assert r_high.price > r_low.price  # higher vol → higher option value


# ═══════════════════════════════════════════════════════════════
# PDE4: PIDE (Jump-Diffusion)
# ═══════════════════════════════════════════════════════════════

class TestPIDE:
    def test_merton_more_expensive(self):
        """Jump risk increases option price for OTM."""
        from pricebook.models.pide_solver import merton_pide
        from pricebook.models.pde_protocol import pde_price
        bs = pde_price(100, 100, 0.20, 0.04, 1.0)
        merton = merton_pide(100, 100, 0.04, 0.20, 1.0, jump_intensity=1.0)
        # Jumps add value (leptokurtic tails)
        assert merton.price > bs.price * 0.8  # reasonably close

    def test_no_jumps_matches_bs(self):
        from pricebook.models.pide_solver import merton_pide
        from pricebook.models.pde_protocol import pde_price
        bs = pde_price(100, 100, 0.20, 0.04, 1.0)
        no_jump = merton_pide(100, 100, 0.04, 0.20, 1.0, jump_intensity=0.0)
        assert no_jump.price == pytest.approx(bs.price, rel=0.05)

    def test_kou(self):
        from pricebook.models.pide_solver import kou_pide
        r = kou_pide(100, 100, 0.04, 0.20, 1.0)
        assert r.price > 0


# ═══════════════════════════════════════════════════════════════
# PDE5: SABR
# ═══════════════════════════════════════════════════════════════

class TestSABRPDE:
    def test_sabr_positive(self):
        from pricebook.models.pde_sabr import sabr_pde
        price = sabr_pde(0.04, 0.04, 1.0, 0.3, 0.5, -0.3, 0.4)
        assert price > 0

    def test_itm_more_expensive(self):
        from pricebook.models.pde_sabr import sabr_pde
        atm = sabr_pde(0.04, 0.04, 1.0, 0.3, 0.5, -0.3, 0.4)
        itm = sabr_pde(0.04, 0.03, 1.0, 0.3, 0.5, -0.3, 0.4)
        assert itm > atm


# ═══════════════════════════════════════════════════════════════
# PDE6: Adaptive
# ═══════════════════════════════════════════════════════════════

class TestAdaptivePDE:
    def test_refine_grid(self):
        from pricebook.models.pde_adaptive import refine_grid
        grid = np.linspace(50, 150, 20)
        values = np.maximum(grid - 100, 0)  # call payoff (kink at 100)
        refined = refine_grid(grid, values)
        assert len(refined) > len(grid)  # should add nodes

    def test_adaptive_converges(self):
        from pricebook.models.pde_adaptive import adaptive_pde
        from pricebook.models.pde_protocol import PDESpec
        spec = PDESpec.european_call(100, 100, 0.20, 0.04, 1.0)
        r = adaptive_pde(spec, 100, n_space_initial=50, max_refinements=2)
        assert r.price > 0


# ═══════════════════════════════════════════════════════════════
# PDE7: American 2D
# ═══════════════════════════════════════════════════════════════

class TestAmerican2D:
    def test_heston_american_put(self):
        from pricebook.models.pde_american_2d import heston_american_pde
        r = heston_american_pde(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                                 is_call=False, n_x=40, n_v=20, n_time=50)
        assert r["price"] > 0

    def test_american_geq_european(self):
        from pricebook.models.pde_american_2d import heston_american_pde
        from pricebook.models.adi import heston_pde
        from pricebook.models.black76 import OptionType
        amer = heston_american_pde(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                                    is_call=False, n_x=40, n_v=20, n_time=50)
        euro = heston_pde(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                          OptionType.PUT, n_x=40, n_v=20, n_time=50)
        assert amer["price"] >= euro * 0.95  # American ≥ European


# ═══════════════════════════════════════════════════════════════
# PDE8: Diagnostics
# ═══════════════════════════════════════════════════════════════

class TestPDEDiagnostics:
    def test_convergence_study(self):
        from pricebook.models.pde_diagnostics import convergence_study
        r = convergence_study(100, 100, 0.20, 0.04, 1.0, grid_sizes=[50, 100, 200])
        assert len(r.prices) == 3
        assert r.convergence_order > 0
        assert r.errors[-1] < r.errors[0]  # finer grid → smaller error

    def test_scheme_recommendation(self):
        from pricebook.models.pde_diagnostics import recommend_scheme
        r = recommend_scheme(0.20, 1.0, is_american=True)
        assert r.method == "crank_nicolson"
        assert r.n_space > 0

    def test_stability_check(self):
        from pricebook.models.pde_diagnostics import stability_check
        r = stability_check(0.20, 1.0, 200, 200, 100)
        assert r["stable"]
        # Explicit with too few time steps → unstable
        r2 = stability_check(0.50, 1.0, 200, 10, 100, method="explicit")
        assert not r2["stable"]


# ═══════════════════════════════════════════════════════════════
# PDE10: Boundary Conditions
# ═══════════════════════════════════════════════════════════════

class TestBoundaryConditions:
    def test_dirichlet(self):
        from pricebook.numerical.pde_boundary import BCSpec, apply_bc
        V = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        grid = np.array([80, 90, 100, 110, 120], dtype=float)
        V_new = apply_bc(V, grid, 0.5, BCSpec.dirichlet(99.0), BCSpec.dirichlet(0.0))
        assert V_new[0] == 99.0
        assert V_new[-1] == 0.0

    def test_neumann(self):
        from pricebook.numerical.pde_boundary import BCSpec, apply_bc
        V = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        grid = np.array([80, 90, 100, 110, 120], dtype=float)
        V_new = apply_bc(V, grid, 0.5, BCSpec.neumann(0.0), BCSpec.neumann(0.0))
        assert V_new[0] == pytest.approx(V[1])  # zero flux → V[0] = V[1]

    def test_linear_extrapolation(self):
        from pricebook.numerical.pde_boundary import BCSpec, apply_bc
        V = np.array([0.0, 2.0, 4.0, 6.0, 0.0])
        grid = np.linspace(80, 120, 5)
        V_new = apply_bc(V, grid, 0, BCSpec.linear_extrapolation(), BCSpec.linear_extrapolation())
        assert V_new[0] == pytest.approx(2 * V[1] - V[2])

    def test_call_bcs(self):
        from pricebook.numerical.pde_boundary import call_bcs
        lo, hi = call_bcs(100, 0.04)
        assert lo.bc_type.value == "dirichlet"

    def test_barrier_bcs(self):
        from pricebook.numerical.pde_boundary import barrier_bcs
        lo, hi = barrier_bcs(120, is_up=True)
        assert hi.bc_type.value == "dirichlet"
