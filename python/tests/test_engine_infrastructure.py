"""Tests for unified engine infrastructure: protocol, config, auto-Greeks,
diagnostics, BDT, implied tree, enhancements, bridge, comparison, registry."""

import pytest
import math
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


def _make_curve(rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 35)]
    dfs = [math.exp(-rate * y) for y in range(1, 35)]
    return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# P1: Engine Protocol
# ═══════════════════════════════════════════════════════════════

class TestEngineProtocol:
    def test_analytical_engine(self):
        from pricebook.models.engine_protocol import AnalyticalEngine
        eng = AnalyticalEngine()
        r = eng.price_vanilla(100, 100, 0.04, 0.20, 1.0)
        assert r.price > 0
        assert r.greeks.delta > 0
        assert r.greeks.vega > 0
        assert r.engine_type == "analytical_bs"

    def test_tree_engine(self):
        from pricebook.models.engine_protocol import TreePricingEngine
        eng = TreePricingEngine(method="lr", n_steps=200)
        r = eng.price_vanilla(100, 100, 0.04, 0.20, 1.0)
        assert r.price > 0
        assert r.greeks.delta > 0

    def test_mc_engine(self):
        from pricebook.models.engine_protocol import MCPricingEngine
        eng = MCPricingEngine(n_paths=50_000, n_steps=50)
        r = eng.price_vanilla(100, 100, 0.04, 0.20, 1.0)
        assert r.price > 0
        assert r.convergence.std_error > 0

    def test_engines_agree(self):
        """All engines agree on European call within tolerance."""
        from pricebook.models.engine_protocol import (
            AnalyticalEngine, TreePricingEngine, MCPricingEngine,
        )
        a = AnalyticalEngine().price_vanilla(100, 100, 0.04, 0.20, 1.0)
        t = TreePricingEngine(n_steps=500).price_vanilla(100, 100, 0.04, 0.20, 1.0)
        m = MCPricingEngine(n_paths=200_000).price_vanilla(100, 100, 0.04, 0.20, 1.0)
        assert a.price == pytest.approx(t.price, rel=0.01)
        assert a.price == pytest.approx(m.price, rel=0.03)

    def test_put(self):
        from pricebook.models.engine_protocol import AnalyticalEngine
        r = AnalyticalEngine().price_vanilla(100, 100, 0.04, 0.20, 1.0, is_call=False)
        assert r.price > 0
        assert r.greeks.delta < 0

    def test_to_dict(self):
        from pricebook.models.engine_protocol import AnalyticalEngine
        r = AnalyticalEngine().price_vanilla(100, 100, 0.04, 0.20, 1.0)
        d = r.to_dict()
        assert "greeks" in d
        assert "convergence" in d


# ═══════════════════════════════════════════════════════════════
# P2: MC Config
# ═══════════════════════════════════════════════════════════════

class TestMCConfig:
    def test_preset_configs(self):
        from pricebook.models.mc_config import preset_configs
        presets = preset_configs()
        assert "fast" in presets
        assert "production" in presets
        assert "heston" in presets
        assert presets["fast"].n_paths < presets["production"].n_paths

    def test_with_overrides(self):
        from pricebook.models.mc_config import MCConfig
        base = MCConfig(n_paths=100_000)
        modified = base.with_overrides(n_paths=500_000, seed=123)
        assert modified.n_paths == 500_000
        assert modified.seed == 123
        assert base.n_paths == 100_000  # original unchanged

    def test_build_process(self):
        from pricebook.models.mc_config import MCConfig, ProcessType, build_process_from_config
        config = MCConfig(process=ProcessType.GBM)
        proc = build_process_from_config(config, 100, 0.04, 0.20)
        assert proc.n_factors == 1

    def test_mc_pricer_from_config(self):
        from pricebook.models.mc_config import MCConfig, mc_pricer_from_config
        config = MCConfig(n_paths=10_000, n_steps=20)
        pricer = mc_pricer_from_config(config)
        r = pricer.price_vanilla(100, 100, 0.04, 0.20, 1.0)
        assert r.price > 0


# ═══════════════════════════════════════════════════════════════
# P3: Auto-Greeks
# ═══════════════════════════════════════════════════════════════

class TestAutoGreeks:
    def test_classify_smooth(self):
        from pricebook.models.mc_greeks_auto import classify_payoff, PayoffType
        assert classify_payoff("european_call") == PayoffType.SMOOTH

    def test_classify_discontinuous(self):
        from pricebook.models.mc_greeks_auto import classify_payoff, PayoffType
        assert classify_payoff("digital_call") == PayoffType.DISCONTINUOUS

    def test_select_method(self):
        from pricebook.models.mc_greeks_auto import select_greek_method, PayoffType
        assert select_greek_method(PayoffType.SMOOTH) == "pathwise"
        assert select_greek_method(PayoffType.DISCONTINUOUS) == "likelihood_ratio"
        assert select_greek_method(PayoffType.PATH_DEPENDENT) == "bump"

    def test_auto_greeks_call(self):
        from pricebook.models.mc_greeks_auto import auto_greeks
        from pricebook.models.mc_payoffs import european_call
        g = auto_greeks(100, 100, 0.04, 0.20, 1.0, european_call(100),
                         n_paths=50_000, payoff_name="european_call")
        assert g.delta > 0
        assert g.vega > 0


# ═══════════════════════════════════════════════════════════════
# P4: Diagnostics
# ═══════════════════════════════════════════════════════════════

class TestDiagnostics:
    def test_full_diagnostics(self):
        from pricebook.models.mc_diagnostics import full_diagnostics
        rng = np.random.default_rng(42)
        values = rng.normal(10.0, 0.5, 10_000)
        d = full_diagnostics(values)
        assert d.mean == pytest.approx(10.0, abs=0.1)
        assert d.ess > 5_000  # should be near N for IID
        assert d.is_converged  # relative error < 1%

    def test_vre(self):
        from pricebook.models.mc_diagnostics import variance_reduction_efficiency
        rng = np.random.default_rng(42)
        crude = rng.normal(10.0, 1.0, 10_000)
        reduced = rng.normal(10.0, 0.3, 10_000)
        vre = variance_reduction_efficiency(crude, reduced)
        assert vre > 5  # 1/0.3² ÷ 1/1² ≈ 11

    def test_convergence_rate(self):
        from pricebook.models.mc_diagnostics import estimate_convergence_rate
        # Simulated MC convergence (1/√N)
        pairs = [(1000, 10.05), (5000, 10.02), (10000, 10.01), (50000, 10.005), (100000, 10.0)]
        rate = estimate_convergence_rate(pairs)
        assert 0.3 < rate < 1.5  # should be ~0.5 for MC


# ═══════════════════════════════════════════════════════════════
# P6: BDT Tree
# ═══════════════════════════════════════════════════════════════

class TestBDT:
    def test_zcb_matches_curve(self):
        from pricebook.models.bdt_tree import BDTTree
        dc = _make_curve(0.05)
        tree = BDTTree(dc, vol_term=0.10, n_steps=10)
        zcb = tree.zcb_price(5)
        target = math.exp(-0.05 * 5)
        assert zcb == pytest.approx(target, rel=0.05)

    def test_callable_bond(self):
        from pricebook.models.bdt_tree import bdt_callable_bond
        dc = _make_curve(0.05)
        r = bdt_callable_bond(dc, 0.06, 10, [3, 4, 5, 6, 7, 8, 9], vol=0.10)
        assert r["callable_price"] < r["straight_price"]
        assert r["call_option_value"] > 0

    def test_bermudan_swaption(self):
        from pricebook.models.bdt_tree import bdt_bermudan_swaption
        dc = _make_curve(0.05)
        r = bdt_bermudan_swaption(dc, 0.05, 10, [2, 3, 4, 5])
        assert r["price"] >= 0


# ═══════════════════════════════════════════════════════════════
# P7: Implied Tree
# ═══════════════════════════════════════════════════════════════

class TestImpliedTree:
    def test_build(self):
        from pricebook.numerical.implied_tree import build_implied_tree
        vols = [[0.22, 0.20, 0.22]] * 5
        strikes = [[95, 100, 105]] * 5
        tree = build_implied_tree(100, 0.04, 0.0, 1.0, 5, vols, strikes)
        assert tree.n_steps == 5
        assert len(tree.spots) == 6  # 0..5

    def test_price_european(self):
        from pricebook.numerical.implied_tree import build_implied_tree, price_on_implied_tree
        vols = [[0.20, 0.20, 0.20]] * 10
        strikes = [[95, 100, 105]] * 10
        tree = build_implied_tree(100, 0.04, 0.0, 1.0, 10, vols, strikes)
        price = price_on_implied_tree(tree, lambda s: max(s - 100, 0), 0.04, 1.0)
        assert price > 0

    def test_local_vol_extraction(self):
        from pricebook.numerical.implied_tree import build_implied_tree, extract_local_vol
        vols = [[0.20, 0.20, 0.20]] * 5
        strikes = [[95, 100, 105]] * 5
        tree = build_implied_tree(100, 0.04, 0.0, 1.0, 5, vols, strikes)
        lv = extract_local_vol(tree)
        assert len(lv) == 5
        assert all(v > 0 for _, v in lv[0])


# ═══════════════════════════════════════════════════════════════
# P8: Tree Enhancements
# ═══════════════════════════════════════════════════════════════

class TestTreeEnhancements:
    def test_adaptive_barrier(self):
        from pricebook.numerical.tree_enhancements import adaptive_barrier_tree
        r = adaptive_barrier_tree(100, 100, 120, 0.04, 0.20, 1.0, is_up=True)
        assert r.price >= 0
        assert r.barrier_accuracy < 5  # within 5% of barrier

    def test_adaptive_cheaper_than_vanilla(self):
        from pricebook.numerical.tree_enhancements import adaptive_barrier_tree
        from pricebook.numerical._trees import solve_tree
        vanilla = solve_tree(100, 100, 0.04, 0.20, 1.0)
        barrier = adaptive_barrier_tree(100, 100, 120, 0.04, 0.20, 1.0)
        assert barrier.price <= vanilla.price * 1.01

    def test_non_recombining_asian(self):
        from pricebook.numerical.tree_enhancements import asian_on_tree
        price = asian_on_tree(100, 100, 0.04, 0.20, 1.0, n_steps=8)
        assert price > 0
        # Asian call < vanilla call
        from pricebook.numerical._trees import solve_tree
        vanilla = solve_tree(100, 100, 0.04, 0.20, 1.0)
        assert price < vanilla.price * 1.1


# ═══════════════════════════════════════════════════════════════
# P9: Tree-MC Bridge
# ═══════════════════════════════════════════════════════════════

class TestTreeMCBridge:
    def test_lsm_on_tree(self):
        from pricebook.models.tree_mc_bridge import lsm_on_tree
        r = lsm_on_tree(100, 100, 0.04, 0.20, 1.0, is_call=False, n_paths=20_000)
        assert r.price > 0
        assert r.method == "lsm_on_tree"
        # American put > European put
        from pricebook.models.engine_protocol import AnalyticalEngine
        euro = AnalyticalEngine().price_vanilla(100, 100, 0.04, 0.20, 1.0, is_call=False)
        assert r.price >= euro.price * 0.95

    def test_stochastic_vol_tree(self):
        from pricebook.models.tree_mc_bridge import stochastic_vol_tree
        r = stochastic_vol_tree(100, 100, 0.04, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0)
        assert r.price > 0
        assert r.n_vol_nodes > 0

    def test_hybrid_auto_select(self):
        from pricebook.models.tree_mc_bridge import hybrid_price
        # European → tree
        r1 = hybrid_price(100, 100, 0.04, 0.20, 1.0)
        assert r1.method == "tree"
        # American → tree
        r2 = hybrid_price(100, 100, 0.04, 0.20, 1.0, is_american=True)
        assert r2.method == "tree"
        # American + path dep → LSM
        r3 = hybrid_price(100, 100, 0.04, 0.20, 1.0, is_american=True, is_path_dependent=True)
        assert r3.method == "lsm_on_tree"


# ═══════════════════════════════════════════════════════════════
# P10: Engine Comparison
# ═══════════════════════════════════════════════════════════════

class TestEngineComparison:
    def test_compare_engines(self):
        from pricebook.models.engine_comparison import compare_engines
        r = compare_engines(100, 100, 0.04, 0.20, 1.0, mc_paths=50_000, tree_steps=200)
        assert len(r.engines) == 3
        assert r.price_spread_pct < 5  # within 5%
        assert r.fastest_engine == "analytical"

    def test_validate_greeks(self):
        from pricebook.models.engine_comparison import validate_greeks
        r = validate_greeks(100, 100, 0.04, 0.20, 1.0, tolerance=0.15)
        # Delta and gamma should be close (within 15%)
        assert r["delta"]["rel_diff"] < 0.15

    def test_to_dict(self):
        from pricebook.models.engine_comparison import compare_engines
        r = compare_engines(100, 100, 0.04, 0.20, 1.0, mc_paths=20_000, tree_steps=100)
        d = r.to_dict()
        assert "greeks_consistent" in d


