"""Tests for portfolio optimisation and game theory extensions."""

import pytest
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════
# P1: CVaR Optimisation
# ═══════════════════════════════════════════════════════════════

class TestCVaR:
    def test_cvar_portfolio(self):
        from pricebook.risk.cvar_optimisation import cvar_portfolio
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (500, 5))
        r = cvar_portfolio(returns, confidence=0.95)
        assert r.n_assets == 5
        assert abs(sum(r.weights) - 1.0) < 0.01
        assert r.cvar > 0

    def test_cvar_with_target(self):
        from pricebook.risk.cvar_optimisation import min_cvar_target_return
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (500, 3))
        r = min_cvar_target_return(returns, 0.0005)
        assert r.expected_return >= 0.0004  # close to target

    def test_risk_budget(self):
        from pricebook.risk.cvar_optimisation import cvar_risk_budget
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (500, 3))
        w = np.array([0.5, 0.3, 0.2])
        budget = cvar_risk_budget(returns, w)
        assert len(budget) == 3
        total_pct = sum(b["pct"] for b in budget)
        assert total_pct == pytest.approx(100, abs=1)

    def test_frontier(self):
        from pricebook.risk.cvar_optimisation import mean_cvar_frontier
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (500, 3))
        frontier = mean_cvar_frontier(returns, n_points=5)
        assert len(frontier) >= 3


# ═══════════════════════════════════════════════════════════════
# P8: Efficient Frontier
# ═══════════════════════════════════════════════════════════════

class TestEfficientFrontier:
    def _inputs(self):
        mu = np.array([0.08, 0.12, 0.06])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.025]])
        return mu, cov

    def test_frontier(self):
        from pricebook.risk.efficient_frontier import efficient_frontier
        mu, cov = self._inputs()
        r = efficient_frontier(mu, cov, n_points=10)
        assert len(r.points) > 5
        assert r.tangency is not None
        assert r.tangency.sharpe_ratio > 0

    def test_tangency(self):
        from pricebook.risk.efficient_frontier import tangency_portfolio
        mu, cov = self._inputs()
        t = tangency_portfolio(mu, cov, risk_free_rate=0.02)
        assert t.sharpe_ratio > 0
        assert abs(sum(t.weights) - 1) < 0.01

    def test_min_variance(self):
        from pricebook.risk.efficient_frontier import minimum_variance_portfolio
        _, cov = self._inputs()
        mv = minimum_variance_portfolio(cov)
        assert mv.volatility > 0

    def test_cml(self):
        from pricebook.risk.efficient_frontier import tangency_portfolio, capital_market_line
        mu, cov = self._inputs()
        t = tangency_portfolio(mu, cov, risk_free_rate=0.02)
        cml = capital_market_line(t, 0.02)
        assert len(cml) > 0
        assert cml[0]["return"] == pytest.approx(0.02)  # starts at rf


# ═══════════════════════════════════════════════════════════════
# P4: HRP
# ═══════════════════════════════════════════════════════════════

class TestHRP:
    def test_hrp_portfolio(self):
        from pricebook.risk.hierarchical_risk_parity import hrp_portfolio
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (250, 5))
        r = hrp_portfolio(returns)
        assert abs(sum(r.weights) - 1) < 0.01
        assert all(w >= 0 for w in r.weights)
        assert r.n_assets == 5

    def test_hrp_diversified(self):
        """HRP should produce non-concentrated weights."""
        from pricebook.risk.hierarchical_risk_parity import hrp_portfolio
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, (250, 10))
        r = hrp_portfolio(returns)
        assert max(r.weights) < 0.5  # no single asset > 50%


# ═══════════════════════════════════════════════════════════════
# P5: Brinson Attribution
# ═══════════════════════════════════════════════════════════════

class TestBrinson:
    def test_attribution(self):
        from pricebook.risk.brinson_attribution import brinson_attribution
        r = brinson_attribution(
            [0.40, 0.35, 0.25], [0.30, 0.40, 0.30],
            [0.10, 0.05, -0.02], [0.08, 0.06, 0.01],
            ["Equity", "Bonds", "Alts"],
        )
        # Allocation + selection + interaction ≈ active return
        assert r.total_allocation + r.total_selection + r.total_interaction == pytest.approx(r.total_active_return, abs=1e-10)

    def test_factor_attribution(self):
        from pricebook.risk.brinson_attribution import factor_based_attribution
        rng = np.random.default_rng(42)
        factors = rng.normal(0, 0.01, (100, 3))
        port = 0.5 * factors[:, 0] + 0.3 * factors[:, 1] + rng.normal(0, 0.001, 100)
        r = factor_based_attribution(port, factors, ["Mkt", "Size", "Value"])
        assert r["r_squared"] > 0.5


# ═══════════════════════════════════════════════════════════════
# P3: Kelly
# ═══════════════════════════════════════════════════════════════

class TestKelly:
    def test_single_asset(self):
        from pricebook.risk.kelly import kelly_fraction
        r = kelly_fraction(0.10, 0.20)
        assert r.kelly_fraction == pytest.approx(0.10 / 0.04, rel=0.01)

    def test_fractional(self):
        from pricebook.risk.kelly import fractional_kelly
        r = fractional_kelly(0.10, 0.20, fraction=0.5)
        full = 0.10 / 0.04
        assert r.kelly_fraction == pytest.approx(0.5 * full, rel=0.01)

    def test_multi_asset(self):
        from pricebook.risk.kelly import multi_asset_kelly
        mu = np.array([0.10, 0.08])
        cov = np.array([[0.04, 0.01], [0.01, 0.03]])
        r = multi_asset_kelly(mu, cov)
        assert len(r["weights"]) == 2


# ═══════════════════════════════════════════════════════════════
# P6: Robust Optimisation
# ═══════════════════════════════════════════════════════════════

class TestRobust:
    def test_ellipsoidal(self):
        from pricebook.risk.robust_optimisation import robust_mean_variance
        mu = np.array([0.08, 0.12, 0.06])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.025]])
        r = robust_mean_variance(mu, cov, epsilon=0.05)
        assert r.worst_case_return < r.nominal_return

    def test_robust_worst_case_below_nominal(self):
        """Robust worst-case return should be below nominal return."""
        from pricebook.risk.robust_optimisation import robust_mean_variance
        mu = np.array([0.08, 0.12, 0.06])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.025]])
        r = robust_mean_variance(mu, cov, epsilon=0.10)
        assert r.worst_case_return < r.nominal_return


# ═══════════════════════════════════════════════════════════════
# P2: Transaction Costs
# ═══════════════════════════════════════════════════════════════

class TestTransactionCost:
    def test_tc_rebalance(self):
        from pricebook.risk.transaction_cost_opt import tc_aware_rebalance
        mu = np.array([0.08, 0.12, 0.06])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.025]])
        current = np.array([0.33, 0.34, 0.33])
        r = tc_aware_rebalance(mu, cov, current, tc_bps=50)
        assert r.turnover >= 0
        assert r.transaction_cost >= 0

    def test_no_trade_bands(self):
        from pricebook.risk.transaction_cost_opt import no_trade_region
        mu = np.array([0.08, 0.12])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        bands = no_trade_region(mu, cov, np.array([0.6, 0.4]), tc_bps=10)
        assert len(bands) == 2
        assert all(b["lower"] <= b["optimal"] <= b["upper"] for b in bands)


# ═══════════════════════════════════════════════════════════════
# P7: Dynamic Allocation
# ═══════════════════════════════════════════════════════════════

class TestDynamic:
    def test_cppi(self):
        from pricebook.risk.dynamic_allocation import cppi_allocation
        r = cppi_allocation(1_000_000, floor_pct=0.80, multiplier=5)
        assert r.final_value > 0
        assert len(r.portfolio_values) > 0

    def test_glide_path(self):
        from pricebook.risk.dynamic_allocation import target_date_glide
        r = target_date_glide(30, 0.90, 0.30)
        assert r.equity_weights[0] == pytest.approx(0.90)
        assert r.equity_weights[-1] == pytest.approx(0.30)
        assert len(r.equity_weights) == 31


# ═══════════════════════════════════════════════════════════════
# X1: Portfolio Analytics
# ═══════════════════════════════════════════════════════════════

class TestAnalytics:
    def test_portfolio_metrics(self):
        from pricebook.risk.portfolio_analytics import portfolio_metrics
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.008, 252)
        m = portfolio_metrics(returns)
        assert m.annualised_vol > 0
        assert m.max_drawdown > 0
        assert 0 < m.hit_ratio < 1
        assert m.max_drawdown > 0

    def test_tracking(self):
        from pricebook.risk.portfolio_analytics import tracking_metrics
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0003, 0.01, 252)
        port = bench + rng.normal(0.0001, 0.002, 252)
        r = tracking_metrics(port, bench)
        assert r["tracking_error"] > 0
        assert "information_ratio" in r


# ═══════════════════════════════════════════════════════════════
# G1: N-Player Nash
# ═══════════════════════════════════════════════════════════════

class TestNPlayerNash:
    def test_fictitious_play_prisoners(self):
        from pricebook.models.n_player_nash import fictitious_play
        # Prisoner's dilemma
        A = np.array([[3, 0], [5, 1]])
        B = np.array([[3, 5], [0, 1]])
        r = fictitious_play([A, B], max_iter=5000)
        assert r.n_players == 2
        # Should converge to (Defect, Defect)
        assert r.strategies[0][1] > 0.5  # mostly defect

    def test_lemke_howson(self):
        from pricebook.models.n_player_nash import lemke_howson_2p
        A = np.array([[3, 0], [5, 1]])
        B = np.array([[3, 5], [0, 1]])
        r = lemke_howson_2p(A, B)
        assert r.n_players == 2

    def test_correlated_equilibrium(self):
        from pricebook.models.n_player_nash import correlated_equilibrium
        A = np.array([[3, 0], [5, 1]])
        B = np.array([[3, 5], [0, 1]])
        r = correlated_equilibrium([A, B])
        assert r.converged


# ═══════════════════════════════════════════════════════════════
# G2: Stackelberg
# ═══════════════════════════════════════════════════════════════

class TestStackelberg:
    def test_cournot(self):
        from pricebook.models.stackelberg import stackelberg_cournot
        r = stackelberg_cournot(a=100, b=1, c_leader=10, c_follower=10)
        assert r.leader_payoff > r.follower_payoff  # first-mover advantage
        assert r.leader_advantage > 0

    def test_bertrand(self):
        from pricebook.models.stackelberg import stackelberg_bertrand
        r = stackelberg_bertrand()
        assert r.leader_payoff > 0
        assert r.follower_payoff > 0

    def test_credit_market(self):
        from pricebook.models.stackelberg import credit_market_stackelberg
        r = credit_market_stackelberg()
        assert r.leader_payoff > 0


# ═══════════════════════════════════════════════════════════════
# G3: Bargaining
# ═══════════════════════════════════════════════════════════════

class TestBargaining:
    def test_nash_bargaining(self):
        from pricebook.models.bargaining import nash_bargaining
        # Linear feasible set
        feasible = np.array([[i, 10 - i] for i in range(11)], dtype=float)
        r = nash_bargaining(feasible)
        assert r.payoff_1 == pytest.approx(5, abs=0.5)  # symmetric → 50/50

    def test_rubinstein(self):
        from pricebook.models.bargaining import rubinstein_alternating
        r = rubinstein_alternating(100, 0.95, 0.90)
        assert r.payoff_1 > r.payoff_2  # player 1 more patient → gets more

    def test_debt_restructuring(self):
        from pricebook.models.bargaining import debt_restructuring_bargain
        r = debt_restructuring_bargain(100, 80, 0.40)
        assert r.payoff_2 > 80 * 0.40  # creditor gets more than liquidation


# ═══════════════════════════════════════════════════════════════
# G4: Market Microstructure
# ═══════════════════════════════════════════════════════════════

class TestMicrostructure:
    def test_kyle_lambda(self):
        from pricebook.models.market_microstructure_games import kyle_lambda
        r = kyle_lambda(1.0, 2.0)
        assert r.lambda_impact == pytest.approx(0.25)
        assert r.insider_profit > 0

    def test_more_informed_less_profit(self):
        from pricebook.models.market_microstructure_games import kyle_lambda
        r1 = kyle_lambda(1.0, 2.0, n_informed=1)
        r5 = kyle_lambda(1.0, 2.0, n_informed=5)
        assert r5.insider_profit < r1.insider_profit

    def test_glosten_milgrom(self):
        from pricebook.models.market_microstructure_games import glosten_milgrom
        r = glosten_milgrom(110, 90, 0.5, 0.3)
        assert r.ask > r.bid
        assert r.spread > 0

    def test_more_informed_wider_spread(self):
        from pricebook.models.market_microstructure_games import glosten_milgrom
        low = glosten_milgrom(110, 90, 0.5, 0.1)
        high = glosten_milgrom(110, 90, 0.5, 0.5)
        assert high.spread > low.spread

    def test_order_splitting(self):
        from pricebook.models.market_microstructure_games import optimal_order_splitting
        r = optimal_order_splitting(100_000, 1_000_000, 0.02, risk_aversion=1e-4)
        assert len(r.schedule) == 10
        assert sum(r.schedule) == pytest.approx(100_000, rel=0.01)
        assert r.total_cost > 0
