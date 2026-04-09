"""Tests for IRC (Incremental Risk Charge)."""

import pytest

from pricebook.regulatory.irc import (
    RATING_CATEGORIES,
    TRANSITION_MATRICES, get_transition_matrix, list_transition_matrices,
    TRANSITION_MATRIX_GLOBAL, TRANSITION_MATRIX_RECESSION, TRANSITION_MATRIX_BENIGN,
    CREDIT_SPREADS, get_credit_spread,
    LGD_BY_SENIORITY,
    IRCPosition, IRCConfig, get_lgd,
    calculate_modified_duration, calculate_spread_pv01,
    simulate_irc_portfolio, calculate_irc, quick_irc, calculate_irc_by_issuer,
)


# ---- Transition matrices ----

class TestTransitionMatrices:
    def test_rows_sum_to_one(self):
        for name in list_transition_matrices():
            m = get_transition_matrix(name)
            for from_r, row in m.items():
                total = sum(row.values())
                assert total == pytest.approx(1.0, abs=1e-6)

    def test_default_absorbing(self):
        for name in list_transition_matrices():
            m = get_transition_matrix(name)
            assert m["D"]["D"] == 1.0

    def test_recession_higher_default(self):
        """Recession matrix has higher default rates than benign."""
        rec = TRANSITION_MATRIX_RECESSION
        ben = TRANSITION_MATRIX_BENIGN
        # BB → D: recession should be higher
        assert rec["BB"]["D"] > ben["BB"]["D"]
        assert rec["B"]["D"] > ben["B"]["D"]

    def test_aliases(self):
        assert get_transition_matrix("default") is TRANSITION_MATRIX_GLOBAL
        assert get_transition_matrix("us_corporate") is TRANSITION_MATRIX_GLOBAL

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_transition_matrix("nonexistent")


# ---- Credit spreads ----

class TestCreditSpreads:
    def test_monotonic_in_rating(self):
        """Lower rating → higher spread."""
        for tenor in [1, 5, 10]:
            assert get_credit_spread("AAA", tenor) < get_credit_spread("BBB", tenor)
            assert get_credit_spread("BBB", tenor) < get_credit_spread("BB", tenor)
            assert get_credit_spread("BB", tenor) < get_credit_spread("CCC", tenor)

    def test_interpolation(self):
        s1 = get_credit_spread("BBB", 1)
        s2 = get_credit_spread("BBB", 2)
        s_mid = get_credit_spread("BBB", 1.5)
        assert s1 < s_mid < s2

    def test_extrapolation_flat(self):
        assert get_credit_spread("BBB", 0.5) == get_credit_spread("BBB", 1)
        assert get_credit_spread("BBB", 50) == get_credit_spread("BBB", 10)


# ---- LGD ----

class TestLGD:
    def test_seniority_lookup(self):
        pos = IRCPosition("p1", "X", 10_000_000, 10_000_000, "BBB", 5, seniority="senior_secured")
        assert get_lgd(pos) == 0.25

    def test_override(self):
        pos = IRCPosition("p1", "X", 10_000_000, 10_000_000, "BBB", 5, lgd=0.6)
        assert get_lgd(pos) == 0.6


# ---- Duration ----

class TestDuration:
    def test_zero_coupon(self):
        d = calculate_modified_duration(5.0, 0.0, 0.05)
        assert d == pytest.approx(5.0 / 1.05)

    def test_pv01_scales(self):
        pv01_1m = calculate_spread_pv01(1_000_000, 5)
        pv01_10m = calculate_spread_pv01(10_000_000, 5)
        assert pv01_10m == pytest.approx(10 * pv01_1m)


# ---- IRC simulation ----

class TestIRCSimulation:
    def test_empty(self):
        r = calculate_irc([])
        assert r["irc"] == 0.0

    def test_single_position(self):
        pos = [IRCPosition("p1", "Corp_A", 10_000_000, 10_000_000, "BBB", 5, coupon_rate=0.05)]
        config = IRCConfig(num_simulations=5_000)
        r = calculate_irc(pos, config)
        assert r["irc"] >= 0
        assert r["num_simulations"] == 5_000

    def test_higher_rating_lower_irc(self):
        config = IRCConfig(num_simulations=10_000)
        pos_aaa = [IRCPosition("p1", "X", 10_000_000, 10_000_000, "AAA", 5, coupon_rate=0.05)]
        pos_b = [IRCPosition("p1", "X", 10_000_000, 10_000_000, "B", 5, coupon_rate=0.05)]
        irc_aaa = calculate_irc(pos_aaa, config)["irc"]
        irc_b = calculate_irc(pos_b, config)["irc"]
        assert irc_b > irc_aaa

    def test_diversification(self):
        """Many issuers → lower IRC than single concentrated issuer."""
        config = IRCConfig(num_simulations=10_000)
        # Single issuer 10M
        single = [IRCPosition("p1", "ONLY", 10_000_000, 10_000_000, "BB", 5, coupon_rate=0.05)]
        # Five issuers 2M each
        diversified = [
            IRCPosition(f"p{i}", f"NAME_{i}", 2_000_000, 2_000_000, "BB", 5, coupon_rate=0.05)
            for i in range(5)
        ]
        irc_single = calculate_irc(single, config)["irc"]
        irc_div = calculate_irc(diversified, config)["irc"]
        assert irc_div < irc_single

    def test_recession_higher_irc(self):
        pos = [IRCPosition("p1", "X", 10_000_000, 10_000_000, "BB", 5, coupon_rate=0.05)]
        config_normal = IRCConfig(num_simulations=10_000, transition_matrix="global")
        config_rec = IRCConfig(num_simulations=10_000, transition_matrix="recession")
        irc_normal = calculate_irc(pos, config_normal)["irc"]
        irc_rec = calculate_irc(pos, config_rec)["irc"]
        assert irc_rec >= irc_normal

    def test_deterministic(self):
        pos = [IRCPosition("p1", "X", 10_000_000, 10_000_000, "BBB", 5, coupon_rate=0.05)]
        config = IRCConfig(num_simulations=5_000, seed=42)
        r1 = calculate_irc(pos, config)
        r2 = calculate_irc(pos, config)
        assert r1["irc"] == r2["irc"]


# ---- Quick IRC ----

class TestQuickIRC:
    def test_basic(self):
        positions = [
            {"issuer": "Corp_A", "notional": 10_000_000, "rating": "BBB", "tenor_years": 5},
            {"issuer": "Corp_B", "notional": 5_000_000, "rating": "BB", "tenor_years": 3},
        ]
        r = quick_irc(positions, num_simulations=5_000)
        assert r["irc"] >= 0
        assert r["num_positions"] == 2
        assert r["num_issuers"] == 2


# ---- IRC by issuer ----

class TestIRCByIssuer:
    def test_decomposition(self):
        positions = [
            IRCPosition("p1", "Corp_A", 5_000_000, 5_000_000, "BBB", 5, coupon_rate=0.05),
            IRCPosition("p2", "Corp_B", 5_000_000, 5_000_000, "BB", 5, coupon_rate=0.05),
        ]
        config = IRCConfig(num_simulations=5_000)
        by_iss = calculate_irc_by_issuer(positions, config)
        assert "Corp_A" in by_iss
        assert "Corp_B" in by_iss
        assert by_iss["Corp_A"] > 0
        assert by_iss["Corp_B"] > 0
