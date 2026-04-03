"""Tests for dependency graph."""

import pytest

from pricebook.dependency_graph import DependencyGraph, GraphNode, NodeCategory


class TestGraphConstruction:
    def test_add_node(self):
        g = DependencyGraph()
        n = g.add_node("rate_5y", "market_data")
        assert n.name == "rate_5y"
        assert n.category == NodeCategory.MARKET_DATA
        assert g.size == 1

    def test_add_with_dependency(self):
        g = DependencyGraph()
        r = g.add_node("rate", "market_data")
        c = g.add_node("curve", "curve", depends_on=[r])
        assert r in c.dependencies
        assert c in r.dependents

    def test_chain(self):
        g = DependencyGraph()
        r = g.add_node("rate", "market_data")
        c = g.add_node("curve", "curve", depends_on=[r])
        s = g.add_node("swap", "instrument", depends_on=[c])
        assert s in c.dependents
        assert c in r.dependents
        assert g.size == 3

    def test_duplicate_raises(self):
        g = DependencyGraph()
        g.add_node("x")
        with pytest.raises(ValueError, match="already exists"):
            g.add_node("x")

    def test_get_node(self):
        g = DependencyGraph()
        n = g.add_node("x")
        assert g.get_node("x") is n

    def test_get_missing_raises(self):
        g = DependencyGraph()
        with pytest.raises(KeyError):
            g.get_node("missing")


class TestDirtyPropagation:
    def test_mark_dirty_propagates(self):
        g = DependencyGraph()
        r = g.add_node("rate", "market_data")
        c = g.add_node("curve", "curve", depends_on=[r])
        s = g.add_node("swap", "instrument", depends_on=[c])

        g.mark_dirty(r)
        assert r.dirty
        assert c.dirty
        assert s.dirty

    def test_mark_mid_chain(self):
        """Marking middle node only dirties downstream."""
        g = DependencyGraph()
        r = g.add_node("rate", "market_data")
        c = g.add_node("curve", "curve", depends_on=[r])
        s = g.add_node("swap", "instrument", depends_on=[c])

        g.mark_dirty(c)
        assert not r.dirty  # upstream not affected
        assert c.dirty
        assert s.dirty

    def test_clean_all(self):
        g = DependencyGraph()
        r = g.add_node("rate")
        c = g.add_node("curve", depends_on=[r])
        g.mark_dirty(r)
        g.clean_all()
        assert not r.dirty
        assert not c.dirty

    def test_get_dirty_nodes_ordered(self):
        g = DependencyGraph()
        r = g.add_node("rate", "market_data")
        c = g.add_node("curve", "curve", depends_on=[r])
        s = g.add_node("swap", "instrument", depends_on=[c])

        g.mark_dirty(r)
        dirty = g.get_dirty_nodes()
        names = [n.name for n in dirty]
        assert names.index("rate") < names.index("curve")
        assert names.index("curve") < names.index("swap")

    def test_mark_by_name(self):
        g = DependencyGraph()
        r = g.add_node("rate")
        c = g.add_node("curve", depends_on=[r])
        g.mark_dirty_by_name("rate")
        assert c.dirty

    def test_isolated_node_not_dirty(self):
        """Node not downstream of change stays clean."""
        g = DependencyGraph()
        r1 = g.add_node("rate_usd")
        r2 = g.add_node("rate_eur")
        c1 = g.add_node("curve_usd", depends_on=[r1])
        c2 = g.add_node("curve_eur", depends_on=[r2])

        g.mark_dirty(r1)
        assert c1.dirty
        assert not r2.dirty
        assert not c2.dirty


class TestRemoveNode:
    def test_remove(self):
        g = DependencyGraph()
        r = g.add_node("rate")
        c = g.add_node("curve", depends_on=[r])
        g.remove_node("curve")
        assert g.size == 1
        assert c not in r.dependents

    def test_remove_missing_raises(self):
        g = DependencyGraph()
        with pytest.raises(KeyError):
            g.remove_node("missing")


class TestCycleDetection:
    def test_no_cycle(self):
        g = DependencyGraph()
        r = g.add_node("rate")
        c = g.add_node("curve", depends_on=[r])
        g.add_node("swap", depends_on=[c])
        assert not g.has_cycle()

    def test_detects_cycle(self):
        g = DependencyGraph()
        a = g.add_node("a")
        b = g.add_node("b", depends_on=[a])
        # Manually create cycle (bypassing the DAG add_node)
        a.add_dependency(b)
        b.add_dependent(a)
        assert g.has_cycle()


class TestCategoryFilter:
    def test_filter(self):
        g = DependencyGraph()
        g.add_node("r1", "market_data")
        g.add_node("r2", "market_data")
        g.add_node("c1", "curve")
        g.add_node("s1", "instrument")

        md = g.nodes_by_category("market_data")
        assert len(md) == 2
        assert all(n.category == NodeCategory.MARKET_DATA for n in md)


class TestPortfolioScenario:
    """Integration: build a realistic portfolio graph."""

    def test_portfolio_graph(self):
        g = DependencyGraph()

        # Market data
        ois_5y = g.add_node("ois_5y_rate", "market_data")
        ois_10y = g.add_node("ois_10y_rate", "market_data")
        vol_5y = g.add_node("vol_5y", "market_data")

        # Curves
        ois_curve = g.add_node("ois_curve", "curve", depends_on=[ois_5y, ois_10y])
        vol_surface = g.add_node("vol_surface", "curve", depends_on=[vol_5y])

        # Instruments
        swap = g.add_node("irs_5y", "instrument", depends_on=[ois_curve])
        swaption = g.add_node("swn_5y10y", "instrument", depends_on=[ois_curve, vol_surface])
        bond = g.add_node("bond_10y", "instrument", depends_on=[ois_curve])

        # Portfolio
        port = g.add_node("portfolio", "aggregation", depends_on=[swap, swaption, bond])

        assert g.size == 9

        # Bump one rate pillar
        g.mark_dirty(ois_5y)
        dirty = g.get_dirty_nodes()
        dirty_names = {n.name for n in dirty}

        # Should dirty: ois_5y, ois_curve, all instruments, portfolio
        assert "ois_5y_rate" in dirty_names
        assert "ois_curve" in dirty_names
        assert "irs_5y" in dirty_names
        assert "swn_5y10y" in dirty_names
        assert "bond_10y" in dirty_names
        assert "portfolio" in dirty_names

        # Should NOT dirty: vol
        assert "vol_5y" not in dirty_names
        assert "vol_surface" not in dirty_names
