"""Tests for capital allocation & RORC."""

import pytest
import numpy as np

from pricebook.regulatory.capital_allocation import (
    DeskCapitalInput, DeskAllocation, CapitalAllocationResult,
    euler_allocation, pro_rata_allocation, calculate_rorc,
    allocate_and_report, capital_limit_monitor,
)


@pytest.fixture
def desks():
    return [
        DeskCapitalInput("rates", 100, 15, rwa=1250),
        DeskCapitalInput("credit", 80, 20, rwa=1000),
        DeskCapitalInput("equity", 60, 8, rwa=750),
    ]


class TestEulerAllocation:
    def test_sums_to_portfolio(self, desks):
        alloc = euler_allocation(desks, portfolio_capital=200)
        assert abs(sum(alloc) - 200) < 1e-6

    def test_with_correlation(self, desks):
        corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
        alloc = euler_allocation(desks, correlation_matrix=corr)
        assert sum(alloc) > 0
        assert all(a > 0 for a in alloc)

    def test_proportional_without_corr(self, desks):
        alloc = euler_allocation(desks, portfolio_capital=240)
        # Without correlation: proportional to standalone
        total_standalone = 240
        assert abs(alloc[0] - 100 / 240 * 240) < 1e-6

    def test_single_desk(self):
        alloc = euler_allocation([DeskCapitalInput("only", 100, 10)], portfolio_capital=100)
        assert abs(alloc[0] - 100) < 1e-6

    def test_empty(self):
        assert euler_allocation([]) == []


class TestProRata:
    def test_proportional(self, desks):
        alloc = pro_rata_allocation(desks, 240)
        total = 240
        assert abs(alloc[0] - 100 / 240 * total) < 1e-6
        assert abs(sum(alloc) - 240) < 1e-6


class TestRORC:
    def test_positive(self):
        assert abs(calculate_rorc(20, 100) - 0.20) < 1e-10

    def test_zero_capital(self):
        assert calculate_rorc(20, 0) == 0.0


class TestAllocateAndReport:
    def test_basic(self, desks):
        result = allocate_and_report(desks, hurdle_rate=0.10)
        assert isinstance(result, CapitalAllocationResult)
        assert len(result.desk_allocations) == 3
        assert result.diversification_benefit > 0

    def test_hurdle_check(self, desks):
        result = allocate_and_report(desks, hurdle_rate=0.10)
        for da in result.desk_allocations:
            if da.rorc >= 0.10:
                assert da.exceeds_hurdle
            else:
                assert not da.exceeds_hurdle

    def test_best_worst(self, desks):
        result = allocate_and_report(desks, hurdle_rate=0.10)
        rorcs = {da.desk_id: da.rorc for da in result.desk_allocations}
        assert rorcs[result.best_rorc_desk] >= rorcs[result.worst_rorc_desk]

    def test_pro_rata_method(self, desks):
        result = allocate_and_report(desks, method="pro_rata", portfolio_capital=200)
        assert result.method == "pro_rata"
        assert abs(result.total_portfolio_capital - 200) < 1e-6

    def test_to_dict(self, desks):
        d = allocate_and_report(desks).to_dict()
        assert "portfolio_rorc" in d
        assert "desk_allocations" in d

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            allocate_and_report([])


class TestCapitalLimitMonitor:
    def test_no_breach(self, desks):
        result = allocate_and_report(desks, portfolio_capital=200)
        breaches = capital_limit_monitor(result.desk_allocations, {"rates": 1000, "credit": 1000})
        assert len(breaches) == 0

    def test_breach(self, desks):
        result = allocate_and_report(desks, portfolio_capital=200)
        breaches = capital_limit_monitor(result.desk_allocations, {"rates": 10})
        assert len(breaches) == 1
        assert breaches[0]["desk_id"] == "rates"
