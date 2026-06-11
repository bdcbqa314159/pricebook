"""Regression — `to_dict()` must return a defensive copy, not a shared `__dict__`.

Sweep closes C.1 B1 (Trade orphan), A.5 B1 (SolverResult), A.7 B1 (3 approximation
dataclasses), and the wider pattern across `book.py`, `daily_pnl.py`, `settlement.py`,
`mandate.py`, `greeks.py`, `dependency_graph.py`, `convergence_framework.py`,
`numerical_safety.py`, `numerical_method_map.py`, `market_data.py` — 30 callsites total.
"""

from __future__ import annotations

from datetime import date


def _assert_to_dict_is_copy(obj, mutate_key, mutate_val):
    """Helper: mutating the returned dict must not mutate `obj.__dict__`."""
    d = obj.to_dict()
    original = getattr(obj, mutate_key, "_SENTINEL_")
    d[mutate_key] = mutate_val
    after = getattr(obj, mutate_key, "_SENTINEL_")
    assert after == original, (
        f"{type(obj).__name__}.to_dict() returned a shared __dict__: "
        f"mutating the returned dict changed obj.{mutate_key} from "
        f"{original!r} to {after!r}"
    )


def test_solver_result_to_dict_is_copy():
    from pricebook.core.solvers import SolverResult
    r = SolverResult(root=1.0, iterations=5, converged=True, function_value=1e-13)
    _assert_to_dict_is_copy(r, "root", 999.0)


def test_chebyshev_to_dict_is_copy():
    import numpy as np
    from pricebook.core.approximation import ChebyshevInterpolant
    c = ChebyshevInterpolant(coefficients=np.array([1.0]), a=0.0, b=1.0, degree=0)
    _assert_to_dict_is_copy(c, "degree", 999)


def test_pade_to_dict_is_copy():
    import numpy as np
    from pricebook.core.approximation import PadeApproximant
    p = PadeApproximant(
        numerator=np.array([1.0]), denominator=np.array([1.0]), L=0, M=0,
    )
    _assert_to_dict_is_copy(p, "L", 999)


def test_richardson_to_dict_is_copy():
    import numpy as np
    from pricebook.core.approximation import RichardsonTable
    r = RichardsonTable(table=np.zeros((2, 2)), best_estimate=1.0, estimates=[1.0, 1.0])
    _assert_to_dict_is_copy(r, "best_estimate", 999.0)


def test_greeks_to_dict_is_copy():
    from pricebook.core.greeks import Greeks
    g = Greeks(price=5.0, delta=0.5, gamma=0.02)
    _assert_to_dict_is_copy(g, "delta", 999.0)


def test_position_to_dict_is_copy():
    from pricebook.core.book import Position
    p = Position(instrument_type="Swap", tenor_bucket="5Y-7Y",
                  net_notional=1e7, trade_count=3)
    _assert_to_dict_is_copy(p, "trade_count", 999)


def test_daily_pnl_to_dict_is_copy():
    from pricebook.core.daily_pnl import DailyPnL
    p = DailyPnL(
        book_name="X", prior_date=date(2024, 1, 1), current_date=date(2024, 1, 2),
        prior_pv=1.0, current_pv=2.0,
        market_move_pnl=1.0, new_trade_pnl=0.0, amendment_pnl=0.0, total_pnl=1.0,
    )
    _assert_to_dict_is_copy(p, "total_pnl", 999.0)


def test_cash_settlement_result_to_dict_is_copy():
    from pricebook.core.settlement import CashSettlementResult, SettlementType
    r = CashSettlementResult(SettlementType.CASH, 100.0, date(2024, 1, 1), "USD")
    _assert_to_dict_is_copy(r, "amount", 999.0)


def test_mandate_holding_to_dict_is_copy():
    from pricebook.core.mandate import PortfolioHolding
    h = PortfolioHolding(trade_id="T1", weight_pct=0.5, notional=1e6)
    _assert_to_dict_is_copy(h, "weight_pct", 999.0)
