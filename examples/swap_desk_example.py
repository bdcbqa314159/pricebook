#!/usr/bin/env python3
"""Swap Desk Operations — Worked Example.

Demonstrates the swap trading desk: DV01 ladder, net risk,
carry, stress testing, and capital.

Run:
    PYTHONPATH=python python examples/swap_desk_example.py
"""

from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.swap_desk import (
    swap_risk_metrics, SwapBook, SwapBookEntry,
    swap_carry_decomposition, swap_dashboard,
    swap_stress_suite, swap_capital,
    swap_hedge_recommendations,
)

REF = date(2024, 7, 15)
print("=" * 72)
print("Swap Desk Operations Example")
print("=" * 72)

# Build OIS curve (upward sloping)
deposits = [(REF + relativedelta(months=3), 0.045), (REF + relativedelta(months=6), 0.043)]
swaps_mkt = [(REF + relativedelta(years=1), 0.041), (REF + relativedelta(years=2), 0.039),
             (REF + relativedelta(years=5), 0.038), (REF + relativedelta(years=10), 0.036)]
ois = bootstrap(REF, deposits, swaps_mkt)

# Three swap positions
book = SwapBook("rates_desk")
swaps = [
    ("2Y Receiver", SwapDirection.RECEIVER, 2, 0.039, 20_000_000, "JPM"),
    ("5Y Payer", SwapDirection.PAYER, 5, 0.038, 50_000_000, "GS"),
    ("10Y Payer", SwapDirection.PAYER, 10, 0.036, 25_000_000, "MS"),
]
for name, direction, tenor, rate, notional, cpty in swaps:
    swap = InterestRateSwap(
        REF, REF + relativedelta(years=tenor),
        fixed_rate=rate, direction=direction, notional=notional)
    book.add(SwapBookEntry(name, swap, cpty))

# 1. Per-position risk
print("\n── Risk Metrics ──")
print(f"  {'Name':>15}  {'Dir':>10}  {'DV01':>10}  {'Gamma':>12}  {'Par Rate':>10}")
for e in book.entries:
    rm = swap_risk_metrics(e.swap, ois)
    print(f"  {e.trade_id:>15}  {rm.direction:>10}  {rm.dv01:>10,.0f}  {rm.gamma:>12,.0f}  {rm.par_rate:>10.4%}")

# 2. Dashboard with DV01 ladder
print("\n── Dashboard ──")
db = swap_dashboard(book, REF, ois)
print(f"  Positions: {db.n_positions}  |  Notional: {db.total_notional:,.0f}")
print(f"  Total DV01: {db.total_dv01:>+,.0f}  |  Net DV01: {db.net_dv01:>+,.0f}")
print(f"  DV01 Ladder:")
for tenor, dv01 in sorted(db.dv01_ladder.items()):
    print(f"    {tenor:>5}: {dv01:>+10,.0f}")

# 3. Carry
print("\n── Carry (5Y Payer) ──")
e5 = book.entries[1]
cd = swap_carry_decomposition(e5.swap, ois)
print(f"  Fixed accrual:   {cd.fixed_accrual:>10,.0f}/day")
print(f"  Float accrual:   {cd.floating_accrual:>10,.0f}/day")
print(f"  Net carry:       {cd.net_carry:>+10,.0f}/day ({cd.direction})")

# 4. Stress
print("\n── Stress Scenarios ──")
for r in swap_stress_suite(book, ois):
    print(f"  {r.description:>20}  →  {r.pnl:>+12,.0f}")

# 5. Capital
print("\n── SA-CCR Capital (5Y Payer) ──")
cap = swap_capital(e5.swap, ois)
print(f"  EAD: {cap.ead:,.0f}  |  Capital: {cap.capital:,.0f}  |  SIMM IM: {cap.simm_im:,.0f}")

# 6. Hedge
print("\n── Hedge Recommendations ──")
recs = swap_hedge_recommendations(book, ois, dv01_limit=5000, net_dv01_limit=5000)
for r in recs:
    print(f"  [{r.risk_type}] {r.current:,.0f} vs {r.limit:,.0f} → {r.action}")
if not recs:
    print("  All within limits.")

print("\n" + "=" * 72)
