#!/usr/bin/env python3
"""Bond Desk Operations — Worked Example.

Demonstrates bond desk: key-rate DV01, carry-and-roll,
funding cost, stress testing, and lifecycle.

Run:
    PYTHONPATH=python python examples/bond_desk_example.py
"""

from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.bond import FixedRateBond
from pricebook.bond_trading_desk import (
    bond_risk_metrics, bond_carry_roll, bond_dashboard,
    bond_stress_suite, bond_funding_cost,
)

REF = date(2024, 7, 15)
print("=" * 72)
print("Bond Desk Operations Example")
print("=" * 72)

# Build yield curve (upward sloping)
deposits = [(REF + relativedelta(months=3), 0.045), (REF + relativedelta(months=6), 0.043)]
swaps = [(REF + relativedelta(years=1), 0.041), (REF + relativedelta(years=2), 0.039),
         (REF + relativedelta(years=5), 0.038), (REF + relativedelta(years=10), 0.036),
         (REF + relativedelta(years=20), 0.035)]
curve = bootstrap(REF, deposits, swaps)

# Three UST positions
bonds = [
    ("2Y Note", FixedRateBond.treasury_note(REF, REF + relativedelta(years=2), 0.0425), 25_000_000),
    ("5Y Note", FixedRateBond.treasury_note(REF, REF + relativedelta(years=5), 0.04125), 50_000_000),
    ("10Y Note", FixedRateBond.treasury_note(REF, REF + relativedelta(years=10), 0.04), 30_000_000),
]
positions = [(name, "UST", bond, face) for name, bond, face in bonds]

# 1. Risk metrics with key-rate DV01
print("\n── Risk Metrics ──")
for name, bond, face in bonds:
    rm = bond_risk_metrics(bond, curve, REF)
    print(f"\n  {name}:")
    print(f"    Dirty price: {rm.pv:.4f}  |  YTM: {rm.ytm:.4%}")
    print(f"    Mod duration: {rm.modified_duration:.4f}  |  Eff duration: {rm.effective_duration:.4f}")
    print(f"    Convexity: {rm.convexity:.4f}  |  DV01: {rm.dv01:.6f}")
    print(f"    Key-rate DV01:")
    for tenor, kr in sorted(rm.key_rate_dv01.items(), key=lambda x: float(x[0].rstrip('YMW') or '0')):
        if abs(kr) > 0.0001:
            print(f"      {tenor:>5}: {kr:>+.6f}")

# 2. Dashboard
print("\n── Dashboard ──")
db = bond_dashboard(positions, REF, curve)
print(f"  Positions: {db.n_positions}  |  Total face: {db.total_face:,.0f}")
print(f"  Total DV01: {db.total_dv01:>+,.0f}  |  W. Duration: {db.weighted_duration:.2f}")
print(f"  By tenor: {db.by_tenor}")

# 3. Carry-and-roll (30 day forecast)
print("\n── Carry & Roll (30-day forecast) ──")
for name, bond, face in bonds:
    cr = bond_carry_roll(bond, curve, repo_rate=0.04, horizon_days=30)
    print(f"\n  {name} (per {face/1e6:.0f}M face):")
    print(f"    Coupon carry:  {cr.coupon_carry:>+10,.0f}")
    print(f"    Funding cost:  {cr.funding_cost:>10,.0f}")
    print(f"    Net carry:     {cr.net_carry:>+10,.0f}")
    print(f"    Roll-down:     {cr.roll_down_return:>+10,.0f}")
    print(f"    Total C&R:     {cr.total_carry_and_roll:>+10,.0f}")

# 4. Funding cost analysis
print("\n── Funding Cost (10Y Note) ──")
_, bond10, _ = bonds[2]
fc = bond_funding_cost(bond10, bond10.dirty_price(curve), repo_rate=0.04, settlement=REF)
print(f"  Coupon income:  {fc.coupon_income:>10,.0f}/year")
print(f"  Repo cost:      {fc.repo_cost:>10,.0f}/year")
print(f"  Net income:     {fc.net_income:>+10,.0f}/year")
print(f"  Breakeven repo: {fc.breakeven_repo:.4%}")

# 5. Stress
print("\n── Stress Scenarios ──")
stress_pos = [(name, bond, face) for name, bond, face in bonds]
for r in bond_stress_suite(stress_pos, curve):
    print(f"  {r.description:>20}  →  {r.total_pnl:>+12,.0f}")

print("\n" + "=" * 72)
