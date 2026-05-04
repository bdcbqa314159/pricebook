#!/usr/bin/env python3
"""CDS Desk Operations — Worked Example.

Demonstrates the unified CDS desk: single-name positions across sectors,
risk aggregation, carry, stress, hedging, and index basis.

Run:
    PYTHONPATH=python python examples/cds_desk_example.py
"""

from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.cds import CDS
from pricebook.cds_market import build_cds_curve
from pricebook.cds_desk import (
    cds_risk_metrics, CDSBook, CDSBookEntry, CDSProductType,
    cds_carry_decomposition, cds_dashboard,
    cds_stress_suite, cds_capital,
    cds_hedge_recommendations,
)

REF = date(2024, 7, 15)
print("=" * 72)
print("CDS Desk Operations Example")
print("=" * 72)

# Build OIS curve
deposits = [(REF + relativedelta(months=3), 0.045), (REF + relativedelta(months=6), 0.043)]
swaps = [(REF + relativedelta(years=1), 0.041), (REF + relativedelta(years=5), 0.038),
         (REF + relativedelta(years=10), 0.036)]
ois = bootstrap(REF, deposits, swaps)

# Five single-name CDS positions
positions = [
    ("AAPL", "tech", 0.0050, 5),
    ("JPM", "financials", 0.0070, 5),
    ("XOM", "energy", 0.0090, 5),
    ("META", "tech", 0.0120, 3),
    ("GS", "financials", 0.0060, 10),
]

book = CDSBook("credit_desk")
for name, sector, spread, tenor in positions:
    cds_spreads = {tenor: spread}
    surv = build_cds_curve(REF, cds_spreads, ois, recovery=0.40)
    cds = CDS(REF, REF + relativedelta(years=tenor), spread=spread,
              notional=10_000_000, recovery=0.40)
    book.add(CDSBookEntry(name, cds, surv, reference_name=name, sector=sector))

# 1. Per-position risk
print("\n── Risk Metrics (per position) ──")
print(f"  {'Name':>6}  {'Spread':>8}  {'CS01':>10}  {'JTD':>12}  {'Carry':>10}")
for e in book.entries:
    rm = cds_risk_metrics(e.instrument, ois, e.survival_curve)
    print(f"  {e.reference_name:>6}  {e.instrument.spread*10000:>6.0f}bp  {rm.cs01:>10,.0f}  {rm.jump_to_default:>12,.0f}  {rm.carry:>10,.0f}")

# 2. Dashboard
print("\n── Dashboard ──")
db = cds_dashboard(book, REF, ois)
print(f"  Positions: {db.n_positions}  |  Notional: {db.total_notional:,.0f}")
print(f"  Total CS01: {db.total_cs01:,.0f}  |  Total JTD: {db.total_jtd:,.0f}")
print(f"  By sector: {db.by_sector}")

# 3. Carry
print("\n── Carry Decomposition (AAPL) ──")
e0 = book.entries[0]
cd = cds_carry_decomposition(e0.instrument, ois, e0.survival_curve)
print(f"  Premium income: {cd.premium_income:>10,.0f}/month")
print(f"  Default risk:   {cd.default_risk:>+10,.0f}/month")
print(f"  Roll-down:      {cd.roll_down:>+10,.0f}/month")
print(f"  Net carry:      {cd.net_carry:>10,.0f}/month")

# 4. Stress
print("\n── Stress Scenarios ──")
for r in cds_stress_suite(book, ois):
    print(f"  {r.description:>30}  →  {r.total_pnl:>+12,.0f}")

# 5. Capital (first position)
print("\n── SA-CCR Capital (AAPL 5Y CDS) ──")
cap = cds_capital(e0.instrument, ois, e0.survival_curve)
print(f"  EAD: {cap.ead:,.0f}  |  RWA: {cap.rwa:,.0f}  |  Capital: {cap.capital:,.0f}  |  SIMM IM: {cap.simm_im:,.0f}")

# 6. Hedge recommendations
print("\n── Hedge Recommendations ──")
recs = cds_hedge_recommendations(book, ois, cs01_limit=3000)
for r in recs:
    print(f"  [{r.risk_type}] {r.current:,.0f} vs limit {r.limit:,.0f} ({r.breach_pct:.0%}) → {r.action}")
if not recs:
    print("  All within limits.")

print("\n" + "=" * 72)
