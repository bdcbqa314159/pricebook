#!/usr/bin/env python3
"""Trader API Workflow — How a trader uses pricebook in 5 minutes.

Shows the ergonomic API: curves from dicts, analyse anything,
build books, run stress, compare CSAs — all without touching internals.

Run:
    PYTHONPATH=python python examples/api_trader_workflow.py
"""

import pricebook.api as pb
import pricebook.api_desk as desk
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 7, 15)
print("=" * 72)
print("Trader API Workflow — pricebook in 5 minutes")
print("=" * 72)

# ═══ STEP 1: Build curves (one line each) ═══
print("\n── Step 1: Build curves ──")
curve = pb.build_curve("USD",
    deposits={"3M": 0.052, "6M": 0.050},
    swaps={"1Y": 0.047, "2Y": 0.043, "5Y": 0.038, "10Y": 0.036},
    reference_date=REF)
print(f"  USD curve: 5Y DF = {curve.df(REF + relativedelta(years=5)):.6f}")

# ═══ STEP 2: Analyse a swap (everything in one call) ═══
print("\n── Step 2: Analyse a 5Y payer swap ──")
result = desk.analyse("irs", curve=curve, tenor="5Y", rate=0.038, notional=50_000_000)
print(f"  PV:       {result['pv']:>+12,.2f}")
print(f"  Par rate: {result['par_rate']:>12.4%}")
print(f"  DV01:     {result['dv01']:>+12,.2f}")
print(f"  Gamma:    {result['gamma']:>12,.0f}")
print(f"  Theta:    {result['theta']:>+12,.2f}")
print(f"  Carry:    {result['carry']}")

# ═══ STEP 3: Analyse a CDS ═══
print("\n── Step 3: Analyse a 5Y CDS (buy protection at 100bp) ──")
result = desk.analyse("cds", curve=curve, tenor="5Y", spread=0.01, hazard=0.02, notional=10_000_000)
print(f"  PV:         {result['pv']:>+12,.2f}")
print(f"  Par spread: {result['par_spread']*10000:>10.1f}bp")
print(f"  CS01:       {result['cs01']:>+12,.2f}")
print(f"  JTD:        {result['jtd']:>+12,.2f}")
print(f"  Carry:      {result['carry']}")

# ═══ STEP 4: CLN with leverage ═══
print("\n── Step 4: CLN (leveraged 2x) ──")
result = desk.cln("5Y", 0.07, curve, hazard=0.02, leverage=2.0, notional=10_000_000)
print(f"  PV:       {result['pv']:>12,.2f}")
print(f"  CS01:     {result['cs01']:>+12,.2f}")
print(f"  JTD:      {result['jtd']:>+12,.2f}")
print(f"  Leverage: {result['leverage']}x")

# ═══ STEP 5: TRS ═══
print("\n── Step 5: Equity TRS (spot=100, 6M) ──")
result = desk.trs("6M", 100.0, curve, funding_spread=0.005, notional=10_000_000, sigma=0.20)
print(f"  PV:       {result['pv']:>+12,.2f}")
print(f"  Delta:    {result['delta']:>12,.2f}")
print(f"  Vega:     {result['vega']:>12,.2f}")

# ═══ STEP 6: Build a swap book (from dicts!) ═══
print("\n── Step 6: Swap book from dicts → instant risk ──")
book = desk.swap_book([
    {"tenor": "2Y", "rate": 0.043, "direction": "receiver", "notional": 20_000_000},
    {"tenor": "5Y", "rate": 0.038, "direction": "payer", "notional": 50_000_000},
    {"tenor": "10Y", "rate": 0.036, "direction": "payer", "notional": 25_000_000},
], curve=curve)
print(f"  Positions:  {book['n_positions']}")
print(f"  Total DV01: {book['total_dv01']:>+12,.2f}")
print(f"  Net DV01:   {book['net_dv01']:>+12,.2f}")
print(f"  DV01 ladder: {book['dv01_ladder']}")
print(f"  Stress scenarios: {len(book['stress'])}")
for s in book['stress'][:3]:
    print(f"    {s['description']:>20}: {s['pnl']:>+12,.0f}")

# ═══ STEP 7: Vol surface from quotes ═══
print("\n── Step 7: FX vol surface from ATM/RR/BF ──")
surface = desk.vol_surface("fx", [
    {"expiry": "1M", "atm": 0.08, "rr25": -0.01, "bf25": 0.003},
    {"expiry": "3M", "atm": 0.09, "rr25": -0.012, "bf25": 0.004},
    {"expiry": "6M", "atm": 0.095, "rr25": -0.015, "bf25": 0.005},
    {"expiry": "1Y", "atm": 0.10, "rr25": -0.018, "bf25": 0.006},
], spot=1.08, ref=REF)
print(f"  Tenors: {len(surface.expiries)}")
vol_3m = surface.vol(surface.expiries[1])
print(f"  3M ATM vol: {vol_3m:.2%}")

# ═══ STEP 8: Multi-curve ═══
print("\n── Step 8: Multi-curve (USD + EUR) ──")
curves = desk.multicurve(ref=REF,
    usd={"swaps": {"1Y": 0.047, "5Y": 0.038, "10Y": 0.036}},
    eur={"swaps": {"1Y": 0.034, "5Y": 0.028, "10Y": 0.026}},
)
t5 = REF + relativedelta(years=5)
print(f"  USD 5Y DF: {curves['USD'].df(t5):.6f}")
print(f"  EUR 5Y DF: {curves['EUR'].df(t5):.6f}")
import math as _m
print(f"  Rate diff: {(-_m.log(curves['USD'].df(t5)) - (-_m.log(curves['EUR'].df(t5))))/5*10000:.0f}bp")

# ═══ STEP 9: Recovery analysis ═══
print("\n── Step 9: CLN recovery analysis ──")
rec = desk.recovery_analysis(
    cds_spreads={1: 0.005, 5: 0.01, 10: 0.012},
    curve=curve, tenor="5Y", coupon=0.05)
print(f"  Direct effect:   {rec['direct_effect']:>+12,.0f} (↑R → ↑recovery payment)")
print(f"  Indirect effect: {rec['indirect_effect']:>+12,.0f} (↑R → ↑h → ↑defaults)")
print(f"  Convexity:       {rec['convexity']:>12,.0f}")
print(f"  Surface points:  {len(rec['surface'])}")

# ═══ STEP 10: Repo ═══
print("\n── Step 10: Quick repo ──")
r = desk.repo(90, 10_000_000, 0.045, haircut=0.05)
print(f"  Cash lent:    {r['cash_lent']:>12,.2f}")
print(f"  Interest:     {r['interest']:>12,.2f}")
print(f"  Maturity amt: {r['maturity_amount']:>12,.2f}")
print(f"  30-day carry: {r['carry_30d']:>12,.2f}")

print("\n" + "=" * 72)
print("Done. Every product in < 3 lines. No imports beyond pb/desk.")
print("=" * 72)
