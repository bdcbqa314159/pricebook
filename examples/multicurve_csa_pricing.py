#!/usr/bin/env python3
"""Multi-Curve CSA Pricing — Post-LIBOR Framework.

Demonstrates the full multi-curve pipeline: RFR curves (SOFR, ESTR, SONIA),
dual-curve swap pricing, CSA discounting comparison, and IBOR fallback.

Run:
    PYTHONPATH=python python examples/multicurve_csa_pricing.py
"""

from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.sofr_curve import build_sofr_curve, build_estr_curve, build_sonia_curve
from pricebook.bootstrap import bootstrap_forward_curve
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.funding_curve import FundingCurve
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention, year_fraction


REF = date(2024, 7, 15)

print("=" * 76)
print("Multi-Curve CSA Pricing — Post-LIBOR Framework")
print(f"Valuation: {REF}")
print("=" * 76)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Build G3 RFR Curves
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 1: Build G3 RFR Curves (SOFR, ESTR, SONIA)")
print("─" * 76)

# USD: SOFR OIS swap rates (realistic post-hike)
sofr_swaps = [
    (REF + relativedelta(months=1), 0.0530),
    (REF + relativedelta(months=3), 0.0525),
    (REF + relativedelta(months=6), 0.0510),
    (REF + relativedelta(years=1), 0.0475),
    (REF + relativedelta(years=2), 0.0430),
    (REF + relativedelta(years=3), 0.0400),
    (REF + relativedelta(years=5), 0.0380),
    (REF + relativedelta(years=7), 0.0370),
    (REF + relativedelta(years=10), 0.0360),
    (REF + relativedelta(years=15), 0.0355),
    (REF + relativedelta(years=20), 0.0350),
    (REF + relativedelta(years=30), 0.0345),
]
sofr_curve = build_sofr_curve(REF, sofr_swaps=sofr_swaps)

# EUR: ESTR swap rates (lower, ECB rate cycle)
estr_swaps = [
    (REF + relativedelta(months=3), 0.0375),
    (REF + relativedelta(months=6), 0.0365),
    (REF + relativedelta(years=1), 0.0340),
    (REF + relativedelta(years=2), 0.0310),
    (REF + relativedelta(years=5), 0.0280),
    (REF + relativedelta(years=10), 0.0260),
    (REF + relativedelta(years=30), 0.0250),
]
estr_curve = build_estr_curve(REF, estr_swaps)

# GBP: SONIA swap rates
sonia_swaps = [
    (REF + relativedelta(months=3), 0.0520),
    (REF + relativedelta(years=1), 0.0480),
    (REF + relativedelta(years=5), 0.0400),
    (REF + relativedelta(years=10), 0.0380),
    (REF + relativedelta(years=30), 0.0370),
]
sonia_curve = build_sonia_curve(REF, sonia_swaps)

t5 = REF + relativedelta(years=5)
t10 = REF + relativedelta(years=10)
print(f"\n  {'Currency':>8}  {'5Y DF':>10}  {'10Y DF':>10}  {'5Y Zero':>10}")
for name, curve in [("USD/SOFR", sofr_curve), ("EUR/ESTR", estr_curve), ("GBP/SONIA", sonia_curve)]:
    df5 = curve.df(t5)
    df10 = curve.df(t10)
    z5 = -__import__('math').log(df5) / 5.0
    print(f"  {name:>8}  {df5:>10.6f}  {df10:>10.6f}  {z5:>10.4%}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Build EUR Projection Curve (EURIBOR 3M off ESTR discount)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 2: Dual-Curve — EURIBOR 3M Projection off ESTR Discount")
print("─" * 76)

# EURIBOR 3M swap rates (higher than ESTR by ~10-15bp basis)
euribor_swaps = [
    (REF + relativedelta(years=1), 0.0352),
    (REF + relativedelta(years=2), 0.0322),
    (REF + relativedelta(years=5), 0.0295),
    (REF + relativedelta(years=10), 0.0275),
]
euribor_curve = bootstrap_forward_curve(REF, euribor_swaps, estr_curve)

print(f"\n  ESTR-EURIBOR basis (5Y):")
estr_5y = estr_curve.forward_rate(t5 - relativedelta(months=3), t5)
euribor_5y = euribor_curve.forward_rate(t5 - relativedelta(months=3), t5)
print(f"    ESTR 5Y fwd:    {estr_5y:.4%}")
print(f"    EURIBOR 5Y fwd: {euribor_5y:.4%}")
print(f"    Basis:           {(euribor_5y - estr_5y)*10000:.1f} bp")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Price EUR IRS under 3 CSA Scenarios
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 3: Price 5Y EUR IRS under Different CSA Scenarios")
print("─" * 76)

swap = InterestRateSwap(
    REF, REF + relativedelta(years=5),
    fixed_rate=0.029, direction=SwapDirection.PAYER,
    notional=50_000_000,
    fixed_frequency=Frequency.ANNUAL,
    float_frequency=Frequency.QUARTERLY,
)

# Scenario 1: EUR cash collateral → ESTR discount
pv_eur_csa = swap.pv(estr_curve, euribor_curve)

# Scenario 2: USD cash collateral → SOFR discount (approximation: use SOFR DF)
# In reality: would adjust via xccy basis
pv_usd_csa = swap.pv(sofr_curve, euribor_curve)

# Scenario 3: No collateral → unsecured funding (OIS + 80bp spread)
funding_spread = 0.0080  # 80bp over OIS (typical bank unsecured)
# Build unsecured curve by bumping OIS by the funding spread
unsecured_dc = estr_curve.bumped(funding_spread)
pv_uncoll = swap.pv(unsecured_dc, euribor_curve)

print(f"\n  5Y EUR IRS: pay {swap.fixed_rate:.2%} fixed, receive EURIBOR 3M")
print(f"  Notional: {swap.notional:,.0f}")
print(f"\n  {'CSA Type':>25}  {'Discount Curve':>15}  {'PV':>14}  {'vs EUR CSA':>12}")
print(f"  {'─'*25}  {'─'*15}  {'─'*14}  {'─'*12}")
print(f"  {'EUR cash (ESTR)':>25}  {'ESTR OIS':>15}  {pv_eur_csa:>+14,.2f}  {'baseline':>12}")
print(f"  {'USD cash (SOFR)':>25}  {'SOFR OIS':>15}  {pv_usd_csa:>+14,.2f}  {pv_usd_csa - pv_eur_csa:>+12,.2f}")
print(f"  {'No collateral':>25}  {'Unsecured':>15}  {pv_uncoll:>+14,.2f}  {pv_uncoll - pv_eur_csa:>+12,.2f}")

csa_switch_value = abs(pv_usd_csa - pv_eur_csa)
print(f"\n  CSA switch value (EUR→USD): {csa_switch_value:,.2f}")
print(f"  Uncollateralised penalty:   {abs(pv_uncoll - pv_eur_csa):,.2f}")
print(f"  → Proper CSA discounting matters: {abs(pv_uncoll - pv_eur_csa)/swap.notional*10000:.1f} bps")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Par Rate Sensitivity to CSA
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 4: Par Rate Depends on CSA")
print("─" * 76)

par_eur = swap.par_rate(estr_curve, euribor_curve)
par_usd = swap.par_rate(sofr_curve, euribor_curve)
par_uncoll = swap.par_rate(unsecured_dc, euribor_curve)

print(f"\n  {'CSA':>25}  {'Par Rate':>10}  {'vs EUR':>10}")
print(f"  {'─'*25}  {'─'*10}  {'─'*10}")
print(f"  {'EUR cash (ESTR)':>25}  {par_eur:>10.4%}  {'baseline':>10}")
print(f"  {'USD cash (SOFR)':>25}  {par_usd:>10.4%}  {(par_usd-par_eur)*10000:>+8.1f}bp")
print(f"  {'No collateral':>25}  {par_uncoll:>10.4%}  {(par_uncoll-par_eur)*10000:>+8.1f}bp")
print(f"\n  → Same swap, different CSA → different par rate!")
print(f"  → The discount curve DIRECTLY affects what 'fair' means.")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: IBOR Fallback
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 5: IBOR Fallback (EURIBOR → ESTR + Spread)")
print("─" * 76)

# ISDA EURIBOR 3M fallback spread (hypothetical, ~8bp)
fallback_spread = 0.0008  # 8bp

print(f"\n  ISDA fallback: EURIBOR 3M → compounded ESTR + {fallback_spread*10000:.0f}bp")
print(f"\n  {'Tenor':>8}  {'EURIBOR fwd':>12}  {'ESTR fwd':>12}  {'Fallback':>12}  {'Diff':>8}")
print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}")

for years in [1, 2, 5, 10]:
    d1 = REF + relativedelta(years=years) - relativedelta(months=3)
    d2 = REF + relativedelta(years=years)
    euribor_fwd = euribor_curve.forward_rate(d1, d2)
    estr_fwd = estr_curve.forward_rate(d1, d2)
    fallback_rate = estr_fwd + fallback_spread
    diff = euribor_fwd - fallback_rate
    print(f"  {years:>6}Y  {euribor_fwd:>12.4%}  {estr_fwd:>12.4%}  {fallback_rate:>12.4%}  {diff*10000:>+6.1f}bp")

print(f"\n  → Non-zero 'Diff' = value transfer from EURIBOR to fallback")
print(f"  → This is why the basis is priced and traded")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: DV01 Decomposition (Rate vs Basis)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 6: DV01 Decomposition — Rate Risk vs Basis Risk")
print("─" * 76)

h = 0.0001

# OIS DV01: bump ESTR curve only (discount risk)
pv_ois_up = swap.pv(estr_curve.bumped(h), euribor_curve)
pv_ois_dn = swap.pv(estr_curve.bumped(-h), euribor_curve)
ois_dv01 = (pv_ois_up - pv_ois_dn) / 2

# Projection DV01: bump EURIBOR curve only (forward risk)
pv_proj_up = swap.pv(estr_curve, euribor_curve.bumped(h))
pv_proj_dn = swap.pv(estr_curve, euribor_curve.bumped(-h))
proj_dv01 = (pv_proj_up - pv_proj_dn) / 2

# Total DV01: bump both
pv_total_up = swap.pv(estr_curve.bumped(h), euribor_curve.bumped(h))
pv_total_dn = swap.pv(estr_curve.bumped(-h), euribor_curve.bumped(-h))
total_dv01 = (pv_total_up - pv_total_dn) / 2

print(f"\n  {'DV01 Component':>20}  {'Value':>12}  {'Share':>8}")
print(f"  {'─'*20}  {'─'*12}  {'─'*8}")
print(f"  {'OIS (discount)':>20}  {ois_dv01:>+12,.2f}  {ois_dv01/total_dv01:>7.0%}")
print(f"  {'Projection (fwd)':>20}  {proj_dv01:>+12,.2f}  {proj_dv01/total_dv01:>7.0%}")
print(f"  {'Total (parallel)':>20}  {total_dv01:>+12,.2f}  {'100%':>8}")
print(f"\n  → Discount DV01 and projection DV01 are DIFFERENT exposures")
print(f"  → Hedging with OIS swaps reduces discount risk")
print(f"  → Hedging with EURIBOR swaps reduces projection risk")
print(f"  → Both needed for complete hedge")

print("\n" + "=" * 76)
print("Example complete.")
print("=" * 76)
