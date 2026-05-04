#!/usr/bin/env python3
"""CLN Recovery-Adjusted Pricing — Rich Worked Example.

Demonstrates the recovery-hazard entanglement and how to exploit it
for repricing credit instruments at non-standard recoveries.

Setup:
- USD OIS curve from deposits + swaps (realistic upward slope)
- Reference entity: IG corporate, CDS par spreads 50-120bp
- CLN: 5Y, 5% coupon, 10M notional, senior unsecured

Run:
    PYTHONPATH=python python examples/cln_recovery_pricing.py
"""

from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.cln import CreditLinkedNote
from pricebook.cln_desk import cln_risk_metrics
from pricebook.cln_xva import cln_mva, cln_kva, cln_analytic_cva, cln_wrong_way_cost
from pricebook.cds_market import build_cds_curve
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from pricebook.recovery_analytics import (
    recovery_curve_family, reprice_at_recovery, recovery_greeks, recovery_pv_surface,
)
from pricebook.recovery_pricing import RecoverySpec
from pricebook.hazard_rate_models import CIRPlusPlus


# ── Market data ──

REF = date(2024, 7, 15)
print("=" * 72)
print("CLN Recovery-Adjusted Pricing Example")
print(f"Valuation date: {REF}")
print("=" * 72)

# 1. Build USD OIS curve from realistic market instruments
deposits = [
    (REF + relativedelta(months=3), 0.045),   # 3M at 4.5%
    (REF + relativedelta(months=6), 0.043),   # 6M at 4.3%
]
swaps = [
    (REF + relativedelta(years=1), 0.041),    # 1Y at 4.1%
    (REF + relativedelta(years=2), 0.039),    # 2Y at 3.9%
    (REF + relativedelta(years=5), 0.038),    # 5Y at 3.8%
    (REF + relativedelta(years=10), 0.036),   # 10Y at 3.6%
]
ois_curve = bootstrap(REF, deposits, swaps)
print(f"\nOIS curve: 3M={4.5}%, 1Y={4.1}%, 5Y={3.8}%, 10Y={3.6}%")

# 2. CDS par spreads for reference entity (IG corporate)
cds_spreads = {1: 0.0050, 3: 0.0080, 5: 0.0100, 10: 0.0120}
print(f"CDS spreads: 1Y=50bp, 3Y=80bp, 5Y=100bp, 10Y=120bp")

# 3. Build CLN
cln = CreditLinkedNote(
    start=REF, end=REF + relativedelta(years=5),
    coupon_rate=0.05, notional=10_000_000, recovery=0.40,
    frequency=Frequency.QUARTERLY,
)
print(f"CLN: 5Y, 5% coupon, 10M notional, R=40%")

# ── Section 1: Base pricing at convention R=40% ──

print("\n" + "─" * 72)
print("SECTION 1: Base Pricing (Convention R = 40%)")
print("─" * 72)

surv_40 = build_cds_curve(REF, cds_spreads, ois_curve, recovery=0.40)
rm = cln_risk_metrics(cln, ois_curve, surv_40)

print(f"  PV         = {rm.pv:>14,.2f}")
print(f"  Clean/100  = {rm.pv / cln.notional * 100:>14.4f}")
print(f"  CS01       = {rm.cs01:>14,.2f}")
print(f"  Rec01      = {rm.recovery_sensitivity:>14,.2f}")
print(f"  JTD        = {rm.jump_to_default_pnl:>14,.2f}")
print(f"  DV01       = {rm.dv01:>14,.2f}")

# ── Section 2: Recovery curve family ──

print("\n" + "─" * 72)
print("SECTION 2: Recovery Curve Family (same CDS spreads, different h)")
print("─" * 72)
print(f"  {'Recovery':>10}  {'5Y Hazard':>12}  {'5Y Survival':>12}")
print(f"  {'─'*10}  {'─'*12}  {'─'*12}")

family = recovery_curve_family(cds_spreads, ois_curve, REF,
    recoveries=[0.20, 0.30, 0.40, 0.50, 0.60])

t5 = REF + relativedelta(years=5)
for R in sorted(family.keys()):
    q = family[R].survival(t5)
    h = -__import__('math').log(max(q, 1e-15)) / 5.0
    print(f"  {R:>10.0%}  {h*10000:>10.1f} bp  {q:>12.6f}")

# ── Section 3: Recovery-adjusted repricing ──

print("\n" + "─" * 72)
print("SECTION 3: Recovery-Adjusted CLN Pricing")
print("─" * 72)
print(f"  {'Recovery':>10}  {'CLN PV':>14}  {'vs Conv':>12}  {'Diff (%)':>10}")
print(f"  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*10}")

for R in [0.20, 0.30, 0.40, 0.50, 0.60]:
    result = reprice_at_recovery(cln, ois_curve, cds_spreads, REF, target_recovery=R)
    print(f"  {R:>10.0%}  {result.pv:>14,.2f}  {result.difference:>+12,.2f}  {result.difference_pct:>+10.4%}")

# ── Section 4: Recovery Greeks ──

print("\n" + "─" * 72)
print("SECTION 4: Recovery Greeks (Direct + Indirect Decomposition)")
print("─" * 72)

rg = recovery_greeks(cln, ois_curve, cds_spreads, REF, base_recovery=0.40)
print(f"  Total dPV/dR    = {rg.total_dPV_dR:>14,.2f}  (per 1% R change)")
print(f"  Direct effect   = {rg.direct_effect:>+14,.2f}  (higher recovery payment)")
print(f"  Indirect effect = {rg.indirect_effect:>+14,.2f}  (more defaults from higher h)")
print(f"  Convexity       = {rg.convexity:>14,.2f}  (d²PV/dR²)")
print()
print(f"  → Direct is POSITIVE: +1% recovery → you get more on default")
print(f"  → Indirect is NEGATIVE: +1% recovery → h goes up → more defaults")
if abs(rg.indirect_effect) > abs(rg.direct_effect):
    print(f"  → Indirect DOMINATES: net effect of higher R is NEGATIVE for CLN holder")
else:
    print(f"  → Direct DOMINATES: net effect of higher R is POSITIVE for CLN holder")

# ── Section 5: Leveraged CLN comparison ──

print("\n" + "─" * 72)
print("SECTION 5: Leveraged CLN — Amplified Recovery Convexity")
print("─" * 72)

cln_lev = CreditLinkedNote(
    start=REF, end=REF + relativedelta(years=5),
    coupon_rate=0.07, notional=10_000_000, recovery=0.40,
    leverage=2.0, frequency=Frequency.QUARTERLY,
)

rg_v = recovery_greeks(cln, ois_curve, cds_spreads, REF)
rg_l = recovery_greeks(cln_lev, ois_curve, cds_spreads, REF)

print(f"  {'':>20}  {'Vanilla':>14}  {'Leveraged 2x':>14}")
print(f"  {'Total dPV/dR':>20}  {rg_v.total_dPV_dR:>14,.2f}  {rg_l.total_dPV_dR:>14,.2f}")
print(f"  {'Direct':>20}  {rg_v.direct_effect:>+14,.2f}  {rg_l.direct_effect:>+14,.2f}")
print(f"  {'Indirect':>20}  {rg_v.indirect_effect:>+14,.2f}  {rg_l.indirect_effect:>+14,.2f}")
print(f"  {'Convexity':>20}  {rg_v.convexity:>14,.2f}  {rg_l.convexity:>14,.2f}")

# ── Section 6: Wrong-way cost ──

print("\n" + "─" * 72)
print("SECTION 6: Wrong-Way Cost (Stochastic Recovery)")
print("─" * 72)

ww_cost = cln_wrong_way_cost(cln, ois_curve, surv_40, n_sims=50_000)
print(f"  Deterministic PV    = {rm.pv:>14,.2f}")
print(f"  Wrong-way cost      = {ww_cost:>14,.2f}")
print(f"  → Stochastic recovery with rho_DR=-0.3 reduces PV by {ww_cost:,.0f}")

# ── Section 7: Stochastic intensity ──

print("\n" + "─" * 72)
print("SECTION 7: Stochastic Intensity Pricing (CIR++)")
print("─" * 72)

det_pv = cln.dirty_price(ois_curve, surv_40)
stoch = cln.price_stochastic_intensity_from_curve(
    ois_curve, surv_40, model_type="cir++", xi=0.10,
    n_paths=20_000, n_steps=100,
)
print(f"  Deterministic PV = {det_pv:>14,.2f}")
print(f"  CIR++ PV (ξ=10%) = {stoch.price:>14,.2f}")
print(f"  Convexity adj    = {stoch.price - det_pv:>+14,.2f}")

# ── Section 8: Bilateral CLN ──

print("\n" + "─" * 72)
print("SECTION 8: Bilateral CLN (Issuer + Reference Default)")
print("─" * 72)

surv_issuer = SurvivalCurve.flat(REF, 0.01)  # 1% issuer hazard
bilateral = cln.price_bilateral_mc(
    ois_curve, surv_40, surv_issuer,
    issuer_recovery=0.45, correlation=0.3, n_paths=20_000,
)
print(f"  Unilateral PV    = {det_pv:>14,.2f}")
print(f"  Bilateral PV     = {bilateral.price:>14,.2f}")
print(f"  Issuer discount  = {bilateral.price - det_pv:>+14,.2f}")

# ── Section 9: All-in cost ──

print("\n" + "─" * 72)
print("SECTION 9: All-In Cost Decomposition")
print("─" * 72)

mva = cln_mva(cln, ois_curve, surv_40)
kva = cln_kva(cln, ois_curve, surv_40)
cva = cln_analytic_cva(cln, ois_curve, surv_40)

print(f"  CVA              = {cva:>14,.2f}")
print(f"  KVA              = {kva:>14,.2f}")
print(f"  MVA              = {mva:>14,.2f}")
print(f"  Wrong-way cost   = {ww_cost:>14,.2f}")
print(f"  Total hidden cost= {cva + kva + mva + ww_cost:>14,.2f}")
hidden_bps = (cva + kva + mva + ww_cost) / cln.notional * 10_000 / 5.0
print(f"  Hidden cost (bps/yr) = {hidden_bps:>10.1f}")

print("\n" + "=" * 72)
print("Example complete.")
print("=" * 72)
