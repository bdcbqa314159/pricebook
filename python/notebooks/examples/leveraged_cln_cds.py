#!/usr/bin/env python3
"""Leveraged CLN & CDS — Mechanics, Comparison, Decomposition.

Shows how leverage amplifies credit risk in CLNs, compares vanilla
vs leveraged pricing, verifies CLN = Bond - CDS relationship,
and demonstrates the leverage-recovery interaction.

Run:
    PYTHONPATH=python python examples/leveraged_cln_cds.py
"""

from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.cds import CDS
from pricebook.cds_market import build_cds_curve
from pricebook.cln import CreditLinkedNote
from pricebook.cln_desk import cln_risk_metrics
from pricebook.schedule import Frequency
from pricebook.recovery_analytics import recovery_greeks


REF = date(2024, 7, 15)
END = REF + relativedelta(years=5)

print("=" * 76)
print("Leveraged CLN & CDS — Mechanics and Comparison")
print("=" * 76)

# ── Market data ──
deposits = [(REF + relativedelta(months=3), 0.052), (REF + relativedelta(months=6), 0.050)]
swaps = [(REF + relativedelta(years=1), 0.047), (REF + relativedelta(years=2), 0.043),
         (REF + relativedelta(years=5), 0.038), (REF + relativedelta(years=10), 0.036)]
ois = bootstrap(REF, deposits, swaps)
cds_spreads = {1: 0.005, 3: 0.008, 5: 0.010, 10: 0.012}
surv = build_cds_curve(REF, cds_spreads, ois, recovery=0.40)

print(f"\nReference entity: IG corporate, 5Y CDS at 100bp, R=40%")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Vanilla vs Leveraged — Side by Side
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 1: Vanilla vs Leveraged CLN — Price Comparison")
print("─" * 76)

print(f"\n  All CLNs: 5Y, 10M notional, 5% coupon, R=40%")
print(f"\n  {'Leverage':>10}  {'CLN PV':>14}  {'Price/100':>10}  {'Credit Disc':>12}  {'JTD':>12}")
print(f"  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*12}  {'─'*12}")

for L in [1.0, 1.5, 2.0, 3.0, 5.0]:
    cln = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=10_000_000,
                           recovery=0.40, leverage=L, frequency=Frequency.QUARTERLY)
    pv = cln.dirty_price(ois, surv)
    rf = cln._risk_free_pv(ois)
    rm = cln_risk_metrics(cln, ois, surv)
    print(f"  {L:>8.1f}x  {pv:>14,.2f}  {pv/10_000_000*100:>10.4f}  {rf-pv:>12,.2f}  {rm.jump_to_default_pnl:>12,.2f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: CLN = Bond - CDS Decomposition
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 2: CLN = Bond - L × CDS Protection (Decomposition)")
print("─" * 76)

for L in [1.0, 2.0, 3.0]:
    cln = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=10_000_000,
                           recovery=0.40, leverage=L, frequency=Frequency.QUARTERLY)
    decomp = cln.decompose_bond_cds(ois, surv)
    print(f"\n  Leverage = {L:.0f}x:")
    print(f"    Risk-free bond PV  = {decomp['risk_free_bond']:>14,.2f}")
    print(f"    Embedded CDS (L×)  = {decomp['embedded_cds_protection']:>14,.2f}")
    print(f"    Synthetic CLN      = {decomp['synthetic_cln']:>14,.2f}")
    print(f"    Actual CLN PV      = {decomp['cln_price']:>14,.2f}")
    print(f"    Basis (actual-syn) = {decomp['basis']:>+14,.2f}")
    print(f"    Credit discount    = {decomp['credit_discount']:>14,.2f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: CDS vs CLN Comparison
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 3: CDS vs CLN — Same Credit, Different Funding")
print("─" * 76)

cds = CDS(REF, END, spread=0.01, notional=10_000_000, recovery=0.40)
cds_pv = cds.pv(ois, surv)
cds_par = cds.par_spread(ois, surv)
cds_cs01 = cds.cs01(ois, surv)

cln_v = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=10_000_000,
                         recovery=0.40, frequency=Frequency.QUARTERLY)
cln_pv = cln_v.dirty_price(ois, surv)
cln_rm = cln_risk_metrics(cln_v, ois, surv)

print(f"\n  {'':>20}  {'CDS (unfunded)':>16}  {'CLN (funded)':>16}")
print(f"  {'─'*20}  {'─'*16}  {'─'*16}")
print(f"  {'PV':>20}  {cds_pv:>16,.2f}  {cln_pv:>16,.2f}")
print(f"  {'Par spread':>20}  {cds_par*10000:>14.1f}bp  {'n/a':>16}")
print(f"  {'CS01':>20}  {cds_cs01:>16,.2f}  {cln_rm.cs01:>16,.2f}")
print(f"  {'Recovery sens':>20}  {cds.rec01(ois, surv):>16,.2f}  {cln_rm.recovery_sensitivity:>16,.2f}")
print(f"  {'Funding':>20}  {'None':>16}  {'5% coupon':>16}")
print(f"\n  Key: CDS is unfunded (no upfront principal). CLN is funded (investor pays par).")
print(f"  CLN CS01 is OPPOSITE sign to CDS CS01 (CLN holder is SHORT protection).")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Leverage Impact on Greeks
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 4: Leverage Impact on Risk Metrics")
print("─" * 76)

print(f"\n  {'Leverage':>10}  {'CS01':>10}  {'Rec01':>10}  {'JTD':>12}  {'DV01':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}")

for L in [1.0, 2.0, 3.0, 5.0]:
    cln = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=10_000_000,
                           recovery=0.40, leverage=L, frequency=Frequency.QUARTERLY)
    rm = cln_risk_metrics(cln, ois, surv)
    print(f"  {L:>8.1f}x  {rm.cs01:>10,.0f}  {rm.recovery_sensitivity:>10,.0f}  {rm.jump_to_default_pnl:>12,.0f}  {rm.dv01:>10,.0f}")

print(f"\n  → CS01 amplifies linearly with leverage")
print(f"  → JTD grows dramatically: at 5x, nearly all capital at risk")
print(f"  → DV01 is UNCHANGED: leverage doesn't affect rate sensitivity")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Recovery Convexity — Leverage Amplification
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 5: Recovery Convexity — Leverage Amplifies Everything")
print("─" * 76)

print(f"\n  {'Leverage':>10}  {'dPV/dR total':>14}  {'Direct':>14}  {'Indirect':>14}  {'Convexity':>12}")
print(f"  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*12}")

for L in [1.0, 2.0, 3.0, 5.0]:
    cln = CreditLinkedNote(REF, END, coupon_rate=max(0.05, 0.02 + L*0.01),
                           notional=10_000_000, recovery=0.40,
                           leverage=L, frequency=Frequency.QUARTERLY)
    rg = recovery_greeks(cln, ois, cds_spreads, REF)
    print(f"  {L:>8.1f}x  {rg.total_dPV_dR:>+14,.0f}  {rg.direct_effect:>+14,.0f}  {rg.indirect_effect:>+14,.0f}  {rg.convexity:>12,.0f}")

print(f"\n  → Direct effect scales with L (more leverage = bigger recovery payment)")
print(f"  → Indirect effect also scales (more defaults hurt more with leverage)")
print(f"  → Convexity amplification: leveraged CLN is MUCH more sensitive to R changes")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: Maximum Loss and Critical Leverage
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 6: Maximum Loss and Critical Leverage")
print("─" * 76)

R = 0.40
N = 10_000_000

print(f"\n  Recovery = {R:.0%}, Notional = {N:,.0f}")
print(f"\n  {'Leverage':>10}  {'On Default':>14}  {'Loss':>14}  {'Loss %':>8}  {'Wipeout?':>10}")
print(f"  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*8}  {'─'*10}")

for L in [1.0, 1.5, 1.67, 2.0, 3.0, 5.0]:
    recovery_pv = R * N
    leverage_loss = (L - 1) * (1 - R) * N
    net = recovery_pv - leverage_loss
    loss = N - max(net, 0)
    pct = loss / N
    wipeout = "YES" if net <= 0 else "no"
    print(f"  {L:>8.2f}x  {max(net,0):>14,.0f}  {loss:>14,.0f}  {pct:>7.0%}  {wipeout:>10}")

critical_L = 1 / (1 - R)
print(f"\n  Critical leverage L* = 1/(1-R) = 1/{1-R:.2f} = {critical_L:.2f}x")
print(f"  At L ≥ L*: investor loses ENTIRE investment on default")
print(f"  At L < L*: partial recovery")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: Bilateral Leveraged CLN
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 76)
print("SECTION 7: Bilateral CLN — Leverage + Issuer Risk")
print("─" * 76)

from pricebook.survival_curve import SurvivalCurve
surv_issuer = SurvivalCurve.flat(REF, 0.01)

print(f"\n  Issuer hazard: 1% | Correlation: 30%")
print(f"\n  {'Leverage':>10}  {'Unilateral':>14}  {'Bilateral':>14}  {'Issuer Disc':>14}")
print(f"  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*14}")

for L in [1.0, 2.0, 3.0]:
    cln = CreditLinkedNote(REF, END, coupon_rate=max(0.05, 0.02 + L*0.01),
                           notional=10_000_000, recovery=0.40,
                           leverage=L, frequency=Frequency.QUARTERLY)
    uni = cln.dirty_price(ois, surv)
    bi = cln.price_bilateral_mc(ois, surv, surv_issuer, correlation=0.3,
                                n_paths=20_000, seed=42)
    print(f"  {L:>8.1f}x  {uni:>14,.2f}  {bi.price:>14,.2f}  {bi.price - uni:>+14,.2f}")

print(f"\n  → Leverage amplifies issuer discount (more at risk if issuer fails too)")

print("\n" + "=" * 76)
print("Example complete.")
print("=" * 76)
