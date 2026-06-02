"""Pricebook Visualisation — Quick Start

Run: PYTHONPATH=python python examples/viz_quickstart.py

Demonstrates:
- plot(instrument, curve) — auto-detects type, shows 2x2 dashboard
- PlotBuilder — fluent API for custom panels
"""

import sys
sys.path.insert(0, "python")

import matplotlib
matplotlib.use("Agg")  # use "TkAgg" or remove this line for interactive display

from datetime import date, timedelta
from pricebook.bootstrap import bootstrap
from pricebook.bond import FixedRateBond
from pricebook.treasury_lock import TreasuryLock
from pricebook.cmasw import CMASWInstrument
from pricebook.cmt import CMTInstrument
from pricebook.index_linked_hybrid import IndexLinkedHybridInstrument
from pricebook.viz import plot, PlotBuilder

# --- Setup ---
REF = date(2026, 4, 26)
deposits = [(REF + timedelta(days=91), 0.04), (REF + timedelta(days=182), 0.039)]
swaps = [
    (REF + timedelta(days=365), 0.038),
    (REF + timedelta(days=1825), 0.035),
    (REF + timedelta(days=3650), 0.034),
]
curve = bootstrap(REF, deposits, swaps)

# --- T-Lock ---
bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
tlock = TreasuryLock(bond, locked_yield=0.03, expiry=REF + timedelta(days=182),
                     repo_rate=0.02, notional=10_000_000)

fig = plot(tlock, curve)
fig.savefig("examples/tlock_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: examples/tlock_dashboard.png")

# Builder example
fig2 = PlotBuilder(tlock, curve).payoff().greeks().figure()
fig2.savefig("examples/tlock_builder.png", dpi=150, bbox_inches="tight")
print("Saved: examples/tlock_builder.png")

# --- CMASW ---
cmasw = CMASWInstrument(
    REF + timedelta(days=1825), REF + timedelta(days=2007),
    swap_tenor=5, bond_price=0.95, sigma_swp=0.30, sigma_asw=0.25, rho=0.5)
fig = plot(cmasw, curve)
fig.savefig("examples/cmasw_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: examples/cmasw_dashboard.png")

# --- CMT ---
cmt = CMTInstrument(
    REF + timedelta(days=1825), REF + timedelta(days=2190),
    bond_tenor=10, sigma=0.20, hazard_rate=0.01)
fig = plot(cmt, curve)
fig.savefig("examples/cmt_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: examples/cmt_dashboard.png")

# --- Index-Linked Hybrid ---
hybrid = IndexLinkedHybridInstrument(
    REF + timedelta(days=1825), swap_tenor=5, index_forward=0.04,
    sigma_F=0.30, sigma_U=0.25, rho=0.3, n_paths=10_000, n_steps=30)
fig = plot(hybrid, curve)
fig.savefig("examples/hybrid_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: examples/hybrid_dashboard.png")

print("\nAll dashboards generated successfully.")
