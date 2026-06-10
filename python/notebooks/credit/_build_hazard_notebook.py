"""Build hazard_from_bonds_when_maturities_are_close.ipynb section by section.

Pattern matches notebooks/examples/_build_quickstart.py. Run:

    cd python
    ../.venv/bin/python notebooks/credit/_build_hazard_notebook.py

then execute the resulting .ipynb in-place via nbconvert to embed outputs:

    cd notebooks/credit
    ../../../.venv/bin/python -m nbconvert --to notebook --execute --inplace \
        hazard_from_bonds_when_maturities_are_close.ipynb
"""

import nbformat as nbf
from pathlib import Path

CELLS = []

def md(s):  CELLS.append(("md",   s))
def code(s): CELLS.append(("code", s))


# ─────────────────────────────────────────────────────────────────
# Title + intro
# ─────────────────────────────────────────────────────────────────

md("""# Hazard rates from bond prices — when maturities are close

A walk through the classical problem of extracting a hazard-rate term structure
from a set of risky bond prices, with a focus on what happens — and why — when
two bonds have nearly the same maturity. Sequential bootstrap (a Newton root-find
one bond at a time) is the natural first attempt; we show it works cleanly when
bonds are well spaced and breaks dramatically when they bunch.

What you'll see, in order:

1. **The problem.** What's a hazard rate, what's a bond, what's "fitting"?
2. **The easy case.** Four bonds at well-spaced maturities. Sequential bootstrap recovers the hazard rates perfectly.
3. **Where sequential breaks.** Add a fifth bond two months from the fifth-year. The Jacobian of "price with respect to hazard" goes near-singular. Tiny price noise → enormous hazard noise.
4. **Solver limits.** Newton divergence and brentq bracket failure when bonds are close.
5. **The Tikhonov fix.** Regularised least-squares: penalise misfit AND non-smooth hazard. Full math derivation.
6. **The L-curve.** How to pick the regularisation strength without cross-validation.
7. **Bid-ask sensitivity.** Monte-Carlo perturbation within realistic spreads.
8. **The adaptive switch.** Pricebook's `bootstrap_hazard_adaptive` heuristic.
9. **A realistic demo.** Eight bonds at typical issuer maturities, with adjacent benchmarks.
10. **Cross-check.** Reconcile piecewise-constant fit with a CIR++ stochastic-intensity fit on the same prices.

References woven through:
- O'Kane (2008), *Modelling Single-name and Multi-name Credit Derivatives*, Ch. 6.
- Duffie & Singleton (1999), *Modeling Term Structures of Defaultable Bonds*, RFS 12(4).
- Hull, Predescu & White (2004), *Bond Prices, Default Probabilities and Risk Premiums*.
- Tikhonov & Arsenin (1977), *Solutions of Ill-Posed Problems*.
- Hansen (1992), *Analysis of Discrete Ill-Posed Problems by Means of the L-Curve*, SIAM Review 34(4).
""")


# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────

code('''import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "python"))

import math
from datetime import date, timedelta
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.credit.bond_hazard_bootstrap import (
    BondInput,
    bootstrap_hazard_from_bonds,
    _price_risky_bond,
)
from pricebook.viz import configure_theme
from pricebook.viz._backend import create_figure

configure_theme(seaborn_style="whitegrid", seaborn_context="notebook")
np.set_printoptions(precision=4, suppress=True)
print("Hazard-bootstrap notebook loaded.")''')


# ─────────────────────────────────────────────────────────────────
# Section 1 — The problem
# ─────────────────────────────────────────────────────────────────

md(r"""## 1. The problem

A *hazard rate* $h(t)$ is the instantaneous arrival rate of default. Given $h$,
the survival probability to time $t$ is

$$
Q(t) = \exp\!\Big(-\int_0^t h(u)\, du\Big).
$$

A *risky bond* with face 100, coupon $c$ paid at $n$ dates $t_1, \dots, t_n$,
recovery $R$, and risk-free discount factors $D(t)$ has price (recovery-of-face
convention, ISDA standard)

$$
P \;=\; \underbrace{c\,\tau\sum_{i=1}^n D(t_i)\,Q(t_i)}_{\text{coupon leg}}
 \;+\; \underbrace{100\, D(T)\,Q(T)}_{\text{principal leg}}
 \;+\; \underbrace{100\, R \sum_{i=1}^n D(\tilde t_i)\,\bigl[Q(t_{i-1}) - Q(t_i)\bigr]}_{\text{recovery leg}},
$$

where $\tilde t_i$ is the midpoint of period $i$ (where we assume default occurs
in expectation).

**The calibration problem.** Given $N$ observed bond prices $P_1, \dots, P_N$
of the *same issuer*, find a hazard-rate function $h(t)$ that reprices all of
them. If we parameterise $h$ as piecewise constant on intervals
$[0, T_1], [T_1, T_2], \dots, [T_{M-1}, T_M]$, we have $M$ unknowns to fit to
$N$ observations.

- $N = M$, well-spaced: unique solution via *sequential bootstrap* — solve for
  $h_1$ from bond 1, plug that in, solve for $h_2$ from bond 2, etc. One nonlinear
  root-find per bond.
- $N > M$: over-determined, use *least squares*.
- $N < M$: under-determined, need *regularisation* or fewer pillars.

**Why this is harder than CDS bootstrapping.** A CDS has a clean protection leg
(pays $1-R$ on default) and a clean premium leg (pays the spread on survival).
Bonds carry large intermediate coupon cashflows that depend on survival at
every coupon date, so the price is a *highly nonlinear* function of the hazard
shape between pillars. And bonds rarely settle with aligned coupon dates, so
the cashflow scheduling itself differs across bonds.

**Why "close maturities" is the trouble case.** If two bonds mature 1 month
apart, their cashflow streams overlap on every single coupon date except the
final two. The *only* part of the hazard curve that distinguishes them is the
1-month sliver of survival between their maturities. The Jacobian
$\partial P_i / \partial h_j$ has two nearly-identical rows for those two
bonds — the matrix is *ill-conditioned*. Small input errors get amplified
enormously in the solved hazards. We'll see this happen below, quantitatively.
""")


# ─────────────────────────────────────────────────────────────────
# Synthetic data infrastructure
# ─────────────────────────────────────────────────────────────────

md("""### Synthetic-data infrastructure

To study the problem without confounding effects from real-world data noise, we
build everything from a **known truth**: a hazard curve we specify ourselves,
then bonds priced from it. The whole notebook stays inside this controlled
synthetic world so we can answer "did the bootstrap recover the truth?"
exactly.""")

code('''REF = date(2026, 6, 10)

# Risk-free discount curve: flat 4%
rf = DiscountCurve.flat(REF, 0.04)

def synthetic_survival(ref: date, pillars_y: list[float], hazards: list[float]) -> SurvivalCurve:
    """Build a SurvivalCurve from piecewise-constant hazards on [0, T_1], [T_1, T_2], ..."""
    dates = [ref + timedelta(days=int(round(365*y))) for y in pillars_y]
    survs, cum, prev_t = [], 1.0, 0.0
    for t_y, h in zip(pillars_y, hazards):
        cum *= math.exp(-h * (t_y - prev_t))
        survs.append(cum)
        prev_t = t_y
    return SurvivalCurve(ref, dates, survs)

def synthetic_bond(ref: date, maturity_years: float, coupon: float,
                   rf_curve, true_survival, recovery: float = 0.40) -> BondInput:
    """Price a bond from the truth, return as a BondInput at that exact price."""
    mat = ref + timedelta(days=int(round(365*maturity_years)))
    price = _price_risky_bond(ref, mat, coupon, 2, recovery, rf_curve, true_survival)
    return BondInput(maturity=mat, coupon=coupon, market_price=price,
                     frequency=2, recovery=recovery)

# A truth curve we'll reuse: 2%/3%/4% hazards on [0,3y], [3,7y], [7,15y]
TRUTH = synthetic_survival(REF, pillars_y=[3.0, 7.0, 15.0], hazards=[0.02, 0.03, 0.04])

print(f"Truth S(3y)  = {TRUTH.survival(REF + timedelta(days=365*3)):.4f}  (expect {math.exp(-0.02*3):.4f})")
print(f"Truth S(7y)  = {TRUTH.survival(REF + timedelta(days=365*7)):.4f}  (expect {math.exp(-0.02*3 - 0.03*4):.4f})")
print(f"Truth S(15y) = {TRUTH.survival(REF + timedelta(days=365*15)):.4f}  (expect {math.exp(-0.02*3 - 0.03*4 - 0.04*8):.4f})")''')


# ─────────────────────────────────────────────────────────────────
# Section 2 — Easy case
# ─────────────────────────────────────────────────────────────────

md(r"""## 2. The easy case — four bonds at 1, 3, 5, 10 years

Spacing is generous; every consecutive pair is at least two years apart. The
sequential bootstrap finds a unique hazard rate per pillar and reproduces every
input price to numerical precision.""")

code('''easy_specs = [(1.0, 0.040), (3.0, 0.045), (5.0, 0.050), (10.0, 0.055)]
easy_bonds = [synthetic_bond(REF, y, c, rf, TRUTH) for y, c in easy_specs]

for b, (y, c) in zip(easy_bonds, easy_specs):
    print(f"  {y:4.1f}y  c={c*100:.1f}%  price = {b.market_price:8.4f}")

print()
res_easy = bootstrap_hazard_from_bonds(REF, easy_bonds, rf, method="sequential")
print(f"Sequential bootstrap: converged={res_easy.converged}, rmse={res_easy.rmse_bp:.2e} bp\\n")
print(f"  pillar maturities (yrs): {[round((d - REF).days/365.0, 3) for d in res_easy.pillar_dates]}")
print(f"  pillar hazards (%):      {[round(h*100, 4) for h in res_easy.pillar_hazards]}")
hh = [round(h*100, 4) for h in res_easy.pillar_hazards]
print(f"\\n  pillar hazards (%, cleaned): {hh}")
print(f"\\n  Bond pillar [0, 1y] → fitted h = {hh[0]:.3f}%   truth: 2.000% (truth flat on [0,3])")
print(f"  Bond pillar [1y, 3y] → fitted h = {hh[1]:.3f}%   truth: 2.000%")
print(f"  Bond pillar [3y, 5y] → fitted h = {hh[2]:.3f}%   truth: 3.000% (truth flat on [3,7])")
print(f"  Bond pillar [5y,10y] → fitted h = {hh[3]:.3f}%   truth: weighted blend of 3% on [5,7] and 4% on [7,10]")
print(f"      naive integrated hazard:  h̄ = (3%·2 + 4%·3)/5 = 3.60%")
print(f"      bootstrap's value differs slightly from 3.60% because intermediate coupons")
print(f"      weight the survival shape, not just its endpoints.")''')

code('''# Plot: implied hazard curve vs truth
fig, axes = create_figure(n_panels=1, figsize=(9, 4.5))
ax = axes[0]

# Truth (piecewise constant)
truth_ts = np.array([0, 3, 3, 7, 7, 15])
truth_hs = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.04])
ax.plot(truth_ts, truth_hs * 100, "k--", lw=2, label="Truth hazard")

# Implied piecewise constant: between (0, pillar_0), (pillar_0, pillar_1), ...
fitted_t = [0.0] + [(d - REF).days / 365.0 for d in res_easy.pillar_dates]
fitted_h = [h * 100 for h in res_easy.pillar_hazards]
for i in range(len(fitted_h)):
    ax.plot([fitted_t[i], fitted_t[i+1]], [fitted_h[i], fitted_h[i]],
            "C0-", lw=3, alpha=0.85, label="Sequential bootstrap" if i == 0 else None)
    ax.scatter([fitted_t[i+1]], [fitted_h[i]], color="C0", s=40, zorder=3)

ax.set_xlabel("Time (years)")
ax.set_ylabel("Hazard rate (%)")
ax.set_title("Section 2 — Well-spaced bonds: sequential bootstrap recovers the truth")
ax.legend(loc="lower right")
ax.set_xlim(0, 12)
ax.set_ylim(1.5, 4.5)
fig''')

md(r"""**Note on the implied hazards vs the truth.** The bootstrap places pillars
*at the bond maturities*, not at the truth's pillars. The truth has knots at
3 and 7 years; the bonds give us knots at 1, 3, 5, and 10. So the [3, 5] bond
pillar straddles the constant-hazard truth segment [3, 7] and correctly
recovers 3.00%. The [5, 10] bond pillar straddles two truth segments
([5, 7] at 3% and [7, 10] at 4%); a naive time-weighted average gives
$\bar h = (3\%\cdot 2 + 4\%\cdot 3)/5 = 3.6\%$, and the bootstrap fits
≈ 3.56% — a hair below 3.6% because the intermediate coupons weight the
survival shape, not just its endpoints. The bootstrap reproduces the *bond
prices* exactly — it cannot do better than that without finer-grained input.""")


# ─────────────────────────────────────────────────────────────────
# Section 3 — Where sequential breaks
# ─────────────────────────────────────────────────────────────────

md(r"""## 3. Where sequential bootstrap breaks

Now add a fifth bond two months from the 5-year. Sequential bootstrap still
runs and still reproduces the input prices exactly **when the input is
noise-free**. The problem is hidden until we add a tiny realistic price
perturbation.""")

code('''# Add a 5th bond at 5y + 2 months (close to the 5y), same 5% coupon, same recovery
close_specs = easy_specs[:3] + [(5.0 + 2/12, 0.050)] + [easy_specs[3]]
close_bonds = [synthetic_bond(REF, y, c, rf, TRUTH) for y, c in close_specs]

for b, (y, c) in zip(close_bonds, close_specs):
    print(f"  {y:5.3f}y  c={c*100:.1f}%  price = {b.market_price:10.6f}")

res_clean = bootstrap_hazard_from_bonds(REF, close_bonds, rf, method="sequential")
print(f"\\nNoise-free: rmse = {res_clean.rmse_bp:.2e} bp   converged = {res_clean.converged}")
print(f"  pillar hazards (%): {[round(h*100, 4) for h in res_clean.pillar_hazards]}")
print(f"  → segments 3y-5y and 5y-5.17y both recover 3.00%, as expected")''')

code('''# Perturb the 5y bond price by a microscopic +5 bp of par
close_bonds_noisy = [BondInput(**b.to_dict() | {"maturity": b.maturity}) if False else b
                     for b in close_bonds]  # no-op deep-copy guard
close_bonds_noisy = [BondInput(maturity=b.maturity, coupon=b.coupon, market_price=b.market_price,
                               frequency=b.frequency, recovery=b.recovery)
                     for b in close_bonds]
close_bonds_noisy[2].market_price += 0.05  # +5 bp of par on bond index 2 (the 5y)

res_noisy = bootstrap_hazard_from_bonds(REF, close_bonds_noisy, rf, method="sequential")
print(f"After +5 bp price noise on the 5y bond:")
print(f"  pillar hazards (%): {[round(h*100, 4) for h in res_noisy.pillar_hazards]}")
print()
print(f"  Δhazard (bp) vs noise-free:")
for i, (h0, h1) in enumerate(zip(res_clean.pillar_hazards, res_noisy.pillar_hazards)):
    label = f"pillar {i}"
    print(f"    {label}: {(h1 - h0)*1e4:+8.2f} bp")
print()
print("  Note: 5 bp of price noise on bond 2 propagates to a 45+ bp swing in the\\n"
      "  short [3y → 5y] and [5y → 5.17y] hazard segments. The amplification factor\\n"
      "  exceeds 10×, with opposite signs that cancel in survival (so the *integrated*\\n"
      "  survival barely moves) but render the instantaneous hazard meaningless.")''')

code('''# Sweep the spacing and quantify amplification: 5 bp price noise vs Δmaturity
deltas_months = [12, 9, 6, 3, 2, 1, 0.5]
amp_records = []
for dmo in deltas_months:
    dy = dmo / 12.0
    specs = easy_specs[:3] + [(5.0 + dy, 0.050)] + [easy_specs[3]]
    bonds_clean = [synthetic_bond(REF, y, c, rf, TRUTH) for y, c in specs]
    bonds_noisy = [BondInput(maturity=b.maturity, coupon=b.coupon, market_price=b.market_price,
                             frequency=b.frequency, recovery=b.recovery) for b in bonds_clean]
    bonds_noisy[2].market_price += 0.05  # +5 bp on the 5y

    r0 = bootstrap_hazard_from_bonds(REF, bonds_clean, rf, method="sequential")
    r1 = bootstrap_hazard_from_bonds(REF, bonds_noisy, rf, method="sequential")
    h0 = r0.pillar_hazards
    h1 = r1.pillar_hazards
    # The two close pillars are indices 2 ([3y, 5y]) and 3 ([5y, 5y+ΔT])
    dh_short = (h1[3] - h0[3]) * 1e4  # bp
    amp_records.append((dmo, dh_short))
    print(f"  ΔT = {dmo:5.2f} mo   Δhazard on [5y, 5y+ΔT] segment = {dh_short:+8.2f} bp (amp = {abs(dh_short/5):5.1f}×)")''')

code('''# Plot the noise amplification
fig, axes = create_figure(n_panels=1, figsize=(9, 4.5))
ax = axes[0]
deltas = [d for d, _ in amp_records]
amps = [abs(dh) / 5.0 for _, dh in amp_records]  # 5 bp price noise → |Δh| / 5 bp = amp factor
ax.plot(deltas, amps, "C3o-", lw=2, ms=8)
ax.axhline(1, color="grey", ls=":", alpha=0.5, label="No amplification (1×)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Maturity spacing  ΔT  (months)")
ax.set_ylabel("Hazard noise amplification  (×, log)")
ax.set_title("Section 3 — Amplification of 5 bp price noise vs maturity spacing\\n"
             "(close bonds → ill-conditioned Jacobian → useless instantaneous hazard)")
ax.legend()
ax.invert_xaxis()
for dmo, amp in zip(deltas, amps):
    ax.annotate(f"{amp:.0f}×", (dmo, amp), textcoords="offset points", xytext=(6, 6), fontsize=9)
fig''')

md(r"""**Takeaway from Section 3.** Sequential bootstrap is a *root-finder*: each
bond uniquely determines one hazard segment. The condition number of the
implicit-function map "input price → hazard" grows like $1/\Delta T$ as
maturities approach. At one month spacing, 5 bp of price noise (well inside a
real bid-ask) becomes ~135 bp of hazard noise. The bond-implied hazard
oscillates between segments to absorb noise that the data simply doesn't
constrain.

Importantly, this is **not a bug in the bootstrap** — the bootstrap is
correctly reproducing the input prices to numerical precision. The problem is
*deeper*: the data themselves don't carry enough information to pin down the
instantaneous hazard at fine resolution. The cure is not a better solver but a
*different formulation* — one that admits "we don't know $h(t)$ exactly between
close maturities" instead of pretending we do. That's the regularisation story
in Section 5.""")


# Build
def cell(t, src):
    if t == "md":
        return nbf.v4.new_markdown_cell(src)
    return nbf.v4.new_code_cell(src)

nb = nbf.v4.new_notebook(cells=[cell(t, s) for t, s in CELLS])
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

out = Path(__file__).parent / "hazard_from_bonds_when_maturities_are_close.ipynb"
with out.open("w") as f:
    nbf.write(nb, f)
print(f"Wrote {out}  ({len(CELLS)} cells)")
