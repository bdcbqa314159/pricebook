"""TRS visualisation panels."""

from __future__ import annotations

import math

import numpy as np

from pricebook.viz._dispatch import register_instrument, register_panels


def plot_value_vs_spot(ax, instrument, curve, *, theme=None, **kwargs):
    """TRS value vs spot — pre-crisis vs repo-adjusted."""
    from pricebook.trs_lou import trs_equity_full_csa, trs_precrisis
    from datetime import timedelta

    S0 = instrument.spot
    mat_date = curve.reference_date + timedelta(days=int(instrument.maturity * 365))
    D = curve.df(mat_date)

    spots = np.linspace(S0 * 0.7, S0 * 1.3, 100)

    v_precrisis = [(S0 * instrument.funding_rate * instrument.maturity + S0) * D - s
                   for s in spots]
    v_repo = [trs_equity_full_csa(s, S0, instrument.funding_rate, instrument.maturity,
               0.0, D, rs_minus_r=instrument.repo_spread).value for s in spots]

    ax.plot(spots, v_precrisis, lw=2, label="Pre-crisis ($r_s = r$)")
    ax.plot(spots, v_repo, "--", lw=2, label=f"Repo-adjusted ($r_s-r$={instrument.repo_spread*10000:.0f}bp)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(S0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Spot")
    ax.set_ylabel("TRS Value (payer)")
    ax.set_title("Value vs Spot")
    ax.legend(fontsize=9)


def plot_fva_vs_repo(ax, instrument, curve, *, theme=None, **kwargs):
    """FVA as function of repo spread."""
    from pricebook.trs_lou import trs_fva

    spreads = np.linspace(0, 0.10, 100)
    fvas = [trs_fva(instrument.spot, rs, instrument.maturity) for rs in spreads]

    ax.plot(spreads * 10000, fvas, lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(instrument.repo_spread * 10000, ls=":", color="gray", alpha=0.5,
               label=f"Current = {instrument.repo_spread*10000:.0f}bp")
    ax.set_xlabel("Repo-OIS Spread (bp)")
    ax.set_ylabel("FVA ($)")
    ax.set_title("Hedge Financing Cost (Eq 8)")
    ax.legend(fontsize=9)


def plot_tree_convergence(ax, instrument, curve, *, theme=None, **kwargs):
    """Tree convergence to analytic."""
    from pricebook.trs_lou import trs_equity_full_csa
    from pricebook.trs_tree import trs_trinomial_tree
    from datetime import timedelta

    mat_date = curve.reference_date + timedelta(days=int(instrument.maturity * 365))
    D = curve.df(mat_date)
    r = -math.log(D) / instrument.maturity

    analytic = trs_equity_full_csa(
        instrument.spot, instrument.spot, instrument.funding_rate,
        instrument.maturity, 0.0, D, rs_minus_r=instrument.repo_spread)

    steps = [10, 20, 50, 100, 200]
    tree_vals = []
    for n in steps:
        tree = trs_trinomial_tree(
            instrument.spot, instrument.funding_rate, instrument.maturity,
            r, instrument.repo_spread, max(instrument.sigma, 0.01),
            n_steps=n, mu=1.0)
        tree_vals.append(tree.value)

    ax.plot(steps, tree_vals, "o-", lw=2, ms=5, label="Tree")
    ax.axhline(analytic.value, ls="--", color="red", lw=2, label=f"Analytic = {analytic.value:.4f}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Value")
    ax.set_title("Tree Convergence")
    ax.legend(fontsize=9)


def _default_dashboard(instrument, curve, *, figsize=None, theme=None, **kwargs):
    from pricebook.viz._backend import create_figure
    from pricebook.viz._generic import plot_summary_table
    fig, axes = create_figure(4, figsize)
    plot_value_vs_spot(axes[0], instrument, curve, theme=theme)
    plot_fva_vs_repo(axes[1], instrument, curve, theme=theme)
    plot_tree_convergence(axes[2], instrument, curve, theme=theme)
    plot_summary_table(axes[3], instrument, curve, theme=theme)
    fig.tight_layout()
    return fig


def _register():
    from pricebook.trs_lou import TotalReturnSwapLou
    register_instrument(TotalReturnSwapLou)(_default_dashboard)
    register_panels(TotalReturnSwapLou, {
        "payoff": plot_value_vs_spot,
        "greeks": plot_fva_vs_repo,
        "comparison": plot_tree_convergence,
    })

_register()
