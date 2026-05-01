"""Treasury Lock visualisation panels."""

from __future__ import annotations

import numpy as np

from pricebook.viz._dispatch import register_instrument, register_panels


def plot_pv_vs_yield(ax, instrument, curve, *, theme=None, **kwargs):
    """T-Lock PV as function of OIS rate (yield proxy)."""
    from pricebook.discount_curve import DiscountCurve

    ref = curve.reference_date
    base_rate = -np.log(curve.df(instrument.expiry)) / (
        (instrument.expiry - ref).days / 365.0
    )

    rates = np.linspace(base_rate - 0.02, base_rate + 0.02, 80)
    pvs = []
    for r in rates:
        c = DiscountCurve.flat(ref, r)
        pvs.append(instrument.price(c).value / 1000)

    ax.plot(rates * 100, pvs, lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(base_rate * 100, ls=":", color="gray", alpha=0.5)
    ax.axvline(instrument.locked_yield * 100, ls="--", color="red", alpha=0.7,
               label=f"Locked = {instrument.locked_yield*100:.2f}%")
    ax.set_xlabel("OIS Rate (%)")
    ax.set_ylabel("PV ($K)")
    ax.set_title("PV vs Yield Level")
    ax.legend(fontsize=9)


def plot_greeks_vs_yield(ax, instrument, curve, *, theme=None, **kwargs):
    """Delta and gamma across yield levels."""
    from pricebook.discount_curve import DiscountCurve

    ref = curve.reference_date
    base_rate = -np.log(curve.df(instrument.expiry)) / (
        (instrument.expiry - ref).days / 365.0
    )

    rates = np.linspace(base_rate - 0.015, base_rate + 0.015, 60)
    deltas = []
    for r in rates:
        c = DiscountCurve.flat(ref, r)
        g = instrument.greeks(c)
        deltas.append(g["delta"])

    ax.plot(rates * 100, deltas, lw=2, color="darkorange")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(instrument.locked_yield * 100, ls="--", color="red", alpha=0.7)
    ax.set_xlabel("OIS Rate (%)")
    ax.set_ylabel("Delta (Pucci)")
    ax.set_title("Delta vs Yield")


def plot_repo_sensitivity(ax, instrument, curve, *, theme=None, **kwargs):
    """PV as function of repo rate."""
    repo_rates = np.linspace(0.0, 0.06, 50)
    pvs = []
    old_repo = instrument.repo_rate
    for rr in repo_rates:
        instrument.repo_rate = rr
        pvs.append(instrument.price(curve).value / 1000)
    instrument.repo_rate = old_repo

    ax.plot(repo_rates * 100, pvs, lw=2, color="seagreen")
    ax.axvline(old_repo * 100, ls="--", color="red", alpha=0.7,
               label=f"Current = {old_repo*100:.1f}%")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Repo Rate (%)")
    ax.set_ylabel("PV ($K)")
    ax.set_title("PV vs Repo Rate")
    ax.legend(fontsize=9)


def _default_dashboard(instrument, curve, *, figsize=None, theme=None, **kwargs):
    from pricebook.viz._backend import create_figure
    from pricebook.viz._generic import plot_summary_table
    fig, axes = create_figure(4, figsize)
    plot_pv_vs_yield(axes[0], instrument, curve, theme=theme)
    plot_greeks_vs_yield(axes[1], instrument, curve, theme=theme)
    plot_repo_sensitivity(axes[2], instrument, curve, theme=theme)
    plot_summary_table(axes[3], instrument, curve, theme=theme)
    fig.tight_layout()
    return fig


def _register():
    from pricebook.treasury_lock import TreasuryLock

    register_instrument(TreasuryLock)(_default_dashboard)
    register_panels(TreasuryLock, {
        "payoff": plot_pv_vs_yield,
        "greeks": plot_greeks_vs_yield,
        "comparison": plot_repo_sensitivity,
    })

_register()
