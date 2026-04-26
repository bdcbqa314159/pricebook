"""CMASW visualisation panels."""

from __future__ import annotations

import numpy as np

from pricebook.viz._dispatch import register_instrument, register_panels


def plot_cc_heatmap(ax, instrument, curve, *, theme=None,
                    sigma_range=(0.10, 0.50, 5), rho_range=(-0.9, 0.9, 5), **kwargs):
    """CC heatmap: sigma_asw vs rho."""
    from pricebook.cmasw import cmasw_cc_lognormal
    import math
    from pricebook.day_count import year_fraction, DayCountConvention

    T0 = year_fraction(curve.reference_date, instrument.fixing_date,
                        DayCountConvention.ACT_365_FIXED)
    n = instrument.swap_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n

    from datetime import timedelta
    dates = [instrument.fixing_date + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    dfs = [curve.df(d) for d in dates]
    annuity = sum(y * d for y, d in zip(yfs, dfs))
    df_tp = curve.df(instrument.payment_date)

    R_asw = instrument.bond_price  # proxy

    sigs = np.linspace(*sigma_range[:2], int(sigma_range[2]))
    rhos = np.linspace(*rho_range[:2], int(rho_range[2]))
    Z = np.zeros((len(sigs), len(rhos)))
    for i, s in enumerate(sigs):
        for j, r in enumerate(rhos):
            Z[i, j] = cmasw_cc_lognormal(R_asw * 0.05, annuity, df_tp, yfs,
                                          instrument.sigma_swp, s, r, T0) * 10000

    im = ax.imshow(Z, cmap="RdBu_r", aspect="auto",
                   extent=[rhos[0], rhos[-1], sigs[-1], sigs[0]])
    ax.set_xlabel("Correlation $\\rho$")
    ax.set_ylabel("$\\sigma_{asw}$")
    ax.set_title("Lognormal CC (bp)")
    ax.figure.colorbar(im, ax=ax, label="CC (bp)")


def plot_cc_vs_rho(ax, instrument, curve, *, theme=None, **kwargs):
    """CC vs rho line plot for multiple sigma_asw."""
    from pricebook.cmasw import cmasw_cc_lognormal
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta

    T0 = year_fraction(curve.reference_date, instrument.fixing_date,
                        DayCountConvention.ACT_365_FIXED)
    n = instrument.swap_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n
    dates = [instrument.fixing_date + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    dfs = [curve.df(d) for d in dates]
    annuity = sum(y * d for y, d in zip(yfs, dfs))
    df_tp = curve.df(instrument.payment_date)

    rhos = np.linspace(-0.95, 0.95, 80)
    for sig in [0.20, 0.30, 0.50]:
        ccs = [cmasw_cc_lognormal(0.049, annuity, df_tp, yfs,
               instrument.sigma_swp, sig, r, T0) * 10000 for r in rhos]
        ax.plot(rhos, ccs, lw=2, label=f"$\\sigma_{{asw}}$={sig*100:.0f}%")

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Correlation $\\rho$")
    ax.set_ylabel("CC (bp)")
    ax.set_title("CC vs Correlation")
    ax.legend(fontsize=9)


def plot_displaced_vs_lognormal(ax, instrument, curve, *, theme=None, **kwargs):
    """Displaced vs lognormal CC comparison."""
    from pricebook.cmasw import cmasw_cc_lognormal, cmasw_convexity_correction
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta

    T0 = year_fraction(curve.reference_date, instrument.fixing_date,
                        DayCountConvention.ACT_365_FIXED)
    n = instrument.swap_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n
    dates = [instrument.fixing_date + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    dfs = [curve.df(d) for d in dates]
    annuity = sum(y * d for y, d in zip(yfs, dfs))
    df_tp = curve.df(instrument.payment_date)
    R_swp = (curve.df(instrument.fixing_date) - dfs[-1]) / annuity if annuity > 0 else 0.04

    cc_ln = cmasw_cc_lognormal(0.049, annuity, df_tp, yfs,
                                instrument.sigma_swp, 0.30, instrument.rho, T0) * 10000

    a_range = np.linspace(-0.02, 0.02, 40)
    displaced = []
    for a in a_range:
        r = cmasw_convexity_correction(0.049, R_swp, annuity, df_tp, yfs, dfs,
                                        instrument.sigma_swp, 0.30, instrument.rho, T0,
                                        a_swp=0.0, a_asw=a)
        displaced.append(r.convexity_correction * 10000)

    ax.plot(a_range * 10000, displaced, lw=2, label="Displaced CC")
    ax.axhline(cc_ln, ls="--", lw=2, label=f"Lognormal = {cc_ln:.1f}bp")
    ax.axvline(0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("$a_{asw}$ (bp)")
    ax.set_ylabel("CC (bp)")
    ax.set_title("Displaced vs Lognormal")
    ax.legend(fontsize=9)


def _default_dashboard(instrument, curve, *, figsize=None, theme=None, **kwargs):
    from pricebook.viz._backend import create_figure
    from pricebook.viz._generic import plot_summary_table
    fig, axes = create_figure(4, figsize)
    plot_cc_heatmap(axes[0], instrument, curve, theme=theme)
    plot_cc_vs_rho(axes[1], instrument, curve, theme=theme)
    plot_displaced_vs_lognormal(axes[2], instrument, curve, theme=theme)
    plot_summary_table(axes[3], instrument, curve, theme=theme)
    fig.tight_layout()
    return fig


def _register():
    from pricebook.cmasw import CMASWInstrument
    register_instrument(CMASWInstrument)(_default_dashboard)
    register_panels(CMASWInstrument, {
        "payoff": plot_cc_vs_rho,
        "heatmap": plot_cc_heatmap,
        "comparison": plot_displaced_vs_lognormal,
    })

_register()
