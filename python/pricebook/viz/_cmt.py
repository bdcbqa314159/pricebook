"""CMT visualisation panels."""

from __future__ import annotations

import math

import numpy as np

from pricebook.viz._dispatch import register_instrument, register_panels


def plot_cc_vs_vol(ax, instrument, curve, *, theme=None, **kwargs):
    """CC vs volatility for all three variants (A/B/C)."""
    from pricebook.cmt import cmt_convexity_corrections
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta

    Ts = year_fraction(curve.reference_date, instrument.fixing_date,
                        DayCountConvention.ACT_365_FIXED)
    n = instrument.bond_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n
    dates = [instrument.fixing_date + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    rf_dfs = [curve.df(d) for d in dates]
    rf_df_Ts = curve.df(instrument.fixing_date)
    rf_df_Tp = curve.df(instrument.payment_date)

    from pricebook.credit_adjustment import risky_annuity, risky_swap_rate
    times = [Ts + dt * (i+1) for i in range(n)]
    cra_dfs = [d * math.exp(-instrument.hazard_rate * t) for d, t in zip(rf_dfs, times)]
    cra_Ts = rf_df_Ts * math.exp(-instrument.hazard_rate * Ts)
    R_cmt = risky_swap_rate(cra_Ts, cra_dfs[-1], risky_annuity(yfs, cra_dfs))

    sigmas = np.linspace(0.01, 0.50, 50)
    cc_a, cc_c = [], []
    for sig in sigmas:
        r = cmt_convexity_corrections(R_cmt, sig, instrument.hazard_rate, Ts,
                                       yfs, rf_dfs, rf_df_Ts, rf_df_Tp)
        cc_a.append(r.cc_A * 10000)
        cc_c.append(r.cc_C * 10000)

    ax.plot(sigmas * 100, cc_a, lw=2, label="CC$^{(A)}$ = CC$^{(B)}$")
    ax.plot(sigmas * 100, cc_c, "--", lw=2, label="CC$^{(C)}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("CMT Vol (%)")
    ax.set_ylabel("CC (bp)")
    ax.set_title("CC vs Volatility")
    ax.legend(fontsize=9)


def plot_cc_vs_hazard(ax, instrument, curve, *, theme=None, **kwargs):
    """CC vs hazard rate."""
    from pricebook.cmt import cmt_convexity_corrections, cmt_cc_no_default
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta

    Ts = year_fraction(curve.reference_date, instrument.fixing_date,
                        DayCountConvention.ACT_365_FIXED)
    n = instrument.bond_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n
    dates = [instrument.fixing_date + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    rf_dfs = [curve.df(d) for d in dates]
    rf_df_Ts = curve.df(instrument.fixing_date)
    rf_df_Tp = curve.df(instrument.payment_date)

    from pricebook.credit_adjustment import risky_annuity, risky_swap_rate

    gammas = np.linspace(0.0, 0.05, 50)
    cc_a, cc_c = [], []
    for g in gammas:
        times = [Ts + dt * (i+1) for i in range(n)]
        cra_dfs = [d * math.exp(-g * t) for d, t in zip(rf_dfs, times)]
        cra_Ts = rf_df_Ts * math.exp(-g * Ts)
        R_g = risky_swap_rate(cra_Ts, cra_dfs[-1], risky_annuity(yfs, cra_dfs))
        r = cmt_convexity_corrections(R_g, instrument.sigma, g, Ts,
                                       yfs, rf_dfs, rf_df_Ts, rf_df_Tp)
        cc_a.append(r.cc_A * 10000)
        cc_c.append(r.cc_C * 10000)

    # Pelsser limit
    rf_ann = sum(y * d for y, d in zip(yfs, rf_dfs))
    alpha = 1.0 / sum(yfs)
    pelsser = cmt_cc_no_default(instrument.sigma, Ts, alpha, rf_ann, rf_df_Tp) * 10000

    ax.plot(gammas * 10000, cc_a, lw=2, label="CC$^{(A)}$")
    ax.plot(gammas * 10000, cc_c, "--", lw=2, label="CC$^{(C)}$")
    ax.axhline(pelsser, ls=":", color="gray", lw=1.5, label=f"Pelsser ($\\gamma$=0)")
    ax.set_xlabel("Hazard Rate (bp)")
    ax.set_ylabel("CC (bp)")
    ax.set_title("CC vs Hazard Rate")
    ax.legend(fontsize=9)


def plot_pelsser_overlay(ax, instrument, curve, *, theme=None, **kwargs):
    """No-default limit: all CCs collapse to Pelsser/Hagan."""
    from pricebook.cmt import cmt_convexity_corrections, cmt_cc_no_default
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta
    from pricebook.credit_adjustment import risky_swap_rate, risky_annuity

    Ts = year_fraction(curve.reference_date, instrument.fixing_date,
                        DayCountConvention.ACT_365_FIXED)
    n = instrument.bond_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n
    dates = [instrument.fixing_date + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    rf_dfs = [curve.df(d) for d in dates]
    rf_df_Ts = curve.df(instrument.fixing_date)
    rf_df_Tp = curve.df(instrument.payment_date)
    rf_ann = sum(y * d for y, d in zip(yfs, rf_dfs))
    alpha = 1.0 / sum(yfs)
    R_rf = risky_swap_rate(rf_df_Ts, rf_dfs[-1], rf_ann)

    sigmas = np.linspace(0.05, 0.50, 30)
    cc_a, pelsser = [], []
    for sig in sigmas:
        r = cmt_convexity_corrections(R_rf, sig, 0.0, Ts, yfs, rf_dfs, rf_df_Ts, rf_df_Tp)
        cc_a.append(r.cc_A * 10000)
        pelsser.append(cmt_cc_no_default(sig, Ts, alpha, rf_ann, rf_df_Tp) * 10000)

    ax.plot(sigmas * 100, cc_a, lw=2, label="CC$^{(A)}$ ($\\gamma$=0)")
    ax.plot(sigmas * 100, pelsser, ":", lw=3, color="black", label="Pelsser/Hagan")
    ax.set_xlabel("Vol (%)")
    ax.set_ylabel("CC (bp)")
    ax.set_title("No-Default Limit Check")
    ax.legend(fontsize=9)


def _default_dashboard(instrument, curve, *, figsize=None, theme=None, **kwargs):
    from pricebook.viz._backend import create_figure
    from pricebook.viz._generic import plot_summary_table
    fig, axes = create_figure(4, figsize)
    plot_cc_vs_vol(axes[0], instrument, curve, theme=theme)
    plot_cc_vs_hazard(axes[1], instrument, curve, theme=theme)
    plot_pelsser_overlay(axes[2], instrument, curve, theme=theme)
    plot_summary_table(axes[3], instrument, curve, theme=theme)
    fig.tight_layout()
    return fig


def _register():
    from pricebook.cmt import CMTInstrument
    register_instrument(CMTInstrument)(_default_dashboard)
    register_panels(CMTInstrument, {
        "payoff": plot_cc_vs_vol,
        "heatmap": plot_cc_vs_hazard,
        "comparison": plot_pelsser_overlay,
    })

_register()
