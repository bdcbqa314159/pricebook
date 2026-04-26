"""Index-linked hybrid visualisation panels."""

from __future__ import annotations

import numpy as np

from pricebook.viz._dispatch import register_instrument, register_panels


def plot_price_vs_rho(ax, instrument, curve, *, theme=None, n_rhos=20, **kwargs):
    """Hybrid price vs correlation (payer + receiver)."""
    from pricebook.index_linked_hybrid import index_linked_hybrid_price
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta
    import math

    T = year_fraction(curve.reference_date, instrument.expiry,
                       DayCountConvention.ACT_365_FIXED)
    df = curve.df(instrument.expiry)
    n = instrument.swap_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n
    taus = [dt * (i+1) for i in range(n)]
    dates = [instrument.expiry + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    dfs = [curve.df(d) for d in dates]
    annuity = sum(y * d for y, d in zip(yfs, dfs))
    F0 = (curve.df(instrument.expiry) - dfs[-1]) / annuity if annuity > 0 else 0.04

    rhos = np.linspace(-0.95, 0.95, n_rhos)
    payer, receiver = [], []
    for rho in rhos:
        r = index_linked_hybrid_price(F0, instrument.index_forward, df, yfs, taus,
                                       instrument.sigma_F, instrument.sigma_U,
                                       rho, T, theta=1, n_paths=10_000, n_steps=20,
                                       seed=instrument.seed)
        payer.append(r.price * 10000)
        r2 = index_linked_hybrid_price(F0, instrument.index_forward, df, yfs, taus,
                                        instrument.sigma_F, instrument.sigma_U,
                                        rho, T, theta=-1, n_paths=10_000, n_steps=20,
                                        seed=instrument.seed)
        receiver.append(r2.price * 10000)

    ax.plot(rhos, payer, "-o", ms=3, lw=2, label="Payer ($\\theta$=+1)")
    ax.plot(rhos, receiver, "-s", ms=3, lw=2, label="Receiver ($\\theta$=-1)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Correlation $\\rho$")
    ax.set_ylabel("Price (bp)")
    ax.set_title("Hybrid Price vs Correlation")
    ax.legend(fontsize=9)


def plot_cash_annuity(ax, instrument, curve, *, theme=None, **kwargs):
    """Cash annuity Â(S) as function of swap rate."""
    from pricebook.cash_settlement import cash_annuity

    n = instrument.swap_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    yfs = [dt] * n
    taus = [dt * (i+1) for i in range(n)]

    rates = np.linspace(0.005, 0.10, 100)
    annuities = [cash_annuity(s, yfs, taus) for s in rates]

    ax.plot(rates * 100, annuities, lw=2)
    ax.set_xlabel("Swap Rate S (%)")
    ax.set_ylabel("Cash Annuity $\\hat{A}(S)$")
    ax.set_title("Cash Annuity Shape")
    ax.grid(alpha=0.3)


def plot_martingale(ax, instrument, curve, *, theme=None, **kwargs):
    """Martingale diagnostics: E[F_T]/F0 and E[U_T]/U0 across seeds."""
    from pricebook.hybrid_mc import simulate_2d_local_vol
    from pricebook.day_count import year_fraction, DayCountConvention
    from datetime import timedelta

    T = year_fraction(curve.reference_date, instrument.expiry,
                       DayCountConvention.ACT_365_FIXED)
    n = instrument.swap_tenor * instrument.frequency
    dt = 1.0 / instrument.frequency
    dates = [instrument.expiry + timedelta(days=int(dt * 365 * (i+1))) for i in range(n)]
    dfs = [curve.df(d) for d in dates]
    yfs = [dt] * n
    annuity = sum(y * d for y, d in zip(yfs, dfs))
    F0 = (curve.df(instrument.expiry) - dfs[-1]) / annuity if annuity > 0 else 0.04

    ratios_F, ratios_U = [], []
    for seed in range(42, 62):
        F_T, U_T = simulate_2d_local_vol(
            F0, instrument.index_forward, instrument.sigma_F, instrument.sigma_U,
            instrument.rho, T, n_paths=5_000, n_steps=20, seed=seed)
        ratios_F.append(float(F_T.mean()) / F0)
        ratios_U.append(float(U_T.mean()) / instrument.index_forward)

    seeds = list(range(42, 62))
    ax.plot(seeds, ratios_F, "-o", ms=4, lw=1.5, label="E[F_T]/F0")
    ax.plot(seeds, ratios_U, "-s", ms=4, lw=1.5, label="E[U_T]/U0")
    ax.axhline(1.0, ls=":", color="gray", lw=1.5, label="Martingale (=1)")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Ratio")
    ax.set_title("Martingale Check")
    ax.legend(fontsize=9)
    ax.set_ylim(0.95, 1.05)


def _default_dashboard(instrument, curve, *, figsize=None, theme=None, **kwargs):
    from pricebook.viz._backend import create_figure
    from pricebook.viz._generic import plot_summary_table
    fig, axes = create_figure(4, figsize)
    plot_price_vs_rho(axes[0], instrument, curve, theme=theme)
    plot_cash_annuity(axes[1], instrument, curve, theme=theme)
    plot_martingale(axes[2], instrument, curve, theme=theme)
    plot_summary_table(axes[3], instrument, curve, theme=theme)
    fig.tight_layout()
    return fig


def _register():
    from pricebook.index_linked_hybrid import IndexLinkedHybridInstrument
    register_instrument(IndexLinkedHybridInstrument)(_default_dashboard)
    register_panels(IndexLinkedHybridInstrument, {
        "payoff": plot_price_vs_rho,
        "heatmap": plot_cash_annuity,
        "martingale": plot_martingale,
    })

_register()
