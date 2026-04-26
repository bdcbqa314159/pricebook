"""T-Lock visualisation panels."""

from __future__ import annotations

import numpy as np

from pricebook.viz._dispatch import register_instrument, register_panels


def plot_payoff(ax, instrument, curve, *, theme=None, **kwargs):
    """T-Lock payoff vs yield with overhedge comparison."""
    from pricebook.bond_yield import bond_price_from_yield, bond_risk_factor

    alphas, times, T_mat = instrument.bond.accrual_schedule(instrument.expiry)
    L = instrument.locked_yield
    c = instrument.bond.coupon_rate

    yields = np.linspace(0.001, max(L * 3, 0.08), 200)
    P_L = bond_price_from_yield(c, alphas, L)

    exact = [bond_risk_factor(c, alphas, y) * (y - L) for y in yields]
    proxy = [P_L - bond_price_from_yield(c, alphas, y) for y in yields]

    ax.plot(yields * 100, exact, lw=2, label="Exact T-Lock payoff")
    ax.plot(yields * 100, proxy, "--", lw=2, label="Forward proxy (overhedge)")
    ax.axvline(L * 100, ls=":", color="gray", alpha=0.5, label=f"L = {L*100:.1f}%")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Yield (%)")
    ax.set_ylabel("Payoff")
    ax.set_title("Payoff vs Forward Proxy")
    ax.legend(fontsize=9)


def plot_greeks(ax, instrument, curve, *, theme=None, **kwargs):
    """Delta and gamma profiles vs yield."""
    from pricebook.treasury_lock import tlock_delta, tlock_gamma

    alphas, times, T_mat = instrument.bond.accrual_schedule(instrument.expiry)
    L = instrument.locked_yield
    c = instrument.bond.coupon_rate

    yields = np.linspace(0.005, max(L * 3, 0.08), 200)
    deltas = [tlock_delta(c, alphas, times, T_mat, y, L, 1) for y in yields]
    gammas = [tlock_gamma(c, alphas, times, T_mat, y, L, 1) for y in yields]

    ax.plot(yields * 100, deltas, lw=2, label="Delta")
    ax.plot(yields * 100, gammas, "--", lw=2, label="Gamma")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(L * 100, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Yield (%)")
    ax.set_ylabel("Greek value")
    ax.set_title("Delta & Gamma Profiles")
    ax.legend(fontsize=9)


def plot_repo_sensitivity(ax, instrument, curve, *, theme=None, **kwargs):
    """PV sensitivity to repo rate."""
    import math
    from pricebook.bond_yield import bond_price_from_yield
    from pricebook.bond_forward import forward_price_repo

    alphas, _, _ = instrument.bond.accrual_schedule(instrument.expiry)
    from pricebook.day_count import year_fraction, DayCountConvention
    tau = year_fraction(curve.reference_date, instrument.expiry, DayCountConvention.ACT_365_FIXED)
    mkt = instrument.bond.dirty_price(curve) / 100.0
    K = bond_price_from_yield(instrument.bond.coupon_rate, alphas, instrument.locked_yield)

    repos = np.linspace(-0.01, 0.06, 100)
    pvs = []
    for r in repos:
        fwd = forward_price_repo(mkt, r, tau, instrument.bond.coupon_rate, [], [])
        pvs.append(math.exp(-0.03 * tau) * (K - fwd))

    ax.plot(repos * 100, pvs, lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(instrument.repo_rate * 100, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Repo Rate (%)")
    ax.set_ylabel("Long T-Lock PV")
    ax.set_title("PV vs Repo Rate")


def plot_roll_surface(ax, instrument, curve, *, theme=None,
                      dc_range=(-0.01, 0.01), dR_range=(-0.005, 0.005),
                      n_grid=40, **kwargs):
    """Roll P&L contour surface."""
    from pricebook.treasury_lock import roll_pnl

    alphas, times, T_mat = instrument.bond.accrual_schedule(instrument.expiry)
    L = instrument.locked_yield
    c = instrument.bond.coupon_rate

    dc = np.linspace(*dc_range, n_grid)
    dR = np.linspace(*dR_range, n_grid)
    DC, DR = np.meshgrid(dc, dR)
    Z = np.zeros_like(DC)

    R_base = L + 0.005
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = roll_pnl(c, c + DC[i, j], R_base, R_base + DR[i, j],
                                L, alphas, times, T_mat)

    cf = ax.contourf(DC * 100, DR * 100, Z * 100, levels=15, cmap="RdBu_r")
    ax.contour(DC * 100, DR * 100, Z * 100, levels=[0], colors="k", linewidths=1.5)
    ax.set_xlabel("Coupon change (%)")
    ax.set_ylabel("Yield change (%)")
    ax.set_title("Roll P&L Surface")
    ax.plot(0, 0, "ko", ms=6)


def _default_dashboard(instrument, curve, *, figsize=None, theme=None, **kwargs):
    """Default 2x2 T-Lock dashboard."""
    from pricebook.viz._backend import create_figure
    fig, axes = create_figure(4, figsize)
    plot_payoff(axes[0], instrument, curve, theme=theme)
    plot_greeks(axes[1], instrument, curve, theme=theme)
    plot_repo_sensitivity(axes[2], instrument, curve, theme=theme)
    plot_roll_surface(axes[3], instrument, curve, theme=theme)
    fig.tight_layout()
    return fig


def _register():
    from pricebook.treasury_lock import TreasuryLock
    register_instrument(TreasuryLock)(_default_dashboard)
    register_panels(TreasuryLock, {
        "payoff": plot_payoff,
        "greeks": plot_greeks,
        "repo_sensitivity": plot_repo_sensitivity,
        "roll_surface": plot_roll_surface,
    })

_register()
