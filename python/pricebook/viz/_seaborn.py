"""Seaborn-powered finance visualisations for pricebook.

Higher-level plots using seaborn for statistical and multi-dimensional data:

    from pricebook.viz import (
        correlation_heatmap, pnl_distribution, recovery_heatmap,
        greeks_profile, sensitivity_grid, exposure_profile,
    )

Heatmaps and distributions require seaborn. Greeks profile and exposure
profile work with pure matplotlib.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pricebook.viz._theme import get_theme, _has_seaborn


def _require_seaborn():
    if not _has_seaborn():
        raise ImportError("seaborn is required for this plot. Install: pip install seaborn")
    import seaborn as sns
    return sns


# ---------------------------------------------------------------------------
# 1. Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(
    data: dict[str, list[float]] | Any,
    title: str = "Correlation Matrix",
    figsize: tuple[float, float] = (8, 6),
    annot: bool = True,
    cmap: str = "RdBu_r",
    vmin: float = -1,
    vmax: float = 1,
):
    """Correlation heatmap from a dict of series or a DataFrame.

    Args:
        data: {name: values} dict or pandas DataFrame.
        title: plot title.
        annot: show correlation values in cells.
        cmap: colormap (RdBu_r = red-blue diverging, good for correlations).
    """
    sns = _require_seaborn()
    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data

    corr = df.corr()
    theme = get_theme()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax,
                center=0, square=True, linewidths=0.5, ax=ax,
                fmt=".2f", cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. P&L distribution
# ---------------------------------------------------------------------------

def pnl_distribution(
    pnl_values: list[float] | Any,
    title: str = "P&L Distribution",
    figsize: tuple[float, float] = (10, 5),
    bins: int = 50,
    kde: bool = True,
    var_quantile: float = 0.01,
):
    """P&L distribution with VaR/CVaR markers.

    Args:
        pnl_values: list or array of P&L values.
        var_quantile: quantile for VaR line (0.01 = 1%).
        kde: show kernel density estimate.
    """
    sns = _require_seaborn()
    import matplotlib.pyplot as plt

    values = np.array(pnl_values)
    if len(values) == 0:
        raise ValueError("pnl_values must not be empty")
    theme = get_theme()
    var = float(np.quantile(values, var_quantile))
    cvar = float(values[values <= var].mean()) if (values <= var).any() else var

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(values, bins=bins, kde=kde, ax=ax, color=theme.colors[0],
                 edgecolor='white', alpha=0.7)

    ax.axvline(var, color=theme.colors[1], linestyle='--', linewidth=2,
               label=f"VaR {var_quantile:.0%}: {var:,.0f}")
    ax.axvline(cvar, color=theme.colors[2], linestyle=':', linewidth=2,
               label=f"CVaR {var_quantile:.0%}: {cvar:,.0f}")
    ax.axvline(float(values.mean()), color=theme.colors[3], linestyle='-', linewidth=1.5,
               label=f"Mean: {values.mean():,.0f}")

    ax.set_xlabel("P&L")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=theme.title_size)
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Recovery heatmap
# ---------------------------------------------------------------------------

def recovery_heatmap(
    surface,
    title: str = "Recovery Surface: R(seniority, tenor)",
    figsize: tuple[float, float] = (10, 6),
    fmt: str = ".1%",
    cmap: str = "YlOrRd_r",
):
    """Heatmap of a RecoverySurface object.

    Args:
        surface: RecoverySurface with .seniorities, .tenors, .recoveries.
    """
    sns = _require_seaborn()
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(
        surface.recoveries,
        index=surface.seniorities,
        columns=[f"{t:.0f}Y" for t in surface.tenors],
    )

    theme = get_theme()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5,
                ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Recovery Rate"})
    ax.set_title(title, fontsize=theme.title_size)
    ax.set_ylabel("Seniority")
    ax.set_xlabel("Tenor")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Greeks profile (multi-line with confidence band)
# ---------------------------------------------------------------------------

def greeks_profile(
    spot_range: list[float] | Any,
    greeks_by_spot: dict[str, list[float]],
    title: str = "Greeks Profile",
    figsize: tuple[float, float] = (12, 5),
):
    """Multi-panel Greeks profile across spot range.

    Args:
        spot_range: list of spot values.
        greeks_by_spot: {greek_name: [values_per_spot]}.
    """
    import matplotlib.pyplot as plt

    if not greeks_by_spot:
        raise ValueError("greeks_by_spot must not be empty")
    n = len(greeks_by_spot)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    theme = get_theme()
    spots = np.array(spot_range)

    keys = list(greeks_by_spot.keys())
    for ax, (name, values) in zip(axes, greeks_by_spot.items()):
        color = theme.colors[keys.index(name) % len(theme.colors)]
        ax.plot(spots, values, color=color, linewidth=theme.line_width)
        ax.fill_between(spots, values, alpha=0.1, color=color)
        ax.set_title(name, fontsize=theme.title_size)
        ax.set_xlabel("Spot")
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Sensitivity grid (2D heatmap of PV across two parameters)
# ---------------------------------------------------------------------------

def sensitivity_grid(
    param1_values: list[float],
    param2_values: list[float],
    pv_grid: list[list[float]] | Any,
    param1_name: str = "Param 1",
    param2_name: str = "Param 2",
    title: str = "PV Sensitivity",
    figsize: tuple[float, float] = (10, 7),
    fmt: str = ",.0f",
    cmap: str = "RdYlGn",
):
    """2D sensitivity heatmap: PV across two parameters.

    Args:
        param1_values: row values (y-axis).
        param2_values: column values (x-axis).
        pv_grid: 2D array of PV values.
    """
    sns = _require_seaborn()
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(
        pv_grid,
        index=[f"{v}" for v in param1_values],
        columns=[f"{v}" for v in param2_values],
    )

    theme = get_theme()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5,
                ax=ax, center=0, cbar_kws={"label": "PV"})
    ax.set_ylabel(param1_name)
    ax.set_xlabel(param2_name)
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Exposure profile (EPE/ENE with confidence bands)
# ---------------------------------------------------------------------------

def exposure_profile(
    times: list[float],
    epe: list[float],
    ene: list[float] | None = None,
    pfe_95: list[float] | None = None,
    pfe_99: list[float] | None = None,
    title: str = "Exposure Profile",
    figsize: tuple[float, float] = (12, 5),
):
    """XVA exposure profile with EPE, ENE, and PFE bands.

    Args:
        times: time points (years).
        epe: expected positive exposure.
        ene: expected negative exposure (optional).
        pfe_95 / pfe_99: potential future exposure at 95th/99th percentile.
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    fig, ax = plt.subplots(figsize=figsize)

    t = np.array(times)
    ax.plot(t, epe, color=theme.colors[0], linewidth=theme.line_width, label="EPE")
    ax.fill_between(t, 0, epe, alpha=0.15, color=theme.colors[0])

    if ene is not None:
        ax.plot(t, ene, color=theme.colors[1], linewidth=theme.line_width, label="ENE")
        ax.fill_between(t, ene, 0, alpha=0.15, color=theme.colors[1])

    if pfe_95 is not None:
        ax.plot(t, pfe_95, color=theme.colors[3 % len(theme.colors)], linewidth=1.0, linestyle='--', label="PFE 95%")

    if pfe_99 is not None:
        ax.plot(t, pfe_99, color=theme.colors[4 % len(theme.colors)], linewidth=1.0, linestyle=':', label="PFE 99%")

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure")
    ax.set_title(title, fontsize=theme.title_size)
    ax.legend()
    plt.tight_layout()
    return fig
