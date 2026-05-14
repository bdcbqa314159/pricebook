"""Risk and desk-level visualisations for pricebook.

Charts for P&L attribution, stress testing, risk decomposition,
Greeks surfaces, and hedge tracking:

    from pricebook.viz import (
        pnl_waterfall, risk_decomposition, stress_comparison,
        tenor_bucketing, vega_ladder, pnl_table,
        greeks_surface, greeks_evolution,
        hedge_pnl_tracking, rolling_correlation,
    )

All functions consume plain data (arrays, dicts) — no instrument imports.
Pure matplotlib (no seaborn dependency).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pricebook.viz._theme import get_theme


# ---------------------------------------------------------------------------
# 1. P&L waterfall / bridge chart
# ---------------------------------------------------------------------------

def pnl_waterfall(
    components: dict[str, float],
    title: str = "P&L Attribution",
    figsize: tuple[float, float] = (12, 6),
    show_total: bool = True,
    fmt: str = ",.0f",
):
    """Waterfall chart decomposing P&L into contributing factors.

    Args:
        components: ordered dict of {factor_name: pnl_value}.
            e.g. {"Carry": 1200, "Rolldown": 300, "Rate": -500, ...}
        show_total: append a Total bar from zero.
        fmt: number format for value labels.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    names = list(components.keys())
    values = list(components.values())

    if show_total:
        names.append("Total")
        values.append(sum(components.values()))

    n = len(names)
    cumulative = np.zeros(n + 1)
    for i, v in enumerate(values):
        if show_total and i == n - 1:
            cumulative[i] = 0.0
        else:
            cumulative[i + 1] = cumulative[i] + v if i < n - 1 else cumulative[i]
            if i > 0:
                cumulative[i] = cumulative[i]

    # Compute bar starts and widths
    starts = np.zeros(n)
    for i in range(n):
        if show_total and i == n - 1:
            starts[i] = 0.0
        else:
            starts[i] = sum(values[:i])

    color_pos = theme.colors[2] if len(theme.colors) > 2 else "#2ca02c"
    color_neg = theme.colors[1] if len(theme.colors) > 1 else "#d62728"
    color_total = theme.colors[0]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n)

    for i in range(n):
        v = values[i]
        if show_total and i == n - 1:
            color = color_total
        elif v >= 0:
            color = color_pos
        else:
            color = color_neg

        ax.barh(y_pos[i], v, left=starts[i], color=color, edgecolor="white",
                height=0.6, linewidth=0.5)

        # Value label
        x_label = starts[i] + v
        ha = "left" if v >= 0 else "right"
        ax.text(x_label, y_pos[i], f" {v:{fmt}} ", ha=ha, va="center",
                fontsize=theme.font_size - 1, color=theme.foreground)

    # Connector lines between bars
    for i in range(n - 1):
        end_x = starts[i] + values[i]
        ax.plot([end_x, end_x], [y_pos[i] - 0.35, y_pos[i + 1] + 0.35],
                color="gray", linewidth=0.5, linestyle=":")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("P&L")
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Risk decomposition (horizontal bar, sorted by abs value)
# ---------------------------------------------------------------------------

def risk_decomposition(
    labels: list[str],
    values: list[float] | Any,
    title: str = "Risk Decomposition",
    figsize: tuple[float, float] = (10, 6),
    fmt: str = ",.0f",
    sort: bool = True,
):
    """Horizontal bar chart of risk contributions sorted by magnitude.

    Args:
        labels: factor names (e.g. curve pillar labels, position names).
        values: risk values (e.g. DV01 per pillar, vega per asset class).
        sort: sort bars by absolute value (largest on top).

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    vals = np.array(values, dtype=float)
    labs = list(labels)

    if sort:
        idx = np.argsort(np.abs(vals))[::-1]
        vals = vals[idx]
        labs = [labs[i] for i in idx]

    color_pos = theme.colors[0]
    color_neg = theme.colors[1] if len(theme.colors) > 1 else "#d62728"
    colors = [color_pos if v >= 0 else color_neg for v in vals]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(vals))
    ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.6, linewidth=0.5)

    for i, v in enumerate(vals):
        ha = "left" if v >= 0 else "right"
        ax.text(v, y_pos[i], f" {v:{fmt}} ", ha=ha, va="center",
                fontsize=theme.font_size - 1, color=theme.foreground)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Stress scenario comparison
# ---------------------------------------------------------------------------

def stress_comparison(
    scenarios: list[dict[str, Any]],
    title: str = "Stress Test Comparison",
    figsize: tuple[float, float] = (12, 6),
    stacked: bool = False,
):
    """Grouped or stacked bar chart comparing stress scenario P&Ls.

    Args:
        scenarios: list of dicts, each with:
            - "name": str — scenario label
            - "total": float — total P&L
            - "breakdown": dict[str, float] | None — optional per-asset-class
        stacked: if True and breakdown present, show stacked bars.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    names = [s["name"] for s in scenarios]
    totals = [s["total"] for s in scenarios]
    has_breakdown = stacked and all(s.get("breakdown") for s in scenarios)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))

    if has_breakdown:
        # Collect all asset class keys
        all_keys: list[str] = []
        for s in scenarios:
            for k in s["breakdown"]:
                if k not in all_keys:
                    all_keys.append(k)

        bottoms_pos = np.zeros(len(names))
        bottoms_neg = np.zeros(len(names))

        for ki, key in enumerate(all_keys):
            color = theme.colors[ki % len(theme.colors)]
            vals = [s["breakdown"].get(key, 0.0) for s in scenarios]
            vals_arr = np.array(vals)

            pos = np.where(vals_arr >= 0, vals_arr, 0)
            neg = np.where(vals_arr < 0, vals_arr, 0)

            ax.bar(x, pos, bottom=bottoms_pos, color=color, label=key,
                   width=0.6, edgecolor="white", linewidth=0.5)
            ax.bar(x, neg, bottom=bottoms_neg, color=color,
                   width=0.6, edgecolor="white", linewidth=0.5, alpha=0.7)

            bottoms_pos += pos
            bottoms_neg += neg

        ax.legend(fontsize=theme.font_size - 1, loc="best")
    else:
        colors = [theme.colors[0] if t >= 0 else theme.colors[1] for t in totals]
        ax.bar(x, totals, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

        for i, t in enumerate(totals):
            va = "bottom" if t >= 0 else "top"
            ax.text(i, t, f" {t:,.0f} ", ha="center", va=va,
                    fontsize=theme.font_size - 1, color=theme.foreground)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("P&L")
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Tenor bucketing
# ---------------------------------------------------------------------------

def tenor_bucketing(
    buckets: list[str],
    values: list[float] | Any,
    title: str = "Tenor Distribution",
    figsize: tuple[float, float] = (10, 5),
    ylabel: str = "DV01",
):
    """Vertical bar chart of risk by tenor bucket with color gradient.

    Args:
        buckets: tenor labels (e.g. ["1M", "3M", "1Y", "5Y", "10Y", "30Y"]).
        values: risk values per bucket (e.g. DV01 per pillar).
        ylabel: y-axis label.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    theme = get_theme()
    vals = np.array(values, dtype=float)
    n = len(vals)

    # Color gradient: blue shading by tenor (short→long)
    cmap = get_cmap("Blues")
    colors = [cmap(0.3 + 0.6 * i / max(n - 1, 1)) for i in range(n)]
    # Override with red for negative values
    for i in range(n):
        if vals[i] < 0:
            colors[i] = theme.colors[1] if len(theme.colors) > 1 else "#d62728"

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n)
    ax.bar(x, vals, color=colors, width=0.7, edgecolor="white", linewidth=0.5)

    for i, v in enumerate(vals):
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v, f"{v:,.1f}", ha="center", va=va,
                fontsize=theme.font_size - 1, color=theme.foreground)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Vega ladder
# ---------------------------------------------------------------------------

def vega_ladder(
    expiry_buckets: list[str],
    vega_values: list[float] | Any,
    vol_premium: list[float] | None = None,
    title: str = "Vega Ladder",
    figsize: tuple[float, float] = (10, 6),
):
    """Horizontal bar chart of vega by expiry bucket.

    Args:
        expiry_buckets: labels (e.g. ["0-3M", "3-6M", "6-12M", "1-2Y", "2-5Y", "5Y+"]).
        vega_values: vega per bucket.
        vol_premium: optional implied-minus-realised per bucket.
            Positive = rich (good for short vega), negative = cheap.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    vals = np.array(vega_values, dtype=float)
    n = len(vals)

    if vol_premium is not None:
        prem = np.array(vol_premium, dtype=float)
        # Rich (premium > 0) = green, Cheap (< 0) = red, neutral = theme default
        colors = []
        for p in prem:
            if p > 0.5:
                colors.append("#2ca02c")  # rich
            elif p < -0.5:
                colors.append("#d62728")  # cheap
            else:
                colors.append(theme.colors[0])
    else:
        colors = [theme.colors[0] if v >= 0 else theme.colors[1] for v in vals]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n)
    ax.barh(y_pos, vals, color=colors, height=0.6, edgecolor="white", linewidth=0.5)

    for i, v in enumerate(vals):
        # Place label at the end of the bar, outside
        if v >= 0:
            ha, x_off = "left", v
        else:
            ha, x_off = "left", 0
        label = f" {v:,.0f}"
        if vol_premium is not None:
            label += f"  (prem: {vol_premium[i]:+.1f}%)"
        ax.text(x_off, y_pos[i], label, ha=ha, va="center",
                fontsize=theme.font_size - 1, color=theme.foreground)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(expiry_buckets)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Vega")
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. P&L explain table
# ---------------------------------------------------------------------------

def pnl_table(
    rows: list[dict[str, Any]],
    title: str = "P&L Explain",
    figsize: tuple[float, float] = (10, 6),
    columns: list[str] | None = None,
):
    """Formatted matplotlib table for P&L attribution.

    Args:
        rows: list of row dicts. Each dict keys map to column names.
            e.g. [{"Risk Factor": "Rates", "Sensitivity": -450,
                    "Move": 0.05, "P&L": -22.5}, ...]
        columns: column names. If None, inferred from first row keys.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    if not rows:
        raise ValueError("rows must not be empty")

    cols = columns or list(rows[0].keys())
    cell_text = []
    for row in rows:
        cell_row = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                cell_row.append(f"{v:,.2f}")
            else:
                cell_row.append(str(v))
        cell_text.append(cell_row)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        loc="center",
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(theme.font_size)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor(theme.colors[0])
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # Alternating row colors
    bg_light = theme.background
    bg_alt = "#f5f5f5" if theme.background == "#ffffff" else "#252540"
    for i in range(len(rows)):
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg_light if i % 2 == 0 else bg_alt)
            cell.set_edgecolor("#dddddd" if theme.background == "#ffffff" else "#444444")
            cell.set_text_props(color=theme.foreground)
            # First column left-aligned
            if j == 0:
                cell.get_text().set_ha("left")

    # Bold last row if it looks like a total
    if len(rows) > 1:
        last_val = str(rows[-1].get(cols[0], "")).lower()
        if last_val in ("total", "sum", "net"):
            for j in range(len(cols)):
                cell = table[len(rows), j]
                cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=theme.title_size, pad=20)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Greeks surface (2D contour: strike × expiry)
# ---------------------------------------------------------------------------

def greeks_surface(
    strikes: list[float] | Any,
    expiries: list[float] | Any,
    greeks_grid: list[list[float]] | Any,
    greek_name: str = "Delta",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 7),
    cmap: str = "RdYlBu_r",
):
    """2D contour plot of a Greek across strike and expiry dimensions.

    Args:
        strikes: strike values (x-axis).
        expiries: expiry values in years (y-axis).
        greeks_grid: 2D array shape (len(expiries), len(strikes)).
        greek_name: name for colorbar label.
        cmap: colormap.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    S = np.array(strikes)
    T = np.array(expiries)
    Z = np.array(greeks_grid)

    if Z.shape != (len(T), len(S)):
        raise ValueError(
            f"greeks_grid shape {Z.shape} must be (len(expiries), len(strikes)) "
            f"= ({len(T)}, {len(S)})"
        )

    X, Y = np.meshgrid(S, T)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
    ax.contour(X, Y, Z, levels=20, colors="k", linewidths=0.3, alpha=0.3)
    fig.colorbar(cf, ax=ax, label=greek_name, shrink=0.8)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry (years)")
    ax.set_title(title or f"{greek_name} Surface", fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Greeks evolution over time
# ---------------------------------------------------------------------------

def greeks_evolution(
    times: list[float] | Any,
    greeks_by_time: dict[str, list[float]],
    title: str = "Greeks Evolution",
    figsize: tuple[float, float] = (14, 8),
    x_label: str = "Days to Expiry",
):
    """Multi-panel line chart of Greeks vs time.

    Args:
        times: time axis (e.g. days to expiry, descending).
        greeks_by_time: {greek_name: [values_per_time]}.
        x_label: label for shared x-axis.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    if not greeks_by_time:
        raise ValueError("greeks_by_time must not be empty")

    theme = get_theme()
    t = np.array(times)
    n = len(greeks_by_time)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    keys = list(greeks_by_time.keys())
    for i, (name, vals) in enumerate(greeks_by_time.items()):
        ax = axes[i]
        color = theme.colors[i % len(theme.colors)]
        ax.plot(t, vals, color=color, linewidth=theme.line_width)
        ax.fill_between(t, vals, alpha=0.1, color=color)
        ax.set_title(name, fontsize=theme.title_size)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(x_label)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=theme.title_size + 1, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Hedge P&L tracking
# ---------------------------------------------------------------------------

def hedge_pnl_tracking(
    dates: list[Any],
    position_pnl: list[float] | Any,
    hedge_pnl: list[float] | Any,
    title: str = "Hedge P&L Tracking",
    figsize: tuple[float, float] = (14, 5),
    cumulative: bool = True,
):
    """Position vs hedge P&L tracking chart.

    Args:
        dates: date labels (strings, datetime, or numeric).
        position_pnl: daily position P&L.
        hedge_pnl: daily hedge P&L.
        cumulative: if True, plot cumulative sums.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    pos = np.array(position_pnl, dtype=float)
    hdg = np.array(hedge_pnl, dtype=float)

    if cumulative:
        pos = np.cumsum(pos)
        hdg = np.cumsum(hdg)

    net = pos + hdg

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(dates))

    ax.plot(x, pos, color=theme.colors[0], linewidth=theme.line_width,
            label="Position")
    ax.plot(x, hdg, color=theme.colors[1], linewidth=theme.line_width,
            label="Hedge")
    ax.plot(x, net, color=theme.colors[2] if len(theme.colors) > 2 else "#2ca02c",
            linewidth=theme.line_width + 0.5, label="Net (hedged)")
    ax.fill_between(x, net, 0, alpha=0.1,
                    color=theme.colors[2] if len(theme.colors) > 2 else "#2ca02c")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L" if cumulative else "P&L")

    # Tick labels: show subset to avoid crowding
    n = len(dates)
    if n > 20:
        step = max(1, n // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(dates[i]) for i in range(0, n, step)],
                           rotation=30, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dates], rotation=30, ha="right")

    ax.legend(fontsize=theme.font_size - 1)
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. Rolling correlation
# ---------------------------------------------------------------------------

def rolling_correlation(
    dates: list[Any],
    corr_series: dict[str, list[float]],
    title: str = "Rolling Correlation",
    figsize: tuple[float, float] = (14, 5),
    confidence_band: float | None = None,
):
    """Multi-line rolling correlation chart.

    Args:
        dates: date labels.
        corr_series: {pair_label: [correlation_values]}.
            e.g. {"SPX/UST": [...], "EUR/GBP": [...]}.
        confidence_band: if provided, shade +-band around each line.

    Returns:
        matplotlib.Figure
    """
    import matplotlib.pyplot as plt

    theme = get_theme()
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(dates))

    for i, (label, vals) in enumerate(corr_series.items()):
        color = theme.colors[i % len(theme.colors)]
        v = np.array(vals)
        ax.plot(x, v, color=color, linewidth=theme.line_width, label=label)
        if confidence_band is not None:
            ax.fill_between(x, v - confidence_band, v + confidence_band,
                            alpha=0.1, color=color)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    ax.axhline(1, color="gray", linewidth=0.3, linestyle=":")
    ax.axhline(-1, color="gray", linewidth=0.3, linestyle=":")
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")

    n = len(dates)
    if n > 20:
        step = max(1, n // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(dates[i]) for i in range(0, n, step)],
                           rotation=30, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dates], rotation=30, ha="right")

    ax.legend(fontsize=theme.font_size - 1)
    ax.set_title(title, fontsize=theme.title_size)
    plt.tight_layout()
    return fig
