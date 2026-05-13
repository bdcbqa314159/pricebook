"""Replay visualization: equity curves, drawdowns, dashboards."""

from __future__ import annotations

import numpy as np

from pricebook.ts._core import TimeSeries
from pricebook.ts._replay import ReplayResult


def _dates_to_floats(dates: np.ndarray) -> list[str]:
    """Convert datetime64 array to string labels for plotting."""
    return [str(d) for d in dates]


def plot_equity_curve(replay: ReplayResult) -> None:
    """Plot cumulative P&L with drawdown shading."""
    from pricebook.viz._backend import apply_theme, create_figure

    with apply_theme():
        fig, ax = create_figure(1)
        labels = _dates_to_floats(replay.cumulative_pnl.dates)
        ax.plot(range(len(labels)), replay.cumulative_pnl.values, linewidth=1.5)
        ax.fill_between(
            range(len(labels)),
            replay.cumulative_pnl.values,
            replay.cumulative_pnl.values * (1 - replay.drawdown.values),
            alpha=0.2, color="red",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L")
        ax.set_title(f"Equity Curve — {replay.pnl.name}")
        # Sparse x labels
        n = len(labels)
        step = max(n // 8, 1)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=45, fontsize=7)
        fig.tight_layout()


def plot_pnl_histogram(replay: ReplayResult) -> None:
    """Plot P&L distribution with VaR markers."""
    from pricebook.viz._seaborn import pnl_distribution
    pnl_distribution(replay.pnl.values, title=f"P&L Distribution — {replay.pnl.name}")


def plot_rolling_sharpe(replay: ReplayResult, window: int = 60) -> None:
    """Plot rolling Sharpe ratio."""
    from pricebook.viz._backend import apply_theme, create_figure
    from pricebook.ts._rolling import rolling_sharpe

    rs = rolling_sharpe(replay.pnl, window)

    with apply_theme():
        fig, ax = create_figure(1)
        labels = _dates_to_floats(rs.dates)
        ax.plot(range(len(labels)), rs.values, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Sharpe")
        ax.set_title(f"Rolling {window}d Sharpe — {replay.pnl.name}")
        n = len(labels)
        step = max(n // 8, 1)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=45, fontsize=7)
        fig.tight_layout()


def plot_drawdowns(replay: ReplayResult, n_worst: int = 5) -> None:
    """Plot drawdown series, highlighting worst episodes."""
    from pricebook.viz._backend import apply_theme, create_figure

    with apply_theme():
        fig, ax = create_figure(1)
        labels = _dates_to_floats(replay.drawdown.dates)
        ax.fill_between(range(len(labels)), -replay.drawdown.values * 100, 0,
                        alpha=0.5, color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"Drawdown — {replay.pnl.name}")
        n = len(labels)
        step = max(n // 8, 1)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=45, fontsize=7)
        fig.tight_layout()


def plot_dashboard(replay: ReplayResult) -> None:
    """2×2 dashboard: equity curve, P&L histogram, rolling Sharpe, drawdown."""
    from pricebook.viz._backend import apply_theme, create_figure
    from pricebook.ts._rolling import rolling_sharpe

    rs = rolling_sharpe(replay.pnl, 60)

    with apply_theme():
        fig, axes = create_figure(4)
        ax1, ax2, ax3, ax4 = axes

        n = len(replay.cumulative_pnl)
        labels = _dates_to_floats(replay.cumulative_pnl.dates)
        step = max(n // 6, 1)

        # Equity curve
        ax1.plot(range(n), replay.cumulative_pnl.values, linewidth=1.5)
        ax1.set_title("Equity Curve")
        ax1.set_xticks(range(0, n, step))
        ax1.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=45, fontsize=6)

        # P&L histogram
        ax2.hist(replay.pnl.values, bins=30, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax2.axvline(0, color="black", linewidth=0.5)
        ax2.set_title("P&L Distribution")

        # Rolling Sharpe
        ax3.plot(range(len(rs)), rs.values, linewidth=1)
        ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax3.set_title("Rolling 60d Sharpe")
        ax3.set_xticks(range(0, len(rs), step))
        ax3.set_xticklabels([str(rs.dates[i]) for i in range(0, len(rs), step)],
                            rotation=45, fontsize=6)

        # Drawdown
        ax4.fill_between(range(n), -replay.drawdown.values * 100, 0,
                         alpha=0.5, color="red")
        ax4.set_title("Drawdown (%)")
        ax4.set_xticks(range(0, n, step))
        ax4.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=45, fontsize=6)

        fig.suptitle(f"Replay Dashboard — {replay.pnl.name}", fontsize=12)
        fig.tight_layout()
