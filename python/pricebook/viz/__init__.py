"""Pricebook visualisation layer.

    from pricebook.viz import plot, PlotBuilder

    # Simple: auto-detects type, shows 2x2 dashboard
    fig = plot(tlock, curve)

    # Advanced: fluent builder
    fig = (PlotBuilder(tlock, curve)
           .payoff()
           .greeks()
           .sensitivity("repo_rate", low=0.0, high=0.05)
           .figure())
"""

from pricebook.viz._dispatch import plot
from pricebook.viz._builder import PlotBuilder
from pricebook.viz._theme import LIGHT, DARK, PricebookTheme, configure_theme
from pricebook.viz._seaborn import (
    correlation_heatmap,
    pnl_distribution,
    recovery_heatmap,
    greeks_profile,
    sensitivity_grid,
    exposure_profile,
)
from pricebook.viz._risk import (
    pnl_waterfall,
    risk_decomposition,
    stress_comparison,
    tenor_bucketing,
    vega_ladder,
    pnl_table,
    greeks_surface,
    greeks_evolution,
    hedge_pnl_tracking,
    rolling_correlation,
)

# Register product modules (triggers @register_instrument decorators)
import pricebook.viz._tlock          # noqa: F401
import pricebook.viz._cmasw          # noqa: F401
import pricebook.viz._cmt            # noqa: F401
import pricebook.viz._hybrid         # noqa: F401
import pricebook.viz._trs            # noqa: F401
import pricebook.viz._treasury_lock  # noqa: F401

__all__ = [
    "plot", "PlotBuilder", "LIGHT", "DARK", "PricebookTheme", "configure_theme",
    "correlation_heatmap", "pnl_distribution", "recovery_heatmap",
    "greeks_profile", "sensitivity_grid", "exposure_profile",
    "pnl_waterfall", "risk_decomposition", "stress_comparison",
    "tenor_bucketing", "vega_ladder", "pnl_table",
    "greeks_surface", "greeks_evolution",
    "hedge_pnl_tracking", "rolling_correlation",
]
