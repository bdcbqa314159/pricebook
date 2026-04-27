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

# Register product modules (triggers @register_instrument decorators)
import pricebook.viz._tlock    # noqa: F401
import pricebook.viz._cmasw    # noqa: F401
import pricebook.viz._cmt      # noqa: F401
import pricebook.viz._hybrid   # noqa: F401
import pricebook.viz._trs      # noqa: F401

__all__ = ["plot", "PlotBuilder", "LIGHT", "DARK", "PricebookTheme", "configure_theme"]
