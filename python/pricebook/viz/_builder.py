"""Fluent PlotBuilder for composing custom dashboards."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelSpec:
    name: str
    kwargs: dict


class PlotBuilder:
    """Fluent builder for composing visualization dashboards.

        fig = (PlotBuilder(tlock, curve)
               .payoff()
               .greeks()
               .sensitivity("repo_rate", low=0.0, high=0.05)
               .figure())
    """

    def __init__(self, instrument, curve=None, *, dark=None):
        self._instrument = instrument
        self._curve = curve
        self._dark = dark
        self._panels: list[PanelSpec] = []

    # -- Product-specific panels --
    def payoff(self, **kwargs) -> PlotBuilder:
        self._panels.append(PanelSpec("payoff", kwargs))
        return self

    def greeks(self, **kwargs) -> PlotBuilder:
        self._panels.append(PanelSpec("greeks", kwargs))
        return self

    def heatmap(self, **kwargs) -> PlotBuilder:
        self._panels.append(PanelSpec("heatmap", kwargs))
        return self

    def comparison(self, **kwargs) -> PlotBuilder:
        self._panels.append(PanelSpec("comparison", kwargs))
        return self

    def roll_surface(self, **kwargs) -> PlotBuilder:
        self._panels.append(PanelSpec("roll_surface", kwargs))
        return self

    def martingale(self, **kwargs) -> PlotBuilder:
        self._panels.append(PanelSpec("martingale", kwargs))
        return self

    # -- Generic panels --
    def summary(self) -> PlotBuilder:
        self._panels.append(PanelSpec("summary", {}))
        return self

    def sensitivity(self, param: str, *, low=None, high=None, n_points=50,
                    **kwargs) -> PlotBuilder:
        self._panels.append(PanelSpec("sensitivity", {
            "param": param, "low": low, "high": high, "n_points": n_points, **kwargs
        }))
        return self

    # -- Terminal --
    def figure(self, figsize=None):
        """Build and return the Figure."""
        from pricebook.viz._backend import apply_theme, create_figure
        from pricebook.viz._dispatch import get_panel_handler
        from pricebook.viz._generic import plot_summary_table, plot_sensitivity
        from pricebook.viz._theme import get_theme

        if not self._panels:
            raise ValueError("No panels added. Use .payoff(), .greeks(), etc.")

        theme = get_theme(self._dark)
        inst_type = type(self._instrument)

        with apply_theme(theme):
            fig, axes = create_figure(len(self._panels), figsize)

            for ax, spec in zip(axes, self._panels):
                # Try product-specific handler first
                handler = get_panel_handler(inst_type, spec.name)
                if handler is not None:
                    handler(ax, self._instrument, self._curve,
                            theme=theme, **spec.kwargs)
                elif spec.name == "summary":
                    plot_summary_table(ax, self._instrument, self._curve, theme=theme)
                elif spec.name == "sensitivity":
                    plot_sensitivity(ax, self._instrument, self._curve,
                                     theme=theme, **spec.kwargs)
                else:
                    raise ValueError(
                        f"Unknown panel '{spec.name}' for {inst_type.__name__}")

            fig.tight_layout()
            return fig

    def show(self, figsize=None):
        """Build, display, and return the Figure."""
        fig = self.figure(figsize)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
        return fig
