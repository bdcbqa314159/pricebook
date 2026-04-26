"""Matplotlib backend wrapper for pricebook charts."""

from __future__ import annotations

from contextlib import contextmanager

from pricebook.viz._theme import PricebookTheme, get_theme


@contextmanager
def apply_theme(theme: PricebookTheme | None = None):
    """Apply pricebook theme to matplotlib rcParams, restore on exit."""
    import matplotlib.pyplot as plt

    t = theme or get_theme()
    old = dict(plt.rcParams)
    try:
        plt.rcParams.update({
            "figure.facecolor": t.background,
            "axes.facecolor": t.background,
            "axes.edgecolor": t.foreground,
            "axes.labelcolor": t.foreground,
            "text.color": t.foreground,
            "xtick.color": t.foreground,
            "ytick.color": t.foreground,
            "axes.grid": True,
            "grid.color": t.grid_color,
            "grid.alpha": t.grid_alpha,
            "font.family": t.font_family,
            "font.size": t.font_size,
            "axes.titlesize": t.title_size,
            "lines.linewidth": t.line_width,
            "axes.prop_cycle": plt.cycler(color=list(t.colors)),
        })
        yield
    finally:
        plt.rcParams.update(old)


def create_figure(n_panels: int, figsize: tuple[float, float] | None = None):
    """Create a figure with a smart grid layout for N panels."""
    import matplotlib.pyplot as plt

    if n_panels <= 0:
        n_panels = 1

    layouts = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
    nrows, ncols = layouts.get(n_panels, (((n_panels - 1) // 3) + 1, 3))

    if figsize is None:
        figsize = (6 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten and trim to exactly n_panels
    import numpy as np
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten().tolist()
    else:
        axes_flat = [axes]

    # Hide unused axes
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    return fig, axes_flat[:n_panels]
