"""Generic plots that work with any InstrumentResult or instrument."""

from __future__ import annotations

import numpy as np


def plot_summary_table(ax, instrument, curve, *, result=None, theme=None, **kwargs):
    """Summary table of result.to_dict() values."""
    if result is None and hasattr(instrument, 'price') and curve is not None:
        result = instrument.price(curve)
    elif result is None and hasattr(instrument, 'to_dict'):
        result = instrument

    if result is None:
        ax.text(0.5, 0.5, "No result to display", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        return

    data = result.to_dict()
    ax.axis("off")

    rows = [(k, f"{v:.6f}" if isinstance(v, float) else str(v))
            for k, v in data.items()]
    if not rows:
        return

    table = ax.table(
        cellText=[[r[1]] for r in rows],
        rowLabels=[r[0] for r in rows],
        colLabels=["Value"],
        loc="center",
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Result Summary")


def plot_sensitivity(ax, instrument, curve, *, param, low=None, high=None,
                     n_points=50, theme=None, **kwargs):
    """Bump a single parameter, reprice, plot price vs param value."""
    original = getattr(instrument, param)
    if low is None:
        low = original * 0.5 if original > 0 else original - 0.01
    if high is None:
        high = original * 1.5 if original > 0 else original + 0.01

    xs = np.linspace(low, high, n_points)
    prices = []
    for x in xs:
        setattr(instrument, param, float(x))
        try:
            r = instrument.price(curve)
            prices.append(r.price)
        except Exception:
            prices.append(float('nan'))
    setattr(instrument, param, original)

    ax.plot(xs, prices, lw=2)
    ax.axvline(original, ls=":", color="gray", alpha=0.7, label=f"current = {original:.4f}")
    ax.set_xlabel(param)
    ax.set_ylabel("Price")
    ax.set_title(f"Sensitivity to {param}")
    ax.legend()
    ax.grid(alpha=0.3)
