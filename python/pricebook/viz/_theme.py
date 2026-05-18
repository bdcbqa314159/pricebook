"""Pricebook visual theme: colors, fonts, light/dark, seaborn integration.

Uses seaborn styling when available, falls back to pure matplotlib.

    from pricebook.viz import configure_theme
    configure_theme(dark=True, seaborn_style="darkgrid")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PricebookTheme:
    """Visual theme for pricebook charts."""
    background: str
    foreground: str
    grid_color: str
    grid_alpha: float
    colors: tuple[str, ...]
    font_family: str
    font_size: int
    title_size: int
    line_width: float
    seaborn_style: str = "whitegrid"
    seaborn_context: str = "notebook"
    seaborn_palette: str | None = None



    def to_dict(self) -> dict:
        return vars(self)
LIGHT = PricebookTheme(
    background="#ffffff",
    foreground="#1a1a2e",
    grid_color="#cccccc",
    grid_alpha=0.3,
    colors=("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#be185d"),
    font_family="sans-serif",
    font_size=11,
    title_size=13,
    line_width=1.8,
    seaborn_style="whitegrid",
    seaborn_context="notebook",
    seaborn_palette="deep",
)

DARK = PricebookTheme(
    background="#1a1a2e",
    foreground="#e0e0e0",
    grid_color="#444444",
    grid_alpha=0.3,
    colors=("#60a5fa", "#f87171", "#34d399", "#fbbf24", "#a78bfa", "#f472b6"),
    font_family="sans-serif",
    font_size=11,
    title_size=13,
    line_width=1.8,
    seaborn_style="darkgrid",
    seaborn_context="notebook",
    seaborn_palette="bright",
)

_current_theme: PricebookTheme = LIGHT
_seaborn_available: bool | None = None


def _has_seaborn() -> bool:
    """Check if seaborn is importable (cached)."""
    global _seaborn_available
    if _seaborn_available is None:
        try:
            import seaborn  # noqa: F401
            _seaborn_available = True
        except ImportError:
            _seaborn_available = False
    return _seaborn_available


def configure_theme(
    theme: PricebookTheme | None = None,
    dark: bool = False,
    seaborn_style: str | None = None,
    seaborn_context: str | None = None,
    seaborn_palette: str | None = None,
) -> None:
    """Set the global pricebook theme.

    Args:
        theme: custom theme (overrides dark).
        dark: use dark mode.
        seaborn_style: "whitegrid", "darkgrid", "white", "dark", "ticks".
        seaborn_context: "paper", "notebook", "talk", "poster".
        seaborn_palette: "deep", "muted", "bright", "pastel", "dark", "colorblind".
    """
    global _current_theme
    if theme is not None:
        _current_theme = theme
    elif dark:
        _current_theme = DARK
    elif seaborn_style is None and seaborn_context is None and seaborn_palette is None:
        # No args at all → reset to LIGHT
        _current_theme = LIGHT
    # else: keep current theme, only update seaborn params below

    # Apply seaborn globally if available
    if _has_seaborn():
        import seaborn as sns
        style = seaborn_style or _current_theme.seaborn_style
        context = seaborn_context or _current_theme.seaborn_context
        palette = seaborn_palette or _current_theme.seaborn_palette
        sns.set_theme(style=style, context=context, palette=palette,
                      font=_current_theme.font_family,
                      font_scale=_current_theme.font_size / 11.0)


def get_theme(dark: bool | None = None) -> PricebookTheme:
    """Get the active theme, optionally overriding dark mode."""
    if dark is None:
        return _current_theme
    return DARK if dark else LIGHT
