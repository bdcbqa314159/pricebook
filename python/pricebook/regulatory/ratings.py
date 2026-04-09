"""Ratings and PD mapping: shared foundation for all regulatory modules.

Single source of truth for rating-to-PD conversion, rating normalisation,
investment grade classification, and PD/rating resolution.

    from pricebook.regulatory.ratings import (
        RATING_TO_PD, resolve_pd, resolve_rating,
        normalize_rating, is_investment_grade,
    )
"""

from __future__ import annotations

import math


# ---- Rating to PD mapping ----

RATING_TO_PD: dict[str, float] = {
    "AAA": 0.0001,
    "AA+": 0.0002,
    "AA": 0.0003,
    "AA-": 0.0005,
    "A+": 0.0007,
    "A": 0.0009,
    "A-": 0.0015,
    "BBB+": 0.0025,
    "BBB": 0.0040,
    "BBB-": 0.0075,
    "BB+": 0.0125,
    "BB": 0.0200,
    "BB-": 0.0350,
    "B+": 0.0550,
    "B": 0.0900,
    "B-": 0.1400,
    "CCC+": 0.2000,
    "CCC": 0.2700,
    "CCC-": 0.3500,
    "CC": 0.4000,
    "C": 0.4500,
    "D": 1.0000,
}

_PD_RATING_SORTED = sorted(RATING_TO_PD.items(), key=lambda x: x[1])


# ---- Rating normalisation ----

_RATING_NORMALIZATION = {
    "AA+": "AA", "AA-": "AA",
    "A+": "A", "A-": "A",
    "BBB+": "BBB", "BBB-": "BBB",
    "BB+": "BB", "BB-": "BB",
    "B+": "B", "B-": "B",
    "CCC+": "CCC", "CCC-": "CCC",
}


def normalize_rating(rating: str) -> str:
    """Normalize a notched rating to its base (e.g. 'AA+' → 'AA')."""
    if not rating:
        return "BBB"
    return _RATING_NORMALIZATION.get(rating.upper().strip(), rating.upper().strip())


# ---- PD <-> Rating conversion ----

def get_rating_from_pd(pd: float) -> str:
    """Find the closest rating for a given PD."""
    if pd <= 0:
        return "AAA"
    if pd >= 1.0:
        return "D"
    best, min_dist = "BBB", float("inf")
    for rating, rpd in _PD_RATING_SORTED:
        d = abs(pd - rpd)
        if d < min_dist:
            min_dist = d
            best = rating
    return best


def get_rating_from_pd_log(pd: float) -> str:
    """Find closest rating using log-scale distance (better for low PDs)."""
    if pd <= 0:
        return "AAA"
    if pd >= 1.0:
        return "D"
    best, min_dist = "BBB", float("inf")
    for rating, rpd in RATING_TO_PD.items():
        d = abs(math.log(max(pd, 1e-8)) - math.log(max(rpd, 1e-8)))
        if d < min_dist:
            min_dist = d
            best = rating
    return best


def get_pd_range(rating: str) -> tuple[float, float]:
    """PD range that maps to a given rating (midpoint boundaries)."""
    if rating not in RATING_TO_PD:
        raise ValueError(f"Unknown rating: {rating}")
    rpd = RATING_TO_PD[rating]
    idx = next(i for i, (r, _) in enumerate(_PD_RATING_SORTED) if r == rating)
    lower = 0.0 if idx == 0 else (_PD_RATING_SORTED[idx - 1][1] + rpd) / 2
    upper = 1.0 if idx == len(_PD_RATING_SORTED) - 1 else (rpd + _PD_RATING_SORTED[idx + 1][1]) / 2
    return (lower, upper)


# ---- Resolve functions ----

def resolve_pd(
    pd: float | None = None,
    rating: str | None = None,
    default_pd: float = 0.004,
) -> float:
    """Resolve PD from explicit value or rating lookup."""
    if pd is not None:
        return pd
    if rating is not None and rating.upper() not in ("UNRATED", "NR", ""):
        return RATING_TO_PD.get(rating, RATING_TO_PD.get(normalize_rating(rating), default_pd))
    return default_pd


def resolve_rating(
    rating: str | None = None,
    pd: float | None = None,
    default_rating: str = "BBB",
) -> str:
    """Resolve rating from explicit value or PD-based estimation."""
    if rating is not None and rating.upper() not in ("UNRATED", "NR", ""):
        return rating
    if pd is not None:
        return get_rating_from_pd(pd)
    return default_rating


# ---- Investment grade classification ----

IG_RATINGS = frozenset({
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
})

HY_RATINGS = frozenset({
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
})


def is_investment_grade(rating: str) -> bool:
    """Check if a rating is investment grade (BBB- or better)."""
    return rating in IG_RATINGS


def is_high_yield(rating: str) -> bool:
    """Check if a rating is high yield (BB+ or worse)."""
    return rating in HY_RATINGS
