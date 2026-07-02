"""Convention data loading — JSON files ↔ convention dataclasses.

Provides :func:`load_conventions` and :func:`save_conventions` for
reading/writing convention reregistries to JSON files in ``pricebook/data/``.

    from pricebook.core.data_registry import load_conventions, save_conventions, DATA_DIR

    # Load sovereign conventions from JSON (or fall back to defaults)
    conventions = load_conventions(
        "sovereign_conventions.json",
        SovereignConventions,
    )

    # Save updated registry back to JSON
    save_conventions("sovereign_conventions.json", conventions)
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, Self, TypeVar


class Convention(Protocol):
    """Structural contract for a convention record: it round-trips through a dict.

    Item types passed to the loaders must provide ``to_dict()`` and a
    ``from_dict()`` classmethod (most get them from the ``@serialisable_convention``
    decorator; a few hand-roll them). Bounding the TypeVar on this Protocol makes
    that contract explicit and type-checkable — ``type[T].from_dict`` is only
    valid because ``T`` is bound here.
    """

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> Self: ...


T = TypeVar("T", bound=Convention)

# Data directory: python/pricebook/data/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _validate_filename(filename: str) -> None:
    """Guard against path traversal and invalid filenames."""
    if not filename or ".." in filename or filename.startswith("/"):
        raise ValueError(f"Invalid filename: {filename!r}. Must be a simple name relative to DATA_DIR.")


def load_conventions(
    filename: str,
    item_type: type[T],
) -> list[T]:
    """Load convention objects from a JSON file.

    Args:
        filename: JSON file name (relative to DATA_DIR). Must not contain '..'.
        item_type: dataclass type with from_dict() classmethod.

    Returns:
        List of convention objects. Empty list if file not found.
    """
    _validate_filename(filename)
    path = DATA_DIR / filename
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        warnings.warn(f"{filename}: expected JSON array, got {type(data).__name__}", RuntimeWarning, stacklevel=2)
        return []

    items = []
    for entry in data:
        try:
            obj = item_type.from_dict(entry)
            items.append(obj)
        except Exception as e:
            warnings.warn(
                f"Skipping invalid entry in {filename}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # A present file whose entries ALL fail is corrupt / schema-drifted — fail
    # loud rather than return [] (which the registry would silently treat as
    # "no file" and replace with hardcoded defaults). An empty file (no entries)
    # is a legitimate "use defaults" and still returns [].
    if data and not items:
        raise ValueError(
            f"{filename}: {len(data)} entries present but none parsed as "
            f"{item_type.__name__} — the data file is corrupt or its schema has "
            f"drifted from the code. Fix the file; do not fall back to defaults."
        )

    return items


def save_conventions(
    filename: str,
    items: Sequence[Convention],
) -> Path:
    """Save convention objects to a JSON file.

    Args:
        filename: JSON file name (relative to DATA_DIR). Must not contain '..'.
        items: list of convention objects with to_dict() method.

    Returns:
        Path to the written file.
    """
    _validate_filename(filename)
    path = DATA_DIR / filename
    data = [item.to_dict() for item in items]

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")  # trailing newline

    return path


def load_registry(
    filename: str,
    item_type: type[T],
    key_fn: Callable[[T], str],
    defaults: dict[str, T],
) -> dict[str, T]:
    """Load a keyed registry from JSON, falling back to hardcoded defaults.

    Args:
        filename: JSON file name (relative to DATA_DIR).
        item_type: dataclass type with from_dict().
        key_fn: function to extract dict key from each item (e.g. lambda c: c.market_code).
            Must not be None.
        defaults: hardcoded default registry dict.

    Returns:
        Dict mapping key → convention object.

    Raises:
        ValueError: if key_fn is None.
    """
    if key_fn is None:
        raise ValueError("key_fn must not be None for load_registry")
    loaded = load_conventions(filename, item_type)
    if loaded:
        return {key_fn(item): item for item in loaded}
    return defaults
