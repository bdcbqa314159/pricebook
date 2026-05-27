"""Static convention data — JSON files loaded at import time.

Convention registries (sovereign bonds, rate indices, CDS specs, etc.)
are stored as JSON arrays in this directory. Each module that owns a
convention dataclass loads its data via :func:`load_conventions`.
"""
