"""Put the repo root on `sys.path` so the tooling tests can import `verify`.

`pyproject.toml` sets `pythonpath = ["src"]` for the library, which deliberately does *not* include
the root — `verify.py` is a tool, not part of the package. A `conftest.py` is the fix rather than a
`sys.path` insert inside the test module: that insert would have to precede the import, tripping
ruff's import-order rule and needing a lint suppression, which §7 forbids. So the cause is fixed
instead of silenced.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
