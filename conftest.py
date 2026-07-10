"""Root conftest — put the new tree (src/) on sys.path for tests_ng.

ponytail: no packaging step for a walking skeleton; add a pyproject for
pricebook_ng when we actually distribute it. pathlib keeps it cross-platform.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
