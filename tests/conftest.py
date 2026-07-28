"""Ensure tests exercise this checkout rather than another installed verl tree."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root = str(PROJECT_ROOT)
if sys.path[0] != project_root:
    sys.path.insert(0, project_root)
