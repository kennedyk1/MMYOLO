from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_ultralytics_repo() -> None:
    """Prefer the local Ultralytics clone in this workspace over any site-packages install."""
    repo_root = Path(__file__).resolve().parents[1] / "ultralytics"
    repo_root_str = str(repo_root)
    if repo_root.exists() and repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
