"""Default locations of data and downloaded artifacts.

Every path the code reads by default is defined here, and each can be moved
with an environment variable. Command-line flags override these defaults.

    LANGSENSOR_ARTIFACTS   checkpoints, canonical splits, LLM caches, archived results
    VLA3D_ROOT             the VLA-3D release (contains 3RScan/, Matterport/, ...)
    CLOSED_LOOP_ROOT       per-scene RGB-D trajectories + episodes for VLMaP
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ[var]).expanduser() if var in os.environ else default


ARTIFACTS = _env_path("LANGSENSOR_ARTIFACTS", REPO_ROOT / "artifacts")
VLA3D_ROOT = _env_path("VLA3D_ROOT", REPO_ROOT / "data" / "VLA-3D")
CLOSED_LOOP_ROOT = _env_path("CLOSED_LOOP_ROOT", REPO_ROOT / "data" / "closed_loop")

CHECKPOINT = ARTIFACTS / "checkpoints" / "lsm.pt"
CLIP_LABEL_MAP = ARTIFACTS / "checkpoints" / "clip_label_map.pt"
SPLITS = ARTIFACTS / "splits"

# LLM response caches. Static benchmark and closed loop use different prompts,
# so they are kept apart (see docs/reproduce.md).
STATIC_CACHE = ARTIFACTS / "cache" / "static"
CLOSED_LOOP_CACHE = ARTIFACTS / "cache" / "closed_loop"

ARCHIVED_RESULTS = ARTIFACTS / "results"
OUTPUTS = REPO_ROOT / "outputs"
