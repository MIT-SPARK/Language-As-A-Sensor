"""The interface between VLMaP and a volumetric mapper.

The vision sensor needs a map that integrates posed RGB-D + per-pixel labels
and reports, for every voxel it has filled, a TSDF distance and its top-k
label weights. The paper uses Hydra (``hydra_map.HydraVoxelMap``); anything
implementing :class:`VoxelMap` can replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from langsensor.vlmap.types import VisionObservation


@dataclass
class VoxelBatch:
    """Every filled voxel after an integration step (N voxels, top-k labels)."""

    keys: np.ndarray            # (N, 3) int32 voxel indices (world position = key * voxel_size)
    weights: np.ndarray         # (N, K) float32 accumulated label weights, zero-padded
    label_ids: np.ndarray       # (N, K) uint32 label ids
    tsdf_distance: np.ndarray   # (N,) float32 signed distance to the nearest surface
    has_semantic: np.ndarray    # (N,) bool
    has_tsdf: np.ndarray        # (N,) bool, TSDF weight above the trust threshold


class VoxelMap(Protocol):
    truncation_distance: float

    def integrate(self, obs: VisionObservation) -> VoxelBatch:
        """Integrate one frame and return all filled voxels."""
        ...
