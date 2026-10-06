"""Data types of the closed-loop VLMaP experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class BeliefModel(ABC):
    """A (possibly unnormalised) spatial likelihood p(observations | x)."""

    @abstractmethod
    def query_batch(self, positions: np.ndarray) -> np.ndarray:
        """(N, 3) positions -> (N,) float32 likelihoods."""

    def query(self, position: np.ndarray) -> float:
        return float(self.query_batch(np.asarray(position)[None])[0])

    def coverage_batch(self, positions: np.ndarray) -> np.ndarray:
        """True where this sensor has observed the position (default: everywhere)."""
        return np.ones(len(positions), dtype=bool)


class UniformBelief(BeliefModel):
    def query_batch(self, positions: np.ndarray) -> np.ndarray:
        return np.full(len(positions), 0.5, dtype=np.float32)


@dataclass(frozen=True)
class VisionObservation:
    timestep: int
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32, metres
    pose: np.ndarray           # (4, 4) world-from-camera
    gt_semantics: np.ndarray   # (H, W) int32 class ids (ADE20K-MP3D label space)


# Episode scene graphs are flat: bboxes are 6-element AABBs and each object
# carries its region id. This is the format written by scripts/make_episodes.py.

@dataclass(frozen=True)
class ObjectEntry:
    id: int
    label: str
    position: tuple[float, float, float]
    bbox: tuple[float, ...] | None = None     # [x_min, y_min, z_min, x_max, y_max, z_max]
    region_id: int | None = None
    nyu40_label: str | None = None


@dataclass(frozen=True)
class RegionEntry:
    id: int
    label: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class SceneGraph:
    objects: tuple[ObjectEntry, ...]
    regions: tuple[RegionEntry, ...] = ()


@dataclass(frozen=True)
class TimestampedUtterance:
    timestep: int
    text: str
    ambiguity: int = 0
    anchor_object_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LanguageEpisode:
    episode_id: str
    utterances: tuple[TimestampedUtterance, ...]
    scene_graph: SceneGraph                       # target removed
    target_xyz: np.ndarray                        # (3,)
    target_label: str = ""
    target_bbox: np.ndarray | None = None
    target_surface_xyz: np.ndarray | None = None   # (M, 3) points on the target's surface
