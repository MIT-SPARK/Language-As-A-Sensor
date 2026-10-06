"""Shared data types: scene graphs, referential statements, queries, groundings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from pydantic import BaseModel


class ObjectInfo(BaseModel):
    id: int
    label: str
    position: list[float]              # (x, y, z) object centre, world frame
    bbox: list[float] | None = None    # 8 corners x 3, flattened
    metadata: dict = {}                # region_id, nyu40_label, raw_label, volume, ...


class RegionInfo(BaseModel):
    id: int
    label: str
    position: list[float]
    metadata: dict = {}                # bbox_{x,y,z}_{min,max}


class SceneGraph(BaseModel):
    objects: list[ObjectInfo]
    regions: list[RegionInfo] = []
    relations: list[dict] = []


class ReferentialStatement(BaseModel):
    text: str
    target_object_id: int
    ambiguity: int
    anchor_object_id: list[int] | None = None
    relation: str | None = None
    region: tuple | None = None        # (region_id, region_label)


@dataclass
class SpatialQuery:
    """One utterance about a target object, with the target removed from the scene."""

    scene_id: str
    scene_graph: SceneGraph
    language: str
    # Supervision and annotations; absent at inference time.
    target_xyz: np.ndarray | None = None          # (3,) target centre
    target_bbox: np.ndarray | None = None
    gt_anchor_object_ids: list[int] | None = None
    gt_anchor_room_id: int | None = None
    pc: np.ndarray | None = None                  # (N, 3) scene points, target removed
    object_split: np.ndarray | None = None        # (N,) per-point object id
    metadata: dict = field(default_factory=dict)


@dataclass
class Grounding:
    """One grounding hypothesis: the region and anchor objects an utterance refers to."""

    anchor_room_id: int
    anchor_object_ids: list[int]
    language: str
    confidence: float = 1.0


@dataclass
class GroundedQuery:
    """Scene context paired with a single grounding — the tensorizer's input."""

    scene_id: str
    scene_graph: SceneGraph
    grounding: Grounding
    target_xyz: np.ndarray | None = None
    target_bbox: np.ndarray | None = None

    @classmethod
    def from_spatial_query(cls, query: SpatialQuery, grounding: Grounding | None = None) -> GroundedQuery:
        """Pair *query* with *grounding*, defaulting to its ground-truth anchors."""
        if grounding is None:
            if query.gt_anchor_room_id is None:
                raise ValueError("SpatialQuery has no gt_anchor_room_id; pass a Grounding explicitly")
            grounding = Grounding(
                anchor_room_id=query.gt_anchor_room_id,
                anchor_object_ids=list(query.gt_anchor_object_ids or []),
                language=query.language,
            )
        return cls(
            scene_id=query.scene_id,
            scene_graph=query.scene_graph,
            grounding=grounding,
            target_xyz=query.target_xyz,
            target_bbox=query.target_bbox,
        )


@dataclass
class CachedSample:
    """Model-ready tensors for one grounded query (no batch dim); text is tokenized at collate time."""

    language: str
    obj_clip_features: torch.Tensor   # (N, 512)
    obj_bboxes: torch.Tensor          # (N, 6) [cx, cy, cz, w, h, l], region frame
    obj_is_anchor: torch.Tensor       # (N,) bool
    obj_padding_mask: torch.Tensor    # (N,) bool, True = padded slot
    coord_shift: torch.Tensor         # (3,) region centre, world frame
    coord_scale: torch.Tensor         # (3,) region size
    target_xyz_world: torch.Tensor    # (3,)
    target_bbox_world: torch.Tensor   # (6,) [x_min, y_min, z_min, x_max, y_max, z_max]


@dataclass
class TensorizerOutput:
    """A batch of CachedSamples plus tokenized text. Every field has a leading batch dim."""

    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor
    obj_clip_features: torch.Tensor
    obj_bboxes: torch.Tensor
    obj_is_anchor: torch.Tensor
    obj_padding_mask: torch.Tensor
    coord_shift: torch.Tensor
    coord_scale: torch.Tensor
    target_xyz_world: torch.Tensor
    target_bbox_world: torch.Tensor


def bbox_to_aabb(bbox) -> np.ndarray:
    """Return a (6,) [min, max] AABB from either a 6-vector or 8x3 corners."""
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size == 6:
        return arr
    if arr.size == 24:
        corners = arr.reshape(8, 3)
        return np.concatenate([corners.min(axis=0), corners.max(axis=0)]).astype(np.float32)
    raise ValueError(f"bbox must have 6 (AABB) or 24 (8 corners) values, got {arr.size}")


def target_aabb(target_bbox, target_xyz) -> np.ndarray:
    """The target's world AABB, or a 0.2 m cube around its centre when no bbox is known."""
    if target_bbox is not None:
        return bbox_to_aabb(target_bbox)
    xyz = np.asarray(target_xyz, dtype=np.float32)
    eps = np.full(3, 0.1, dtype=np.float32)
    return np.concatenate([xyz - eps, xyz + eps])
