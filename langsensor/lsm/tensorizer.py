"""Grounded queries -> model-ready tensors (text is tokenized later, in CollateFn)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from langsensor.core.schema import CachedSample, GroundedQuery, SceneGraph, target_aabb

_REGION_BBOX_KEYS = ("bbox_x_min", "bbox_y_min", "bbox_z_min", "bbox_x_max", "bbox_y_max", "bbox_z_max")


def build_clip_label_map(labels: list[str], clip_model: str = "ViT-B/32", device: str = "cpu") -> dict[str, np.ndarray]:
    """Embed each object label with the frozen CLIP text encoder (L2-normalised, 512-d)."""
    import clip

    model, _ = clip.load(clip_model, device=device)
    model.eval()
    with torch.no_grad():
        feats = model.encode_text(clip.tokenize(labels).to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return {label: f.cpu().float().numpy() for label, f in zip(labels, feats)}


def region_frame(scene_graph: SceneGraph, region_id: int, scene_id: str = "") -> tuple[np.ndarray, np.ndarray]:
    """(shift, scale) of the anchor region's bbox: p_region = (p_world - shift) / scale."""
    for region in scene_graph.regions:
        if region.id == region_id and all(k in region.metadata for k in _REGION_BBOX_KEYS):
            bbox = np.array([region.metadata[k] for k in _REGION_BBOX_KEYS], dtype=np.float32)
            shift = ((bbox[:3] + bbox[3:]) / 2.0).astype(np.float32)
            scale = np.maximum(bbox[3:] - bbox[:3], 1e-3).astype(np.float32)
            return shift, scale
    raise ValueError(f"Region {region_id} not found in scene graph of '{scene_id}'")


class Tensorizer:
    """Crop the scene to the grounding's region, normalise to its frame, pad to max_objects.

    Args:
        max_objects:        Objects kept per region (the rest are dropped, in scene-graph order).
        clip_embedding_map: label -> (clip_dim,) embedding; unknown labels get zeros.
    """

    def __init__(self, max_objects: int = 150, clip_embedding_map: dict[str, np.ndarray] | None = None,
                 clip_dim: int = 512) -> None:
        self.max_objects = max_objects
        self.clip_embedding_map = clip_embedding_map or {}
        self.clip_dim = clip_dim

    def __call__(self, queries: Sequence[GroundedQuery]) -> list[CachedSample]:
        return [self._one(q) for q in queries]

    def _one(self, q: GroundedQuery) -> CachedSample:
        N, g = self.max_objects, q.grounding
        shift, scale = region_frame(q.scene_graph, g.anchor_room_id, q.scene_id)
        anchors = set(g.anchor_object_ids)

        clip = np.zeros((N, self.clip_dim), dtype=np.float32)
        boxes = np.zeros((N, 6), dtype=np.float32)
        is_anchor = np.zeros(N, dtype=bool)
        padding = np.ones(N, dtype=bool)

        objs = [o for o in q.scene_graph.objects if o.metadata.get("region_id") == g.anchor_room_id][:N]
        for i, obj in enumerate(objs):
            clip[i] = self.clip_embedding_map.get(obj.label, np.zeros(self.clip_dim, dtype=np.float32))
            if obj.bbox is not None:
                corners = np.array(obj.bbox, dtype=np.float32).reshape(8, 3)
                center, size = corners.mean(axis=0), corners.max(axis=0) - corners.min(axis=0)
            else:
                center, size = np.array(obj.position, dtype=np.float32), np.zeros(3, dtype=np.float32)
            boxes[i] = np.concatenate([(center - shift) / scale, size / scale])
            is_anchor[i] = obj.id in anchors
            padding[i] = False

        if q.target_xyz is None and q.target_bbox is None:     # pure inference: no supervision
            tgt_xyz, tgt_box = np.zeros(3, dtype=np.float32), np.zeros(6, dtype=np.float32)
        else:
            tgt_xyz = (np.asarray(q.target_xyz, dtype=np.float32) if q.target_xyz is not None
                       else np.zeros(3, dtype=np.float32))
            tgt_box = target_aabb(q.target_bbox, q.target_xyz)

        return CachedSample(
            language=g.language,
            obj_clip_features=torch.from_numpy(clip),
            obj_bboxes=torch.from_numpy(boxes),
            obj_is_anchor=torch.from_numpy(is_anchor),
            obj_padding_mask=torch.from_numpy(padding),
            coord_shift=torch.from_numpy(shift),
            coord_scale=torch.from_numpy(scale),
            target_xyz_world=torch.from_numpy(tgt_xyz),
            target_bbox_world=torch.from_numpy(tgt_box),
        )
