from __future__ import annotations

import numpy as np

from langsensor.core.schema import ReferentialStatement, SceneGraph, SpatialQuery


def build_spatial_query(
    scene_id: str,
    scene_graph: SceneGraph,
    statement: ReferentialStatement,
    points: np.ndarray | None = None,
    object_split: np.ndarray | None = None,
) -> SpatialQuery:
    """Turn a statement into a query whose scene no longer contains the target.

    The target's position and bbox become supervision; the target is removed
    from the scene graph (and from the point cloud, when one is given).
    """
    target_id = statement.target_object_id
    target = next((o for o in scene_graph.objects if o.id == target_id), None)
    if target is None:
        raise ValueError(f"target_object_id {target_id} not in scene graph of '{scene_id}'")

    masked_graph = SceneGraph(
        objects=[o for o in scene_graph.objects if o.id != target_id],
        regions=scene_graph.regions,
        relations=scene_graph.relations,
    )
    pc = split = None
    if points is not None and object_split is not None:
        keep = object_split != target_id
        pc, split = points[keep], object_split[keep]

    return SpatialQuery(
        scene_id=scene_id,
        scene_graph=masked_graph,
        pc=pc,
        object_split=split,
        language=statement.text,
        target_xyz=np.array(target.position, dtype=np.float32),
        target_bbox=np.array(target.bbox, dtype=np.float32) if target.bbox is not None else None,
        gt_anchor_object_ids=[int(a) for a in statement.anchor_object_id] if statement.anchor_object_id else None,
        gt_anchor_room_id=int(statement.region[0]) if statement.region else None,
    )
