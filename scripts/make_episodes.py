"""Build closed-loop language episodes for Matterport scenes that have a recorded trajectory.

For each scene this (1) links the VLA-3D scene files into ``<scene>/scene/``,
(2) samples referential statements into episode bundles under
``<scene>/episodes/`` (target removed from the scene graph), and (3) extracts
the target's surface points (``gt_surface.npy``) used by the metrics.

Defaults are the settings of the paper's 54 episodes over 13 scenes.

    python scripts/make_episodes.py --closed-loop-root data/closed_loop --vla3d-root data/VLA-3D
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from langsensor import paths
from langsensor.core.ontology import VALID_RELATIONS, semantic_label
from langsensor.core.transforms import build_spatial_query
from langsensor.data.vla3d import VLA3DScene, expand_split
from langsensor.vlmap.episodes import load_ply_xyz

SCENE_FILES = ("_pc_result.ply", "_object_split.npy", "_scene_graph.json")
PAPER_SCENES = ["17DRP5sb8fy", "2t7WUuJeko7", "GdvgFV5R1Z5", "gZ6f7yhEvPG", "HxpKQynjfin", "JF19kD82Mey",
                "jh4fc5c5qoQ", "pLe4wQe7qrG", "Pm6F8kyY3z2", "RPmz2sHmrrY", "x8F5xyUWy9e", "XcA2TqTSSAj", "YmJkqBEsHnH"]
PAPER_TARGETS = ["bed", "sofa", "desk", "cabinet", "table", "toilet", "television", "mirror", "chair",
                 "bookshelf", "dresser", "refrigerator"]


def flat_scene_graph(sg) -> dict:
    """The episode scene-graph format: 6-element AABBs and a region id per object."""
    objects = []
    for o in sg.objects:
        d = {"id": o.id, "label": o.label, "position": list(o.position)}
        if o.bbox is not None:
            c = np.array(o.bbox, dtype=np.float32).reshape(-1, 3)
            d["bbox"] = [float(v) for v in (*c.min(axis=0), *c.max(axis=0))]
        if o.metadata.get("region_id") is not None:
            d["region_id"] = int(o.metadata["region_id"])
        if o.metadata.get("nyu40_label"):
            d["nyu40_label"] = str(o.metadata["nyu40_label"])
        objects.append(d)
    return {"objects": objects, "regions": [{"id": r.id, "label": r.label, "position": list(r.position)} for r in sg.regions]}


def sample_statements(scene: VLA3DScene, args, rng: random.Random):
    sg = scene.load_scene_graph()
    objs = {o.id: o for o in sg.objects}

    def keep(s) -> bool:
        target = objs.get(s.target_object_id)
        if not (args.ambiguity_min <= s.ambiguity <= args.ambiguity_max) or s.relation not in VALID_RELATIONS:
            return False
        if target is None or (target.metadata.get("nyu40_label") or "").lower() not in args.target_labels:
            return False
        if target.bbox is None:
            return False
        c = np.asarray(target.bbox, dtype=np.float32).reshape(-1, 3)
        return float(np.prod(np.maximum(c.max(axis=0) - c.min(axis=0), 0))) >= args.min_target_volume \
            and s.relation in args.relation_types

    by_target: dict[int, list] = {}
    for s in scene.load_statements(sg):
        if keep(s):
            by_target.setdefault(s.target_object_id, []).append(s)
    unique = [rng.choice(group) for group in by_target.values()]         # one statement per target object
    return sg, rng.sample(unique, min(args.n_samples, len(unique)))


def target_surface(xyz: np.ndarray, per_point: np.ndarray, target_id: int, resolution: float) -> np.ndarray:
    """The target's points, one per ``resolution``-sized voxel."""
    pts = xyz[per_point == target_id]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    _, idx = np.unique(np.floor(pts / resolution).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(idx)].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closed-loop-root", type=Path, default=paths.CLOSED_LOOP_ROOT)
    ap.add_argument("--vla3d-root", type=Path, default=paths.VLA3D_ROOT)
    ap.add_argument("--scene-ids", nargs="+", default=PAPER_SCENES)
    ap.add_argument("--ambiguity-min", type=int, default=1)
    ap.add_argument("--ambiguity-max", type=int, default=4)
    ap.add_argument("--target-labels", nargs="+", default=PAPER_TARGETS)
    ap.add_argument("--relation-types", nargs="+", default=["near", "between", "above", "below", "on"])
    ap.add_argument("--min-target-volume", type=float, default=0.4, help="m³; smaller targets barely register in 0.1 m voxels")
    ap.add_argument("--n-samples", type=int, default=10, help="max episodes per scene")
    ap.add_argument("--seed", type=int, default=121)
    ap.add_argument("--surface-resolution", type=float, default=0.025)
    args = ap.parse_args()

    for sid in args.scene_ids:
        scene_dir = args.closed_loop_root / sid
        src = args.vla3d_root / "Matterport" / sid
        (scene_dir / "scene").mkdir(parents=True, exist_ok=True)
        for suffix in SCENE_FILES:
            link = scene_dir / "scene" / f"{sid}{suffix}"
            if not link.exists():
                link.symlink_to((src / f"{sid}{suffix}").resolve())

        # A fresh RNG per scene, as the paper's episodes were generated one scene per run.
        sg, statements = sample_statements(VLA3DScene(src), args, random.Random(args.seed))
        labels = {o.id: semantic_label(o.metadata).lower().strip() for o in sg.objects}
        xyz = load_ply_xyz(scene_dir / "scene" / f"{sid}_pc_result.ply")
        per_point = expand_split(np.load(scene_dir / "scene" / f"{sid}_object_split.npy"))

        bundle = scene_dir / "episodes"
        bundle.mkdir(exist_ok=True)
        manifest = []
        for i, s in enumerate(statements):
            ep_id = f"{sid}_{i:07d}"
            ep_dir = bundle / sid / ep_id
            ep_dir.mkdir(parents=True, exist_ok=True)
            q = build_spatial_query(sid, sg, s)
            (ep_dir / "metadata.json").write_text(json.dumps({
                "episode_id": ep_id, "scene_id": sid, "target_object_id": s.target_object_id,
                "target_label": labels.get(s.target_object_id, ""),
                "target_xyz": q.target_xyz.tolist(),
                "target_bbox": q.target_bbox.tolist() if q.target_bbox is not None else None,
                "utterances": [{"timestep": 0, "text": s.text, "anchor_object_ids": s.anchor_object_id or [],
                                "ambiguity": s.ambiguity, "relation": s.relation,
                                "region_id": int(s.region[0]) if s.region else None}],
            }, indent=2))
            (ep_dir / "scene_graph.json").write_text(json.dumps(flat_scene_graph(q.scene_graph), indent=2))
            np.save(ep_dir / "gt_surface.npy", target_surface(xyz, per_point, s.target_object_id, args.surface_resolution))
            manifest.append({"episode_id": ep_id, "scene_id": sid})
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{sid}: {len(manifest)} episodes")


if __name__ == "__main__":
    main()
