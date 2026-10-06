"""Load closed-loop scenes: an RGB-D trajectory plus language episodes (see docs/data.md).

    <scene_dir>/
      trajectory/  camera_info.yaml, poses.csv, color/rgb_%07d.png, depth/depth_%07d.tiff, labels/labels_%07d.png
      scene/       <sid>_pc_result.ply, <sid>_object_split.npy, <sid>_scene_graph.json   (VLA-3D Matterport)
      episodes/    manifest.json, <sid>/<episode_id>/{metadata.json, scene_graph.json, gt_surface.npy}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from langsensor.vlmap.types import (
    LanguageEpisode,
    ObjectEntry,
    RegionEntry,
    SceneGraph,
    TimestampedUtterance,
    VisionObservation,
)

log = logging.getLogger(__name__)

# 90° rotation about z that aligns the trajectory poses with the VLA-3D point-cloud frame.
_T_WORLD_FIX = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def load_ply_xyz(path: Path) -> np.ndarray:
    """Vertices of a binary little-endian PLY with (x, y, z float32, r, g, b uint8)."""
    with open(path, "rb") as f:
        n = 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


class Trajectory:
    """Posed RGB-D frames with ground-truth semantics, read on demand."""

    def __init__(self, scene_dir: str | Path, max_steps: int | None = None) -> None:
        import pandas as pd

        self.dir = Path(scene_dir) / "trajectory"
        poses = pd.read_csv(self.dir / "poses.csv")
        self.poses = poses.iloc[:max_steps] if max_steps else poses
        self.camera_info = yaml.safe_load((self.dir / "camera_info.yaml").read_text())

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, t: int) -> VisionObservation:
        import tifffile
        from PIL import Image

        row = self.poses.iloc[t]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
        T[:3, 3] = [row.tx, row.ty, row.tz]
        return VisionObservation(
            timestep=t,
            rgb=np.array(Image.open(self.dir / "color" / f"rgb_{t:07d}.png")),
            depth=tifffile.imread(self.dir / "depth" / f"depth_{t:07d}.tiff").astype(np.float32),
            pose=_T_WORLD_FIX @ T,
            gt_semantics=np.array(Image.open(self.dir / "labels" / f"labels_{t:07d}.png")).astype(np.int32),
        )


def scene_extent(scene_dir: str | Path) -> tuple[np.ndarray, tuple[float, float]]:
    """([x_min, x_max, y_min, y_max], (z_min, z_max)) of the scene point cloud."""
    scene_dir = Path(scene_dir)
    ply = sorted((scene_dir / "scene").glob("*_pc_result.ply"))
    if not ply:
        raise FileNotFoundError(f"no *_pc_result.ply under {scene_dir / 'scene'}")
    xyz = load_ply_xyz(ply[0])
    bounds = np.array([xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].min(), xyz[:, 1].max()], dtype=np.float32)
    return bounds, (float(xyz[:, 2].min()), float(xyz[:, 2].max()))


def _scene_graph(path: Path) -> SceneGraph:
    data = json.loads(path.read_text())
    return SceneGraph(
        objects=tuple(ObjectEntry(
            id=int(o["id"]), label=str(o["label"]), position=tuple(float(v) for v in o["position"]),
            bbox=tuple(float(v) for v in o["bbox"]) if o.get("bbox") else None,
            region_id=int(o["region_id"]) if o.get("region_id") is not None else None,
            nyu40_label=str(o["nyu40_label"]) if o.get("nyu40_label") else None,
        ) for o in data.get("objects", [])),
        regions=tuple(RegionEntry(int(r["id"]), str(r["label"]), tuple(float(v) for v in r["position"]))
                      for r in data.get("regions", [])),
    )


def load_episodes(scene_dir: str | Path) -> list[LanguageEpisode]:
    bundle = Path(scene_dir) / "episodes"
    manifest = bundle / "manifest.json"
    if not manifest.exists():
        return []
    episodes = []
    for entry in json.loads(manifest.read_text()):
        ep_dir = bundle / entry["scene_id"] / entry["episode_id"]
        meta = json.loads((ep_dir / "metadata.json").read_text())
        surface = ep_dir / "gt_surface.npy"
        surface_xyz = np.load(surface).astype(np.float32) if surface.exists() else None
        episodes.append(LanguageEpisode(
            episode_id=str(meta["episode_id"]),
            utterances=tuple(TimestampedUtterance(
                int(u["timestep"]), str(u["text"]), int(u.get("ambiguity", 0)),
                tuple(int(x) for x in u.get("anchor_object_ids", [])),
            ) for u in meta["utterances"]),
            scene_graph=_scene_graph(ep_dir / "scene_graph.json"),
            target_xyz=np.array(meta["target_xyz"], dtype=np.float32),
            target_label=str(meta.get("target_label", "")),
            target_bbox=np.array(meta["target_bbox"], dtype=np.float32) if meta.get("target_bbox") else None,
            target_surface_xyz=surface_xyz if surface_xyz is not None and surface_xyz.size else None,
        ))
    log.info("loaded %d episodes from %s", len(episodes), bundle)
    return episodes
