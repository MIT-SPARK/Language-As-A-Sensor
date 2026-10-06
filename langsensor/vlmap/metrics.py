"""Per-frame quality of the fused belief in one episode.

The belief is normalised over the fusion grid. The target is represented by its
surface points; a grid voxel "is on the target" if it lies within
``surface_radius`` of one. Recorded each frame:

    surface_mass            P(on target)                         (Table 2, Fig 4b)
    info_gain_at_surface    log(P(on target) / P_uniform(on target)), nats  (Table 2, Fig 4a)
    argmax_dist_to_surface  distance from the belief's mode to the target surface (Table 2)
    entropy, gt_likelihood, argmax_value, argmax_dist_to_gt, vision_surface_mass
"""

from __future__ import annotations


import numpy as np
from scipy.spatial import cKDTree

from langsensor.vlmap.types import BeliefModel

_EPS = 1e-12


def _normalise(raw: np.ndarray) -> np.ndarray:
    total = float(raw.sum())
    return raw / np.float32(total) if total > _EPS else np.full(len(raw), np.float32(1.0 / len(raw)), dtype=np.float32)


class MetricsTracker:
    def __init__(self, grid: np.ndarray, target_xyz: np.ndarray, target_surface_xyz: np.ndarray | None,
                 surface_radius: float = 0.15) -> None:
        self.grid = np.asarray(grid, dtype=np.float32)
        self.target = np.asarray(target_xyz, dtype=np.float32)
        self._gt_idx = int(np.argmin(np.sum((self.grid - self.target[None]) ** 2, axis=1)))
        self.series: dict[str, list[float]] = {k: [] for k in (
            "entropy", "gt_likelihood", "argmax_value", "argmax_dist_to_gt", "surface_mass",
            "info_gain_at_surface", "argmax_dist_to_surface", "vision_surface_mass")}
        self._surface_tree = None
        if target_surface_xyz is not None and len(target_surface_xyz):
            self._surface_tree = cKDTree(np.asarray(target_surface_xyz, dtype=np.float32))
            self._on_surface = self._surface_tree.query(self.grid, k=1)[0] <= float(surface_radius)
            self._p_uniform = float(self._on_surface.sum()) / len(self.grid)

    def record(self, belief: BeliefModel, vision_belief: BeliefModel | None = None) -> None:
        probs = _normalise(belief.query_batch(self.grid))
        s = self.series
        s["entropy"].append(-float(np.sum(probs * np.log(probs + _EPS))))
        s["gt_likelihood"].append(float(probs[self._gt_idx]))
        mode = int(np.argmax(probs))
        s["argmax_value"].append(float(probs[mode]))
        s["argmax_dist_to_gt"].append(float(np.linalg.norm(self.grid[mode] - self.target)))
        if self._surface_tree is None:
            return
        mass = float(probs[self._on_surface].sum())
        s["surface_mass"].append(mass)
        s["info_gain_at_surface"].append(float(np.log(max(mass, _EPS) / max(self._p_uniform, _EPS)))
                                         if self._p_uniform > 0 else float("nan"))
        s["argmax_dist_to_surface"].append(float(self._surface_tree.query(self.grid[mode][None], k=1)[0][0]))
        if vision_belief is None:
            s["vision_surface_mass"].append(float("nan"))
        else:
            vis = vision_belief.query_batch(self.grid)
            total = float(vis.sum())
            s["vision_surface_mass"].append(float((vis / np.float32(total))[self._on_surface].sum()) if total > _EPS else 0.0)

    def record_dict(self, scene_id: str, episode, success_threshold: float) -> dict:
        """One entry of metrics_timeseries.json."""
        sm = self.series["surface_mass"]
        terminal = sm[-1] if sm else (self.series["gt_likelihood"][-1] if self.series["gt_likelihood"] else None)
        return {
            "scene_id": scene_id,
            "episode_id": episode.episode_id,
            "target_label": episode.target_label,
            "utterances": [u.text for u in episode.utterances],
            "timesteps": list(range(len(self.series["entropy"]))),
            **{k: (v if v else None) for k, v in self.series.items()},
            "success": terminal is not None and terminal >= success_threshold,
            "success_threshold": success_threshold,
            "terminal_surface_mass": terminal,
        }
