"""Semantic vision sensor: a voxel map's label histograms -> a likelihood of the target at each voxel.

For every filled voxel v with label weights P(c | v) and TSDF distance d:

    score(v) = Σ_c P(c | v) exp(s_c) / Σ_{c in label space} P(c | v)  ·  σ(-d / τ),
    s_c = max(cos(φ(c), φ("a <target>")), 0)

where φ is a sentence embedding (all-MiniLM-L6-v2). The first factor says "this
voxel looks like the target", the second "this voxel is on a surface". Scores
are EMA-blended into a dense grid over the scene; never-observed voxels get a
uniform 1/N_voxels ("uniform" mode) so unseen space neither wins nor vanishes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from scipy.special import log_expit, logsumexp

from langsensor.vlmap.types import BeliefModel, VisionObservation
from langsensor.vlmap.voxel_map import VoxelMap

log = logging.getLogger(__name__)

_TINY = np.finfo(np.float64).tiny
_LOG_TINY = np.log(_TINY)


class VoxelBelief(BeliefModel):
    """View of the sensor's dense score grid. Shares its buffers: consume before the next frame."""

    def __init__(self, score: np.ndarray, observed: np.ndarray, voxel_size: float, offset: np.ndarray,
                 unobserved_score: float, grid_cache: dict | None = None) -> None:
        self._score, self._observed = score, observed
        self._voxel_size = np.float32(voxel_size)
        self._offset = offset
        self._shape = np.array(score.shape, dtype=np.int32)
        self._unobserved = np.float32(unobserved_score)
        self._grid_cache = grid_cache

    def _indices(self, positions: np.ndarray):
        local = np.floor(np.asarray(positions, dtype=np.float32) / self._voxel_size).astype(np.int32) - self._offset
        return local, np.all((local >= 0) & (local < self._shape), axis=1)

    def query_batch(self, positions: np.ndarray) -> np.ndarray:
        c = self._grid_cache
        if c is not None and id(positions) == c["id"] and positions.shape[0] == c["in_bounds"].shape[0]:
            result = np.full(len(positions), self._unobserved, dtype=np.float32)
            idx = c["linear_idx"]
            if idx.size:
                result[c["in_bounds"]] = np.where(self._observed.reshape(-1)[idx], self._score.reshape(-1)[idx],
                                                  self._unobserved)
            return result
        local, inb = self._indices(positions)
        result = np.full(len(positions), self._unobserved, dtype=np.float32)
        if inb.any():
            i = local[inb]
            result[inb] = np.where(self._observed[i[:, 0], i[:, 1], i[:, 2]],
                                   self._score[i[:, 0], i[:, 1], i[:, 2]], self._unobserved)
        return result

    def coverage_batch(self, positions: np.ndarray) -> np.ndarray:
        local, inb = self._indices(positions)
        result = np.zeros(len(positions), dtype=bool)
        if inb.any():
            i = local[inb]
            result[inb] = self._observed[i[:, 0], i[:, 1], i[:, 2]]
        return result


def sbert_embedder(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda text: model.encode(text, normalize_embeddings=True)


class SemanticVisionSensor:
    """Turns a :class:`VoxelMap` into a per-frame :class:`VoxelBelief` about one target label."""

    _BLOCK_PADDING = 1   # extra map blocks around the scene bounds

    def __init__(
        self,
        voxel_map: VoxelMap,
        target_label: str,
        label_names: dict[int, str],
        scene_bounds: Sequence[float],       # [x_min, x_max, y_min, y_max]
        z_bounds: Sequence[float],
        voxel_size: float = 0.1,
        voxels_per_side: int = 16,
        tsdf_sigma: float | None = 0.15,  # None -> the map's truncation distance
        score_ema_alpha: float = 0.25,
        unobserved_mode: str = "uniform",    # "uniform" (1/N_voxels) | "fixed"
        unobserved_score: float = 0.01,
        embed=None,
    ) -> None:
        if unobserved_mode not in ("fixed", "uniform"):
            raise ValueError(f"unobserved_mode must be 'fixed' or 'uniform', got {unobserved_mode!r}")
        self.map = voxel_map
        self._voxel_size = float(voxel_size)
        self._alpha = np.float32(score_ema_alpha)
        self._sigma = float(tsdf_sigma) if tsdf_sigma else float(voxel_map.truncation_distance)
        self._sim = self._label_similarity(label_names, target_label, embed or sbert_embedder())

        # Dense grid over the scene, padded and rounded up to whole map blocks.
        V = int(voxels_per_side)
        lo = np.floor(np.array([scene_bounds[0], scene_bounds[2], z_bounds[0]], dtype=np.float64) / voxel_size)
        hi = np.ceil(np.array([scene_bounds[1], scene_bounds[3], z_bounds[1]], dtype=np.float64) / voxel_size)
        lo, hi = lo.astype(np.int64) - self._BLOCK_PADDING * V, hi.astype(np.int64) + self._BLOCK_PADDING * V
        shape = tuple(int(s) for s in ((hi - lo + V - 1) // V * V))
        self._offset = lo.astype(np.int32)
        self._unobserved = 1.0 / float(np.prod(shape, dtype=np.int64)) if unobserved_mode == "uniform" \
            else float(unobserved_score)
        self._score = np.full(shape, self._unobserved, dtype=np.float32)
        self._observed = np.zeros(shape, dtype=bool)
        self._grid_cache: dict | None = None

    @staticmethod
    def _label_similarity(label_names: dict[int, str], target_label: str, embed) -> np.ndarray:
        """Per-label log-weight s_c = max(cos(φ(label), φ("a <target>")), tiny); unknown ids get log(tiny)."""
        target = embed(f"a {target_label}")
        target_norm = np.linalg.norm(target) + 1e-8
        table = np.full(max(label_names) + 1 if label_names else 1, _LOG_TINY, dtype=np.float64)
        for label_id, name in label_names.items():
            e = embed(name)
            table[label_id] = max(np.dot(e, target) / ((np.linalg.norm(e) + 1e-8) * target_norm), _TINY)
        return table

    def set_query_grid(self, grid: np.ndarray) -> None:
        """Precompute flat indices for a fixed query grid (the fusion grid) so per-frame queries are cheap."""
        local = np.floor(np.asarray(grid, dtype=np.float32) / self._voxel_size).astype(np.int64) - self._offset
        inb = np.all((local >= 0) & (local < np.asarray(self._score.shape)), axis=1)
        idx = np.ravel_multi_index(local[inb].T, self._score.shape).astype(np.intp) if inb.any() \
            else np.empty(0, dtype=np.intp)
        self._grid_cache = {"id": id(grid), "in_bounds": inb, "linear_idx": idx}

    def belief(self) -> VoxelBelief:
        return VoxelBelief(self._score, self._observed, self._voxel_size, self._offset, self._unobserved,
                           self._grid_cache)

    def log_semantic(self, weights: np.ndarray, label_ids: np.ndarray) -> np.ndarray:
        """log of the semantic factor for each voxel; -inf where the voxel has no label weight."""
        w = weights.astype(np.float64, copy=False)
        ids = label_ids.astype(np.int64, copy=False)
        totals = w.sum(axis=1)
        valid = totals > 1e-8
        log_p = np.where(w > 0, np.log(np.maximum(w / np.where(valid, totals, 1.0)[:, None], _TINY)), _LOG_TINY)
        in_range = (ids >= 0) & (ids < len(self._sim))
        log_sim = np.where(in_range, self._sim[np.where(in_range, ids, 0)], _LOG_TINY)
        log_in_space = np.where(in_range, 0.0, _LOG_TINY)
        score = logsumexp(log_p + log_sim, axis=1) - logsumexp(log_p + log_in_space, axis=1)
        return np.where(valid, score, -np.inf)

    def __call__(self, obs: VisionObservation) -> VoxelBelief:
        try:
            batch = self.map.integrate(obs)
        except Exception as e:                   # keep the previous grid rather than aborting the episode
            log.warning("voxel map integration failed at t=%d: %s", obs.timestep, e)
            return self.belief()
        if batch.keys.shape[0] == 0:
            return self.belief()

        log_sem = self.log_semantic(batch.weights, batch.label_ids)
        log_occ = np.where(batch.has_tsdf, log_expit(-batch.tsdf_distance.astype(np.float64) / self._sigma), 0.0)
        # Voxels without semantic evidence rely on occupancy alone.
        new = np.exp(np.where(batch.has_semantic & (log_sem > -np.inf), log_sem + log_occ, log_occ)).astype(np.float32)

        local = batch.keys.astype(np.int64) - self._offset.astype(np.int64)
        inb = np.all((local >= 0) & (local < np.asarray(self._score.shape, dtype=np.int64)), axis=1)
        local, new = local[inb], new[inb]
        if local.shape[0] == 0:
            return self.belief()
        ix, iy, iz = local[:, 0], local[:, 1], local[:, 2]
        prev_obs = self._observed[ix, iy, iz]
        blended = np.where(prev_obs, self._alpha * new + (np.float32(1.0) - self._alpha) * self._score[ix, iy, iz], new)
        self._score[ix, iy, iz] = blended.astype(np.float32)
        self._observed[ix, iy, iz] = True
        return self.belief()
