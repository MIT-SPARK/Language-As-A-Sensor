"""Beliefs and their fusion on a voxel grid.

VLMaP fuses sensors as a product of experts in log space,

    log p(x) = Σ_i w_i log p_i(x) - log Z,

normalised over a fixed grid covering the scene.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

from langsensor.vlmap.types import BeliefModel

_EPS = 1e-12


def _to_numpy(a) -> np.ndarray:
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


class GMMBelief(BeliefModel):
    """A language belief: Gaussian mixture density (world frame).

    Results are memoised for the last two query arrays, because the same belief
    is evaluated on the same fusion grid every frame until the next utterance.
    """

    _LOG2PI = math.log(2.0 * math.pi)

    def __init__(self, mus, Ls, weights) -> None:
        self.mus = _to_numpy(mus).astype(np.float32)                   # (K, 3)
        Ls64 = _to_numpy(Ls).astype(np.float64)                        # (K, 3, 3)
        self.Ls = Ls64.astype(np.float32)
        w = _to_numpy(weights).astype(np.float64)
        w = w / w.sum()
        self.weights = w.astype(np.float32)

        self._L_inv = np.linalg.inv(Ls64).astype(np.float32)
        self._log_w = np.log(w + 1e-12).astype(np.float32)
        _, log_dets = np.linalg.slogdet(Ls64 @ Ls64.transpose(0, 2, 1))
        self._log_norm = (-0.5 * (3.0 * self._LOG2PI + log_dets.astype(np.float32))).astype(np.float32)
        self._cache: OrderedDict = OrderedDict()

    def query_batch(self, positions: np.ndarray) -> np.ndarray:
        key = (id(positions), positions.shape)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pts = np.asarray(positions, dtype=np.float32)
        log_p = np.empty((len(self.mus), len(pts)), dtype=np.float32)
        for k in range(len(self.mus)):
            z = (pts - self.mus[k]) @ self._L_inv[k].T
            log_p[k] = self._log_w[k] + self._log_norm[k] - 0.5 * np.einsum("ni,ni->n", z, z)
        result = np.exp(logsumexp(log_p, axis=0)).astype(np.float32)
        if len(self._cache) >= 2:
            self._cache.popitem(last=False)
        self._cache[key] = result
        return result


def _log_product(beliefs: list[BeliefModel], pts: np.ndarray, weights: Sequence[float] | None) -> np.ndarray:
    log_sum = np.zeros(len(pts), dtype=np.float32)
    for i, b in enumerate(beliefs):
        w = np.float32(1.0 if weights is None else weights[i])
        log_sum += w * np.log(b.query_batch(pts).astype(np.float32, copy=False) + np.float32(_EPS))
    return log_sum


class FusedBelief(BeliefModel):
    """Normalised product of beliefs; queries on the fusion grid itself return the cached result."""

    def __init__(self, beliefs, log_Z: float, weights, grid: np.ndarray | None, grid_probs: np.ndarray | None):
        self._beliefs, self._log_Z, self._weights = beliefs, log_Z, weights
        self._grid_id = id(grid) if grid is not None else None
        self._grid_probs = grid_probs

    def query_batch(self, positions: np.ndarray) -> np.ndarray:
        if (self._grid_probs is not None and id(positions) == self._grid_id
                and positions.shape[0] == self._grid_probs.shape[0]):
            return self._grid_probs
        log_p = _log_product(self._beliefs, np.asarray(positions, dtype=np.float32), self._weights)
        return np.exp(log_p - self._log_Z).astype(np.float32)


class FusionPipeline:
    """Log-product fusion normalised over a grid spanning ``scene_bounds`` x ``z_bounds``."""

    def __init__(self, scene_bounds: np.ndarray, z_bounds: tuple[float, float], resolution: float = 0.1) -> None:
        xs = np.arange(scene_bounds[0], scene_bounds[1], resolution, dtype=np.float32)
        ys = np.arange(scene_bounds[2], scene_bounds[3], resolution, dtype=np.float32)
        zs = np.arange(z_bounds[0], z_bounds[1], resolution, dtype=np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])       # (N, 3) float32
        self.resolution = resolution

    def step(self, beliefs: list[BeliefModel], weights: Sequence[float] | None = None) -> BeliefModel:
        if len(beliefs) == 1:
            return beliefs[0]
        log_vals = _log_product(beliefs, self.grid, weights)
        log_Z = float(logsumexp(log_vals))
        return FusedBelief(beliefs, log_Z, weights, self.grid, np.exp(log_vals - log_Z).astype(np.float32))
