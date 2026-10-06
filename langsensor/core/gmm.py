"""Gaussian mixture over a 3D position: the output of every language predictor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from langsensor.core.schema import Grounding


@dataclass
class GMMResult:
    groundings: list[Grounding]
    mus: torch.Tensor       # (K, 3) component means, world frame
    Ls: torch.Tensor        # (K, 3, 3) lower-triangular Cholesky factors, Σ = L Lᵀ
    weights: torch.Tensor   # (K,) mixture weights, sum to 1

    def _components(self):
        for k in range(len(self.weights)):
            yield torch.distributions.MultivariateNormal(self.mus[k], scale_tril=self.Ls[k])

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw *n* points, (n, 3) float32."""
        rng = np.random.RandomState(seed)
        counts = rng.multinomial(n, self.weights.numpy())
        parts = [dist.sample((int(c),)) for dist, c in zip(self._components(), counts) if c > 0]
        return torch.cat(parts).numpy() if parts else np.zeros((0, 3), dtype=np.float32)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density at *x*, shape (*, 3) -> (*)."""
        log_w = torch.log(self.weights.clamp(min=1e-12))
        per_component = torch.stack([d.log_prob(x) for d in self._components()], dim=-1)
        return torch.logsumexp(per_component + log_w, dim=-1)

    def cdf_bbox(self, bbox: np.ndarray) -> float:
        """Mass inside an AABB [xmin, ymin, zmin, xmax, ymax, zmax], product-of-marginals."""
        box = torch.as_tensor(bbox, dtype=torch.float32)
        lo, hi = box[:3], box[3:]
        total = 0.0
        for k in range(len(self.weights)):
            # Marginal std is sqrt(diag(L Lᵀ)), the row norms of L -- not diag(L).
            sigma = (self.Ls[k] ** 2).sum(dim=-1).clamp(min=1e-12).sqrt()
            z_lo = (lo - self.mus[k]) / (sigma * math.sqrt(2.0))
            z_hi = (hi - self.mus[k]) / (sigma * math.sqrt(2.0))
            total += float(self.weights[k]) * float((0.5 * (torch.erf(z_hi) - torch.erf(z_lo))).prod())
        return total

    @property
    def mean(self) -> np.ndarray:
        return (self.weights.unsqueeze(-1) * self.mus).sum(0).numpy()
