"""Post-hoc covariance rescaling of a predictor, fitted on held-out data.

With residual r = y - μ and whitened residual z = L⁻¹ r (Σ = L Lᵀ):

    scale       L' = s · L          s²  = mean(‖z‖²) / 3        ("global rescale")
    axis_scale  L' = L · diag(s)    s_j² = mean(z_j²)           ("per-axis rescale")

Both are the exact maximum-likelihood fit in their family, so ANEES on the
calibration set is 3 by construction. ``axis_scale`` rescales the base
predictor's own whitened axes (right-multiplication), which is what makes it
separable and closed-form. Means are never changed.

A fit is cached as JSON and silently reused; delete the file to refit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from langsensor.core.schema import Grounding, SpatialQuery
from langsensor.predictors.base import DistributionPredictor, Proposer

MODES = ("scale", "axis_scale")
_MIN_PARAM = 1e-3


@dataclass
class CovarianceCalibration:
    mode: str
    params: list[float]
    n_calibration: int
    predictor_name: str
    anees_before: float = float("nan")
    anees_after: float = float("nan")


def fit_params(mode: str, z: np.ndarray) -> list[float]:
    if mode == "scale":
        return [max(float(np.sqrt((z ** 2).sum(axis=1).mean() / 3.0)), _MIN_PARAM)]
    if mode == "axis_scale":
        return [max(float(np.sqrt(v)), _MIN_PARAM) for v in (z ** 2).mean(axis=0)]
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def apply_params(mode: str, params: list[float], Ls: torch.Tensor) -> torch.Tensor:
    Ls = Ls.to(torch.float32)
    if mode == "scale":
        return Ls * float(params[0])
    # Right-multiplying by diag(s) scales column j of L by s_j.
    return Ls * torch.tensor(params, dtype=torch.float32).view(1, 1, 3)


class CovarianceCalibrated(DistributionPredictor):
    """Wrap *base*, rescaling its covariance with a fit over *proposer*'s best grounding.

    The proposer must be the one the row is evaluated with, so the fit sees the
    same grounding errors it is scored on.
    """

    def __init__(self, base: DistributionPredictor, proposer: Proposer, mode: str,
                 predictor_name: str, cache_dir: str | Path) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.base, self.proposer, self.mode = base, proposer, mode
        self.path = Path(cache_dir) / (f"{predictor_name}_{mode}".replace("/", "_") + ".json")
        self.predictor_name = predictor_name
        self.calibration: CovarianceCalibration | None = None
        if self.path.exists():
            cal = CovarianceCalibration(**json.loads(self.path.read_text()))
            self.calibration = cal if cal.mode == mode else None

    def calibrate(self, queries: list[SpatialQuery], force: bool = False) -> CovarianceCalibration:
        if self.calibration is not None and not force:
            print(f"[covcal] cached {self.mode} params={[round(p, 4) for p in self.calibration.params]} "
                  f"(n={self.calibration.n_calibration}) from {self.path}")
            return self.calibration

        whitened = []
        for q in tqdm(queries, desc=f"[covcal] {self.predictor_name} ({self.mode})", unit="q"):
            try:
                groundings = self.proposer.propose(q)
                if not groundings:
                    continue
                mus, Ls = self.base.predict(q, groundings[:1])
                r = torch.as_tensor(q.target_xyz, dtype=torch.float32) - mus[0].float()
                z = torch.linalg.solve_triangular(Ls[0].float(), r.unsqueeze(-1), upper=False).squeeze(-1).numpy()
                if np.isfinite(z).all():
                    whitened.append(z)
            except Exception as e:
                tqdm.write(f"[covcal] skip {q.scene_id}: {e}")
        if not whitened:
            raise RuntimeError("calibration produced no residuals")

        z = np.stack(whitened)
        params = fit_params(self.mode, z)
        before = float((z ** 2).sum(axis=1).mean())
        after = before / params[0] ** 2 if self.mode == "scale" else float((z ** 2 / np.array(params) ** 2).sum(axis=1).mean())
        self.calibration = CovarianceCalibration(self.mode, params, len(z), self.predictor_name, before, after)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self.calibration), indent=2))
        print(f"[covcal] {self.predictor_name} ({self.mode}): params={[round(p, 4) for p in params]} "
              f"n={len(z)}  ANEES {before:.2f} -> {after:.2f}")
        return self.calibration

    def predict(self, query: SpatialQuery, groundings: list[Grounding]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.calibration is None:
            raise RuntimeError("CovarianceCalibrated.predict called before calibrate()")
        mus, Ls = self.base.predict(query, groundings)
        return mus, apply_params(self.mode, self.calibration.params, Ls)
