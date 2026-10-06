from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from langsensor import paths
from langsensor.core.schema import GroundedQuery, Grounding, SpatialQuery
from langsensor.lsm.dataset import CollateFn, to_device
from langsensor.lsm.model import load_lsm
from langsensor.lsm.tensorizer import Tensorizer
from langsensor.predictors.base import DistributionPredictor


class LSMPredictor(DistributionPredictor):
    """The trained Language Sensor Model; all groundings go through one batched forward pass."""

    def __init__(
        self,
        checkpoint: str | Path = paths.CHECKPOINT,
        clip_label_map: str | Path | dict[str, np.ndarray] = paths.CLIP_LABEL_MAP,
        device: str = "cuda",
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = load_lsm(checkpoint, self.device)
        cfg = self.model.cfg
        if not isinstance(clip_label_map, dict):
            clip_label_map = torch.load(clip_label_map, weights_only=False)
        self.tensorizer = Tensorizer(cfg.max_objects, clip_label_map, cfg.clip_dim)
        self.collate = CollateFn(cfg.text_model, cfg.max_text_len)

    @torch.no_grad()
    def predict(self, query: SpatialQuery, groundings: list[Grounding]) -> tuple[torch.Tensor, torch.Tensor]:
        if not groundings:
            raise ValueError("LSMPredictor.predict called with no groundings")
        grounded = [GroundedQuery(query.scene_id, query.scene_graph, g, query.target_xyz, query.target_bbox)
                    for g in groundings]
        batch = to_device(self.collate(self.tensorizer(grounded)), self.device)
        # bf16 on GPU (as for every result in the paper), plain fp32 on CPU.
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext():
            pred = self.model.forward_batch(batch)
        return pred.mu.float().cpu(), pred.L.float().cpu()
