"""The two halves of every language sensor.

A :class:`Proposer` decides *what* an utterance refers to (region + anchor
objects, possibly several hypotheses). A :class:`DistributionPredictor` turns
each hypothesis into a Gaussian over the target's position. A
:class:`GMMApproach` combines them into a mixture weighted by proposer confidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from langsensor.core.gmm import GMMResult
from langsensor.core.schema import Grounding, SpatialQuery


class Proposer(ABC):
    @abstractmethod
    def propose(self, query: SpatialQuery) -> list[Grounding]:
        """Grounding hypotheses for *query*, most plausible first. Must not read gt fields."""


class DistributionPredictor(ABC):
    @abstractmethod
    def predict(self, query: SpatialQuery, groundings: list[Grounding]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mus (K, 3), Ls (K, 3, 3))``, float32 CPU tensors in world frame, one per grounding."""


class GroundTruthProposer(Proposer):
    """Oracle proposer: the query's annotated anchors, with confidence 1."""

    def propose(self, query: SpatialQuery) -> list[Grounding]:
        if query.gt_anchor_room_id is None:
            raise ValueError(f"no gt_anchor_room_id for scene '{query.scene_id}'")
        return [Grounding(query.gt_anchor_room_id, list(query.gt_anchor_object_ids or []), query.language)]


class GMMApproach:
    """One benchmark row: ``(proposer, predictor)`` -> GMMResult.

    With ``proposer=None`` the predictor must itself map a query to a full
    GMMResult (the end-to-end LLM baseline).
    """

    def __init__(self, proposer: Proposer | None, predictor) -> None:
        self.proposer = proposer
        self.predictor = predictor

    def predict(self, query: SpatialQuery) -> GMMResult:
        if self.proposer is None:
            return self.predictor.predict(query)
        groundings = self.proposer.propose(query)
        if not groundings:
            raise RuntimeError(f"proposer returned no groundings for {query.scene_id}")
        mus, Ls = self.predictor.predict(query, groundings)
        w = np.asarray([g.confidence for g in groundings], dtype=np.float32)
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))
        return GMMResult(groundings=groundings, mus=mus, Ls=Ls, weights=torch.from_numpy(w))
