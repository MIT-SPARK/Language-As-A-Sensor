"""LLM baselines for the static benchmark.

* :class:`LLMGaussianPredictor` ("Scaffolded-LLM"): given a grounding, the LLM
  emits one Gaussian per hypothesis.
* :class:`LLMFullGMMPredictor` ("LLM-E2E"): no proposer; the LLM emits the whole
  mixture from the utterance and scene graph in one call.
"""

from __future__ import annotations

import numpy as np
import torch

from langsensor.core.gmm import GMMResult
from langsensor.core.schema import Grounding, SceneGraph, SpatialQuery
from langsensor.predictors.base import DistributionPredictor
from langsensor.predictors.llm_client import (
    DEFAULT_SIGMA,
    ChatClient,
    JsonCache,
    as_tensors,
    extract_json_object,
    parse_mu_sigma,
    sha256,
)
from langsensor.predictors.proposer import scene_graph_to_json

GAUSS_SYSTEM = """You are a spatial reasoning module that predicts the 3D location of an object described by a natural-language utterance.

Given:
- A scene graph: objects (with world-frame centres) grouped by region.
- An utterance introducing a new (target) object.
- A grounding: the region and anchor object(s) the utterance refers to (assumed correct).

Output a 3D Gaussian distribution N(μ, Σ) over the TARGET object's world-frame position.

Format strictly as JSON:
{
  "mu":    [x, y, z],                                    # world-frame centre, in metres
  "sigma": [[s11,s12,s13],[s12,s22,s23],[s13,s23,s33]],  # symmetric PSD covariance, in m²
  "explanation": "..."                                    # one short sentence
}

Guidance for Σ:
- Use tighter σ (e.g. 0.05–0.2 m along each axis) for precise relations like "on", "in".
- Use looser σ (e.g. 0.5–2.0 m) for vague relations like "near", "around".
- Σ must be symmetric and positive semi-definite.
- Use anchor object centres + typical placement priors to decide μ."""

GAUSS_USER_TEMPLATE = """Scene Graph:
{scene_json}

Utterance:
"{utterance}"

Grounding (assumed correct):
{grounding_desc}

Return ONE JSON object with keys "mu", "sigma", "explanation"."""

FULL_GMM_SYSTEM = """You are a spatial reasoning module that predicts the 3D location of a target object described by a natural-language utterance, *without* being told which anchor objects the utterance refers to.

Given:
- A scene graph: objects (with world-frame centres) grouped by region.
- An utterance introducing a new target object.

Output a Gaussian mixture model over the new target object's world-frame position — one component per plausible grounding of the utterance.

Format strictly as JSON:
{
  "components": [
    {
      "mu":    [x, y, z],
      "sigma": [[...3x3...]],
      "weight": 0.5,
      "explanation": "..."
    },
    ...
  ]
}

- Weights must be non-negative and sum to 1.
- Return at least 1 and at most 5 components (one per plausible anchor).
- Σ must be symmetric and positive semi-definite."""

FULL_GMM_USER_TEMPLATE = """Scene Graph:
{scene_json}

Utterance:
"{utterance}"

Return ONE JSON object with key "components"."""


def grounding_summary(g: Grounding, sg: SceneGraph) -> str:
    """How a grounding is described to the LLM, e.g. ``region_id=3 ("kitchen"), anchor_object_ids=[12:table]``."""
    labels = {o.id: o.label for o in sg.objects}
    regions = {r.id: r.label for r in sg.regions}
    anchors = ", ".join(f"{a}:{labels.get(a, '?')}" for a in g.anchor_object_ids)
    region = regions.get(g.anchor_room_id, f"region_{g.anchor_room_id}")
    return f'region_id={g.anchor_room_id} ("{region}"), anchor_object_ids=[{anchors}]'


def anchor_centroid_fallback(g: Grounding, sg: SceneGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """μ at the anchors' centroid (else the region centre), σ = 0.5 m."""
    pos = {o.id: np.asarray(o.position, dtype=np.float32) for o in sg.objects}
    anchors = [pos[a] for a in g.anchor_object_ids if a in pos]
    if anchors:
        mu = np.mean(anchors, axis=0)
    else:
        region = {r.id: np.asarray(r.position, dtype=np.float32) for r in sg.regions}.get(g.anchor_room_id)
        mu = region if region is not None else np.zeros(3, dtype=np.float32)
    return as_tensors(mu, np.eye(3, dtype=np.float32) * DEFAULT_SIGMA)


class LLMGaussianPredictor(DistributionPredictor):
    system_prompt = GAUSS_SYSTEM

    def __init__(self, provider: str = "openai", model: str = "gpt-5.2", cache_dir=None,
                 base_url: str | None = None, temperature: float | None = None, seed: int = 42) -> None:
        self.client = ChatClient(provider, model, base_url, temperature=temperature, seed=seed)
        self.cache = JsonCache(cache_dir)

    def cache_key(self, scene_json: str, utterance: str, grounding_desc: str) -> str:
        return sha256(f"{self.client.model}|{utterance}|{grounding_desc}|{scene_json}")

    def _predict_one(self, scene_json: str, query: SpatialQuery, g: Grounding) -> tuple[torch.Tensor, torch.Tensor]:
        desc = grounding_summary(g, query.scene_graph)
        key = self.cache_key(scene_json, query.language, desc)
        parsed = self.cache.get(key)
        if parsed is None:
            user = GAUSS_USER_TEMPLATE.format(scene_json=scene_json, utterance=query.language, grounding_desc=desc)
            raw, _ = self.client.complete([{"role": "system", "content": self.system_prompt},
                                           {"role": "user", "content": user}])
            parsed = extract_json_object(raw)
            if parsed is not None:
                self.cache.put(key, parsed)
        result = parse_mu_sigma(parsed) if parsed is not None else None
        return as_tensors(*result) if result is not None else anchor_centroid_fallback(g, query.scene_graph)

    def predict(self, query: SpatialQuery, groundings: list[Grounding]) -> tuple[torch.Tensor, torch.Tensor]:
        scene_json = scene_graph_to_json(query.scene_graph)
        mus, Ls = zip(*(self._predict_one(scene_json, query, g) for g in groundings))
        return torch.stack(mus), torch.stack(Ls)


class LLMFullGMMPredictor:
    """Straight-shot baseline: ``predict(query) -> GMMResult`` (used with ``proposer=None``)."""

    def __init__(self, provider: str = "openai", model: str = "gpt-5.2", cache_dir=None,
                 base_url: str | None = None, temperature: float | None = None, seed: int = 42,
                 max_components: int = 5) -> None:
        self.client = ChatClient(provider, model, base_url, temperature=temperature, seed=seed)
        self.cache = JsonCache(cache_dir)
        self.max_components = max_components

    def predict(self, query: SpatialQuery) -> GMMResult:
        sg = query.scene_graph
        scene_json = scene_graph_to_json(sg)
        key = sha256(f"full_gmm|{self.client.model}|{query.language}|{scene_json}")
        parsed = self.cache.get(key)
        if parsed is None:
            user = FULL_GMM_USER_TEMPLATE.format(scene_json=scene_json, utterance=query.language)
            raw, _ = self.client.complete([{"role": "system", "content": FULL_GMM_SYSTEM},
                                           {"role": "user", "content": user}])
            parsed = extract_json_object(raw)
            if parsed is not None:
                self.cache.put(key, parsed)

        mus, Ls, weights = [], [], []
        for comp in (parsed or {}).get("components", [])[: self.max_components]:
            result = parse_mu_sigma(comp) if isinstance(comp, dict) else None
            if result is not None:
                mu, L = as_tensors(*result)
                mus.append(mu)
                Ls.append(L)
                weights.append(max(float(comp.get("weight", 1.0)), 0.0))
        if not mus:     # scene centroid, σ = 1 m
            centroid = np.mean([o.position for o in sg.objects], axis=0) if sg.objects else np.zeros(3)
            mus, Ls, weights = [torch.tensor(centroid, dtype=torch.float32)], [torch.eye(3)], [1.0]

        w = np.asarray(weights, dtype=np.float32)
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))
        region = sg.regions[0].id if sg.regions else 0
        return GMMResult(
            groundings=[Grounding(region, [], query.language, confidence=float(x)) for x in w],
            mus=torch.stack(mus), Ls=torch.stack(Ls), weights=torch.from_numpy(w),
        )
