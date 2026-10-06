"""The language sensor used in the closed-loop experiments (Figure 4, Table 2).

It is the same idea as the static benchmark's sensor — an LLM proposer
followed by a per-grounding predictor — but it was run with slightly different
settings, kept here so the closed-loop results reproduce:

* the proposer prompt adds a "relational sparsity" instruction and a third
  demonstration, and serialises the scene graph with region-tagged keys;
* the LLM predictors see a compact scene graph (label + centre only);
* the LSM's region frame is the anchor region's object extent padded by 1 m
  (the episode scene graphs carry no region boxes), with at most 100 objects;
* mixture weights are ``softmax(confidence)`` rather than normalised confidences.
"""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer

from langsensor import paths
from langsensor.core.ontology import NYU40_CATCHALL
from langsensor.core.schema import Grounding
from langsensor.lsm.model import load_lsm
from langsensor.predictors.llm import FULL_GMM_SYSTEM, FULL_GMM_USER_TEMPLATE, GAUSS_USER_TEMPLATE, grounding_summary
from langsensor.predictors.llm_client import (
    DEFAULT_SIGMA,
    ChatClient,
    JsonCache,
    extract_json_object,
    parse_mu_sigma,
    sha256,
)
from langsensor.predictors.proposer import (
    BEDROOM_SCENE,
    K_PLACEHOLDER,
    OFFICE_SCENE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    parse_hypotheses,
)
from langsensor.vlmap.fusion import GMMBelief
from langsensor.vlmap.types import BeliefModel, SceneGraph, TimestampedUtterance, UniformBelief

# ── Proposer ──────────────────────────────────────────────────────────────────

USER_TEMPLATE_CL = USER_TEMPLATE.replace(
    "- Use only objects present in the scene graph.\n",
    "- Use only objects present in the scene graph.\n"
    "- Apply a relational sparsity / non-redundancy prior: among multiple valid anchors, prefer objects "
    "that do not already participate in the same relation with an object of the target type.\n",
)


def _demo(scene: str, utterance: str) -> dict:
    return {"role": "user", "content": USER_TEMPLATE_CL.format(scene_json=scene, utterance=utterance,
                                                               k_placeholder=K_PLACEHOLDER)}


FEW_SHOT_CL = [
    _demo(OFFICE_SCENE, "there is a mouse on the desk"),
    {"role": "assistant", "content": '{"hypotheses": [{"utterance": "there is a mouse on the desk", "referred_object_id": ["0"], "region_id": 0, "confidence": 0.8}, {"utterance": "there is a mouse on the table", "referred_object_id": ["7"], "region_id": 1, "confidence": 0.3}]}'},
    _demo(BEDROOM_SCENE, "there is a bag on the nightstand"),
    {"role": "assistant", "content": '{"hypotheses": [{"utterance": "there is a bag on the nightstand", "referred_object_id": ["0"], "region_id": 0, "confidence": 0.88}, {"utterance": "there is a bag on the nightstand", "referred_object_id": ["2"], "region_id": 0, "confidence": 0.72}]}'},
    _demo(OFFICE_SCENE, "there is a desk near the chair"),
    {"role": "assistant", "content": '{"hypotheses": [{"utterance": "there is a desk near the chair", "referred_object_id": ["5"], "region_id": 0, "confidence": 0.9}]}'},
]


def proposer_scene_json(sg: SceneGraph) -> str:
    regions = {r.id: r for r in sg.regions}
    by_region: dict[int, list] = defaultdict(list)
    for obj in sg.objects:
        if obj.region_id is not None:
            by_region[int(obj.region_id)].append(obj)
    out = {}
    for rid, objects in by_region.items():
        obj_map = {}
        for obj in objects:
            vol = 0.0
            if obj.bbox is not None and len(obj.bbox) == 6:
                vol = float(np.prod(np.maximum(np.asarray(obj.bbox[3:], np.float32) - np.asarray(obj.bbox[:3], np.float32), 0)))
            sem = obj.nyu40_label or obj.label
            obj_map[str(obj.id)] = {
                "semantics": obj.label if sem in NYU40_CATCHALL else sem,
                "raw_label": obj.label,
                "center": [round(float(c), 2) for c in obj.position],
                "volume": round(vol, 3),
                "object_id": str(obj.id),
            }
        name = regions[rid].label if rid in regions else f"region_{rid}"
        out[f"{name} [{rid}]"] = {"objects": obj_map, "region_id": rid}
    return json.dumps(out, indent=2)


class ClosedLoopProposer:
    def __init__(self, provider: str = "openai", model: str = "gpt-5.2", cache_dir=None,
                 base_url: str | None = None) -> None:
        self.client = ChatClient(provider, model, base_url)
        self.cache = JsonCache(cache_dir)

    def propose(self, utterance: TimestampedUtterance, sg: SceneGraph) -> list[Grounding]:
        scene_json = proposer_scene_json(sg)
        # "att" and "prior" are the fields of the (unused) re-proposal loop; they stay in the key.
        key = sha256(f"{self.client.model}|{utterance.text}|{scene_json}|t={self.client.temperature}|att=0|prior=none")
        cached = self.cache.get(key)
        hypotheses = cached.get("hypotheses", []) if cached is not None else None
        if hypotheses is None:
            user = USER_TEMPLATE_CL.format(scene_json=scene_json, utterance=utterance.text, k_placeholder=K_PLACEHOLDER)
            raw, _ = self.client.complete([{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOT_CL,
                                           {"role": "user", "content": user}])
            hypotheses = parse_hypotheses(raw)
            if hypotheses:
                self.cache.put(key, {"hypotheses": hypotheses})

        valid = {o.id for o in sg.objects}
        region_of = {o.id: o.region_id for o in sg.objects if o.region_id is not None}
        groundings = []
        for h in hypotheses:
            ids = [int(x) for x in h["referred_object_id"] if int(x) in valid]
            if ids:
                groundings.append(Grounding(region_of.get(ids[0], h.get("region_id", 0)), ids,
                                            h.get("utterance", "").strip() or utterance.text, h.get("confidence", 0.5)))
        if not groundings:
            return [Grounding(sg.regions[0].id if sg.regions else 0, [], utterance.text, 0.1)]
        return sorted(groundings, key=lambda g: g.confidence, reverse=True)


# ── Predictors ────────────────────────────────────────────────────────────────

def predictor_scene_json(sg: SceneGraph) -> str:
    regions = {r.id: r.label for r in sg.regions}
    by_region: dict[int, list] = defaultdict(list)
    for obj in sg.objects:
        if obj.region_id is not None:
            by_region[int(obj.region_id)].append(obj)
    return json.dumps({
        regions.get(rid, f"region_{rid}"): {
            "objects": {str(o.id): {"label": o.label, "center": [round(float(c), 2) for c in o.position]} for o in objs},
            "region_id": rid,
        }
        for rid, objs in by_region.items()
    }, indent=2)


GAUSS_SYSTEM_CL = """You are a spatial reasoning module that predicts the 3D location of an object described by a natural-language utterance.

Given:
- A scene graph: objects (with world-frame centres) grouped by region.
- An utterance introducing a new (target) object.
- A grounding: the region and anchor object(s) the utterance refers to (assumed correct).

Output a 3D Gaussian distribution N(μ, Σ) over the TARGET object's world-frame position.

Format strictly as JSON:
{
  "mu":    [x, y, z],
  "sigma": [[s11,s12,s13],[s12,s22,s23],[s13,s23,s33]],
  "explanation": "..."
}

Guidance for Σ:
- Use tighter σ (e.g. 0.05–0.2 m) for precise relations like "on", "in".
- Use looser σ (e.g. 0.5–2.0 m) for vague relations like "near", "around".
- Σ must be symmetric and positive semi-definite."""


def _anchor_mean(g: Grounding, sg: SceneGraph, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    pos = {o.id: o.position for o in sg.objects}
    anchors = [pos[a] for a in g.anchor_object_ids if a in pos]
    mu = np.mean(np.array(anchors, dtype=np.float32), axis=0) if anchors else np.zeros(3, dtype=np.float32)
    return mu, np.eye(3, dtype=np.float32) * sigma


class ClosedLoopLLMPredictor:
    """Scaffolded-LLM: one Gaussian per grounding."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", cache_dir=None,
                 base_url: str | None = None, fallback_sigma: float = DEFAULT_SIGMA) -> None:
        self.client = ChatClient(provider, model, base_url)
        self.cache = JsonCache(cache_dir)
        self.fallback_sigma = fallback_sigma

    def predict(self, utterance: TimestampedUtterance, sg: SceneGraph, groundings: list[Grounding]):
        scene_json = predictor_scene_json(sg)
        mus, Ls = [], []
        for g in groundings:
            desc = grounding_summary(g, sg)
            key = sha256(f"{self.client.model}|{utterance.text}|{desc}|{scene_json}")
            parsed = self.cache.get(key)
            if parsed is None:
                user = GAUSS_USER_TEMPLATE.format(scene_json=scene_json, utterance=utterance.text, grounding_desc=desc)
                raw, _ = self.client.complete([{"role": "system", "content": GAUSS_SYSTEM_CL},
                                               {"role": "user", "content": user}])
                parsed = extract_json_object(raw)
                if parsed is not None:
                    self.cache.put(key, parsed)
            result = parse_mu_sigma(parsed) if parsed is not None else None
            mu, L = result if result is not None else _anchor_mean(g, sg, self.fallback_sigma)
            mus.append(mu)
            Ls.append(L)
        return np.stack(mus), np.stack(Ls)


class ClosedLoopE2EPredictor:
    """LLM-E2E: one call returns the whole mixture. Acts as both proposer and predictor.

    Proposed confidences are log(weight), so the language sensor's softmax
    recovers the LLM's weights exactly.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", cache_dir=None,
                 base_url: str | None = None, fallback_sigma: float = DEFAULT_SIGMA, max_components: int = 5) -> None:
        self.client = ChatClient(provider, model, base_url)
        self.cache = JsonCache(cache_dir)
        self.fallback_sigma = fallback_sigma
        self.max_components = max_components
        self._memo: dict[str, list] = {}

    def _components(self, utterance: TimestampedUtterance, sg: SceneGraph) -> list[tuple]:
        scene_json = predictor_scene_json(sg)
        key = sha256(f"e2e|{self.client.model}|{utterance.text}|{scene_json}")
        if key in self._memo:
            return self._memo[key]
        parsed = self.cache.get(key)
        if parsed is None:
            raw, _ = self.client.complete([
                {"role": "system", "content": FULL_GMM_SYSTEM},
                {"role": "user", "content": FULL_GMM_USER_TEMPLATE.format(scene_json=scene_json, utterance=utterance.text)},
            ])
            parsed = extract_json_object(raw)
            if parsed is not None:
                self.cache.put(key, parsed)

        comps = []
        for c in (parsed or {}).get("components", [])[: self.max_components]:
            result = parse_mu_sigma(c) if isinstance(c, dict) else None
            if result is not None:
                comps.append((*result, max(float(c.get("weight", 1.0)), 0.0)))
        if not comps:
            pos = np.array([o.position for o in sg.objects], dtype=np.float32)
            comps = [(pos.mean(axis=0) if pos.size else np.zeros(3, np.float32),
                      np.eye(3, dtype=np.float32) * self.fallback_sigma, 1.0)]
        total = sum(w for *_, w in comps)
        comps = [(mu, L, w / total if total > 0 else 1.0 / len(comps)) for mu, L, w in comps]
        self._memo[key] = comps
        return comps

    def propose(self, utterance: TimestampedUtterance, sg: SceneGraph) -> list[Grounding]:
        region = sg.regions[0].id if sg.regions else 0
        return [Grounding(region, [], utterance.text, float(np.log(max(w, 1e-8))))
                for *_, w in self._components(utterance, sg)]

    def predict(self, utterance: TimestampedUtterance, sg: SceneGraph, groundings: list[Grounding]):
        comps = self._components(utterance, sg)
        return np.stack([c[0] for c in comps]), np.stack([c[1] for c in comps])


class ClosedLoopLSMPredictor:
    """The released LSM, tensorised from an episode scene graph."""

    def __init__(self, checkpoint: str | Path, clip_label_map: str | Path, device: str = "cuda",
                 max_objects: int = 100) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = load_lsm(checkpoint, self.device)
        self.clip = torch.load(clip_label_map, weights_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.cfg.text_model)
        self.max_objects = max_objects

    @staticmethod
    def region_frame(sg: SceneGraph, region_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Extent of the region's object centres, padded by 1 m (all objects if the region is empty)."""
        objs = [o for o in sg.objects if o.region_id == region_id] or list(sg.objects)
        if not objs:
            return np.zeros(3, np.float32), np.ones(3, np.float32)
        pos = np.array([o.position for o in objs], dtype=np.float32)
        lo, hi = pos.min(axis=0) - 1.0, pos.max(axis=0) + 1.0
        return ((lo + hi) / 2.0).astype(np.float32), np.maximum(hi - lo, 1e-3).astype(np.float32)

    @torch.no_grad()
    def predict(self, utterance: TimestampedUtterance, sg: SceneGraph, groundings: list[Grounding]):
        K, N, D = len(groundings), self.max_objects, self.model.cfg.clip_dim
        clip = np.zeros((K, N, D), np.float32)
        boxes = np.zeros((K, N, 6), np.float32)
        anchor = np.zeros((K, N), bool)
        pad = np.ones((K, N), bool)
        shift = np.zeros((K, 3), np.float32)
        scale = np.ones((K, 3), np.float32)
        for k, g in enumerate(groundings):
            shift[k], scale[k] = self.region_frame(sg, g.anchor_room_id)
            ids = set(g.anchor_object_ids)
            for i, o in enumerate([o for o in sg.objects if o.region_id == g.anchor_room_id][:N]):
                clip[k, i] = self.clip.get(o.label, np.zeros(D, np.float32))
                if o.bbox is not None:
                    b = np.array(o.bbox, dtype=np.float32)
                    center, size = (b[:3] + b[3:]) / 2.0, b[3:] - b[:3]
                else:
                    center, size = np.array(o.position, dtype=np.float32), np.zeros(3, np.float32)
                boxes[k, i] = np.concatenate([(center - shift[k]) / scale[k], size / scale[k]])
                anchor[k, i], pad[k, i] = o.id in ids, False

        enc = self.tokenizer([g.language for g in groundings], padding=True, truncation=True,
                             max_length=self.model.cfg.max_text_len, return_tensors="pt")
        to = lambda a: (torch.from_numpy(a) if isinstance(a, np.ndarray) else a).to(self.device)  # noqa: E731
        # bf16 on GPU (as for every result in the paper), plain fp32 on CPU.
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext():
            pred = self.model(to(enc["input_ids"]), to(enc["attention_mask"]), to(clip), to(boxes),
                              to(anchor), to(pad), to(scale), to(shift))
        return pred.mu.float().cpu().numpy(), pred.L.float().cpu().numpy()


class LanguageSensor:
    """utterance + scene graph -> GMMBelief, weighted by softmax(proposer confidence)."""

    def __init__(self, proposer, predictor) -> None:
        self.proposer, self.predictor = proposer, predictor

    def __call__(self, utterance: TimestampedUtterance, sg: SceneGraph) -> BeliefModel:
        groundings = self.proposer.propose(utterance, sg)
        if not groundings:
            return UniformBelief()
        mus, Ls = self.predictor.predict(utterance, sg, groundings)
        return GMMBelief(mus, Ls, softmax([g.confidence for g in groundings]))


def make_language_sensor(cfg, cache_root: Path) -> LanguageSensor | None:
    """Build the sensor for ``cfg.predictor.type`` in {none, lsm, llm, llm_e2e}."""
    p = cfg.predictor
    if p.type == "none":
        return None
    if p.type == "llm_e2e":
        e2e = ClosedLoopE2EPredictor(p.provider, p.model, cache_root / "predictor_e2e",
                                     fallback_sigma=p.fallback_sigma, max_components=p.max_components)
        return LanguageSensor(e2e, e2e)
    proposer = ClosedLoopProposer(cfg.proposer.provider, cfg.proposer.model, cache_root / "proposer")
    if p.type == "lsm":
        return LanguageSensor(proposer, ClosedLoopLSMPredictor(
            p.checkpoint or paths.CHECKPOINT, p.clip_label_map or paths.CLIP_LABEL_MAP, p.device))
    if p.type == "llm":
        return LanguageSensor(proposer, ClosedLoopLLMPredictor(p.provider, p.model, cache_root / "predictor",
                                                               fallback_sigma=p.fallback_sigma))
    raise ValueError(f"unknown predictor type {p.type!r}")
