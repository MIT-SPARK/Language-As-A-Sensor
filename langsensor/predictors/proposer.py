"""LLM grounding proposer: scene graph + utterance -> ranked grounding hypotheses.

The prompt below is the one used for every static-benchmark result in the
paper (Appendix, "Hypothesis Generation LLM Prompt"). Its text is part of the
response-cache key, so editing it invalidates the shipped cache.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from langsensor.core.ontology import semantic_label
from langsensor.core.schema import Grounding, SceneGraph, SpatialQuery
from langsensor.predictors.base import Proposer
from langsensor.predictors.llm_client import ChatClient, JsonCache, extract_json_object, sha256

SYSTEM_PROMPT = """You are a spatial reasoning module.

You are given:

A scene graph describing a region and its objects.

A natural language utterance introducing a new object.

Your task is to generate K grounded spatial hypotheses that explain how the new object could relate to existing objects in the scene.

A hypothesis must include:

utterance: A fully specified sentence of the form
"there is a <target object> <relation> <existing object(s)>". Infer the target object from the provided utterance.

referred_object_id: A list of object ids the relation is grounded to. Use exactly one id for relations like "near", "on", "against", "above", "below". Use exactly two ids for "between" (both target objects, in any order).

region_id: The region id (integer) of the referred object(s). Same region for all referred objects in that hypothesis.

confidence: A number between 0 and 1 representing plausibility given the scene.

Only use objects that exist in the scene graph.
Do NOT invent objects.
Output MUST be valid JSON.
Return exactly K hypotheses."""

USER_TEMPLATE = """Scene Graph:
{scene_json}

Utterance:
"{utterance}"

Number of hypotheses (K):
{k_placeholder}

Instructions:
- Generate K diverse and plausible spatial hypotheses.
- Use only relations such as: near, between, above, below, in, or on. Avoid all other relations.
- referred_object_id must always be a list: one object id for single-object relations (e.g. "near the desk"), exactly two object ids for "between" (e.g. "between the desk and the wall" -> ["id1", "id2"]).
- Use only objects present in the scene graph.
- Confidence should reflect spatial plausibility and typical region layout priors.
- Consider different rooms and object arrangements to generate diverse hypotheses.
- Output format (referred_object_id is always a list; region_id is a single integer):

{{
  "hypotheses": [
    {{
      "utterance": "...",
      "referred_object_id": ["<object_id>"],
      "region_id": 0,
      "confidence": 0.8
    }}
  ]
}}"""

K_PLACEHOLDER = (
    "as many as appropriate (at least 1, up to 5). "
    "Generate the minimum number of hypotheses necessary to disambiguate the utterance."
)

OFFICE_SCENE = """{
  "Office": {
    "objects": {
      "0": {"semantics": "desk", "raw_label": "wooden desk", "center": [0.0, -1.0, 0.4], "volume": 2.5, "object_id": "0"},
      "1": {"semantics": "monitor", "raw_label": "computer monitor", "center": [0.1, -1.0, 0.9], "volume": 0.04, "object_id": "1"},
      "2": {"semantics": "keyboard", "raw_label": "keyboard", "center": [0.1, -0.85, 0.7], "volume": 0.003, "object_id": "2"},
      "3": {"semantics": "chair", "raw_label": "office chair", "center": [0.8, -0.6, 0.5], "volume": 0.3, "object_id": "3"},
      "4": {"semantics": "wall", "raw_label": "wall", "center": [0.0, -2.5, 1.5], "volume": 6.0, "object_id": "4"},
      "5": {"semantics": "chair", "raw_label": "folding chair", "center": [1.8, -1.6, 0.5], "volume": 0.3, "object_id": "5"}
    },
    "region_id": 0
  },
  "Lobby": {
    "objects": {
      "6": {"semantics": "sofa", "raw_label": "sofa", "center": [0.0, -1.0, 0.5], "volume": 1.0, "object_id": "6"},
      "7": {"semantics": "table", "raw_label": "coffee table", "center": [0.0, -0.5, 0.3], "volume": 0.5, "object_id": "7"},
      "8": {"semantics": "plant", "raw_label": "plant", "center": [0.0, -0.5, 0.8], "volume": 0.1, "object_id": "8"}
    },
    "region_id": 1
  }
}"""

BEDROOM_SCENE = """{
  "Bedroom": {
    "objects": {
      "0": {"semantics": "nightstand", "raw_label": "wooden nightstand", "center": [-0.8, -1.1, 0.6], "volume": 0.35, "object_id": "0"},
      "1": {"semantics": "bed", "raw_label": "queen bed", "center": [0.0, -1.2, 0.5], "volume": 3.5, "object_id": "1"},
      "2": {"semantics": "nightstand", "raw_label": "wooden nightstand", "center": [0.8, -1.1, 0.6], "volume": 0.35, "object_id": "2"},
      "3": {"semantics": "lamp", "raw_label": "table lamp", "center": [0.8, -1.1, 1.1], "volume": 0.05, "object_id": "3"},
      "4": {"semantics": "dresser", "raw_label": "dresser", "center": [-1.2, -0.4, 0.9], "volume": 1.8, "object_id": "4"},
      "5": {"semantics": "chair", "raw_label": "armchair", "center": [1.4, -0.3, 0.5], "volume": 0.35, "object_id": "5"},
      "6": {"semantics": "window", "raw_label": "window", "center": [0.0, -2.5, 1.4], "volume": 3.0, "object_id": "6"}
    },
    "region_id": 0
  }
}"""

FEW_SHOT = [
    {"role": "user", "content": USER_TEMPLATE.format(
        scene_json=OFFICE_SCENE, utterance="there is a mouse on the desk", k_placeholder=K_PLACEHOLDER)},
    {"role": "assistant", "content": """{
  "hypotheses": [
    {"utterance": "there is a mouse on the desk", "referred_object_id": ["0"], "region_id": 0, "confidence": 0.8},
    {"utterance": "there is a mouse on the table", "referred_object_id": ["7"], "region_id": 1, "confidence": 0.3}
  ]
}"""},
    {"role": "user", "content": USER_TEMPLATE.format(
        scene_json=BEDROOM_SCENE, utterance="there is a bag on the nightstand", k_placeholder=K_PLACEHOLDER)},
    {"role": "assistant", "content": """{
  "hypotheses": [
    {"utterance": "there is a bag on the nightstand", "referred_object_id": ["0"], "region_id": 0, "confidence": 0.88},
    {"utterance": "there is a bag on the nightstand", "referred_object_id": ["2"], "region_id": 0, "confidence": 0.72}
  ]
}"""},
]


def prompt_signature() -> str:
    """Hash of everything that shapes the rendered prompt; part of the cache key."""
    payload = json.dumps({
        "k_mode": "dynamic", "k": None, "few_shot": "synthetic2",
        "system": SYSTEM_PROMPT, "user_template": USER_TEMPLATE,
        "k_placeholder": K_PLACEHOLDER, "demos": FEW_SHOT,
    }, sort_keys=True)
    return sha256(payload)[:16]


def scene_graph_to_json(sg: SceneGraph) -> str:
    """Objects grouped by region, in the format of the few-shot scenes above."""
    region_map = {r.id: r for r in sg.regions}
    by_region: dict[int, list] = defaultdict(list)
    for obj in sg.objects:
        if obj.metadata.get("region_id") is not None:
            by_region[int(obj.metadata["region_id"])].append(obj)

    out: dict = {}
    for rid, objects in by_region.items():
        region = region_map.get(rid)
        obj_map = {}
        for obj in objects:
            vol = obj.metadata.get("volume")
            if vol is None and obj.bbox is not None:
                corners = np.array(obj.bbox, dtype=np.float32).reshape(8, 3)
                vol = float(np.prod(corners.max(axis=0) - corners.min(axis=0)))
            obj_map[str(obj.id)] = {
                "semantics": semantic_label(obj.metadata) or obj.label,
                "raw_label": obj.metadata.get("raw_label") or obj.label,
                "center": [round(c, 2) for c in obj.position],
                "volume": round(float(vol), 3) if vol else 0.0,
                "object_id": str(obj.id),
            }
        out[region.label if region else f"region_{rid}"] = {"objects": obj_map, "region_id": rid}
    return json.dumps(out, indent=2)


def parse_hypotheses(text: str) -> list[dict]:
    data = extract_json_object(text)
    raw = data.get("hypotheses", []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return []
    out = []
    try:
        for h in raw:
            if not isinstance(h, dict):
                continue
            ref = h.get("referred_object_id", h.get("referred_object_ids", []))
            ref = [ref] if isinstance(ref, (str, int)) else ref
            out.append({
                "utterance": str(h.get("utterance", "")).strip(),
                "referred_object_id": [str(x) for x in ref],
                "region_id": int(h.get("region_id", 0)),
                "confidence": float(h.get("confidence", 0.5)),
            })
    except (TypeError, ValueError):
        return []
    return out


class LLMProposer(Proposer):
    """Ask an LLM for 1-5 grounding hypotheses; invalid object ids are discarded.

    Falls back to a single anchor-less hypothesis in the first region when the
    LLM returns nothing usable.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-5.2", cache_dir=None,
                 base_url: str | None = None, temperature: float | None = None, seed: int = 42) -> None:
        self.client = ChatClient(provider, model, base_url, temperature=temperature, seed=seed)
        self.cache = JsonCache(cache_dir)
        self._signature = prompt_signature()

    def messages(self, scene_json: str, utterance: str) -> list[dict]:
        user = USER_TEMPLATE.format(scene_json=scene_json, utterance=utterance, k_placeholder=K_PLACEHOLDER)
        return [{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOT, {"role": "user", "content": user}]

    def propose(self, query: SpatialQuery) -> list[Grounding]:
        sg, utterance = query.scene_graph, query.language
        scene_json = scene_graph_to_json(sg)
        key = sha256(f"{self.client.model}|{self._signature}|t={self.client.temperature}|{utterance}|{scene_json}")

        cached = self.cache.get(key)
        hypotheses = cached.get("hypotheses", []) if cached is not None else None
        if hypotheses is None:
            raw, usage = self.client.complete(self.messages(scene_json, utterance))
            hypotheses = parse_hypotheses(raw)
            if hypotheses:
                self.cache.put(key, {"hypotheses": hypotheses, **usage})

        valid_ids = {o.id for o in sg.objects}
        obj_region = {o.id: int(o.metadata["region_id"]) for o in sg.objects if "region_id" in o.metadata}
        groundings = []
        for h in hypotheses:
            ids = [int(x) for x in h["referred_object_id"] if int(x) in valid_ids]
            if ids:
                groundings.append(Grounding(
                    anchor_room_id=obj_region.get(ids[0], h.get("region_id", 0)),
                    anchor_object_ids=ids,
                    language=h.get("utterance", "").strip() or utterance,
                    confidence=h.get("confidence", 0.5),
                ))
        if not groundings:
            return [Grounding(sg.regions[0].id if sg.regions else 0, [], utterance, confidence=0.1)]
        return sorted(groundings, key=lambda g: g.confidence, reverse=True)
