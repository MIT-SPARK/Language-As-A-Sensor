"""In-context calibrated LLM: measured covariances go into the prompt; the output is used as-is.

Unlike post-hoc rescaling, nothing corrects the LLM afterwards. On a held-out
split we measure, per relation, the spread of the true target around its anchor
centroid, and show the LLM that table plus a few worked examples in place of the
hand-written sigma advice.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from langsensor.core.schema import Grounding, SpatialQuery
from langsensor.predictors.base import Proposer
from langsensor.predictors.llm import LLMGaussianPredictor, grounding_summary
from langsensor.predictors.llm_client import sha256

_MIN_SIGMA = 0.05   # m


def anchor_centroid(query: SpatialQuery, g: Grounding) -> np.ndarray | None:
    pts = [np.asarray(o.position, dtype=np.float32) for o in query.scene_graph.objects if o.id in set(g.anchor_object_ids)]
    return np.mean(np.stack(pts), axis=0) if pts else None


@dataclass
class InContextCalibration:
    relation_sigma: dict[str, list[float]] = field(default_factory=dict)   # metres, per axis
    relation_n: dict[str, int] = field(default_factory=dict)
    global_sigma: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    examples: list[dict] = field(default_factory=list)
    n_calibration: int = 0

    def signature(self) -> str:
        """Hash of the demonstrations; keeps these responses apart from the zero-shot ones in the cache."""
        return sha256(json.dumps(asdict(self), sort_keys=True))[:16]

    def system_prompt(self) -> str:
        table = [
            f'  "{rel}": sigma = [{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}] m  '
            f"(measured on {self.relation_n.get(rel, 0)} held-out cases)"
            for rel, s in sorted(self.relation_sigma.items())
        ]
        g = self.global_sigma
        table.append(f"  (any other relation): sigma = [{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}] m")

        examples = []
        for i, ex in enumerate(self.examples, 1):
            c, off, s = ex["anchor_centroid"], ex["offset_from_anchor"], ex["sigma"]
            examples.append(
                f"Example {i}\n"
                f'  Utterance: "{ex["utterance"]}"\n'
                f"  Grounding: {ex['grounding_desc']}\n"
                f"  Anchor centroid (world): [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]\n"
                f"  TRUE target offset from that centroid: [{off[0]:.2f}, {off[1]:.2f}, {off[2]:.2f}] m\n"
                f"  Well-calibrated sigma for this relation "
                f'("{ex["relation"]}"): [{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}] m'
            )
        table_block, examples_block = "\n".join(table), "\n\n".join(examples)

        return f"""You are a spatial reasoning module that predicts the 3D location of an object described by a natural-language utterance.

Given:
- A scene graph: objects (with world-frame centres) grouped by region.
- An utterance introducing a new (target) object.
- A grounding: the region and anchor object(s) the utterance refers to (assumed correct).

Output a 3D Gaussian distribution N(μ, Σ) over the TARGET object's world-frame position.

Format strictly as JSON:
{{
  "mu":    [x, y, z],                                    # world-frame centre, in metres
  "sigma": [[s11,s12,s13],[s12,s22,s23],[s13,s23,s33]],  # symmetric PSD covariance, in m²
  "explanation": "..."                                    # one short sentence
}}

CALIBRATION DATA (measured on {self.n_calibration} held-out cases, not guessed).
These are the empirical per-axis standard deviations of the true target position
around its anchor centroid, broken down by relation:

{table_block}

Your Σ should reflect these spreads. Note sigma is a standard deviation in metres,
while the "sigma" field above is a COVARIANCE in m²: a per-axis standard deviation
of s gives diagonal entries s². Use off-diagonal terms only when the relation
implies a correlated direction.

Worked examples showing where targets actually fall relative to their anchors:

{examples_block}

- Σ must be symmetric and positive semi-definite.
- Use anchor object centres + the calibration data above to decide μ and Σ."""


def fit_incontext_calibration(queries: list[SpatialQuery], proposer: Proposer, n_examples: int = 8,
                              seed: int = 0, cache_path: str | Path | None = None) -> InContextCalibration:
    """Per-relation spreads + stratified worked examples from held-out *queries* (cached as JSON)."""
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        cal = InContextCalibration(**json.loads(cache_path.read_text()))
        print(f"[incontext] cached fit ({cal.n_calibration} cases, {len(cal.examples)} examples) from {cache_path}")
        return cal

    by_relation: dict[str, list[np.ndarray]] = defaultdict(list)
    candidates: list[dict] = []
    for q in tqdm(queries, desc="[incontext] fit", unit="q"):
        try:
            groundings = proposer.propose(q)
            centroid = anchor_centroid(q, groundings[0]) if groundings else None
            if centroid is None:
                continue
            offset = np.asarray(q.target_xyz, dtype=np.float32) - centroid
            if not np.isfinite(offset).all():
                continue
            relation = str(q.metadata.get("relation") or "unknown")
            by_relation[relation].append(offset)
            candidates.append({
                "relation": relation,
                "utterance": q.language,
                "grounding_desc": grounding_summary(groundings[0], q.scene_graph),
                "anchor_centroid": [float(x) for x in centroid],
                "offset_from_anchor": [float(x) for x in offset],
            })
        except Exception as e:
            tqdm.write(f"[incontext] skip {q.scene_id}: {e}")
    if not candidates:
        raise RuntimeError("in-context calibration produced no usable cases")

    all_offsets = np.stack([o for v in by_relation.values() for o in v])
    global_sigma = [max(float(s), _MIN_SIGMA) for s in all_offsets.std(axis=0)]
    relation_sigma = {
        rel: [max(float(s), _MIN_SIGMA) for s in np.stack(offs).std(axis=0)] if len(offs) >= 3 else list(global_sigma)
        for rel, offs in by_relation.items()
    }

    # Round-robin across relations so the examples cover every behaviour.
    rng = np.random.RandomState(seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        pools[c["relation"]].append(c)
    examples: list[dict] = []
    rels = sorted(pools)
    while len(examples) < n_examples and any(pools[r] for r in rels):
        for rel in rels:
            if pools[rel] and len(examples) < n_examples:
                pick = pools[rel].pop(rng.randint(len(pools[rel])))
                pick["sigma"] = relation_sigma.get(rel, global_sigma)
                examples.append(pick)

    cal = InContextCalibration(relation_sigma, {r: len(v) for r, v in by_relation.items()},
                               global_sigma, examples, len(candidates))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(asdict(cal), indent=2))
    return cal


class LLMInContextPredictor(LLMGaussianPredictor):
    def __init__(self, *args, calibration: InContextCalibration, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = calibration.system_prompt()
        self._sig = calibration.signature()

    def cache_key(self, scene_json: str, utterance: str, grounding_desc: str) -> str:
        return sha256(f"{self.client.model}|incontext:{self._sig}|{utterance}|{grounding_desc}|{scene_json}")
