"""Accuracy and calibration of predicted mixtures against the true target location.

* RMSE    distance from the mixture mean to the target centre.
* NLL     -log p(target centre) under the mixture.
* NEES    (x - μ)ᵀ Σ⁻¹ (x - μ) under the moment-matched single Gaussian; its
          mean (ANEES) is 3 for a calibrated 3D predictor (>3 overconfident).
* NEES_min / NEES_w   per-component variants for multimodal predictions: the
          best-matching component, and the weight-averaged one.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from langsensor.core.gmm import GMMResult
from langsensor.core.schema import Grounding, SpatialQuery, target_aabb


def ambiguity_sort_key(level: Any) -> tuple:
    """Integer levels numerically, then "N+" buckets, then anything else."""
    s = str(level)
    if s.endswith("+") and s[:-1].isdigit():
        return (1, int(s[:-1]))
    try:
        return (0, int(s))
    except ValueError:
        return (2, s)


def proposer_precision_recall(groundings_per_query: list[list[Grounding]], queries: list[SpatialQuery]) -> dict:
    """Precision / recall of the union of proposed anchor ids against the annotated anchors."""
    precision, recall = [], []
    for gs, q in zip(groundings_per_query, queries):
        gt = set(q.gt_anchor_object_ids or [])
        proposed = {a for g in gs for a in g.anchor_object_ids}
        tp = len(gt & proposed)
        precision.append(tp / len(proposed) if proposed else 0.0)
        recall.append(tp / len(gt) if gt else 0.0)
    return {"proposer_precision": float(np.mean(precision)), "proposer_recall": float(np.mean(recall)),
            "n": len(queries)}


def rmse(r: GMMResult, q: SpatialQuery) -> float:
    return float(np.linalg.norm(r.mean - q.target_xyz.astype(np.float32)))


def nll(r: GMMResult, q: SpatialQuery) -> float:
    return -float(r.log_prob(torch.as_tensor(q.target_xyz, dtype=torch.float32)))


def nees(r: GMMResult, q: SpatialQuery) -> float:
    """NEES under the moment-matched mixture: μ = Σ w μ_k,  Σ = Σ w (Σ_k + (μ_k-μ)(μ_k-μ)ᵀ)."""
    w = r.weights.to(torch.float64)
    mus = r.mus.to(torch.float64)
    Ls = r.Ls.to(torch.float64)
    mu = (w.unsqueeze(-1) * mus).sum(0)
    d = mus - mu
    sigma = (w.view(-1, 1, 1) * (Ls @ Ls.transpose(-2, -1) + d.unsqueeze(-1) @ d.unsqueeze(-2))).sum(0)
    delta = torch.as_tensor(q.target_xyz, dtype=torch.float64) - mu
    try:
        sol = torch.linalg.solve(sigma, delta)
    except torch._C._LinAlgError:
        sol = torch.linalg.pinv(sigma) @ delta
    return float(delta @ sol)


def component_nees(r: GMMResult, q: SpatialQuery) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-component NEES ε_k and normalised weights."""
    diff = torch.as_tensor(q.target_xyz, dtype=torch.float64).unsqueeze(0) - r.mus.to(torch.float64)
    y = torch.linalg.solve_triangular(r.Ls.to(torch.float64), diff.unsqueeze(-1), upper=False).squeeze(-1)
    w = r.weights.to(torch.float64)
    total = float(w.sum())
    return (y * y).sum(-1), (w / total if total > 0 else torch.full_like(w, 1.0 / len(w)))


def cdf_mass(r: GMMResult, q: SpatialQuery) -> float:
    return r.cdf_bbox(target_aabb(q.target_bbox, q.target_xyz))


def mean_in_bbox(r: GMMResult, q: SpatialQuery) -> bool:
    mu = r.mean
    if q.target_bbox is None:
        return float(np.linalg.norm(mu - q.target_xyz)) < 0.2
    box = target_aabb(q.target_bbox, q.target_xyz)
    return bool(np.all(mu >= box[:3]) and np.all(mu <= box[3:]))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25, 75])
    return {"mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr)),
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "count": len(values)}


def group_by(values: list[float], groups: list) -> dict[Any, dict[str, float]]:
    buckets: dict[Any, list[float]] = defaultdict(list)
    for v, g in zip(values, groups):
        buckets[g].append(v)
    return {g: summarize(vs) for g, vs in sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def compute_all_metrics(results: list[GMMResult], queries: list[SpatialQuery]) -> dict[str, Any]:
    """Overall, per-relation and per-ambiguity summaries of every metric."""
    per_query: dict[str, list[float]] = {
        "cdf": [cdf_mass(r, q) for r, q in zip(results, queries)],
        "rmse": [rmse(r, q) for r, q in zip(results, queries)],
        "nll": [nll(r, q) for r, q in zip(results, queries)],
        "nees": [nees(r, q) for r, q in zip(results, queries)],
    }
    comp = [component_nees(r, q) for r, q in zip(results, queries)]
    per_query["nees_min"] = [float(eps.min()) for eps, _ in comp]
    per_query["nees_w"] = [float((w * eps).sum()) for eps, w in comp]

    rm, nl, ne = (per_query[k] for k in ("rmse", "nll", "nees"))
    overall = {
        "n": len(queries),
        "cdf_mean": float(np.mean(per_query["cdf"])),
        "accuracy": float(np.mean([mean_in_bbox(r, q) for r, q in zip(results, queries)])),
        "rmse_mean": float(np.mean(rm)), "rmse_std": float(np.std(rm)), "rmse_median": float(np.median(rm)),
        "nll_mean": float(np.mean(nl)), "nll_std": float(np.std(nl)), "nll_median": float(np.median(nl)),
        "anees": float(np.mean(ne)), "nees_std": float(np.std(ne)), "nees_median": float(np.median(ne)),
    }
    relations = [q.metadata.get("relation", "unknown") for q in queries]
    ambiguities = [q.metadata.get("ambiguity", 0) for q in queries]
    return {
        "overall": overall,
        "per_query": per_query,
        "by_relation": {k: group_by(v, relations) for k, v in per_query.items()},
        "by_ambiguity": {k: group_by(v, ambiguities) for k, v in per_query.items()},
    }
