"""Run several approaches on the same queries and collect metrics."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from langsensor.core.gmm import GMMResult
from langsensor.core.schema import SpatialQuery
from langsensor.core.transforms import build_spatial_query
from langsensor.data.splits import SplitRecord
from langsensor.eval.metrics import compute_all_metrics, proposer_precision_recall


def records_to_queries(records: list[SplitRecord], desc: str = "loading queries") -> list[SpatialQuery]:
    """Materialise queries (target removed), reading each scene graph once.

    Query order follows the first appearance of each scene in *records*, which
    is the order every benchmark in the paper was run in.
    """
    by_scene: dict[str, list[SplitRecord]] = defaultdict(list)
    for rec in records:
        by_scene[rec.scene.scene_id].append(rec)

    queries: list[SpatialQuery] = []
    for scene_id, recs in tqdm(by_scene.items(), desc=desc):
        try:
            sg = recs[0].scene.load_scene_graph()
        except Exception as e:
            print(f"[eval] skipping {scene_id}: {e}")
            continue
        for rec in recs:
            s = rec.statement
            try:
                q = build_spatial_query(scene_id, sg, s)
            except ValueError as e:
                print(f"[eval] skipping statement in {scene_id}: {e}")
                continue
            q.metadata.update(relation=s.relation, ambiguity=s.ambiguity, target_object_id=s.target_object_id)
            queries.append(q)
    return queries


def run_benchmark(approaches: dict[str, Any], queries: list[SpatialQuery], verbose: bool = True) -> dict[str, dict]:
    """``{name: metrics}`` for each approach (anything with ``.predict(query) -> GMMResult``).

    A query that raises is counted as failed and excluded from that approach's metrics.
    """
    out: dict[str, dict] = {}
    for name, approach in approaches.items():
        results: list[GMMResult] = []
        ok: list[SpatialQuery] = []
        failed = 0
        for q in tqdm(queries, desc=name, disable=not verbose):
            try:
                r = approach.predict(q)
            except Exception as e:
                if verbose:
                    tqdm.write(f"  [warn] {name} / {q.scene_id}: {e}")
                failed += 1
                continue
            if r.groundings:
                results.append(r)
                ok.append(q)
        if not results:
            out[name] = {"error": "all queries failed"}
            continue
        m = compute_all_metrics(results, ok)
        m["proposer"] = proposer_precision_recall([r.groundings for r in results], ok)
        m["n_failed"] = failed
        if verbose:
            o = m["overall"]
            print(f"  {name}: RMSE={o['rmse_mean']:.3f}  NLL={o['nll_mean']:.2f}  "
                  f"ANEES={o['anees']:.3f}  n={o['n']}  failed={failed}")
        out[name] = m
    return out


def write_rows(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")
