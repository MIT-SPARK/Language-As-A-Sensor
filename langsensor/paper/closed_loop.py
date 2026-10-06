"""Reading closed-loop sweeps and reducing them to per-method numbers (shared by Figure 4 and Table 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from langsensor.paper import style as ps


def discover(out_roots: list[Path], drop: set[str] = frozenset()) -> dict[str, list[dict]]:
    """``{display label: episode records}`` from ``<root>/<method>/metrics_timeseries.json``.

    Several roots may be given; per method, the newest file wins.
    """
    found: dict[str, tuple[float, Path]] = {}
    for root in out_roots:
        if not root.is_dir():
            print(f"  WARNING: no such directory: {root}")
            continue
        for ts in sorted(root.glob("*/metrics_timeseries.json")):
            method = ts.parent.name
            label = ps.APPROACH_LABELS.get(method)
            if method in drop or label in drop:
                continue
            if label is None:
                print(f"  skipping unmapped method directory '{method}'")
                continue
            mtime = ts.stat().st_mtime
            if label not in found or mtime > found[label][0]:
                found[label] = (mtime, ts)
    data = {}
    for label, (_, ts) in found.items():
        data[label] = json.loads(ts.read_text())
        print(f"  {label:<36} {len(data[label]):>3} episodes  ({ts.parent})")
    return data


def _finite(rec: dict, key: str) -> np.ndarray | None:
    arr = np.asarray(rec.get(key) or [], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr if arr.size else None


def episode_stats(rec: dict) -> dict:
    """Terminal surface mass, and trajectory means of argmax distance and information gain."""
    sm, dist, ig = (_finite(rec, k) for k in ("surface_mass", "argmax_dist_to_surface", "info_gain_at_surface"))
    return {
        "episode_id": rec.get("episode_id"),
        "terminal_surface_mass": float(sm[-1]) if sm is not None else None,
        "mean_argmax_dist": float(dist.mean()) if dist is not None else None,
        "mean_info_gain": float(ig.mean()) if ig is not None else None,
    }


def aggregate(records: list[dict], success_threshold: float = 0.15, miss_threshold: float = 0.01) -> dict:
    """Success = terminal surface mass ≥ success_threshold; miss = below miss_threshold."""
    rows = [episode_stats(r) for r in records]

    def col(key):
        return [r[key] for r in rows if r[key] is not None]

    sm, dist, ig = col("terminal_surface_mass"), col("mean_argmax_dist"), col("mean_info_gain")
    n = len(sm)
    nan = float("nan")
    return {
        "n_eps": n,
        "success_pct": 100.0 * sum(v >= success_threshold for v in sm) / n if n else nan,
        "miss_pct": 100.0 * sum(v < miss_threshold for v in sm) / n if n else nan,
        "mean_sm": float(np.mean(sm)) if sm else nan,
        "mean_argmax_dist": float(np.mean(dist)) if dist else nan,
        "mean_info_gain": float(np.mean(ig)) if ig else nan,
        "rows": rows,
    }
