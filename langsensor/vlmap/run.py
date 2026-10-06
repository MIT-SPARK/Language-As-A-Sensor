"""Closed-loop VLMaP evaluation: fuse a language belief with vision along a recorded trajectory.

    python -m langsensor.vlmap.run configs/vlmap/closed_loop.yaml predictor=lsm
    python -m langsensor.vlmap.run configs/vlmap/closed_loop.yaml predictor=none scenes=[17DRP5sb8fy]

``predictor`` selects ``configs/vlmap/predictor/<name>.yaml``. For every scene
and episode the language sensor is queried once at the utterance's timestep;
from then on each frame's vision belief is fused with it and the metrics in
:mod:`langsensor.vlmap.metrics` are recorded. Results go to
``<out_dir>/<name>/metrics_timeseries.json``, the input of Figure 4 and Table 2.

Requires Hydra for the voxel map (docs/hydra.md).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from langsensor import paths
from langsensor.config import load_config
from langsensor.vlmap.episodes import Trajectory, load_episodes, scene_extent
from langsensor.vlmap.fusion import FusionPipeline
from langsensor.vlmap.language import make_language_sensor
from langsensor.vlmap.metrics import MetricsTracker
from langsensor.vlmap.vision_sensor import SemanticVisionSensor, sbert_embedder

log = logging.getLogger(__name__)

# ADE20K classes as labelled in the Matterport3D trajectories.
ADE20K_MP3D = {
    0: "void", 1: "wall", 2: "floor", 3: "chair", 4: "door", 5: "table", 6: "picture", 7: "cabinet",
    8: "cushion", 9: "window", 10: "sofa", 11: "bed", 12: "curtain", 13: "chest_of_drawers", 14: "plant",
    15: "sink", 16: "stairs", 17: "ceiling", 18: "toilet", 19: "stool", 20: "towel", 21: "mirror",
    22: "tv_monitor", 23: "shower", 24: "column", 25: "bathtub", 26: "counter", 27: "fireplace",
    28: "lighting", 29: "railing", 30: "shelving", 31: "blinds", 32: "seating", 33: "board_panel",
    34: "furniture", 35: "appliances", 36: "clothes", 37: "objects", 38: "misc",
}


def z_bounds_for(pc_z: tuple[float, float], episodes, pad: float) -> tuple[float, float]:
    """The point cloud's z-range, stretched to contain every target, padded on both sides."""
    zs = [pc_z[0], pc_z[1], *(float(e.target_xyz[2]) for e in episodes)]
    return min(zs) - pad, max(zs) + pad


def run_episode(cfg: DictConfig, traj: Trajectory, episode, lang_sensor, pipeline: FusionPipeline,
                bounds, z_bounds, embed) -> MetricsTracker:
    from langsensor.vlmap.hydra_map import HydraVoxelMap

    s = cfg.sensor
    voxel_map = HydraVoxelMap(traj.camera_info, ADE20K_MP3D, s.voxel_size, s.voxels_per_side, s.top_k, s.min_tsdf_weight)
    vision = SemanticVisionSensor(voxel_map, episode.target_label, ADE20K_MP3D, bounds, z_bounds,
                                  s.voxel_size, s.voxels_per_side, s.tsdf_sigma, s.score_ema_alpha,
                                  s.unobserved_mode, s.unobserved_score, embed=embed)
    vision.set_query_grid(pipeline.grid)
    tracker = MetricsTracker(pipeline.grid, episode.target_xyz, episode.target_surface_xyz, cfg.metrics.surface_radius)

    schedule = {}
    if lang_sensor is not None:
        schedule = {u.timestep: lang_sensor(u, episode.scene_graph) for u in episode.utterances}

    language = None
    for t in tqdm(range(len(traj)), desc=episode.episode_id, leave=False):
        language = schedule.get(t, language)
        vision_belief = vision(traj[t])
        beliefs, weights = [vision_belief], [1.0]
        if language is not None:
            beliefs.append(language)
            weights.append(cfg.fusion.language_weight)
        fused = pipeline.step(beliefs, weights if len(beliefs) > 1 else None)
        tracker.record(fused, vision_belief)
    return tracker


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or "=" in argv[0]:
        raise SystemExit("usage: python -m langsensor.vlmap.run <config.yaml> predictor=<name> [key=value ...]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(argv[0], argv[1:])
    name = cfg.get("name") or cfg.predictor
    cfg.predictor = OmegaConf.load(Path(argv[0]).parent / "predictor" / f"{cfg.predictor}.yaml")

    data_root = Path(cfg.data_root) if cfg.data_root else paths.CLOSED_LOOP_ROOT
    out_dir = Path(cfg.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_sensor = make_language_sensor(cfg, Path(cfg.cache_dir) if cfg.cache_dir else paths.CLOSED_LOOP_CACHE)
    embed = sbert_embedder()

    records = []
    for sid in cfg.scenes:
        scene_dir = data_root / sid
        episodes = load_episodes(scene_dir)
        if not episodes:
            log.warning("%s: no episodes, skipped", sid)
            continue
        bounds, pc_z = scene_extent(scene_dir)
        z_bounds = z_bounds_for(pc_z, episodes, cfg.fusion.z_pad)
        pipeline = FusionPipeline(bounds, z_bounds, cfg.fusion.resolution)
        traj = Trajectory(scene_dir, cfg.max_steps)
        for ep in episodes:
            log.info("%s %s [%s]: %s", sid, ep.episode_id, ep.target_label, ep.utterances[0].text)
            tracker = run_episode(cfg, traj, ep, lang_sensor, pipeline, bounds, z_bounds, embed)
            records.append(tracker.record_dict(sid, ep, cfg.metrics.success_threshold))
            sm = records[-1]["terminal_surface_mass"]
            log.info("  terminal surface mass %.3f", float("nan") if sm is None else sm)

    path = out_dir / "metrics_timeseries.json"
    path.write_text(json.dumps(records))
    n_success = sum(r["success"] for r in records)
    log.info("wrote %s (%d episodes, %d successes)", path, len(records), n_success)


if __name__ == "__main__":
    main()
