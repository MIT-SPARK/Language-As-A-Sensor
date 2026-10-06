# Data

Two datasets are used. The static benchmark (Table 1, Figure 3) and LSM training
need only **VLA-3D**. The closed-loop experiments (Figure 4, Table 2) also need
**RGB-D trajectories** through 13 Matterport3D scenes.

## VLA-3D

The public release of [VLA-3D](https://github.com/HaochenZ11/VLA-3D) (Zhang et al., 2024),
unmodified. Download it following that repository's instructions and unpack it so that

```
$VLA3D_ROOT/
  3RScan/  ARKitScenes/  HM3D/  Matterport/  Scannet/  Unity/
    <scene_id>/
      <scene_id>_scene_graph.json
      <scene_id>_referential_statements.json
      <scene_id>_pc_result.ply
      <scene_id>_object_split.npy
```

and point `VLA3D_ROOT` at it (default `data/VLA-3D`). Evaluation reads only the
scene graphs; point clouds are used for closed-loop episode generation.

### The canonical split

Every number in the paper is computed on one fixed partition, shipped in the
artifacts as `artifacts/splits/` (train 1,399,294 / val_seen 74,389 / val_unseen
116,084 statements over 7,147 scenes). It is what
`scripts/build_canonical_split.py` produces from VLA-3D:

* 5% of all scenes (pooled across the six datasets, seed 42) are held out entirely
  → `val_unseen`; 5% of each remaining scene's statements → `val_seen`; the rest → `train`.
* A statement is kept if its relation is one of *on, in, near, above, below,
  between* and its target and anchors have a NYU40 label in
  `langsensor.core.ontology.VALID_NYU40_LABELS`.
* Its **ambiguity** is the number of ways to resolve its anchors: the product,
  over anchors, of how many scene objects share the anchor's label.

Two details matter if you rebuild it:

* The label set includes `otherprop`, `otherfurniture`, `otherstructure` and
  `person`. The released checkpoint was trained with them; without them the
  corpus shrinks 2.46x and most of `val_seen` would overlap its training data.
* VLA-3D statements read "Find a ...", rewritten to "There is a ...". When
  several annotations share one statement text, the original loader applied the
  rewrite cumulatively ("There is a e is a ..."), which affects ~2% of statements.
  The checkpoint was trained on these strings, so the loader keeps the behaviour
  (`langsensor/data/vla3d.py`).

The partition depends on the *whole* dataset list because scenes are shuffled
together; to evaluate on a subset, filter the loaded records instead of
narrowing `--datasets`.

## Closed-loop scenes

The closed loop replays posed RGB-D frames with ground-truth semantic labels
through 13 Matterport3D scenes of VLA-3D. Set `CLOSED_LOOP_ROOT` (default
`data/closed_loop`) to a directory with one folder per scene:

```
$CLOSED_LOOP_ROOT/<scene_id>/
  trajectory/
    camera_info.yaml          fx, fy, cx, cy, width, height (640x480, fx = fy = 320)
    poses.csv                 timestamp_ns, tx, ty, tz, qw, qx, qy, qz (world <- camera)
    color/rgb_%07d.png        uint8 RGB
    depth/depth_%07d.tiff     float32 metres
    labels/labels_%07d.png    ADE20K-MP3D class ids (langsensor/vlmap/run.py: ADE20K_MP3D)
  scene/                      symlinks to the scene's VLA-3D files (created by make_episodes.py)
  episodes/                   created by make_episodes.py
```

Poses are rotated by 90° about z on load to align them with the VLA-3D point
clouds (`langsensor/vlmap/episodes.py`).

> **Trajectories.** The paper uses one trajectory per scene (259-1,370 frames,
> ~16 GB in total). **TODO(authors): state where they come from and how to
> obtain them.** They are derived from Matterport3D, whose license terms apply.

### Episodes

```bash
python scripts/make_episodes.py
```

samples up to 10 statements per scene (ambiguity 1-4; relations near, between,
above, below, on; one of 12 furniture-sized target classes with a bounding box of
at least 0.4 m³; seed 121), writes each as an episode with the target removed
from its scene graph, and stores the target's surface points
(`gt_surface.npy`, one point per 2.5 cm voxel). With the defaults it regenerates
the paper's 54 episodes exactly. Which episodes the trajectory never observes
(the "unobserved" block of Table 2) is recorded in
`artifacts/results/closed_loop_2026-05-14/_summary.json`.
