# Hydra (closed-loop experiments only)

VLMaP uses [Hydra](https://github.com/MIT-SPARK/Hydra) only as a volumetric
mapper: a TSDF with a top-k semantic integrator, fed posed depth and
ground-truth labels. No scene-graph layers are used. The single file that
touches it is `langsensor/vlmap/hydra_map.py`, which implements the
`VoxelMap` interface in `langsensor/vlmap/voxel_map.py`; any other mapper that
returns, per filled voxel, a TSDF distance and top-k label weights can replace it.

Nothing else in the repository needs Hydra: training, the static benchmark and
rendering every figure and table work without it.

## What is needed

The Python bindings (`hydra_python`) with a `map.get_filled_semantic_voxels()`
method, which is not in upstream Hydra. It is on the branch
`feature/semantic_window_python` of MIT-SPARK/Hydra; the paper's sweep used
commit **`8bd55289`**.

The bindings link against Hydra's C++ libraries, so build the ROS 2 workspace
first. Versions contemporaneous with the sweep (last commit on or before
2026-05-14, unverified as a set):

| Package | Repository | Commit |
|---|---|---|
| hydra | MIT-SPARK/Hydra | `8bd55289` (branch `feature/semantic_window_python`) |
| config_utilities | MIT-SPARK/config_utilities | `629688a` |
| spark_dsg | MIT-SPARK/Spark-DSG | `c6e7430` |
| spatial_hash | MIT-SPARK/Spatial-Hash | `b5be4d4` |
| kimera_pgmo | MIT-SPARK/Kimera-PGMO | `8fcfd86` |
| kimera_rpgo | MIT-SPARK/Kimera-RPGO | `f1fee09` |
| pose_graph_tools | MIT-SPARK/pose_graph_tools | `23b7876` |
| teaser_plusplus | MIT-SPARK/TEASER-plusplus | `52a9c52` |
| ianvs | MIT-SPARK/Ianvs | `e76fc7c` |

(ROS 2 Jazzy, which also provides GTSAM.)

## Build

1. Set up a ROS 2 workspace for Hydra following the installation instructions of
   [Hydra-ROS](https://github.com/MIT-SPARK/Hydra-ROS), then check out the
   commits in the table (Hydra on `feature/semantic_window_python`).

2. Build it and put its libraries on the loader path:

   ```bash
   cd ~/hydra_ws && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
   source install/setup.bash
   ```

3. Install the Python bindings (the Hydra repository root is a Python project)
   into this repository's environment, and check them:

   ```bash
   CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH" pip install ~/hydra_ws/src/hydra
   python -c "from hydra_python._hydra_bindings import HydraReconstruction; print('ok')"
   ```

`install/setup.bash` must be sourced in every shell that runs
`langsensor.vlmap.run`.

## Troubleshooting

* **`Could not find a package configuration file provided by hydra`** when
  building the bindings: pass the workspace prefix through `CMAKE_ARGS` as
  above; `CMAKE_PREFIX_PATH` alone does not reach pip's isolated build.
* **Segmentation fault on the first `HydraReconstruction.step()`**: the
  bindings were built against a different `libhydra.so` than the one on the
  loader path (e.g. the workspace was rebuilt from another branch). Rebuild the
  workspace and the bindings from the same commits.
* **`Cannot register already existent type ...` errors** at start-up are
  harmless; the paper's sweep printed them too.
