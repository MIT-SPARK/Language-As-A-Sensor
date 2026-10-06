"""Hydra's TSDF + top-k semantic integrator as a :class:`VoxelMap`.

Requires the ``hydra_python`` bindings built from MIT-SPARK/Hydra with the
``get_filled_semantic_voxels`` extension — see docs/hydra.md. Only the
volumetric reconstruction is used (no scene-graph layers), fed with
ground-truth semantic labels.
"""

from __future__ import annotations

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from langsensor.vlmap.types import VisionObservation
from langsensor.vlmap.voxel_map import VoxelBatch


class HydraVoxelMap:
    def __init__(
        self,
        camera_info: dict,
        label_names: dict[int, str],
        voxel_size: float = 0.1,
        voxels_per_side: int = 16,
        top_k: int = 15,
        min_tsdf_weight: float = 1.0,
    ) -> None:
        try:
            from hydra_python._hydra_bindings import Camera, CameraConfig, ExtrinsicsConfig, HydraReconstruction
        except ImportError as e:
            raise ImportError("hydra_python is not installed; see docs/hydra.md") from e

        cfg = {
            "visualize_mesh": False,
            "reconstruction": {
                "volumetric_map": {"voxel_size": voxel_size, "voxels_per_side": voxels_per_side,
                                   "with_semantics": True},
                "tsdf": {"semantic_integrator": {"type": "FirstKSemanticIntegrator", "k": top_k}},
            },
            "label_space": {"total_labels": len(label_names)},
            "label_names": [{"label": k, "name": v} for k, v in sorted(label_names.items())],
        }
        cam = CameraConfig()
        cam.width, cam.height = int(camera_info["width"]), int(camera_info["height"])
        cam.fx, cam.fy = float(camera_info["fx"]), float(camera_info["fy"])
        cam.cx, cam.cy = float(camera_info["cx"]), float(camera_info["cy"])
        cam.min_range, cam.max_range = 0.1, 10.0
        cam.extrinsics = ExtrinsicsConfig()

        self._recon = HydraReconstruction.from_config(yaml.dump(cfg), Camera(cam, "camera"))
        if not hasattr(self._recon.map, "get_filled_semantic_voxels"):
            raise RuntimeError("hydra_python lacks map.get_filled_semantic_voxels; build the branch in docs/hydra.md")
        self._top_k = top_k
        self._min_tsdf_weight = min_tsdf_weight
        self.truncation_distance = float(self._recon.map.truncation_distance)

    def integrate(self, obs: VisionObservation) -> VoxelBatch:
        q = Rotation.from_matrix(obs.pose[:3, :3]).as_quat()           # xyzw
        self._recon.step(
            obs.timestep, obs.pose[:3, 3], np.array([q[3], q[0], q[1], q[2]]),
            obs.depth.astype(np.float32), obs.gt_semantics.astype(np.int32), obs.rgb,
        )
        b = self._recon.map.get_filled_semantic_voxels(top_k=self._top_k, min_tsdf_weight=self._min_tsdf_weight)
        return VoxelBatch(b["keys"], b["weights"], b["label_ids"], b["tsdf_distance"], b["has_semantic"], b["has_tsdf"])
