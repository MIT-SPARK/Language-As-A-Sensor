import numpy as np
import pytest

from langsensor.vlmap.fusion import FusionPipeline, GMMBelief
from langsensor.vlmap.metrics import MetricsTracker
from langsensor.vlmap.types import VisionObservation
from langsensor.vlmap.vision_sensor import SemanticVisionSensor
from langsensor.vlmap.voxel_map import VoxelBatch

BOUNDS, Z = np.array([0.0, 2.0, 0.0, 2.0], np.float32), (0.0, 1.0)


def test_gmm_density_integrates_to_one():
    belief = GMMBelief(np.array([[1.0, 1.0, 0.5]]), np.eye(3)[None] * 0.1, np.array([1.0]))
    pipe = FusionPipeline(np.array([-1, 3, -1, 3], np.float32), (-1.5, 2.5), resolution=0.05)
    assert float(belief.query_batch(pipe.grid).sum()) * 0.05 ** 3 == pytest.approx(1.0, rel=1e-3)


def test_fused_belief_is_normalised_on_the_grid():
    pipe = FusionPipeline(BOUNDS, Z, 0.1)
    a = GMMBelief(np.array([[0.5, 0.5, 0.5]]), np.eye(3)[None] * 0.3, np.array([1.0]))
    b = GMMBelief(np.array([[1.5, 1.5, 0.5]]), np.eye(3)[None] * 0.3, np.array([1.0]))
    assert float(pipe.step([a, b], [1.0, 1.0]).query_batch(pipe.grid).sum()) == pytest.approx(1.0, rel=1e-5)


class _OneVoxelMap:
    """Every frame observes the same voxel, labelled 'sofa' (id 1) on a surface."""
    truncation_distance = 0.3

    def integrate(self, obs):
        return VoxelBatch(keys=np.array([[5, 5, 5]], np.int32), weights=np.array([[3.0, 0.0]], np.float32),
                          label_ids=np.array([[1, 0]], np.uint32), tsdf_distance=np.zeros(1, np.float32),
                          has_semantic=np.array([True]), has_tsdf=np.array([True]))


def _embed(text):
    return np.array([1.0, 0.0]) if "sofa" in text else np.array([0.0, 1.0])


def test_vision_sensor_scores_matching_voxel_above_unobserved():
    sensor = SemanticVisionSensor(_OneVoxelMap(), "sofa", {0: "wall", 1: "sofa"}, BOUNDS, Z, embed=_embed)
    obs = VisionObservation(0, None, None, np.eye(4), None)
    belief = sensor(obs)
    seen, unseen = belief.query_batch(np.array([[0.55, 0.55, 0.55], [1.55, 1.55, 0.55]], np.float32))
    # exp(cos=1) * sigmoid(0) = e / 2
    assert seen == pytest.approx(np.e / 2, rel=1e-5) and unseen < 1e-4
    assert belief.coverage_batch(np.array([[0.55, 0.55, 0.55]], np.float32)).all()


def test_metrics_surface_mass_and_information_gain():
    pipe = FusionPipeline(BOUNDS, Z, 0.1)
    surface = np.array([[1.0, 1.0, 0.5]], np.float32)
    tracker = MetricsTracker(pipe.grid, surface[0], surface, surface_radius=0.15)
    tracker.record(GMMBelief(surface, np.eye(3)[None] * 0.05, np.array([1.0])))
    sm, ig = tracker.series["surface_mass"][0], tracker.series["info_gain_at_surface"][0]
    assert 0.5 < sm <= 1.0 and ig > 3.0
