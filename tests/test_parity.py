"""The released checkpoint and split, run end to end, give the recorded predictions.

Needs the artifacts (scripts/download_artifacts.py) and VLA-3D (VLA3D_ROOT);
skipped otherwise. The fixture was recorded on CPU (float32) with this code, whose
GPU predictions match the original implementation bit for bit.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from langsensor import paths

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "lsm_val_seen_cpu.json").read_text())

pytestmark = pytest.mark.skipif(
    not (paths.CHECKPOINT.exists() and paths.SPLITS.exists() and paths.VLA3D_ROOT.exists()),
    reason="needs artifacts/ and VLA-3D",
)


def test_split_size():
    from langsensor.data.canonical import load_split
    assert len(load_split("val_seen")) == FIXTURE["n_val_seen"]


def test_lsm_predictions_match_fixture():
    from langsensor.data.canonical import load_split
    from langsensor.eval.benchmark import records_to_queries
    from langsensor.predictors.base import GroundTruthProposer
    from langsensor.predictors.lsm import LSMPredictor

    queries = records_to_queries(load_split("val_seen")[: len(FIXTURE["queries"])])
    predictor = LSMPredictor(device="cpu")
    for q, expected in zip(queries, FIXTURE["queries"]):
        assert q.language == expected["text"]
        mu, L = predictor.predict(q, GroundTruthProposer().propose(q))
        np.testing.assert_allclose(mu[0].numpy(), expected["mu"], atol=1e-4)
        np.testing.assert_allclose(L[0].numpy().ravel(), expected["L"], atol=1e-4)
