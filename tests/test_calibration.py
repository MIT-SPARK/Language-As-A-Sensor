import numpy as np
import pytest
import torch

from langsensor.predictors.calibration import apply_params, fit_params
from langsensor.predictors.llm_client import extract_json_object, parse_mu_sigma, psd_cholesky


def _whitened(sigmas, n=20000, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 3)) * np.asarray(sigmas)


@pytest.mark.parametrize("mode", ["scale", "axis_scale"])
def test_fit_makes_anees_three(mode):
    z = _whitened([2.0, 0.5, 1.0])
    params = np.asarray(fit_params(mode, z))
    anees = float((z ** 2 / params ** 2).sum(axis=1).mean())
    assert anees == pytest.approx(3.0, rel=1e-6)


def test_axis_scale_recovers_per_axis_spread():
    params = fit_params("axis_scale", _whitened([2.0, 0.5, 1.0]))
    assert params == pytest.approx([2.0, 0.5, 1.0], rel=0.03)


def test_apply_keeps_cholesky_structure():
    L = torch.tensor([[[1.0, 0, 0], [0.3, 0.8, 0], [-0.2, 0.1, 0.5]]])
    for mode, params in (("scale", [2.0]), ("axis_scale", [2.0, 0.5, 3.0])):
        out = apply_params(mode, params, L)
        assert torch.equal(out, torch.tril(out)) and (out.diagonal(dim1=-2, dim2=-1) > 0).all()


def test_psd_projection_of_bad_covariance():
    L = psd_cholesky([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    S = L @ L.T
    assert np.all(np.linalg.eigvalsh(S) >= 0.01 - 1e-6)


def test_parse_llm_output():
    obj = extract_json_object('Sure! {"mu": [1, 2, 3], "sigma": [0.1, 0.2, 0.3], "explanation": "x"} done')
    mu, L = parse_mu_sigma(obj)
    assert mu.tolist() == [1, 2, 3]
    assert np.allclose(np.diag(L @ L.T), [0.1, 0.2, 0.3])
    assert parse_mu_sigma({"mu": [1, 2]}) is None
