import pytest
import torch

from langsensor.lsm.config import LSMConfig
from langsensor.lsm.model import LSMModel, center_nll_loss


def _inputs(B=2, N=10, L=8):
    pad = torch.zeros(B, N, dtype=torch.bool)
    pad[:, 7:] = True
    anchor = torch.zeros(B, N, dtype=torch.bool)
    anchor[:, 0] = True
    return (torch.randint(1000, 2000, (B, L)), torch.ones(B, L, dtype=torch.long), torch.randn(B, N, 512),
            torch.rand(B, N, 6) - 0.5, anchor, pad, torch.rand(B, 3) + 1, torch.randn(B, 3))


@pytest.mark.parametrize("overrides", [
    {},
    {"backbone_type": "identity"},
    {"use_global_fusion": False, "pooling_type": "mean"},
    {"use_film": False, "use_anchor_centric_coords": False, "zero_bboxes": True},
])
def test_forward_gives_valid_gaussians(overrides):
    torch.manual_seed(0)
    model = LSMModel(LSMConfig(**overrides)).eval()
    pred = model(*_inputs())
    assert pred.mu.shape == (2, 3) and pred.L.shape == (2, 3, 3)
    assert torch.equal(pred.L, torch.tril(pred.L)) and (pred.L.diagonal(dim1=-2, dim2=-1) > 0).all()
    assert torch.isfinite(center_nll_loss(pred, torch.randn(2, 3)))


def test_config_accepts_legacy_checkpoint_keys():
    cfg = LSMConfig.from_dict({"hidden_dim": 256, "bert_dim": 768, "head_type": "gaussian_cholesky"})
    assert cfg.hidden_dim == 256
    with pytest.raises(ValueError):
        LSMConfig.from_dict({"head_type": "gaussian_diagonal"})
