"""The Language Sensor Model (LSM): utterance + grounded scene graph -> 3D Gaussian.

Forward pass (B = batch, N = objects, D = hidden_dim):

    text  -> BERT [CLS]                                  (B, D)
    objects: CLIP label + anchor role (+ anchor offset)  (B, N, D)
    pairwise geometry, FiLM'd by the text                (B, N, N, 12)
    spatial-attention backbone over objects              (B, N, D)
    fusion transformer over [text, Q_target, objects]    -> Q_target state (B, D)
    FiLM on the region frame, Gaussian head              -> μ (B, 3), L (B, 3, 3), world frame

The model works in a region-normalised frame, p_region = (p_world - shift) / scale,
and the head maps its prediction back to world coordinates.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from langsensor.lsm.backbone import BACKBONES
from langsensor.lsm.config import LSMConfig
from langsensor.lsm.encoders import AnchorCentricEmbedding, SpatialRelationMLP, TextEncoder, VisionEncoder


class GaussianPrediction(NamedTuple):
    mu: torch.Tensor   # (B, 3) world frame
    L: torch.Tensor    # (B, 3, 3) lower-triangular, Σ = L Lᵀ, world frame


# ── Pooling ───────────────────────────────────────────────────────────────────

class QueryTokenPooling(nn.Module):
    """A learnable target-query token inserted after the text token; its fused state is the context."""

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(cfg.hidden_dim))
        nn.init.normal_(self.query, std=cfg.hidden_dim ** -0.5)

    def prepend(self, seq: torch.Tensor, mask: torch.Tensor):
        B = seq.size(0)
        q = self.query.to(seq.dtype).view(1, 1, -1).expand(B, 1, -1)
        q_mask = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        return (torch.cat([seq[:, :1], q, seq[:, 1:]], dim=1),
                torch.cat([mask[:, :1], q_mask, mask[:, 1:]], dim=1))

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        return x[:, 1]


class MeanPooling(nn.Module):
    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        if padding_mask is None:
            return x.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


POOLING = {"query_token": QueryTokenPooling, "mean": MeanPooling}


# ── Region conditioning and output head ───────────────────────────────────────

class FiLMLayer(nn.Module):
    """γ(region) * x + β(region), from [coord_scale, coord_shift]; identity at init."""

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, cfg.film_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.film_hidden_dim, 2 * cfg.hidden_dim),
        )
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            bias = torch.zeros(2 * cfg.hidden_dim)
            bias[: cfg.hidden_dim] = 1.0
            self.mlp[-1].bias.copy_(bias)

    def forward(self, x, coord_scale, coord_shift) -> torch.Tensor:
        out = self.mlp(torch.cat([coord_scale, coord_shift], dim=-1))
        D = x.size(1)
        return out[:, :D] * x + out[:, D:]


class GaussianCholeskyHead(nn.Module):
    """Predict μ (tanh-bounded, region frame) and a Cholesky factor, then map to world frame.

        μ_world = μ_region * scale + shift,   L_world = diag(scale) @ L_region
    """

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 9),        # 3 for μ, 6 for L
        )
        self.min_sigma = max(float(cfg.head_min_sigma), 1e-4)
        self.register_buffer("tril_idx", torch.tensor([[0, 1, 2, 1, 2, 2],
                                                       [0, 1, 2, 0, 0, 1]], dtype=torch.long))

    def forward(self, x, coord_scale, coord_shift) -> GaussianPrediction:
        out = self.mlp(x)
        mu_region = torch.tanh(out[:, :3])
        l_raw = out[:, 3:]

        rows, cols = self.tril_idx
        diag = rows == cols
        L_region = torch.zeros(x.size(0), 3, 3, device=x.device, dtype=out.dtype)
        L_region[:, rows[diag], cols[diag]] = F.softplus(l_raw[:, diag]).to(L_region.dtype) + self.min_sigma
        L_region[:, rows[~diag], cols[~diag]] = l_raw[:, ~diag]

        return GaussianPrediction(
            mu=mu_region * coord_scale + coord_shift,
            L=coord_scale.unsqueeze(-1) * L_region,
        )


def center_nll_loss(
    pred: GaussianPrediction,
    target_xyz: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_mahal: float = 0.1,
    lambda_vol: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gaussian NLL of the target centre + λ₁·L1(μ) + λ_m·Mahalanobis + λ_v·log det Σ."""
    diff = target_xyz - pred.mu
    log_det = 2.0 * pred.L.diagonal(dim1=-2, dim2=-1).clamp(min=eps).log().sum(dim=-1)
    z = torch.linalg.solve_triangular(pred.L, diff.unsqueeze(-1), upper=False).squeeze(-1)
    quad = (z * z).sum(dim=-1)
    nll = 0.5 * (log_det + quad + 3 * math.log(2.0 * math.pi))
    loss = nll + lambda_l1 * diff.abs().sum(dim=-1) + lambda_mahal * quad.clamp(min=1e-12).sqrt() + lambda_vol * log_det
    return loss.mean()


# ── Model ─────────────────────────────────────────────────────────────────────

class GlobalFusionTransformer(nn.Module):
    """Pre-norm transformer over [text, (query), objects]; lets the text attend to anchor-tagged objects."""

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim, nhead=cfg.num_heads, dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=cfg.num_fusion_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x, src_key_padding_mask=None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class LSMModel(nn.Module):
    _MODALITY_TEXT = 0

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        if not cfg.use_global_fusion and cfg.pooling_type == "query_token":
            raise ValueError("use_global_fusion=False requires pooling_type='mean'")
        self.cfg = cfg
        self.text_enc = TextEncoder(cfg)
        self.vision_enc = VisionEncoder(cfg)
        self.spatial_enc = SpatialRelationMLP(cfg)
        self.anchor_centric = AnchorCentricEmbedding(cfg) if cfg.use_anchor_centric_coords else None
        self.backbone = BACKBONES[cfg.backbone_type](cfg)
        self.fusion = GlobalFusionTransformer(cfg)
        self.modality_embed = nn.Embedding(2, cfg.hidden_dim)
        self.pool = POOLING[cfg.pooling_type](cfg)
        self.film = FiLMLayer(cfg) if cfg.use_film else None
        self.head = GaussianCholeskyHead(cfg)

    def forward(
        self,
        text_input_ids, text_attention_mask,
        obj_clip_features, obj_bboxes, obj_is_anchor, obj_padding_mask,
        coord_scale, coord_shift,
    ) -> GaussianPrediction:
        if self.cfg.zero_bboxes:
            obj_bboxes = torch.zeros_like(obj_bboxes)

        text_cls = self.text_enc(text_input_ids, text_attention_mask)
        obj = self.vision_enc(obj_clip_features, obj_is_anchor)
        if self.anchor_centric is not None:
            obj = self.anchor_centric(obj, obj_bboxes, obj_is_anchor, obj_padding_mask)
        obj = self.backbone(obj, self.spatial_enc(obj_bboxes, text_cls), obj_padding_mask)

        if self.cfg.use_global_fusion:
            text_tok = (text_cls + self.modality_embed.weight[self._MODALITY_TEXT]).unsqueeze(1)
            seq = torch.cat([text_tok, obj], dim=1)
            mask = torch.cat([torch.zeros_like(obj_padding_mask[:, :1]), obj_padding_mask], dim=1)
            if isinstance(self.pool, QueryTokenPooling):
                seq, mask = self.pool.prepend(seq, mask)
            ctx = self.pool(self.fusion(seq, src_key_padding_mask=mask), mask)
        else:
            ctx = self.pool(obj, obj_padding_mask) + text_cls

        if self.film is not None:
            ctx = self.film(ctx, coord_scale, coord_shift)
        return self.head(ctx, coord_scale, coord_shift)

    def forward_batch(self, b) -> GaussianPrediction:
        """Forward a :class:`TensorizerOutput` batch."""
        return self(b.text_input_ids, b.text_attention_mask, b.obj_clip_features, b.obj_bboxes,
                    b.obj_is_anchor, b.obj_padding_mask, b.coord_scale, b.coord_shift)


def load_lsm(checkpoint: str | Path, device: str | torch.device = "cpu") -> LSMModel:
    """Build an LSMModel from a checkpoint ({"model": state_dict, "cfg": {"model": ...}}).

    A frozen text encoder may be left out of the checkpoint (as in the released
    one); it is then the pretrained Hugging Face model at ``text_model_revision``.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = LSMConfig.from_dict(dict(ckpt["cfg"]["model"]))
    model = LSMModel(cfg)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bert_only = all(k.startswith("text_enc.bert.") for k in missing)
    if unexpected or (missing and not (bert_only and cfg.freeze_text)):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()
