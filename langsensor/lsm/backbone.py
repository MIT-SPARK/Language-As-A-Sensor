"""Object-level spatial encoder: self-attention with an additive pairwise-geometry bias (3D-VisTA style)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from langsensor.lsm.config import LSMConfig


class MultiHeadAttentionSpatial(nn.Module):
    """Multi-head attention whose logits get one learned bias per head from pairwise features."""

    def __init__(self, d_model: int, num_heads: int, spatial_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.spatial_proj = nn.Linear(spatial_dim, num_heads, bias=False)

    def forward(self, x, spatial_relations, key_padding_mask=None) -> torch.Tensor:
        B, N, D = x.shape

        def heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N, d)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        logits = q @ k.transpose(-2, -1) * self.head_dim ** -0.5
        logits = logits + self.spatial_proj(spatial_relations).permute(0, 3, 1, 2)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        # A row whose keys are all padding softmaxes to NaN; zero it instead.
        attn = torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        return self.out_proj((attn @ v).transpose(1, 2).contiguous().view(B, N, D))


class SpatialEncoderLayer(nn.Module):
    """Pre-norm block: spatial attention, then FFN."""

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttentionSpatial(
            cfg.hidden_dim, cfg.num_heads, cfg.spatial_relation_dim, cfg.dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, spatial_relations, key_padding_mask=None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), spatial_relations, key_padding_mask))
        return x + self.dropout(self.ffn(self.norm2(x)))


class SceneSpatialEncoder(nn.Module):
    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SpatialEncoderLayer(cfg) for _ in range(cfg.num_spatial_layers)])

    def forward(self, obj_features, spatial_relations, key_padding_mask=None) -> torch.Tensor:
        x = obj_features
        for layer in self.layers:
            x = layer(x, spatial_relations, key_padding_mask)
        return x


class IdentityBackbone(nn.Module):
    """'No spatial backbone' ablation: object features pass through unchanged."""

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()

    def forward(self, obj_features, spatial_relations, key_padding_mask=None) -> torch.Tensor:
        return obj_features


BACKBONES = {"scene_spatial": SceneSpatialEncoder, "identity": IdentityBackbone}
