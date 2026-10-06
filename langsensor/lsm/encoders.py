"""Input encoders: text, object labels, pairwise geometry, anchor-relative offsets.

Tensor conventions (B = batch, N = max_objects, L = text length):
    text_input_ids / text_attention_mask   (B, L)
    obj_clip_features                      (B, N, 512)
    obj_bboxes                             (B, N, 6)  [cx, cy, cz, w, h, l], region frame
    obj_is_anchor / obj_padding_mask       (B, N)     padding mask: True = empty slot
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from langsensor.lsm.config import LSMConfig


class TextEncoder(nn.Module):
    """BERT [CLS] embedding projected to hidden_dim."""

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        kw = {"revision": cfg.text_model_revision} if cfg.text_model_revision else {}
        try:
            self.bert = AutoModel.from_pretrained(cfg.text_model, attn_implementation="sdpa", **kw)
        except (ValueError, TypeError):
            self.bert = AutoModel.from_pretrained(cfg.text_model, **kw)
        if cfg.freeze_text:
            self.bert.requires_grad_(False)
        self.proj = nn.Linear(int(self.bert.config.hidden_size), cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.dropout(self.proj(cls))                                  # (B, D)


class VisionEncoder(nn.Module):
    """Project frozen CLIP label embeddings and add a shared anchor/non-anchor role embedding.

    All anchors share one role vector, so the model is invariant to anchor order.
    """

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.clip_proj = nn.Linear(cfg.clip_dim, cfg.hidden_dim)
        self.role_embed = nn.Embedding(2, cfg.hidden_dim)                   # 0 = other, 1 = anchor
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, clip_features: torch.Tensor, is_anchor: torch.Tensor) -> torch.Tensor:
        x = self.clip_proj(clip_features) + self.role_embed(is_anchor.long())
        return self.dropout(self.norm(x))                                    # (B, N, D)


class SpatialRelationMLP(nn.Module):
    """Pairwise geometry [box_i, box_j] -> MLP, with the hidden layer FiLM-modulated by the text.

    Text conditioning lets the query reshape the spatial bias (e.g. "above"
    up-weights vertical pairs). The FiLM generator starts at identity.
    """

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.hidden = cfg.spatial_mlp_hidden
        self.fc1 = nn.Linear(cfg.spatial_relation_dim, cfg.spatial_mlp_hidden)
        self.fc2 = nn.Linear(cfg.spatial_mlp_hidden, cfg.spatial_relation_dim)
        self.text_film: nn.Linear | None = None
        if cfg.condition_spatial_on_text:
            self.text_film = nn.Linear(cfg.hidden_dim, 2 * cfg.spatial_mlp_hidden)
            with torch.no_grad():
                nn.init.zeros_(self.text_film.weight)
                bias = torch.zeros(2 * cfg.spatial_mlp_hidden)
                bias[: cfg.spatial_mlp_hidden] = 1.0                        # γ = 1, β = 0
                self.text_film.bias.copy_(bias)

    def forward(self, bboxes: torch.Tensor, text_cls: torch.Tensor) -> torch.Tensor:
        n = bboxes.size(1)
        pairs = torch.cat([
            bboxes.unsqueeze(2).expand(-1, -1, n, -1),
            bboxes.unsqueeze(1).expand(-1, n, -1, -1),
        ], dim=-1)                                                          # (B, N, N, 12)
        h = F.relu(self.fc1(pairs))
        if self.text_film is not None:
            film = self.text_film(text_cls).unsqueeze(1).unsqueeze(1)       # (B, 1, 1, 2H)
            h = film[..., : self.hidden] * h + film[..., self.hidden:]
        return self.fc2(h)                                                  # (B, N, N, R)


class AnchorCentricEmbedding(nn.Module):
    """Add an embedding of each object's offset from the anchors' centroid (origin if no anchor)."""

    def __init__(self, cfg: LSMConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, obj_feat, bboxes, is_anchor, padding_mask) -> torch.Tensor:
        centers = bboxes[..., :3]
        w = (is_anchor.bool() & ~padding_mask.bool()).to(centers.dtype).unsqueeze(-1)
        centroid = (centers * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return obj_feat + self.mlp(centers - centroid.unsqueeze(1))
