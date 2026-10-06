from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class LSMConfig:
    """Architecture of the Language Sensor Model. Defaults are the released checkpoint's."""

    hidden_dim: int = 256
    clip_dim: int = 512                   # CLIP ViT-B/32 label embeddings (frozen)
    text_model: str = "bert-base-uncased"
    text_model_revision: str | None = None   # Hugging Face commit; pins the frozen encoder's weights
    freeze_text: bool = True

    # Pairwise spatial features: [box_i, box_j] (12-d) -> MLP, FiLM-conditioned on the text.
    spatial_relation_dim: int = 12
    spatial_mlp_hidden: int = 64
    condition_spatial_on_text: bool = True

    use_anchor_centric_coords: bool = True   # add an embedding of (centre - anchor centroid)
    backbone_type: str = "scene_spatial"     # "scene_spatial" | "identity" (ablation)
    num_spatial_layers: int = 3

    num_fusion_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 1024
    use_global_fusion: bool = True           # ablation: pool objects alone, add text CLS
    zero_bboxes: bool = False                # ablation: remove all 3D geometry
    pooling_type: str = "query_token"        # "query_token" | "mean"

    head_min_sigma: float = 0.05             # σ floor, region-normalised units
    use_film: bool = True                    # condition the context on the region frame
    film_hidden_dim: int = 64
    dropout: float = 0.3

    max_objects: int = 150
    max_text_len: int = 64

    # Keys found in older checkpoints, with the only value this code supports.
    _LEGACY_FIXED = {
        "pairwise_rel_type": "mlp",
        "spatial_conditioning_type": "film",
        "head_type": "gaussian_cholesky",
    }
    # Keys found in older checkpoints that have no effect on the model.
    _LEGACY_UNUSED = {"bert_dim", "spatial_pairwise_dist_norm", "head_hidden_sizes", "head_dropout"}

    @classmethod
    def from_dict(cls, d: dict) -> LSMConfig:
        known = {f.name for f in fields(cls)}
        for key, required in cls._LEGACY_FIXED.items():
            if key in d and d[key] != required:
                raise ValueError(f"{key}={d[key]!r} is not supported (only {required!r})")
        unknown = set(d) - known - set(cls._LEGACY_FIXED) - cls._LEGACY_UNUSED
        if unknown:
            raise ValueError(f"Unknown LSMConfig keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})
