"""YAML configs with ``key=value`` command-line overrides.

    cfg = load_config("configs/lsm/train.yaml", ["training.epochs=5", "model.use_film=false"])

A config may name a base file with a top-level ``base:`` key (resolved relative
to itself); the base is loaded first and the file's own values are merged on top.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    path = Path(path)
    cfg = OmegaConf.load(path)
    base = cfg.pop("base", None)
    if base is not None:
        cfg = OmegaConf.merge(load_config(path.parent / base), cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
