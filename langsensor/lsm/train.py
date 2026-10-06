"""Train the LSM.

    python -m langsensor.lsm.train configs/lsm/train.yaml
    python -m langsensor.lsm.train configs/lsm/ablations/no_film.yaml training.epochs=20

Any config value can be overridden as ``key=value``. Checkpoints are written to
``checkpoint.dir`` as ``best.pt`` (lowest val_seen loss) and ``epoch_NNNN.pt``.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from langsensor.config import load_config
from langsensor.core.schema import TensorizerOutput
from langsensor.lsm.config import LSMConfig
from langsensor.lsm.dataset import CollateFn, Compose, RandomMaskObjects, RandomObjectJitter, open_split, to_device
from langsensor.lsm.model import GaussianPrediction, LSMModel, center_nll_loss

PRECISION = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(cfg: DictConfig, pred: GaussianPrediction, batch: TensorizerOutput) -> torch.Tensor:
    return center_nll_loss(pred, batch.target_xyz_world, lambda_l1=cfg.lambda_l1,
                           lambda_mahal=cfg.lambda_mahal, lambda_vol=cfg.lambda_vol)


@torch.no_grad()
def evaluate(model, loader, device, dtype, loss_cfg) -> dict[str, float]:
    model.eval()
    total, n = 0.0, 0
    dist, nll, quad = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            pred = model.forward_batch(batch)
            loss = loss_fn(loss_cfg, pred, batch)
        B = pred.mu.shape[0]
        total, n = total + loss.item() * B, n + B

        mu, L = pred.mu.float(), pred.L.float()
        diff = batch.target_xyz_world.float() - mu
        z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
        q = (z * z).sum(-1)
        log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6).log().sum(-1)
        dist.append(diff.norm(dim=-1).cpu())
        quad.append(q.cpu())
        nll.append((0.5 * (log_det + q + 3.0 * math.log(2.0 * math.pi))).cpu())
    model.train()
    dist, nll, quad = torch.cat(dist), torch.cat(nll), torch.cat(quad)
    return {
        "loss": total / max(n, 1),
        "mean_dist": dist.mean().item(),
        "nll_mean": nll.mean().item(),
        "nll_median": nll.median().item(),
        "anees": quad.mean().item(),       # per-sample NEES (single Gaussian); 3 = calibrated
    }


def train(cfg: DictConfig) -> float:
    """Train and return the best val_seen loss."""
    tr = cfg.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = PRECISION[tr.precision]
    seed_everything(tr.seed)

    aug = cfg.augmentation
    augment = Compose([t for t in (
        RandomMaskObjects(aug.mask_prob) if aug.mask_prob > 0 else None,
        RandomObjectJitter(aug.jitter_std) if aug.jitter_std > 0 else None,
    ) if t is not None])
    cache = Path(cfg.cache_dir)
    train_ds = open_split(cache / "train", augment, shuffle=True)
    val_sets = {name: open_split(cache / name) for name in ("val_seen", "val_unseen") if (cache / name).exists()}

    model_cfg = LSMConfig.from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    collate = CollateFn(model_cfg.text_model, model_cfg.max_text_len)
    loader_kw = dict(batch_size=tr.batch_size, num_workers=tr.num_workers, collate_fn=collate,
                     pin_memory=device.type == "cuda", persistent_workers=tr.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=not isinstance(train_ds, IterableDataset),
                              drop_last=True, **loader_kw)
    val_loaders = {name: DataLoader(ds, shuffle=False, **loader_kw) for name, ds in val_sets.items()}

    model = LSMModel(model_cfg).to(device)
    # BERT gets lr * bert_lr_scale (0 for the released model: BERT is frozen).
    bert = {id(p) for p in model.text_enc.bert.parameters()}
    if device.type == "cuda" and tr.compile:
        model = torch.compile(model, dynamic=False)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in bert], "lr": tr.lr * tr.bert_lr_scale},
        {"params": [p for p in model.parameters() if id(p) not in bert], "lr": tr.lr},
    ], weight_decay=tr.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, tr.warmup_steps, len(train_loader) * tr.epochs)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    wandb = None
    if cfg.wandb.enabled:
        import wandb
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, tags=list(cfg.wandb.tags),
                   config=OmegaConf.to_container(cfg, resolve=True))

    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    def save(path: Path, epoch: int, **extra) -> None:
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cfg": cfg_dict, **extra}, path)

    best, step = float("inf"), 0
    max_steps = int(tr.max_train_steps_per_epoch or 0)
    for epoch in range(1, tr.epochs + 1):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{tr.epochs}", leave=False)
        for batch in pbar:
            if max_steps and n_steps >= max_steps:
                break
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                loss = loss_fn(cfg.loss, model.forward_batch(batch), batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss, n_steps, step = epoch_loss + loss.item(), n_steps + 1, step + 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if wandb and step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[1], "step": step})

        results = {name: evaluate(model, dl, device, dtype, cfg.loss) for name, dl in val_loaders.items()}
        line = "  ".join(f"{name}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items()) for name, m in results.items())
        print(f"[epoch {epoch:3d}] train_loss={epoch_loss / max(n_steps, 1):.4f}  {line}", flush=True)
        if wandb:
            wandb.log({"epoch": epoch, "train/epoch_loss": epoch_loss / max(n_steps, 1),
                       **{f"{s}/{k}": v for s, m in results.items() for k, v in m.items()}})

        val_loss = results.get("val_seen", {}).get("loss", float("inf"))
        if val_loss < best:
            best = val_loss
            save(ckpt_dir / "best.pt", epoch, val_loss=val_loss)
        if epoch % cfg.checkpoint.save_every == 0:
            save(ckpt_dir / f"epoch_{epoch:04d}.pt", epoch)

    if wandb:
        wandb.finish()
    return best


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or "=" in argv[0]:
        raise SystemExit("usage: python -m langsensor.lsm.train <config.yaml> [key=value ...]")
    train(load_config(argv[0], argv[1:]))


if __name__ == "__main__":
    main()
