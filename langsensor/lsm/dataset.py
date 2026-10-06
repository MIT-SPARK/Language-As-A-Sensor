"""Training data: pre-tensorized samples on disk, augmentations, and batch collation.

``scripts/preprocess.py`` writes one of two layouts per split:

    <cache>/<split>/manifest.json + *.pt       one CachedSample per file
    <cache>/<split>/shards.json + shard-*.tar  WebDataset shards (better on network storage)
"""

from __future__ import annotations

import io
import json
from dataclasses import fields, replace
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer

from langsensor.core.schema import CachedSample, TensorizerOutput


class CachedDataset(Dataset):
    def __init__(self, split_dir: str | Path, augment=None) -> None:
        self.split_dir = Path(split_dir)
        self.augment = augment
        manifest = self.split_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} not found; run scripts/preprocess.py first")
        self.files: list[str] = json.loads(manifest.read_text())

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> CachedSample:
        item = torch.load(self.split_dir / self.files[idx], weights_only=False)
        return self.augment(item) if self.augment else item


class ShardedDataset(IterableDataset):
    def __init__(self, split_dir: str | Path, augment=None, shuffle_buffer: int = 0) -> None:
        super().__init__()
        self.split_dir = Path(split_dir)
        self.augment = augment
        self.shuffle_buffer = shuffle_buffer
        index = json.loads((self.split_dir / "shards.json").read_text())
        self.shards: list[str] = index["shards"]
        self.n_samples = int(index["n_samples"])

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):
        import webdataset as wds

        urls = [str(self.split_dir / s) for s in self.shards]
        pipeline = wds.WebDataset(urls, shardshuffle=len(urls) if self.shuffle_buffer else False,
                                  nodesplitter=wds.split_by_node)
        if self.shuffle_buffer:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        for record in pipeline:
            if "sample" not in record:
                continue
            item = torch.load(io.BytesIO(record["sample"]), weights_only=False, map_location="cpu")
            yield self.augment(item) if self.augment else item


def open_split(split_dir: str | Path, augment=None, shuffle: bool = False):
    split_dir = Path(split_dir)
    if (split_dir / "shards.json").is_file():
        return ShardedDataset(split_dir, augment, shuffle_buffer=8000 if shuffle else 0)
    return CachedDataset(split_dir, augment)


def to_device(batch: TensorizerOutput, device) -> TensorizerOutput:
    return replace(batch, **{f.name: getattr(batch, f.name).to(device) for f in fields(TensorizerOutput)})


class CollateFn:
    """Stack CachedSamples and tokenize their text into a TensorizerOutput."""

    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_text_len: int = 64) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = max_text_len
        self._tensor_fields = [f.name for f in fields(CachedSample) if f.name != "language"]

    def __call__(self, batch: list[CachedSample]) -> TensorizerOutput:
        enc = self.tokenizer([s.language for s in batch], max_length=self.max_text_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return TensorizerOutput(
            text_input_ids=enc["input_ids"],
            text_attention_mask=enc["attention_mask"],
            **{name: torch.stack([getattr(s, name) for s in batch]) for name in self._tensor_fields},
        )


# ── Augmentations (train split only) ──────────────────────────────────────────

class RandomMaskObjects:
    """Hide each non-anchor object with probability *p*, never hiding every object."""

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, item: CachedSample) -> CachedSample:
        eligible = ~item.obj_is_anchor & ~item.obj_padding_mask
        drop = eligible & (torch.rand(item.obj_padding_mask.shape[0]) < self.p)
        if not drop.any():
            return item
        clip, boxes, pad = item.obj_clip_features.clone(), item.obj_bboxes.clone(), item.obj_padding_mask.clone()
        clip[drop], boxes[drop], pad[drop] = 0.0, 0.0, True

        was_valid = ~item.obj_padding_mask
        if was_valid.any() and bool(pad[was_valid].all()):           # restore one object
            j = int(torch.nonzero(drop & was_valid).view(-1)[torch.randint(int(drop.sum()), (1,))].item())
            pad[j], clip[j], boxes[j] = False, item.obj_clip_features[j], item.obj_bboxes[j]
        return replace(item, obj_clip_features=clip, obj_bboxes=boxes, obj_padding_mask=pad)


class RandomObjectJitter:
    """Gaussian noise on object centres (region-normalised units)."""

    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def __call__(self, item: CachedSample) -> CachedSample:
        if self.std <= 0:
            return item
        boxes = item.obj_bboxes.clone()
        valid = ~item.obj_padding_mask
        boxes[valid, :3] += (torch.randn(int(valid.sum()), 3) * self.std).to(boxes.dtype)
        return replace(item, obj_bboxes=boxes)


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item
