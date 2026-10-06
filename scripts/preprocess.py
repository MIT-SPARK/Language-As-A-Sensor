"""Tensorize the canonical splits into the training cache read by ``langsensor.lsm.train``.

Each statement becomes one CachedSample (the target removed from its scene,
the anchor region cropped and normalised), saved as ``<split>/<scene>_<idx>.pt``
with a ``manifest.json``. ``--shards`` additionally packs each split into
WebDataset tar shards, which stream far faster from network storage.

    python scripts/preprocess.py --out-dir outputs/tensor_cache
"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from langsensor import paths
from langsensor.core.schema import GroundedQuery
from langsensor.core.transforms import build_spatial_query
from langsensor.data.canonical import SPLIT_NAMES, load_canonical
from langsensor.lsm.tensorizer import Tensorizer


def tensorize_split(records, tensorizer: Tensorizer, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_scene = defaultdict(list)
    for idx, rec in enumerate(records):
        by_scene[rec.scene.scene_id].append((idx, rec))

    manifest, failed = [], 0
    for scene_id, recs in tqdm(by_scene.items(), desc=out.name):
        try:
            sg = recs[0][1].scene.load_scene_graph()
        except Exception as e:
            print(f"  [warn] skipping {scene_id}: {e}")
            failed += len(recs)
            continue
        for idx, rec in recs:
            try:
                sample = tensorizer([GroundedQuery.from_spatial_query(build_spatial_query(scene_id, sg, rec.statement))])[0]
            except Exception as e:
                print(f"  [warn] {scene_id} #{idx}: {e}")
                failed += 1
                continue
            name = f"{scene_id}_{idx:07d}.pt"
            torch.save(sample, out / name)
            manifest.append(name)
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"  {out.name}: {len(manifest):,} samples ({failed} failed)")


def shard_split(split_dir: Path, samples_per_shard: int = 1024) -> None:
    """Pack ``<split_dir>/*.pt`` into ``shard-NNNNNN.tar`` + ``shards.json``, in place."""
    files = json.loads((split_dir / "manifest.json").read_text())
    shards = []
    for start in range(0, len(files), samples_per_shard):
        name = f"shard-{start // samples_per_shard:06d}.tar"
        with tarfile.open(split_dir / name, "w") as tar:
            for f in files[start:start + samples_per_shard]:
                tar.add(split_dir / f, arcname=f"{f[:-3]}.sample")
        shards.append(name)
    (split_dir / "shards.json").write_text(json.dumps(
        {"shards": shards, "n_samples": len(files), "samples_per_shard": samples_per_shard}))
    print(f"  {split_dir.name}: {len(shards)} shards")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=paths.SPLITS)
    ap.add_argument("--data-root", type=Path, default=paths.VLA3D_ROOT)
    ap.add_argument("--clip-label-map", type=Path, default=paths.CLIP_LABEL_MAP)
    ap.add_argument("--out-dir", type=Path, default=paths.OUTPUTS / "tensor_cache")
    ap.add_argument("--max-objects", type=int, default=150)
    ap.add_argument("--shards", action="store_true", help="also write WebDataset tar shards")
    ap.add_argument("--limit", type=int, default=None, help="first N records per split (smoke tests)")
    args = ap.parse_args()

    tensorizer = Tensorizer(args.max_objects, torch.load(args.clip_label_map, weights_only=False))
    splits = load_canonical(args.splits, args.data_root, SPLIT_NAMES)
    for name in SPLIT_NAMES:
        tensorize_split(getattr(splits, name)[: args.limit], tensorizer, args.out_dir / name)
        if args.shards:
            shard_split(args.out_dir / name)


if __name__ == "__main__":
    main()
