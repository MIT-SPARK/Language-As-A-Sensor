"""Build the canonical train / val_seen / val_unseen partition and the CLIP label map from raw VLA-3D.

Both come with the artifacts (docs/artifacts.md); rebuild them only to verify
or to change the partition. Walks every scene (~7.6k, 1-3 h).

    python scripts/build_canonical_split.py --data-root data/VLA-3D --out-dir outputs/splits

Smoke test (15 scenes, about a minute):

    python scripts/build_canonical_split.py --datasets Unity --out-dir /tmp/split_smoke --no-clip-map
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from langsensor import paths
from langsensor.data.canonical import save_canonical
from langsensor.data.splits import PAPER_DATASETS, SplitConfig, build_splits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=paths.VLA3D_ROOT)
    ap.add_argument("--datasets", nargs="+", default=list(PAPER_DATASETS),
                    help="Changing this re-shuffles the whole partition; narrow it only for smoke tests.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=paths.OUTPUTS / "splits")
    ap.add_argument("--clip-map-out", type=Path, default=paths.OUTPUTS / "clip_label_map.pt")
    ap.add_argument("--no-clip-map", action="store_true")
    args = ap.parse_args()

    cfg = SplitConfig(data_root=args.data_root, datasets=tuple(args.datasets), seed=args.seed)
    labels: set[str] = set()
    splits = build_splits(cfg, label_sink=labels)
    print(splits.summary())
    manifest = save_canonical(splits, args.out_dir, cfg)
    print(f"wrote {manifest}: {json.loads(manifest.read_text())['counts']}")

    if not args.no_clip_map:
        from langsensor.lsm.tensorizer import build_clip_label_map

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_map = build_clip_label_map(sorted(labels), device=device)
        args.clip_map_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clip_map, args.clip_map_out)
        print(f"wrote {args.clip_map_out} ({len(clip_map):,} labels)")


if __name__ == "__main__":
    main()
