"""Download and unpack the released artifacts into ``artifacts/`` (see docs/artifacts.md).

    python scripts/download_artifacts.py                 # everything (~65 MB)
    python scripts/download_artifacts.py checkpoints     # only some parts

Set ``LANGSENSOR_ARTIFACT_URL`` to download from a mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from langsensor import paths

BASE_URL = os.environ.get(
    "LANGSENSOR_ARTIFACT_URL",
    "https://github.com/MIT-SPARK/Language-As-A-Sensor/releases/download/v1.0",
)

PARTS = {
    "checkpoints": ("LSM checkpoint + CLIP label map", "067b1c7379ef809bb507fe573e1de000a6af37d414c89c112a24035c0b56632f"),
    "splits": ("canonical train/val_seen/val_unseen partition", "9398585c80686fce15a20704b33a24423a00e642246c5a861a15a6fc67e2094b"),
    "llm_cache": ("cached LLM responses + calibration fits", "ca7963b5a26daf71d81cbfafb6f0b7abcb4261d3bd7b03bdf6201342d484b9c3"),
    "results": ("archived results (3D-ViSTA rows, closed-loop sweep)", "f3d74f3cfa6d54b8fe4ffb3baa10e14d201db83353545047bce43b0810089f4b"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, dest: Path) -> None:
    what, expected = PARTS[name]
    url = f"{BASE_URL}/{name}.tar.gz"
    print(f"{name}: {what}\n  {url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.tar.gz"
        urllib.request.urlretrieve(url, archive)
        if sha256(archive) != expected:
            raise SystemExit(f"  checksum mismatch for {name}.tar.gz")
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
    print(f"  -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", help=f"any of {list(PARTS)} (default: all)")
    ap.add_argument("--dest", type=Path, default=paths.ARTIFACTS)
    args = ap.parse_args()
    unknown = set(args.parts) - set(PARTS)
    if unknown:
        ap.error(f"unknown parts {sorted(unknown)}; choose from {list(PARTS)}")
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in args.parts or PARTS:
        fetch(name, args.dest)


if __name__ == "__main__":
    main()
