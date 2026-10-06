"""Table 1: unambiguous utterances (ambiguity = 1) with the ground-truth grounding.

Every row uses the oracle proposer, so only the distribution predictor differs.
Test sets are 400 records each from val_seen and val_unseen. The rescaled rows
are calibrated once, on a separate 200-record slice of val_seen, and applied to
both test sets (so val_unseen measures how that calibration transfers).

    python -m langsensor.eval.table1 --out-dir outputs/table1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from langsensor import paths
from langsensor.data.canonical import load_split
from langsensor.eval.benchmark import records_to_queries, run_benchmark, write_rows
from langsensor.eval.common import calibrated, incontext_fit_records, incontext_predictor, select_rows
from langsensor.predictors.base import GMMApproach, GroundTruthProposer
from langsensor.predictors.llm import LLMGaussianPredictor
from langsensor.predictors.lsm import LSMPredictor

ROWS = ("gt_lss", "gt_llm", "gt_llm_scale", "gt_llm_axis_scale", "gt_llm_incontext")
STATS = ("rmse_mean", "rmse_std", "rmse_median", "nll_mean", "nll_std", "nll_median")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=paths.CHECKPOINT)
    p.add_argument("--clip-label-map", type=Path, default=paths.CLIP_LABEL_MAP)
    p.add_argument("--splits", type=Path, default=paths.SPLITS)
    p.add_argument("--data-root", type=Path, default=paths.VLA3D_ROOT)
    p.add_argument("--cache-dir", type=Path, default=paths.STATIC_CACHE)
    p.add_argument("--out-dir", type=Path, default=paths.OUTPUTS / "table1")
    p.add_argument("--only", nargs="+", default=list(ROWS), metavar="ROW",
                   help="rows to run; excluded rows cost no API calls")
    p.add_argument("--max-samples", type=int, default=400, help="test records per split")
    p.add_argument("--calib-max-samples", type=int, default=200)
    p.add_argument("--llm-incontext-k", type=int, default=8)
    p.add_argument("--llm-incontext-per-relation", type=int, default=40)
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--llm-model", default="gpt-5.2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def write_csvs(results: dict, out_dir: Path, split: str) -> None:
    write_rows(out_dir / f"table1_{split}.csv",
               ["approach", "n", *STATS, "anees", "nees_std", "nees_median"],
               [[name, m["overall"]["n"], *(f"{m['overall'][k]:.6f}" for k in (*STATS, "anees", "nees_std", "nees_median"))]
                for name, m in results.items() if "error" not in m])
    rows = []
    for name, m in results.items():
        if "error" in m:
            continue
        br = m["by_relation"]
        for rel in sorted(br["rmse"], key=str):
            r, n, e = br["rmse"][rel], br["nll"][rel], br["nees"][rel]
            rows.append([name, rel, r["count"], *(f"{s[k]:.6f}" for s in (r, n, e) for k in ("mean", "std", "median"))])
    write_rows(out_dir / f"table1_{split}_by_relation.csv",
               ["approach", "relation", "n", "rmse_mean", "rmse_std", "rmse_median",
                "nll_mean", "nll_std", "nll_median", "anees", "nees_std", "nees_median"], rows)


def main(argv=None) -> None:
    args = parse_args(argv)

    # val_seen -> [calibration slice | test candidates]; val_unseen -> test.
    seen = [r for r in load_split("val_seen", args.splits, args.data_root) if r.statement.ambiguity == 1]
    rng = random.Random(args.seed)
    rng.shuffle(seen)
    calib_records = seen[: args.calib_max_samples]
    seen_test = seen[args.calib_max_samples:][: args.max_samples]

    unseen = [r for r in load_split("val_unseen", args.splits, args.data_root) if r.statement.ambiguity == 1]
    random.Random(args.seed + 100).shuffle(unseen)
    unseen_test = unseen[: args.max_samples]
    print(f"calibration={len(calib_records)}  val_seen test={len(seen_test)}  val_unseen test={len(unseen_test)}")

    gt = GroundTruthProposer()
    want = set(args.only)
    llm = LLMGaussianPredictor(args.llm_provider, args.llm_model, args.cache_dir / "llm_gauss")
    calib = records_to_queries(calib_records, desc="calibration") if want & {"gt_llm_scale", "gt_llm_axis_scale"} else []

    rows: dict = {}
    if "gt_lss" in want:
        rows["gt_lss"] = GMMApproach(gt, LSMPredictor(args.checkpoint, args.clip_label_map, args.device))
    rows["gt_llm"] = GMMApproach(gt, llm)
    for mode in ("scale", "axis_scale"):
        if f"gt_llm_{mode}" in want:
            rows[f"gt_llm_{mode}"] = GMMApproach(
                gt, calibrated(llm, gt, mode, f"llm_{args.llm_model}", args.cache_dir, calib))
    if "gt_llm_incontext" in want:
        fit = incontext_fit_records(seen, seen_test + unseen_test, args.llm_incontext_per_relation)
        rows["gt_llm_incontext"] = GMMApproach(gt, incontext_predictor(args, fit, gt, f"llm_{args.llm_model}.json"))
    rows = select_rows(rows, args.only)

    for split, records in (("seen", seen_test), ("unseen", unseen_test)):
        print(f"\n=== Table 1, val_{split} ===")
        write_csvs(run_benchmark(rows, records_to_queries(records, desc=f"val_{split}")), args.out_dir, split)


if __name__ == "__main__":
    main()
