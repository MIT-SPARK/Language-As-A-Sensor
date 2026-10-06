"""Figure 3: accuracy and calibration as referential ambiguity grows (LLM proposer, no GT grounding).

val_seen records are bucketed by ambiguity (1, 2, 3, 4, 5+). From each shuffled
bucket the first 40 records form the calibration slice for the rescaled rows and
the next 400 are the test set, so calibration matches the test set's ambiguity mix.

    python -m langsensor.eval.fig3 --out-dir outputs/fig3
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from langsensor import paths
from langsensor.data.canonical import load_split
from langsensor.eval.benchmark import records_to_queries, run_benchmark, write_rows
from langsensor.eval.common import calibrated, incontext_fit_records, incontext_predictor, select_rows
from langsensor.eval.metrics import ambiguity_sort_key
from langsensor.predictors.base import GMMApproach
from langsensor.predictors.llm import LLMFullGMMPredictor, LLMGaussianPredictor
from langsensor.predictors.lsm import LSMPredictor
from langsensor.predictors.proposer import LLMProposer

ROWS = ("framework", "straight_llm", "scaffolded_llm", "scaffolded_llm_scale",
        "scaffolded_llm_axis_scale", "scaffolded_llm_incontext")
BUCKETS = ("1", "2", "3", "4", "5+")


def bucket(level: int) -> str:
    return str(level) if int(level) < 5 else "5+"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=paths.CHECKPOINT)
    p.add_argument("--clip-label-map", type=Path, default=paths.CLIP_LABEL_MAP)
    p.add_argument("--splits", type=Path, default=paths.SPLITS)
    p.add_argument("--data-root", type=Path, default=paths.VLA3D_ROOT)
    p.add_argument("--cache-dir", type=Path, default=paths.STATIC_CACHE)
    p.add_argument("--out-dir", type=Path, default=paths.OUTPUTS / "fig3")
    p.add_argument("--only", nargs="+", default=list(ROWS), metavar="ROW")
    p.add_argument("--split", default="val_seen", choices=["val_seen", "val_unseen"])
    p.add_argument("--per-ambiguity", type=int, default=400, help="test records per ambiguity bucket")
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--calib-per-ambiguity", type=int, default=40)
    p.add_argument("--llm-incontext-k", type=int, default=8)
    p.add_argument("--llm-incontext-per-relation", type=int, default=40)
    p.add_argument("--proposer-model", default="gpt-5.2")
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--llm-model", default="gpt-5.2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def select_records(records, per_bucket: int, calib_per_bucket: int, max_samples: int, rng: random.Random):
    """Split into (calibration, test): calibration comes off the front of each shuffled bucket."""
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[bucket(r.statement.ambiguity)].append(r)
    for b in by_bucket.values():
        rng.shuffle(b)

    calib = []
    for label in BUCKETS:
        calib.extend(by_bucket[label][:calib_per_bucket])
        by_bucket[label] = by_bucket[label][calib_per_bucket:]
    rng.shuffle(calib)

    test = []
    for label in BUCKETS:
        take = by_bucket[label][:per_bucket]
        print(f"  ambiguity {label:>2}: {len(take)}/{len(by_bucket[label])}")
        test.extend(take)
    rng.shuffle(test)
    return calib, test[:max_samples]


def write_csv(results: dict, path: Path) -> None:
    rows = []
    for name, m in results.items():
        if "error" in m:
            continue
        ba = m["by_ambiguity"]
        for lvl in sorted(ba["rmse"], key=ambiguity_sort_key):
            r, n, e, emin, ew = (ba[k][lvl] for k in ("rmse", "nll", "nees", "nees_min", "nees_w"))
            rows.append([name, lvl, r["count"], *(f"{v:.6f}" for v in (
                r["mean"], r["std"], n["mean"], n["std"],
                e["mean"], e["std"], e["median"], e["q25"], e["q75"], e["iqr"],
                emin["median"], emin["q25"], emin["q75"], emin["iqr"], ew["mean"], ew["std"]))])
    write_rows(path, ["approach", "ambiguity", "n", "rmse_mean", "rmse_std", "nll_mean", "nll_std",
                      "anees_mean", "anees_std", "anees_median", "anees_q25", "anees_q75", "anees_iqr",
                      "anees_min_median", "anees_min_q25", "anees_min_q75", "anees_min_iqr",
                      "anees_w_mean", "anees_w_std"], rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    records = load_split(args.split, args.splits, args.data_root)
    calib_records, test_records = select_records(
        records, args.per_ambiguity, args.calib_per_ambiguity, args.max_samples, random.Random(args.seed))
    print(f"calibration={len(calib_records)}  test={len(test_records)}")

    queries = records_to_queries(test_records, desc="test")
    for q in queries:
        q.metadata["ambiguity"] = bucket(q.metadata["ambiguity"])

    want = set(args.only)
    proposer = LLMProposer("openai", args.proposer_model, args.cache_dir / "proposer")
    llm = LLMGaussianPredictor(args.llm_provider, args.llm_model, args.cache_dir / "llm_gauss")
    needs_calib = want & {"scaffolded_llm_scale", "scaffolded_llm_axis_scale"}
    calib = records_to_queries(calib_records, desc="calibration") if needs_calib else []

    rows: dict = {}
    if "framework" in want:
        rows["framework"] = GMMApproach(proposer, LSMPredictor(args.checkpoint, args.clip_label_map, args.device))
    rows["straight_llm"] = GMMApproach(None, LLMFullGMMPredictor(args.llm_provider, args.llm_model,
                                                                 args.cache_dir / "llm_full_gmm"))
    rows["scaffolded_llm"] = GMMApproach(proposer, llm)
    # The fit uses the LLM proposer, so it sees the grounding errors the row is scored with.
    for mode in ("scale", "axis_scale"):
        if f"scaffolded_llm_{mode}" in want:
            rows[f"scaffolded_llm_{mode}"] = GMMApproach(proposer, calibrated(
                llm, proposer, mode, f"llm_{args.llm_model}_ambig_{args.split}", args.cache_dir, calib))
    if "scaffolded_llm_incontext" in want:
        fit = incontext_fit_records(records, test_records + calib_records, args.llm_incontext_per_relation)
        rows["scaffolded_llm_incontext"] = GMMApproach(
            proposer, incontext_predictor(args, fit, proposer, f"llm_{args.llm_model}_ambig_{args.split}.json"))
    rows = select_rows(rows, args.only)

    write_csv(run_benchmark(rows, queries), args.out_dir / "fig3_by_ambiguity.csv")


if __name__ == "__main__":
    main()
