"""Row builders shared by the Table 1 and Figure 3 drivers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from langsensor.data.splits import SplitRecord
from langsensor.eval.benchmark import records_to_queries
from langsensor.predictors.base import DistributionPredictor, Proposer
from langsensor.predictors.calibration import CovarianceCalibrated
from langsensor.predictors.incontext import LLMInContextPredictor, fit_incontext_calibration


def select_rows(built: dict, only: list[str] | None) -> dict:
    if not only:
        return built
    missing = [n for n in only if n not in built]
    if missing:
        raise SystemExit(f"unknown rows {missing}; available: {sorted(built)}")
    return {k: built[k] for k in only}


def calibrated(base: DistributionPredictor, proposer: Proposer, mode: str, name: str,
               cache_dir: Path, calib_queries) -> CovarianceCalibrated:
    wrapped = CovarianceCalibrated(base, proposer, mode, name, cache_dir / "covcal")
    wrapped.calibrate(calib_queries)
    return wrapped


def incontext_fit_records(pool: list[SplitRecord], exclude: list[SplitRecord], per_relation: int) -> list[SplitRecord]:
    """Up to *per_relation* records of each relation, excluding test/calibration records.

    Stratified because `near` is ~88% of the corpus and `in` ~0.3%: a random
    draw could not estimate the rare relations' spread.
    """
    excluded = {id(r) for r in exclude}
    by_rel: dict[str, list[SplitRecord]] = defaultdict(list)
    for r in pool:
        if id(r) not in excluded:
            by_rel[str(r.statement.relation)].append(r)
    out = [r for rel in sorted(by_rel) for r in by_rel[rel][:per_relation]]
    print(f"[incontext] fit pool: {len(out)} records")
    return out


def incontext_predictor(args, fit_records, proposer: Proposer, cache_name: str) -> LLMInContextPredictor:
    cal = fit_incontext_calibration(
        records_to_queries(fit_records, desc="incontext fit"),
        proposer=proposer,
        n_examples=args.llm_incontext_k,
        seed=args.seed,
        cache_path=args.cache_dir / "incontext" / cache_name,
    )
    return LLMInContextPredictor(args.llm_provider, args.llm_model, args.cache_dir / "llm_incontext", calibration=cal)
