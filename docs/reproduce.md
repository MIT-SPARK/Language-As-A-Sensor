# Reproducing the paper

```bash
bash scripts/reproduce_paper.sh           # everything below except re-running the closed loop
```

With the artifacts and VLA-3D in place this takes about ten minutes on one GPU,
makes no API calls, and writes `outputs/paper/{table1.tex, table2.tex,
ambiguity_vs_{anees,rmse}.pdf, information_gain.pdf, target_object_probability.pdf}`,
which are identical to the paper's.

| Paper | Command | Notes |
|---|---|---|
| Table 1 | `python -m langsensor.eval.table1` | Ambiguity-1 statements, ground-truth grounding, 400 per split. Rescaled rows are fitted on a separate 200-record slice of val_seen. |
| Figure 3 | `python -m langsensor.eval.fig3` | 400 val_seen statements per ambiguity bucket (1, 2, 3, 4, 5+), LLM proposer. Rescaled rows are fitted on 40 more per bucket. |
| Figure 4, Table 2 | `python -m langsensor.vlmap.run configs/vlmap/closed_loop.yaml predictor=<none\|lsm\|llm\|llm_e2e>` | 54 episodes in 13 scenes. Needs Hydra and the trajectories; see below. |
| Appendix: ablations | `bash scripts/run_ablations.sh` | Five 50-epoch trainings; compare their `val_seen` metrics with the full model's. |
| LSM training | `python scripts/preprocess.py && python -m langsensor.lsm.train configs/lsm/train.yaml` | Needs VLA-3D, the CLIP label map, and ~1.4M cached samples on disk. |

Renderers: `python -m langsensor.paper.{table1,fig3,fig4,table2} --help`.

## LLM calls and the cache

Every LLM-based row uses gpt-5.2 through the OpenAI API, except the closed-loop
Scaffolded-LLM and LLM-E2E predictors, which used **gpt-4o-mini** (their proposer
is gpt-5.2). The shipped cache contains every response, so the numbers
reproduce exactly and for free. `reproduce_paper.sh` sets `LANGSENSOR_OFFLINE=1`,
which turns any cache miss into an error; pass `ALLOW_API=1` to allow calls
(needs `OPENAI_API_KEY`). A fresh run without the cache will not give the same
numbers: the API is not deterministic.

The static benchmark and the closed loop prompt the LLM slightly differently
(see `langsensor/vlmap/language.py`) and have separate caches.

## The closed loop

By default Figure 4 and Table 2 are rendered from the archived sweep
(`artifacts/results/closed_loop_2026-05-14/`). To re-run it:

1. Build Hydra ([hydra.md](hydra.md)) and install the extras: `pip install -e ".[closed-loop]"`.
2. Put the trajectories under `$CLOSED_LOOP_ROOT` and run `python scripts/make_episodes.py` ([data.md](data.md)).
3. `RUN_CLOSED_LOOP=1 bash scripts/reproduce_paper.sh`

Each method took 2.5-3.5 hours in the paper's sweep. Fresh results override the archived ones
per method in the rendered figure and table; Scaffolded-VLM always comes from
the archive.

The re-implementation has been checked component by component against the code
that produced the sweep (identical language beliefs for all 54 episodes and
all three language predictors, identical vision-sensor scores, fusion and
metrics on synthetic data). Two inputs differ from the original run: the CLIP
label map is a later rebuild of the same embeddings, and Hydra is whatever
build you install, so expect agreement to within numerical noise rather than
bit for bit.

## What is not included

* **3D-ViSTA** and **Scaffolded-VLM** baselines (their paper rows come from the archived results).
* The **Scaffolded-VLM row of Table 1** reports an earlier run whose raw output was not kept; it renders as "--".
* The **real-robot (Spot) experiments** and the qualitative Figures 1 and 2.
