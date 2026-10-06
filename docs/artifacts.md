# Artifacts

`python scripts/download_artifacts.py` fetches ~65 MB into `artifacts/`
(or `$LANGSENSOR_ARTIFACTS`):

| Part | Contents |
|---|---|
| `checkpoints` | `lsm.pt`: the released LSM (epoch 46, lowest val_seen loss of a 50-epoch run). The frozen BERT encoder is not stored; it is `bert-base-uncased` at the Hugging Face commit recorded in the checkpoint's config, downloaded on first use. `clip_label_map.pt`: CLIP ViT-B/32 text embedding of every object label in VLA-3D. |
| `splits` | The canonical partition (see [data.md](data.md)). |
| `llm_cache` | Every LLM response behind the paper's numbers, and the fitted calibrations. |
| `results` | Results that this repository cannot regenerate, or that need hardware it cannot assume. |

```
artifacts/
  checkpoints/
    lsm.pt
    clip_label_map.pt
  splits/
    manifest.json  train.jsonl.gz  val_seen.jsonl.gz  val_unseen.jsonl.gz
  cache/
    static/                      used by langsensor.eval.{table1,fig3}
      proposer/                  grounding hypotheses (gpt-5.2)
      llm_gauss/                 Scaffolded-LLM Gaussians (gpt-5.2)
      llm_incontext/             in-context calibrated LLM Gaussians
      llm_full_gmm/              LLM-E2E mixtures
      covcal/                    global / per-axis rescale fits
      incontext/                 measured per-relation spreads + worked examples
    closed_loop/                 used by langsensor.vlmap.run
      proposer/                  grounding hypotheses (gpt-5.2)
      predictor/                 Scaffolded-LLM Gaussians (gpt-4o-mini)
      predictor_e2e/             LLM-E2E mixtures (gpt-4o-mini)
  results/
    static/                      the paper's Table 1 / Figure 3 CSVs (include the 3D-ViSTA rows)
    closed_loop_2026-05-14/      the closed-loop sweep behind Figure 4 / Table 2
      <method>/metrics_timeseries.json
      _summary.json              ids of the episodes whose target is never observed
```

## Caches

Responses are stored as `<sha256 key>.json`. The key hashes the model name, the
utterance, the serialised scene graph and (for predictors) the grounding, plus a
hash of the prompt for the proposer and the in-context predictor; see
`langsensor/predictors/`. A query that is not in the cache calls the API, unless
`LANGSENSOR_OFFLINE=1`, in which case it raises.

The calibration fits in `covcal/` and `incontext/` are reused whenever they
exist. Delete them to refit; the fit then costs one LLM pass over the
calibration slice (200 records for Table 1, 200 for Figure 3).

## Why some results are archived rather than regenerated

* **3D-ViSTA** (Table 1, Figure 3) is a fine-tuned third-party model and is not
  part of this repository; its rows are read from `results/static/`.
* **Scaffolded-VLM** (Figure 4, Table 2) used a local Ollama `qwen2.5vl:32b`
  and is read from the archived sweep. Its Table 1 row in the paper comes from
  an earlier run whose output was not kept, so it renders as "--".
* **The closed-loop sweep** needs Hydra and the RGB-D trajectories. Figure 4 and
  Table 2 are rendered from the archived sweep by default; re-running it is
  described in [reproduce.md](reproduce.md).
