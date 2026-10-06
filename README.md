# Language as a Sensor

Code for **Language as a Sensor: Calibrated Spatial Belief Estimation in 3D
Scenes from Natural Language** (CoRL 2026) — Aryan Naveen, Jason Xinyu Liu,
Luca Carlone, Andreea Bobu.

A robot told *"I left my backpack on the table"* should treat the sentence as a
measurement. The **Language Sensor Model (LSM)** turns an utterance and a 3D
scene graph into a calibrated Gaussian mixture over the referent's position:
an LLM proposes which objects the utterance refers to (the mixture
components), and a learned spatial model predicts where the target lies
relative to each (their means and covariances). **VLMaP** fuses these
distributions with streaming vision in a voxel belief map.

## Install

```bash
git clone https://github.com/MIT-SPARK/Language-As-A-Sensor.git && cd Language-As-A-Sensor
pip install -e .                       # add ".[train]" to train, ".[closed-loop]" for VLMaP
python scripts/download_artifacts.py   # checkpoint, splits, LLM cache, archived results (~65 MB)
```

Then download [VLA-3D](https://github.com/HaochenZ11/VLA-3D) and set
`VLA3D_ROOT` to it ([docs/data.md](docs/data.md)). The OpenAI key
(`OPENAI_API_KEY`) is only needed for LLM queries that are not in the shipped cache.

## Use the LSM

```python
from langsensor import paths
from langsensor.core.schema import SpatialQuery
from langsensor.data.vla3d import VLA3DScene
from langsensor.predictors.base import GMMApproach
from langsensor.predictors.lsm import LSMPredictor
from langsensor.predictors.proposer import LLMProposer

scene = VLA3DScene(paths.VLA3D_ROOT / "Unity" / "arabic_room")
query = SpatialQuery(scene.scene_id, scene.load_scene_graph(), "there is a pillow on the sofa")

sensor = GMMApproach(LLMProposer(model="gpt-5.2", cache_dir="outputs/llm_cache"), LSMPredictor())
gmm = sensor.predict(query)      # one weighted Gaussian per grounding hypothesis, world frame
gmm.mus, gmm.Ls, gmm.weights     # means (K, 3), Cholesky factors (K, 3, 3), weights (K,)
gmm.sample(1000)                 # (1000, 3)
```

Any scene graph in the VLA-3D format works; see `langsensor/core/schema.py`.

## Reproduce the paper

```bash
bash scripts/reproduce_paper.sh
```

regenerates Tables 1-2 and Figures 3-4 in about ten minutes on one GPU, with no
API calls (every LLM response is in the shipped cache); the output is identical
to the paper's. See [docs/reproduce.md](docs/reproduce.md) for each experiment
on its own, for training, and for re-running the closed loop.

| Paper | Code |
|---|---|
| LSM architecture, loss (Sec. 4, App.) | `langsensor/lsm/` |
| Grounding proposer (App. prompt) | `langsensor/predictors/proposer.py` |
| Baselines: Scaffolded-LLM, LLM-E2E, post-hoc and in-context calibration | `langsensor/predictors/{llm,calibration,incontext}.py` |
| Table 1, Figure 3 | `langsensor/eval/{table1,fig3}.py` |
| VLMaP: vision sensor, fusion, closed loop (Sec. 5) | `langsensor/vlmap/` |
| Figure 4, Table 2 | `langsensor/vlmap/run.py`, `langsensor/paper/{fig4,table2}.py` |
| Ablations (App.) | `configs/lsm/ablations/`, `scripts/run_ablations.sh` |

## Repository

```
langsensor/
  core/        scene graphs, queries, Gaussian mixtures
  data/        VLA-3D reader, train/val partition
  lsm/         model, tensorizer, training
  predictors/  proposers and distribution predictors (LSM and LLM baselines)
  eval/        static benchmark: metrics, Table 1 and Figure 3 drivers
  vlmap/       closed-loop fusion with vision (Hydra behind a small interface)
  paper/       renders the paper's tables and figures
configs/       training, ablation and closed-loop configs (plain YAML, key=value overrides)
scripts/       artifacts, data preparation, paper reproduction
docs/          data, artifacts, reproduction, Hydra
```

The closed-loop experiments additionally need [Hydra](https://github.com/MIT-SPARK/Hydra),
installed separately ([docs/hydra.md](docs/hydra.md)). Nothing else depends on it.

## Citation

```bibtex
@inproceedings{naveen2026language,
  title     = {Language as a Sensor: Calibrated Spatial Belief Estimation in 3D Scenes from Natural Language},
  author    = {Naveen, Aryan and Liu, Jason Xinyu and Carlone, Luca and Bobu, Andreea},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)). VLA-3D, Matterport3D and the models this code
downloads (BERT, CLIP, all-MiniLM-L6-v2) are under their own licenses.
