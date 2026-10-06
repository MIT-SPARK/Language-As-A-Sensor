#!/usr/bin/env bash
# Train the five LSM ablations of the appendix (each 50 epochs, like the full model).
# Extra key=value overrides apply to every run, e.g.  bash scripts/run_ablations.sh wandb.enabled=true
set -euo pipefail
cd "$(dirname "$0")/.."

for ablation in no_spatial_backbone no_global_fusion no_film no_anchor_centric no_geometry; do
  echo "== $ablation"
  python -m langsensor.lsm.train "configs/lsm/ablations/$ablation.yaml" "wandb.name=$ablation" "$@"
done
