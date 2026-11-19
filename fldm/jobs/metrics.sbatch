#!/bin/bash
#SBATCH --job-name=fldm_metrics
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=jobs/logs/%x_%j.out

# Computes SIFID and LPIPS on the renders of one run.
# Usage (from fldm/), MAXS is optional and uses only the first samples:
#   sbatch --export=ALL,RUN=ref_ddpm1000,MAXS=0 jobs/metrics.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/fldm_eval_env.sh

RUN=${RUN:?Set RUN, for example RUN=ref_ddpm1000}
MAXS=${MAXS:-0}
OUT=mesh_diffusion/runs/$RUN

python mesh_diffusion/evaluation/metrics.py \
  --render_dir "$OUT/renders" --out "$OUT/metrics.json" \
  --max_samples "$MAXS" 2>&1 | tee "$OUT/metrics.log"