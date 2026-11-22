#!/bin/bash
#SBATCH --job-name=fldm_blocks
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/%x_%j.out

# Recomputes SIFID with other Inception blocks on existing renders.
# Usage (from fldm/), FORCE=1 recomputes existing files:
#   sbatch --export=ALL,RUNS="ref_ddpm1000 ddim_n10_eta0.0_leading",BLOCKS="1 2" \
#       jobs/metrics_blocks.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/fldm_eval_env.sh

RUNS=${RUNS:?Set RUNS to a space separated list of run names}
BLOCKS=${BLOCKS:-"1 2"}
FORCE=${FORCE:-0}

for RUN in $RUNS; do
  DIR=mesh_diffusion/runs/$RUN
  for B in $BLOCKS; do
    OUT=$DIR/metrics_block$B.json
    if [ -f "$OUT" ] && [ "$FORCE" != 1 ]; then
      echo "skip $RUN block $B, file exists"; continue
    fi
    echo "=== $RUN block $B $(date -Is)"
    python mesh_diffusion/evaluation/metrics.py \
      --render_dir "$DIR/renders" --out "$OUT" --sifid_block "$B" \
      2>&1 | tee "$DIR/metrics_block$B.log" \
      || echo "FAILED $RUN block $B"
  done
done