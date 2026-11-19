#!/bin/bash
#SBATCH --job-name=fldm_render
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=jobs/logs/%x_%j.out

# Renders the reference texture and the samples of one run under 24 fixed views.
# Usage (from fldm/), LIMIT is optional and renders only the first samples:
#   sbatch --export=ALL,RUN=ref_ddpm1000,LIMIT=0 jobs/render_set.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/fldm_env.sh

RUN=${RUN:?Set RUN, for example RUN=ref_ddpm1000}
LIMIT=${LIMIT:-0}
ID=fe973fc8e8c049c6b9ab884137fc9463
OUT=mesh_diffusion/runs/$RUN/renders
mkdir -p "$OUT"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "slurm_job: $SLURM_JOB_ID"
  echo "run: $RUN  limit: $LIMIT"
  echo "git_commit: $(git rev-parse HEAD)"
} > "$OUT/meta.txt"

CUDA_VISIBLE_DEVICES="" python mesh_diffusion/evaluation/render_set.py \
  --tfrecord mesh_diffusion/data/meshes/$ID/test/geom_0.tfrecords \
  --reference mesh_diffusion/data/meshes/$ID/valid_tex.jpg \
  --samples_dir mesh_diffusion/runs/$RUN/samples \
  --out_dir "$OUT" --limit "$LIMIT" \
  --views 24 --elevation_deg 45 --size 512 --up z --splat 1 \
  2>&1 | tee "$OUT/render.log"