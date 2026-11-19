#!/bin/bash
#SBATCH --job-name=fldm_eval
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=jobs/logs/%x_%j.out

# Renders and evaluates (SIFID, LPIPS) several sampling runs.
# Usage (from fldm/), FORCE=1 recomputes runs that already have metrics.json:
#   sbatch --export=ALL,RUNS="ddim_n10_eta0.0_trailing ddim_n50_eta0.0_trailing" \
#       jobs/eval_runs.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

RUNS=${RUNS:?Set RUNS to a space separated list of run names}
LIMIT=${LIMIT:-0}
MAXS=${MAXS:-0}
FORCE=${FORCE:-0}
ID=fe973fc8e8c049c6b9ab884137fc9463

eval_run() {
  local run=$1
  local dir=mesh_diffusion/runs/$run
  if [ ! -d "$dir/samples" ]; then
    echo "skip $run, no samples directory"; return 0
  fi
  if [ -f "$dir/metrics.json" ] && [ "$FORCE" != 1 ]; then
    echo "skip $run, metrics.json exists (FORCE=1 to redo)"; return 0
  fi
  echo "=== $run $(date -Is)"
  mkdir -p "$dir/renders"
  {
    echo "date: $(date -Is)"
    echo "slurm_job: $SLURM_JOB_ID  host: $(hostname)"
    echo "git_commit: $(git rev-parse HEAD)"
  } > "$dir/eval_meta.txt"

  (
    source ~/fldm_env.sh
    CUDA_VISIBLE_DEVICES="" python mesh_diffusion/evaluation/render_set.py \
      --tfrecord mesh_diffusion/data/meshes/$ID/test/geom_0.tfrecords \
      --reference mesh_diffusion/data/meshes/$ID/valid_tex.jpg \
      --samples_dir "$dir/samples" --out_dir "$dir/renders" --limit "$LIMIT" \
      --views 24 --elevation_deg 45 --size 512 --up z --splat 1
  ) 2>&1 | tee "$dir/renders/render.log" || return 1

  (
    source ~/fldm_eval_env.sh
    python mesh_diffusion/evaluation/metrics.py \
      --render_dir "$dir/renders" --out "$dir/metrics.json" \
      --max_samples "$MAXS"
  ) 2>&1 | tee "$dir/metrics.log" || return 1
}

FAILED=""
for RUN in $RUNS; do
  if ! eval_run "$RUN"; then
    FAILED="$FAILED $RUN"
    echo "FAILED $RUN"
  fi
done
echo "failed runs:${FAILED:- none}"