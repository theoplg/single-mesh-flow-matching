#!/bin/bash
#SBATCH --job-name=fldm_ddim
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/%x_%j.out

# Sweeps the number of DDIM steps on the published weights.
# Usage (from fldm/), all variables are optional:
#   sbatch --export=ALL,STEPS_LIST="10 50",ETA=0.0,SPACING=trailing,NSAMP=8 \
#       jobs/sample_ddim.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/fldm_env.sh

STEPS_LIST=${STEPS_LIST:-"10 20 50 100 250"}
ETA=${ETA:-0.0}
SPACING=${SPACING:-trailing}
NSAMP=${NSAMP:-64}
ID=fe973fc8e8c049c6b9ab884137fc9463

for N in $STEPS_LIST; do
  RUN=ddim_n${N}_eta${ETA}_${SPACING}
  OUT=mesh_diffusion/runs/$RUN
  mkdir -p "$OUT/samples"
  {
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "slurm_job: $SLURM_JOB_ID"
    echo "run: $RUN  steps: $N  eta: $ETA  spacing: $SPACING  num_samples: $NSAMP"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"; git status --short
  } > "$OUT/meta.txt" 2>&1
  python -m pip freeze > "$OUT/pip_freeze.txt"

  START=$(date +%s)
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 python -m mesh_diffusion.sin_mesh_ddm.main \
    --config=mesh_diffusion/sin_mesh_ddm/configs/default_sample_labeled.py \
    --workdir="$OUT/" --logtostderr \
    --config.trainer_checkpoint_dir=mesh_diffusion/data/flvae/flvae_checkpoint/ \
    --config.model_checkpoint_dir=mesh_diffusion/data/meshes/$ID/fldm_sealion/ \
    --config.geom_record_path=mesh_diffusion/data/meshes \
    --config.obj_labels=mesh_diffusion/data/meshes/$ID/${ID}_sel.ply \
    --config.sample_save_dir="$OUT/samples/" \
    --config.num_train_samples=100 --config.num_test_samples=16 \
    --config.num_samples="$NSAMP" \
    --config.sampler=ddim --config.num_sampling_steps="$N" \
    --config.ddim_eta="$ETA" --config.ddim_spacing="$SPACING" \
    2>&1 | tee "$OUT/run.log"
  echo "wall_seconds: $(( $(date +%s) - START ))" | tee -a "$OUT/meta.txt"
done
