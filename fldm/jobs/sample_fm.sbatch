#!/bin/bash
#SBATCH --job-name=fldm_fm_sample
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/%x_%j.out

# Samples textures with the flow matching sampler from a training run.
# Usage (from fldm/), only TRAIN is required, the others are optional:
#   sbatch --export=ALL,TRAIN=fm_smoke,STEPS_LIST="10",NSAMP=2 jobs/sample_fm.sbatch
# Each number of steps writes mesh_diffusion/runs/flow_<TRAIN>_<SOLVER>_n<N>.
# The Heun solver uses 2 N - 1 network evaluations for N steps.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/fldm_env.sh

TRAIN=${TRAIN:?Set TRAIN to the name of the training run}
STEPS_LIST=${STEPS_LIST:-"10 20 50 100 250"}
SOLVER=${SOLVER:-euler}
NSAMP=${NSAMP:-64}
CKPT=${CKPT:-mesh_diffusion/runs/$TRAIN/checkpoints-0}
ID=fe973fc8e8c049c6b9ab884137fc9463

for N in $STEPS_LIST; do
  RUN=flow_${TRAIN}_${SOLVER}_n${N}
  OUT=mesh_diffusion/runs/$RUN
  mkdir -p "$OUT/samples"
  {
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "slurm_job: $SLURM_JOB_ID"
    echo "run: $RUN  steps: $N  solver: $SOLVER  num_samples: $NSAMP  ckpt: $CKPT"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"; git status --short
  } > "$OUT/meta.txt" 2>&1
  python -m pip freeze > "$OUT/pip_freeze.txt"

  START=$(date +%s)
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 python -m mesh_diffusion.sin_mesh_ddm.main \
    --config=mesh_diffusion/sin_mesh_ddm/configs/default_sample_labeled.py \
    --workdir="$OUT/" --logtostderr \
    --config.trainer_checkpoint_dir=mesh_diffusion/data/flvae/flvae_checkpoint/ \
    --config.model_checkpoint_dir="$CKPT" \
    --config.geom_record_path=mesh_diffusion/data/meshes \
    --config.obj_labels=mesh_diffusion/data/meshes/$ID/${ID}_sel.ply \
    --config.sample_save_dir="$OUT/samples/" \
    --config.num_train_samples=100 --config.num_test_samples=16 \
    --config.num_samples="$NSAMP" \
    --config.sampler=flow --config.num_sampling_steps="$N" \
    --config.fm_solver="$SOLVER" \
    2>&1 | tee "$OUT/run.log"
  echo "wall_seconds: $(( $(date +%s) - START ))" | tee -a "$OUT/meta.txt"
done