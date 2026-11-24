#!/bin/bash
#SBATCH --job-name=fldm_fm_train
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/logs/%x_%j.out

# Trains the latent model with the flow matching objective on one GPU.
# Usage (from fldm/), only RUN is required. STEPS counts training steps (one mesh
# each), so 16 steps make one optimizer update.
#   Smoke test, warm start from the published diffusion weights: see the README
#   of the journal. Variables: RUN, STEPS, WARMUP, LOSS, LOG_EVERY, CKPT_EVERY,
#   INIT (checkpoint directory for a warm start), EXTRA (more --config flags).
#   From scratch, one full job: sbatch --export=ALL,RUN=fm_scratch jobs/train_fm.sbatch
# Submitting again with the same RUN resumes from the last checkpoint.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/fldm_env.sh

RUN=${RUN:?Set RUN to the name of the training run}
STEPS=${STEPS:-84000}
WARMUP=${WARMUP:-1000}
LOSS=${LOSS:-l1}
LOG_EVERY=${LOG_EVERY:-160}
CKPT_EVERY=${CKPT_EVERY:-2000}
INIT=${INIT:-}
EXTRA=${EXTRA:-}
ID=fe973fc8e8c049c6b9ab884137fc9463
OUT=mesh_diffusion/runs/$RUN
mkdir -p "$OUT"

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "slurm_job: $SLURM_JOB_ID"
  echo "run: $RUN  steps: $STEPS  warmup: $WARMUP  loss: $LOSS  init: ${INIT:-scratch}"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_status:"; git status --short
  nvidia-smi
} >> "$OUT/meta.txt" 2>&1
python -m pip freeze > "$OUT/pip_freeze.txt"

START=$(date +%s)
# INIT and EXTRA are left unquoted on purpose, so that empty values disappear.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 python -m mesh_diffusion.sin_mesh_ddm.main \
  --config=mesh_diffusion/sin_mesh_ddm/configs/default_train_labeled.py \
  --workdir="$OUT/" --logtostderr \
  --config.trainer_checkpoint_dir=mesh_diffusion/data/flvae/flvae_checkpoint/ \
  --config.geom_record_path=mesh_diffusion/data/meshes \
  --config.obj_labels=mesh_diffusion/data/meshes/$ID/${ID}_sel.ply \
  --config.num_train_samples=100 --config.num_test_samples=16 \
  --config.sample_during_training=False \
  --config.objective=flow_matching --config.fm_loss="$LOSS" \
  --config.lr_schedule_unit=steps \
  --config.num_steps="$STEPS" --config.warmup_steps="$WARMUP" \
  --config.log_loss_every_steps="$LOG_EVERY" \
  --config.checkpoint_every_steps="$CKPT_EVERY" \
  ${INIT:+--config.init_checkpoint_dir="$INIT"} $EXTRA \
  2>&1 | tee -a "$OUT/run.log"
echo "wall_seconds: $(( $(date +%s) - START ))" | tee -a "$OUT/meta.txt"