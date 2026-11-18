#!/bin/bash
#SBATCH --job-name=fldm_bench
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=jobs/logs/%x_%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/fldm_env.sh

RUN=${RUN:-bench1}
STEPS=${STEPS:-300}
ID=fe973fc8e8c049c6b9ab884137fc9463
OUT=mesh_diffusion/runs/$RUN
mkdir -p "$OUT/ckpt"

# Trace de l'exécution
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "slurm_job: $SLURM_JOB_ID"
  echo "run: $RUN  steps: $STEPS"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_status:"; git status --short
  nvidia-smi
} > "$OUT/meta.txt" 2>&1
python -m pip freeze > "$OUT/pip_freeze.txt"

# Suivi de la mémoire GPU toutes les 5 s
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$OUT/gpu.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 python -m mesh_diffusion.sin_mesh_ddm.main \
  --config=mesh_diffusion/sin_mesh_ddm/configs/default_train_labeled.py \
  --workdir="$OUT/" --logtostderr \
  --config.trainer_checkpoint_dir=mesh_diffusion/data/flvae/flvae_checkpoint/ \
  --config.model_checkpoint_dir="$OUT/ckpt/" \
  --config.geom_record_path=mesh_diffusion/data/meshes \
  --config.obj_labels=mesh_diffusion/data/meshes/$ID/${ID}_sel.ply \
  --config.num_train_samples=100 --config.num_test_samples=16 \
  --config.sample_during_training=False \
  --config.num_steps="$STEPS" --config.warmup_steps=50 --config.log_loss_every_steps=20 \
  --config.checkpoint_every_steps=1000 2>&1 | tee "$OUT/run.log"

awk -F, 'NR>1{gsub(/[^0-9]/,"",$2); if($2+0>m)m=$2+0} END{print "peak_gpu_MiB:", m}' "$OUT/gpu.csv" | tee -a "$OUT/meta.txt"
