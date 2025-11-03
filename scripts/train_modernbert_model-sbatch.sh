#!/bin/bash
#SBATCH --job-name=train-modernbert
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --account=<project_number>
#SBATCH --output=../logs/production-logs/%x_%j.output
#SBATCH --error=../logs/production-logs/%x_%j.error
module purge
export EBU_USER_PREFIX="/scratch/<project_number>/<user_name>/EasyBuild"
module load LUMI/24.03
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250911
set -euo pipefail
export PYTHONWARNINGS=ignore
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export TORCH_DISABLE_ADDR2LINE=1

CONFIG_PATH=$1
LATEST_CHECKPOINT=$2

if [[ -n "$LATEST_CHECKPOINT" ]]; then
    echo "Resuming from: $LATEST_CHECKPOINT"
    srun singularity exec $SIF ./laucher.sh ../src/training/train_modernbert.py --config $CONFIG_PATH --checkpoint_path $LATEST_CHECKPOINT
else
    echo "No checkpoint given. Starting training from scratch."
    srun singularity exec $SIF ./laucher.sh ../src/training/train_modernbert.py --config $CONFIG_PATH