#!/bin/bash
#SBATCH --job-name=anneal_model
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G
#SBATCH --partition=standard-g
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --account=<project_number>
#SBATCH --output=../logs/production-logs/%x_%j.output
#SBATCH --error=../logs/production-logs/%x_%j.error
module purge
export EBU_USER_PREFIX=/scratch/<project_number>/<user_name>/EasyBuild
module load CrayEnv
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250117
set -euo pipefail

CONFIG_PATH=$1
CHECKPOINT_PATH=$2

srun singularity exec $SIF ./laucher.sh ../src/training/train_modernbert.py --config $CONFIG_PATH --checkpoint_path $CHECKPOINT_PATH --annealing

