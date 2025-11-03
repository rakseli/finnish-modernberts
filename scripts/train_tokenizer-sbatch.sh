#!/bin/bash
#SBATCH --job-name=train_tokenizer-hf-tokens-in-gib
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --mem=5G
#SBATCH --account=<project_number>
#SBATCH --output=../logs/tokenizer-logs/%x_%j.output
#SBATCH --error=../logs/tokenizer-logs/%x_%j.error
export EBU_USER_PREFIX="/projappl/<project_number>/Easybuild"
module purge
module load CrayEnv
module load PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20250117
set -euo pipefail
data_path=$1
#
srun singularity exec $SIF conda-python-simple ../src/tokenizer/train_tokenizer_hf.py --input_path $data_path --model_size experimental
