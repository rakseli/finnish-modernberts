#!/bin/bash
#SBATCH --job-name=eval_tokenizer
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --account=project_462000615
#SBATCH --output=../logs/tokenizer-logs/%x_%j.output
#SBATCH --error=../logs/tokenizer-logs/%x_%j.error
module purge
module load CrayEnv
module load PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20250117
set -euo pipefail
data_path=$2
tokenizer_path=$1
srun singularity exec $SIF \
    conda-python-simple ../src/tokenizer/evaluate_tokenizer.py \
    --tokenizer_path $tokenizer_path \
    --data_path $data_path \
    --save \
    --force
