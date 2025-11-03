#!/bin/bash
#SBATCH --job-name=convert_tokenizer
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --account=<project_number>
#SBATCH --output=../logs/tokenizer-logs/%x_%j.output
#SBATCH --error=../logs/tokenizer-logs/%x_%j.error
module purge
export EBU_USER_PREFIX="<easybuild_modelue_location>"
module load CrayEnv
module load PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20250117
set -euo pipefail
tokenizer_path=$1
srun singularity exec $SIF \
    conda-python-simple ../src/tokenizer/convert_tokenizer_hf.py \
    --tokenizer_path $tokenizer_path \

