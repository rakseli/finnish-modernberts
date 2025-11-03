#!/bin/bash
#SBATCH --job-name=create-long-document-subset
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/data-logs/%x_%j.output
#SBATCH --error=../logs/data-logs/%x_%j.error

module purge
module load CrayEnv
module load PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20250117
set -euo pipefail
input_file=$1
output_file=$2
index_file=$3

srun singularity exec $SIF conda-python-simple ../src/data-tools/create_long_doc_subset.py --input-file $input_file --output-file $output_file --index-file $index_file