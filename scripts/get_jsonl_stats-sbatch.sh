#!/bin/bash
#SBATCH --job-name=get_jsonl_stats
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error
module purge
module load cray-python
set -euo pipefail
input_root=$1
output_file=$2
srun python ../src/data-tools/jsonl_stats.py --input_root $input_root --output_file $output_file --per_source
