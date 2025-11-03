#!/bin/bash
#SBATCH --job-name=fix_jsonls
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=small
#SBATCH --time=00:15:00
#SBATCH --account=project_462000615
#SBATCH --output=../logs/data-logs/fix_jsonl_%A_%a.output
#SBATCH --error=../logs/data-logs/fix_jsonl_%A_%a.error
#SBATCH --array=799-920%100

module purge
module load cray-python
set -euo pipefail
FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" file_list.txt)

python ../src/data-tools/remove_double_quote.py --input_file "$FILE"