#!/bin/bash
#SBATCH --job-name=run-pii
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --account=project_462000444
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error
module purge
ml use /appl/local/csc/modulefiles/
ml pytorch/2.4
source /scratch/<project_number>/<user_name>/venvs/pii-tool/bin/activate
set -euo pipefail
srun \
    python ../src/data-tools/run_pii.py \
    --input-file $1 \
    --lang $2 \
    --safe-mode
