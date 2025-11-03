#!/bin/bash
#SBATCH --job-name=get_tokens_stats
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
module use /appl/local/csc/modulefiles/
module load pytorch
LOG_DIR=$1
srun python /scratch/<project_number>/<user_name>/finnish-modernberts/src/data-tools/get_tokens_stats.py --log_dir $LOG_DIR
