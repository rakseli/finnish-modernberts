#!/bin/bash
#SBATCH --job-name=filter-smollm
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G 
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error


#* fail the script if:
#* -e exit if any command [1] has a non-zero exit status
#* -u a reference to any variable you haven't previously defined - with the exceptions of $* and $@ - is an error
#* -o pipefail  If any command in a pipeline fails, that return code will be used as the return code of the whole pipeline
set -euo pipefail
module purge
module load LUMI/24.03
ml use /appl/local/csc/modulefiles/
ml load pytorch/2.4

srun python ../src/data-tools/filter_smollm.py --score 5