#!/bin/bash
#SBATCH --job-name=process-snt-data
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=20G 
#SBATCH --partition=debug
#SBATCH --time=00:30:00
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
module load cray-python

srun python ../src/data-tools/process_snt_data.py