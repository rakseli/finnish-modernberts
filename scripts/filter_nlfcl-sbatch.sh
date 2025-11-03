#!/bin/bash
#SBATCH --job-name=filter-nlfcl
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G 
#SBATCH --partition=small
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

python ../src/data-tools/filter_jsonl.py \
    --foreign-ratio 0.0001 \
    --nlfcl \
    --jsonl \
    ../data/source/fi/nlfcl-fi/nlfcl-fi-shard-0.jsonl > ../data/source/fi/nlfcl-fi/nlfcl_fi_filtered.jsonl

python ../src/data-tools/filter_jsonl.py \
    --foreign-ratio 0.001 \
    --nlfcl \
    --jsonl \
    ../data/source/sv/nlfcl-sv/nlfcl_sv.jsonl > ../data/source/sv/nlfcl-sv/nlfcl_sv_filtered.jsonl