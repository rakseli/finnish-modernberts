#!/bin/bash
#SBATCH --job-name=extract-wiki
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error
module load cray-python/3.9.12.1
wikifile=$1
output_dir=$(dirname "$wikifile")
filename=$(basename "$wikifile")
output_prefix=$(echo "$filename" | awk -F'-' '{print $1}')
output_suffix=$(echo "$filename" | awk -F'-' '{print $2}')

echo "wikifile $wikifile"
echo "output_dir $output_dir"
echo "filename $filename"
echo "output_prefix $output_prefix"
echo "output_prefix $output_suffix"
export PYTHONPATH="${PYTHONPATH}:/scratch/<project_number>/<user_name>/finnish-modernberts/external_code/wikiextractor"
srun python -m wikiextractor.WikiExtractor \
    $wikifile \
    --processes $SLURM_CPUS_PER_TASK \
    --output "$output_dir/$output_prefix-$output_suffix"

