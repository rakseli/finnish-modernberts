#!/bin/bash
#SBATCH --job-name=downsample_jsonl
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=70G
#SBATCH --partition=small
#SBATCH --time=03:00:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

output_filename=$(basename $1 .jsonl)
output_filepath=$(dirname $1)
echo $output_filename
echo $output_filepath
echo "Starting $(date)"
srun perl -pe '$_="" unless rand()<'"$2"  "$1" > "$output_filepath/$output_filename-downsampled.jsonl"
echo "Done  $(date)"