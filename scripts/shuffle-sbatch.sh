#!/bin/bash
#SBATCH --job-name shuffle
#SBATCH --account=<project_number>
#SBATCH --partition=small         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00     
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --output=../logs/data-logs/%x_%j.output
#SBATCH --error=../logs/data-logs/%x_%j.error

echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: <dir or file>, no arguments given, exiting.."
    exit 1
fi

if [ -d "$1" ]; then
  echo "$1 is a folder, shards are combined..."
  dir_path=$1
  basename_without_suffix=$(basename $dir_path)
  if [ -e "$dir_path/$basename_without_suffix-combined.jsonl" ]; then
    echo "File $dir_path/$basename_without_suffix-combined.jsonl exists, skipping combination..."
    input_file=$dir_path/$basename_without_suffix-combined.jsonl
  else
    cat $dir_path/*.jsonl > $dir_path/$basename_without_suffix-combined.jsonl
    input_file=$dir_path/$basename_without_suffix-combined.jsonl
  fi
else
  echo "$1 is not a folder, will work on individual file..."
  basename_without_suffix=$(basename $1 .jsonl)
  dir_path=$(dirname $1)
  input_file=$1
fi

if [ -e "$dir_path/$basename_without_suffix-shuffled.jsonl" ]; then
    echo "File $dir_path/$basename_without_suffix-shuffled.jsonl exists, skipping shuffling..."
else
    echo "Shuffling..."
    srun shuf --random-source="../data/random-sources/random_source_large.bin" $input_file -o $dir_path/$basename_without_suffix-shuffled.jsonl
    echo "Done shuffling"
fi

echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
