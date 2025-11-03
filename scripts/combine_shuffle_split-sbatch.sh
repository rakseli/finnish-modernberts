#!/bin/bash
#SBATCH --job-name combine-shuffle-and-split
#SBATCH --account=<project_number>
#SBATCH --partition=small         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=1
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

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

# Get total lines
total_lines=$(wc -l < $input_file | cut -d " " -f1)
if [ "$total_lines" -lt 10 ]; then
    echo "There is less than 10 lines in file $input_file, check the source manually..."
    exit 1
fi
# Calculate split sizes
# Define fraction based on line N
if [ "$total_lines" -lt 100 ]; then
    val_fraction=0.2
    if [ -e "$dir_path/$basename_without_suffix-shuffled.jsonl" ]; then
        echo "File $dir_path/$basename_without_suffix-shuffled.jsonl exists, skipping shuffling..."
    else
        echo "Shuffling..."
        srun shuf --random-source="../data/random-sources/random_source_small.bin" $input_file -o $dir_path/$basename_without_suffix-shuffled.jsonl
        echo "Done shuffling"
    fi
elif [ "$total_lines" -lt 1000000 ]; then
    val_fraction=0.01
    if [ -e "$dir_path/$basename_without_suffix-shuffled.jsonl" ]; then
        echo "File $dir_path/$basename_without_suffix-shuffled.jsonl exists, skipping shuffling..."
    else
        echo "Shuffling..."
        srun shuf --random-source="../data/random-sources/random_source_small.bin" $input_file -o $dir_path/$basename_without_suffix-shuffled.jsonl
        echo "Done shuffling"
    fi
else
    val_fraction=0.0001
    if [ -e "$dir_path/$basename_without_suffix-shuffled.jsonl" ]; then
        echo "File $dir_path/$basename_without_suffix-shuffled.jsonl exists, skipping shuffling..."
    else
        echo "Shuffling..."
        srun shuf --random-source="../data/random-sources/random_source_large.bin" $input_file -o $dir_path/$basename_without_suffix-shuffled.jsonl
        echo "Done shuffling"
    fi
fi

echo "Using val frac $val_fraction"
train_fraction=$(awk "BEGIN {print 1 - 2 * $val_fraction}")
echo "Using train frac $train_fraction"
train_lines=$(awk "BEGIN {print int($total_lines * $train_fraction + 0.5)}")
echo "Train lines: $train_lines"
valid_lines=$(awk "BEGIN {print int($total_lines * $val_fraction + 0.5)}")
echo "Valid lines: $valid_lines"

# Split the file
echo "Splitting..."
srun head -n $train_lines $dir_path/$basename_without_suffix-shuffled.jsonl > $dir_path/$basename_without_suffix-train.jsonl
srun tail -n +$((train_lines + 1)) $dir_path/$basename_without_suffix-shuffled.jsonl | head -n $valid_lines > $dir_path/$basename_without_suffix-validation.jsonl
srun tail -n +$((train_lines + valid_lines + 1)) $dir_path/$basename_without_suffix-shuffled.jsonl > $dir_path/$basename_without_suffix-test.jsonl
echo "Done splitting"

echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
