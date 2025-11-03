#!/bin/bash
#SBATCH --job-name=exact-dedup-sme-crawls
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error
module purge
ml use /appl/local/csc/modulefiles/
ml pytorch/2.4
source /scratch/<project_number>/<user_name>/venvs/text-dedup/bin/activate
set -euo pipefail
data_path=$1
if [ -d "$data_path" ]; then
  echo "$data_path is a directory"
  output_dir=$data_path
  f_name=$(basename $data_path)
  echo $output_dir
  echo $f_name
  srun \
   python ../external_code/text-dedup/text_dedup/exact_hash.py \
       --path "json" \
       --data_dir $data_path \
       --split "train" \
       --output "${output_dir}/${f_name}-deduplicated.jsonl" \
       --column "text" \
       --batch_size 1000 \
       --use_auth_token true \
       --num_proc $SLURM_CPUS_PER_TASK \
       --compatible_mode true
else
  output_dir=$(dirname $data_path)
  f_name=$(basename $data_path |awk -F'.' '{print $1}')
  echo $output_dir
  echo $f_name
  srun \
   python ../external_code/text-dedup/text_dedup/exact_hash.py \
       --path "json" \
       --data_files $data_path \
       --split "train" \
       --output "${output_dir}/${f_name}-deduplicated.jsonl" \
       --column "text" \
       --batch_size 1000 \
       --use_auth_token true \
       --num_proc $SLURM_CPUS_PER_TASK

fi
