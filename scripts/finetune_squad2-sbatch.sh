#!/bin/bash
#SBATCH --job-name=train_squad
#SBATCH --account=<project number>
#SBATCH --time=02:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --partition=gpumedium
#SBATCH --output=../logs/finsquad/%x_%j.output
#SBATCH --error=../logs/finsquad/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
set -euo pipefail
CONTAINER_PATH=/scratch/<project number>/csc-containers/scripts/basic_pytorch_transformers_20261103_195648.sif
export HF_HOME="/scratch/<project number>/.cache/hf_cache"
#Models
#TurkuNLP/finnish-modernbert-large-short
#TurkuNLP/finnish-modernbert-large
#TurkuNLP/bert-large-finnish-cased-v1
# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Place and bind CPU threads to single CPU cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

apptainer exec --bind="/users,/projappl,/scratch" \
  --nv $CONTAINER_PATH \
  python ../src/finsquad/run_qa.py \
  --model_name_or_path FacebookAI/xlm-roberta-large \
  --dataset_name TurkuNLP/finbenchv2-squad-strip-fi-mt \
  --do_train --do_eval \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../results/qa-finsquad2-xlm-roberta-large \
  --version_2_with_negative \
  --save_steps 5000 \
  

echo "End time: $(date)"
