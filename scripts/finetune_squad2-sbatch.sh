#!/bin/bash
#SBATCH --job-name=train_squad
#SBATCH --account=<project_number>
#SBATCH --time=12:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpusmall
#SBATCH --output=../logs/finsquad/%x_%j.output
#SBATCH --error=../logs/finsquad/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
set -euo pipefail
#Mahti
#https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
CONTAINER_PATH=<container_path>
export LOCAL_SCRATCH=<cache_path>
export HF_HOME=<hf_cache>
TMPDIR=$LOCAL_SCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
#Models
#TurkuNLP/finnish-modernbert-large-short
#TurkuNLP/finnish-modernbert-large
#TurkuNLP/bert-large-finnish-cased-v1

apptainer exec --bind="/users,/projappl,/scratch" \
  --nv $CONTAINER_PATH \
  python ../src/finsquad/run_qa.py \
  --model_name_or_path TurkuNLP/finnish-modernbert-large \
  --dataset_name TurkuNLP/finbenchv2-squad-strip-fi-mt \
  --do_train --do_eval \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../results/qa-finsquad2-finnish-modernbert-large \
  --version_2_with_negative \
  --save_steps 5000 \

echo "End time: $(date)"
