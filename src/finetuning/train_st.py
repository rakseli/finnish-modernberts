# Adapted from https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/evaluate_st.py
# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
# 

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from training.distributed_utils import is_main_process
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

def main():
    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir",type=str,default="../../results/finetuned-models")
    
    args = parser.parse_args()
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]
    results_dir = os.path.join(args.output_dir,model_shortname+"_msmarco")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir,exist_ok=True)

    # 1. Load a model to finetune
    model = SentenceTransformer(model_name,config_kwargs={"reference_compile":False})

    # 2. Load a dataset to finetune on
    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)  # Increase mini_batch_size if you have enough VRAM

    run_name = f"{model_shortname}-{lr}"
    # 4. (Optional) Specify training arguments
    run_dir = os.path.join(results_dir,run_name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=run_dir,
        # Optional training parameters:
        dataloader_drop_last=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=512,
        warmup_ratio=0.05,
        fp16=False,  # Set to False if GPU can't handle FP16
        bf16=True,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        learning_rate=lr,
        # Optional tracking/debugging parameters:
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=1000,
        report_to='tensorboard',
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. (Optional) Evaluate the trained model on the evaluator after training
    dev_evaluator(model)

    # 8. Save the model
    final_model_dir = os.path.join(run_dir,"final")
    model.save_pretrained(final_model_dir)

    #if is_main_process():
    #    model.push_to_hub(f"{run_name}-msmarco", private=True)

if __name__ == "__main__":
    main()