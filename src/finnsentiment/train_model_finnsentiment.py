import argparse
import json
import os
import torch
import evaluate
import numpy as np
from datasets import Dataset,DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ray import tune
import ray
ray.init(include_dashboard=False,num_cpus=32, num_gpus=1)

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    p.add_argument("--output_base", default="/scratch/<project_name>/finnish-modernberts/results/finnsentiment")

    # Learning-rate search
    p.add_argument("--n_trials", type=int, default=10,
                   help="ray trials over the learning rate. 0 disables the search "
                        "and uses --lr directly.")
    p.add_argument("--lr_min", type=float, default=5e-6, help="Search lower bound.")
    p.add_argument("--lr_max", type=float, default=1e-4, help="Search upper bound.")
    p.add_argument("--objective", default="eval_f1",
                   choices=["eval_accuracy", "eval_f1"],
                   help="Dev metric the search maximises.")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Learning rate used when --n-trials 0.")

    p.add_argument("--epochs", type=float, default=7.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dev_ratio", type=float, default=0.25,
                   help="Fraction of train held out as dev.")
    p.add_argument("--label_all_subwords", action="store_true",
                   help="Supervise every subword instead of only the first one.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do_test",action="store_true")
    return p.parse_args()


def load_splits(dev_ratio, seed):
    """Load the data"""
    data = []
    with open("/scratch/<project_name>/finnish-modernberts/data/finnsentiment_minimal.jsonl") as f:
        for l in f:
            data.append(json.loads(l))

    raw = Dataset.from_list(data)
    split = raw.train_test_split(test_size=dev_ratio, shuffle=True,seed=seed)
    dataset = DatasetDict({
    "train":      split['train'],
    "test": split['test'],
    })
    split = dataset['train'].train_test_split(test_size=dev_ratio,shuffle=True,seed=seed)
    dataset['train'], dataset['validation'] = split['train'], split['test']
    print("[data]", {k: len(v) for k, v in dataset.items()},flush=True)
    return dataset


def make_compute_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(
                predictions=preds, references=labels
            )["accuracy"],
            "f1": f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
        }

    return compute_metrics

def make_training_args(args, learning_rate, *, for_search):
    """Search trials skip checkpointing; the final run keeps the best model."""
    output_dir = os.path.join(args.output_base, f"ray_{os.path.basename(args.model)}" if for_search else f"all_best_lr_checkpoints_{os.path.basename(args.model)}")
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no" if for_search else "epoch",
        save_total_limit=3,
        load_best_model_at_end=not for_search,
        metric_for_best_model=args.objective.removeprefix("eval_"),
        greater_is_better=True,
        logging_steps=1,
        disable_tqdm=for_search,
        seed=args.seed,
        bf16=False if any(m in args.model for m in ["xlm-roberta-large","bert-large-finnish-cased-v1","bert-base-finnish-cased-v1"]) else True,
        report_to=[],
    )


def search_learning_rate(args, training_args, tokenized, tokenizer, model_init,
                         collator, compute_metrics):

    def hp_space(trial):
        return {"learning_rate": tune.loguniform(args.lr_min, args.lr_max)}
    

    reporter = tune.CLIReporter(
    metric_columns=["objective", "eval_loss", "eval_accuracy", "training_iteration"],
    parameter_columns=["learning_rate"],
    )

    trainer = Trainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    
    output_dir = os.path.join(args.output_base, f"ray_{os.path.basename(args.model)}")
    best = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        n_trials=args.n_trials,
        hp_space=hp_space,
        progress_reporter=reporter,
        verbose=2,
        compute_objective=lambda metrics: metrics[args.objective],
        **{'max_concurrent_trials':1,
        'storage_path':output_dir,
        'resources_per_trial':{"cpu": 32, "gpu": 1}}
    )
    return best


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = load_splits(args.dev_ratio, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize_function(examples):
        return tokenizer(examples["text"],max_length=args.max_length,truncation=True)

    tokenized = dataset.map(tokenize_function,batched=True)

    def model_init(trial=None):
        if any(m in args.model for m in ["xlm-roberta-large","bert-large-finnish-cased-v1","bert-base-finnish-cased-v1"]):
            additional_args = {"dtype":torch.float32}
        else: 
            additional_args = {"attn_implementation":"flash_attention_2", "dtype":torch.bfloat16}

        return AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=3,
            **additional_args)

    compute_metrics = make_compute_metrics()
    collator = DataCollatorWithPadding(tokenizer)

    best_run = None
    if args.n_trials > 0:
        print(f"\n=== learning-rate search: {args.n_trials} trials on dev "
              f"({args.lr_min:.1e} – {args.lr_max:.1e}, maximising {args.objective}) ===",flush=True)
        best_run = search_learning_rate(
            args,
            make_training_args(args, args.lr, for_search=True),
            tokenized, tokenizer, model_init, collator, compute_metrics,
        )
        learning_rate = best_run.hyperparameters["learning_rate"]
        print(f"\n[search] best lr = {learning_rate:.5e} "
              f"({args.objective} = {best_run.objective:.4f})",flush=True)
    else:
        learning_rate = args.lr
        print(f"[search] skipped; using lr = {learning_rate:.5e}",flush=True)
    if args.do_test:
        print("\n=== final training ===",flush=True)
        trainer = Trainer(
            model=None,
            model_init=model_init,
            args=make_training_args(args, learning_rate, for_search=False),
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        print("\n=== test set ===",flush=True)
        metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
        print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, float)}, indent=2),flush=True)

        final_dir = os.path.join(args.output_base,f"best_checkpoint_{os.path.basename(args.model)}")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "results.json"), "w") as f:
            json.dump(
                {
                    "learning_rate": learning_rate,
                    "search_trials": args.n_trials,
                    "search_objective": best_run.objective if best_run else None,
                    "test": {k: v for k, v in metrics.items() if isinstance(v, float)},
                },
                f, indent=2,
            )
        print(f"\nSaved to {final_dir}/",flush=True)


if __name__ == "__main__":
    main()