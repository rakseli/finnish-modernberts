import argparse
import json
import os
import torch
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ray import tune
import ray
ray.init(include_dashboard=False,num_cpus=32, num_gpus=1)
LABEL_COLUMN = "upos"

UPOS_INVENTORY = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET",
    "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX",
]

EXPECTED_TAGS = {
    "sme_giella": [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="FacebookAI/xlm-roberta-large",
                   help="Any encoder checkpoint that has a fast tokenizer.")
    p.add_argument("--output_base", default="/scratch/<project_name>/finnish-modernberts/results")

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

    p.add_argument("--epochs", type=float, default=10.0)
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


def load_splits(dataset_name, config, dev_ratio, seed,do_split=True):
    """Load the treebank, always carving dev out of train."""
    raw = load_dataset(dataset_name, config)
    if do_split:
        split = raw["train"].train_test_split(test_size=dev_ratio, seed=seed)
        raw["train"], raw["validation"] = split["train"], split["test"]

    print("[data]", {k: len(v) for k, v in raw.items()},flush=True)
    return raw


def _make_decoder(raw):
    """Map whatever is in the upos column onto tag-name strings."""

    sample = next(
        (t for split in raw.values() for row in split[LABEL_COLUMN] for t in row),
        None,
    )
    if isinstance(sample, (int, np.integer)):
        return lambda tag: UPOS_INVENTORY[tag]
    return lambda tag: "_" if tag is None else tag


def resolve_labels(raw):
    """
    Return (label_names, encode_fn) covering only the UPOS tags this treebank
    actually uses, 14 of 17 for sme_giella.
    """
    decode = _make_decoder(raw)
    observed = sorted(
        {decode(t) for split in raw.values() for row in split[LABEL_COLUMN] for t in row}
    )
    lookup = {name: i for i, name in enumerate(observed)}
    return observed, (lambda tags: [lookup[decode(t)] for t in tags])


def build_tokenize_fn(tokenizer, args, encode_labels):
    """Tokenize pre-split words and project word-level tags onto subwords."""

    def tokenize_and_align(batch):
        enc = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        all_labels = []
        for i, tags in enumerate(batch[LABEL_COLUMN]):
            tags = encode_labels(tags)
            word_ids = enc.word_ids(batch_index=i)
            labels, previous = [], None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)                  # [CLS] / [SEP] / pad
                elif word_id != previous:
                    labels.append(tags[word_id])         # first subword of a word
                else:
                    labels.append(tags[word_id] if args.label_all_subwords else -100)
                previous = word_id
            all_labels.append(labels)
        enc["labels"] = all_labels
        return enc

    return tokenize_and_align


def make_compute_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        mask = labels != -100
        references, predictions = labels[mask], preds[mask]

        return {
            "accuracy": accuracy.compute(
                predictions=predictions, references=references
            )["accuracy"],
            "f1": f1.compute(
                predictions=predictions, references=references, average="macro"
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
        bf16=False if "xlm-roberta-large" in args.model else True,
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
    raw = load_splits("universal-dependencies/universal_dependencies", "sme_giella", args.dev_ratio, args.seed)
    label_names, encode_labels = resolve_labels(raw)
    print(f"[data] {len(label_names)} UPOS tags in use: {' '.join(label_names)}",flush=True)
    expected = EXPECTED_TAGS.get("sme_giella")
    if expected and label_names != expected:
        print(f"[warn] expected {len(expected)} tags for {args.config}; "
              f"missing {sorted(set(expected) - set(label_names))}, "
              f"unexpected {sorted(set(label_names) - set(expected))}",flush=True)

    # add_prefix_space matters for byte-level BPE tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    if not tokenizer.is_fast:
        raise ValueError(
            f"{args.model} has no fast tokenizer, but word_ids() is required "
            "to align word-level tags with subwords."
        )
    tokenized = raw.map(
        build_tokenize_fn(tokenizer, args, encode_labels),
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing and aligning labels",
    )

    def model_init(trial=None):
        if "xlm-roberta-large" in args.model:
            additional_args = {"dtype":torch.float32}
        else: 
            additional_args = {"attn_implementation":"flash_attention_2", "dtype":torch.bfloat16}

        return AutoModelForTokenClassification.from_pretrained(
            args.model,
            num_labels=len(label_names),
            id2label={i: n for i, n in enumerate(label_names)},
            label2id={n: i for i, n in enumerate(label_names)},**additional_args
        )

    compute_metrics = make_compute_metrics()
    collator = DataCollatorForTokenClassification(tokenizer)

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
                    "labels": label_names,
                    "test": {k: v for k, v in metrics.items() if isinstance(v, float)},
                },
                f, indent=2,
            )
        print(f"\nSaved to {final_dir}/",flush=True)


if __name__ == "__main__":
    main()