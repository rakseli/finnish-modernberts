"""
Train a spaCy dependency parser (transformer + parser only) on a UD treebank from
the Hugging Face Hub, with a manual learning-rate grid search.

Steps:
  1. Load the treebank from HF and carve dev out of train (same --dev_ratio and
     --seed as train_ud.py, so the dev split is identical to your tagger runs).
  2. Convert train/dev/test to spaCy DocBin files (gold tokens, heads, deprels).
  3. Generate a transformer + parser config with `spacy init config`.
  4. Run `spacy train` once per learning rate in --lrs. spaCy keeps model-best
     per run, selected on dev LAS.
  5. Score every run's model-best on dev with CoNLL-style UAS/LAS
     (all words, punctuation included), pick the best LR, then score it on test.
  6. Write results.json, copy the best run to best_model/, and delete everything
     else in the work dir (corpus, config, every lr_* run).

Usage:
    pip install spacy spacy-transformers datasets
    python spacy_parser_lr_search.py --model FacebookAI/xlm-roberta-large \
        --lrs 1e-5 2e-5 3e-5 5e-5 1e-4 --gpu_id 0

Finished runs are skipped on restart, so an interrupted search can be resumed
by re-running the same command. Once a search completes, the run directories are
deleted; re-running then exits early unless --overwrite is given.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import spacy
from datasets import load_dataset
from spacy.tokens import Doc, DocBin

DATASET = "universal-dependencies/universal_dependencies"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--treebank", default="sme_giella")
    p.add_argument("--model", default="FacebookAI/xlm-roberta-large",
                   help="Hugging Face encoder used by spacy-transformers.")
    p.add_argument("--output_base",
                   default="/scratch/<project number>/finnish-modernberts/results")
    p.add_argument("--lrs", type=float, nargs="+",
                   default=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
                   help="Learning rates to try (one full training run each).")
    p.add_argument("--dev_ratio", type=float, default=0.25,
                   help="Fraction of train held out as dev.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0, help="-1 for CPU.")
    p.add_argument("--max_steps", type=int, default=10000,
                   help="Training steps per run; also the length of the LR schedule.")
    p.add_argument("--warmup_steps", type=int, default=250)
    p.add_argument("--patience", type=int, default=1000,
                   help="Stop a run after this many steps without dev improvement.")
    p.add_argument("--eval_frequency", type=int, default=200)
    p.add_argument("--mixed_precision", action="store_true",
                   help="Enable spacy-transformers mixed precision.")
    p.add_argument("--overwrite", action="store_true",
                   help="Retrain runs that already have a model-best.")
    return p.parse_args()


# =================================================================== data

def syntactic_words(tokens, heads, deprels):
    """
    Keep syntactic words only. Multiword-token ranges (e.g. 1-2) and empty nodes
    (e.g. 8.1) have a non-integer HEAD; dropping them leaves words 1..n in order,
    so HEAD values index straight into the list (0 = ROOT).
    """
    keep = [i for i, h in enumerate(heads) if h is not None and str(h).isdigit()]
    return ([tokens[i] for i in keep],
            [int(heads[i]) for i in keep],
            [deprels[i] for i in keep])


def load_gold(args):
    """Return {split: [(words, heads, deprels), ...]} with UD conventions (1-based heads)."""
    raw = load_dataset(DATASET, args.treebank)
    split = raw["train"].train_test_split(test_size=args.dev_ratio, seed=args.seed)
    raw["train"], raw["validation"] = split["train"], split["test"]
    print("[data]", {k: len(v) for k, v in raw.items()}, flush=True)

    gold = {}
    for name, ds in raw.items():
        sents = []
        for toks, hs, rs in zip(ds["tokens"], ds["head"], ds["deprel"]):
            words, heads, rels = syntactic_words(toks, hs, rs)
            if words:
                sents.append((words, heads, rels))
        gold[name] = sents
    return gold


def to_spacy_doc(vocab, words, heads, rels):
    """
    spaCy convention: heads are absolute 0-based token indices, the root points to
    itself and is labelled "ROOT" (this matches `spacy convert --converter conllu`).
    """
    sp_heads = [i if h == 0 else h - 1 for i, h in enumerate(heads)]
    sp_deps = ["ROOT" if h == 0 else r for h, r in zip(heads, rels)]
    return Doc(vocab, words=words, heads=sp_heads, deps=sp_deps)


def write_docbins(gold, corpus_dir):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    nlp = spacy.blank("xx")
    paths = {}
    for name, sents in gold.items():
        docs = [to_spacy_doc(nlp.vocab, *s) for s in sents]
        path = corpus_dir / f"{name}.spacy"
        DocBin(docs=docs).to_disk(path)
        paths[name] = path
        print(f"[data] wrote {len(docs)} docs to {path}", flush=True)
    return paths


# =================================================================== evaluation

def parse_gold_tokens(nlp, gold_sents, batch_size=32):
    """Run the pipeline on gold tokenization (one sentence per Doc)."""
    docs = [Doc(nlp.vocab, words=words) for words, _, _ in gold_sents]
    for _, proc in nlp.pipeline:
        docs = list(proc.pipe(docs, batch_size=batch_size))
    return docs


def predicted_tree(doc):
    """Convert a parsed Doc back to UD conventions: 1-based heads, 0 = ROOT, 'root' label."""
    heads, rels = [], []
    for tok in doc:
        is_root = tok.head.i == tok.i
        heads.append(0 if is_root else tok.head.i + 1)
        rels.append("root" if is_root or tok.dep_ == "ROOT" else tok.dep_)
    return heads, rels


def attachment_scores(pred_docs, gold_sents):
    """CoNLL-style UAS/LAS over all syntactic words, punctuation included, full labels."""
    total = uas = las = 0
    for doc, (_, g_heads, g_rels) in zip(pred_docs, gold_sents):
        p_heads, p_rels = predicted_tree(doc)
        for ph, pr, gh, gr in zip(p_heads, p_rels, g_heads, g_rels):
            total += 1
            if ph == gh:
                uas += 1
                if pr == gr:
                    las += 1
    return {"uas": uas / total, "las": las / total, "words": total}


def load_pipeline(path, gpu_id):
    if gpu_id >= 0:
        spacy.require_gpu(gpu_id)
    return spacy.load(path)


def free_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


# =================================================================== training

def init_config(path, model_name, mixed_precision):
    """
    Generate the transformer + parser config, then write the encoder name into it.
    `spacy init config --lang xx` defaults to bert-base-multilingual-uncased, so the
    name is set in the file itself rather than only as a `spacy train` override.
    """
    if not path.exists():
        cmd = [sys.executable, "-m", "spacy", "init", "config", str(path),
               "--lang", "xx", "--pipeline", "parser", "--optimize", "accuracy",
               "--gpu", "--force"]
        print("[config]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    cfg = spacy.util.load_config(path, interpolate=False)  # keep ${paths.*} variables
    model_cfg = cfg["components"]["transformer"]["model"]
    model_cfg["name"] = model_name
    if mixed_precision:
        model_cfg["mixed_precision"] = True
    cfg.to_disk(path)
    print(f"[config] {path}: transformer = {model_cfg['name']}", flush=True)


def lr_tag(lr):
    return f"lr_{lr:.0e}".replace("+", "").replace("-0", "-")


def train_one(args, base_cfg, corpus, run_dir, lr):
    best = run_dir / "model-best"
    if best.exists() and not args.overwrite:
        check_encoder(best, args.model)
        print(f"[train] {run_dir.name}: model-best exists, skipping", flush=True)
        return best
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    cmd = [
        sys.executable, "-m", "spacy", "train", str(base_cfg),
        "--output", str(run_dir),
        "--gpu-id", str(args.gpu_id),
        "--paths.train", str(corpus["train"]),
        "--paths.dev", str(corpus["validation"]),
        "--system.seed", str(args.seed),
        "--training.seed", str(args.seed),
        "--training.optimizer.learn_rate.initial_rate", str(lr),
        "--training.optimizer.learn_rate.warmup_steps", str(args.warmup_steps),
        "--training.optimizer.learn_rate.total_steps", str(args.max_steps),
        "--training.max_steps", str(args.max_steps),
        "--training.max_epochs", "0",
        "--training.patience", str(args.patience),
        "--training.eval_frequency", str(args.eval_frequency),
        # Select model-best on LAS only (spaCy's scorer excludes punct here;
        # final model comparison below uses CoNLL-style scores).
        "--training.score_weights.dep_uas", "0.0",
        "--training.score_weights.dep_las", "1.0",
    ]
    log_path = run_dir / "train.log"
    print(f"[train] {run_dir.name}: lr={lr:.2e}, log -> {log_path}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not best.exists():
        tail = log_path.read_text().splitlines()[-30:]
        print("\n".join(tail), flush=True)
        raise RuntimeError(f"spacy train failed for lr={lr} (see {log_path})")
    check_encoder(best, args.model)
    return best


def check_encoder(model_dir, expected):
    """Fail loudly if a trained pipeline used a different encoder than requested."""
    cfg = spacy.util.load_config(model_dir / "config.cfg", interpolate=False)
    used = cfg["components"]["transformer"]["model"]["name"]
    if used != expected:
        raise RuntimeError(f"{model_dir} was trained with '{used}', not '{expected}'. "
                           "Delete it (or pass --overwrite) and retrain.")


# =================================================================== cleanup

KEEP = {"results.json", "best_model"}


def clean_work_dir(work):
    """Delete everything in the work dir except results.json and best_model/."""
    for p in work.iterdir():
        if p.name in KEEP:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"[cleanup] removed {p}", flush=True)


# =================================================================== main

def main():
    args = parse_args()
    work = Path(args.output_base) / f"spacy_deps_{args.treebank}_{os.path.basename(args.model)}"
    work.mkdir(parents=True, exist_ok=True)

    if (work / "results.json").exists() and (work / "best_model").exists() \
            and not args.overwrite:
        print(f"[done] {work} already has results.json and best_model/; "
              "pass --overwrite to rerun the search.", flush=True)
        return

    gold = load_gold(args)
    corpus = write_docbins(gold, work / "corpus")
    base_cfg = work / "base.cfg"
    init_config(base_cfg, args.model, args.mixed_precision)

    # ---- learning-rate search
    search = []
    for lr in args.lrs:
        run_dir = work / lr_tag(lr)
        model_path = train_one(args, base_cfg, corpus, run_dir, lr)
        nlp = load_pipeline(model_path, args.gpu_id)
        dev = attachment_scores(parse_gold_tokens(nlp, gold["validation"]), gold["validation"])
        del nlp
        free_gpu()
        print(f"[search] lr={lr:.2e}  dev UAS {dev['uas']:.4f}  LAS {dev['las']:.4f}",
              flush=True)
        search.append({"learning_rate": lr, "run": run_dir.name, "dev": dev})

    best = max(search, key=lambda r: r["dev"]["las"])
    print(f"\n[search] best lr = {best['learning_rate']:.2e} "
          f"(dev LAS {best['dev']['las']:.4f})", flush=True)

    # ---- test
    best_src = work / best["run"] / "model-best"
    nlp = load_pipeline(best_src, args.gpu_id)
    test = attachment_scores(parse_gold_tokens(nlp, gold["test"]), gold["test"])
    del nlp
    free_gpu()
    print(f"[test] UAS {test['uas']:.4f}  LAS {test['las']:.4f}", flush=True)

    final_dir = work / "best_model"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best_src, final_dir)

    results = {
        "treebank": args.treebank,
        "model": args.model,
        "best_learning_rate": best["learning_rate"],
        "search": search,
        "test": test,
        "note": "UAS/LAS over all syntactic words incl. punctuation, full deprel labels.",
    }
    # Write to a temp file and rename, so results.json only appears once complete.
    tmp = work / "results.json.tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(work / "results.json")

    # Only now, with both outputs safely in place, delete everything else.
    clean_work_dir(work)
    print(f"\nSaved best pipeline to {final_dir}/ and results to {work}/results.json",
          flush=True)


if __name__ == "__main__":
    main()