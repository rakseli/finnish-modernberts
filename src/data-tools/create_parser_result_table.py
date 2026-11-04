#!/usr/bin/env python3
"""Build a LaTeX table of test UAS/LAS from spacy_deps_<treebank>_<model>/results.json.

Usage:
    python make_deps_table.py RESULTS_ROOT [-o table.tex] [--decimals 1]
"""
import argparse
import json
import re
from pathlib import Path

model_map = {
    "finnish-modernbert-large-3e-05": "Finnish-ModernBERT-large",
    "finnish-modernbert-base-8e-05": "Finnish-ModernBERT-base",
    "finnish-modernbert-tiny-0.0001": "Finnish-ModernBERT-tiny",
    "xlm-roberta-large-2e-05": "XLM-RoBERTa-large",
    "mmBERT-base-1e-05": "mmBERT-base",
    "finnish-modernbert-tiny-short-8e-05": "Finnish-ModernBERT-tiny-short",
    "finnish-modernbert-base-short-0.0001": "Finnish-ModernBERT-base-short",
    "finnish-modernbert-large-short-5e-05": "Finnish-ModernBERT-large-short",
    "finnish-modernbert-tiny-short-cpt-5e-05": "Finnish-ModernBERT-tiny-short-cpt",
    "finnish-modernbert-base-short-cpt-0.0001": "Finnish-ModernBERT-base-short-cpt",
    "finnish-modernbert-large-short-cpt-2e-05": "Finnish-ModernBERT-large-short-cpt",
    "finnish-modernbert-tiny-short-edu-8e-05": "Finnish-ModernBERT-tiny-short-edu",
    "finnish-modernbert-large-short-edu-5e-05": "Finnish-ModernBERT-large-short-edu",
    "finnish-modernbert-base-short-edu-3e-05": "Finnish-ModernBERT-base-short-edu",
    "finnish-modernbert-tiny-edu-8e-05": "Finnish-ModernBERT-tiny-edu",
    "finnish-modernbert-large-edu-3e-05": "Finnish-ModernBERT-large-edu",
    "finnish-modernbert-base-edu-0.0001": "Finnish-ModernBERT-base-edu",
    "qwen3-embedding-0.6B-None": "Qwen3-Embedding-0.6B",
}

# Strip the trailing learning rate ("-8e-05", "-0.0001", "-None") from the keys
LR_SUFFIX = re.compile(r"-(?:\d+(?:\.\d+)?(?:e-?\d+)?|None)$")
name_map = {LR_SUFFIX.sub("", k): v for k, v in model_map.items()}
model_order = list(name_map)  # row order follows model_map

TREEBANK_NAMES = {"sme_giella": "North Sámi (Giella)"}


def tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def collect(root: Path):
    """Return {treebank: {model_key: (uas, las)}}."""
    data = {}
    for f in sorted(root.glob("spacy_deps_*/results.json")):
        res = json.loads(f.read_text())
        tb = res["treebank"]
        prefix = f"spacy_deps_{tb}_"
        folder = f.parent.name
        if not folder.startswith(prefix):
            print(f"% skipping {folder}: does not match treebank {tb}")
            continue
        model = folder[len(prefix):]
        if model not in name_map:
            print(f"% warning: {model} not in model_map, using raw name")
        test = res["test"]
        data.setdefault(tb, {})[model] = (test["uas"], test["las"])
    return data


def make_table(data, decimals=1, caption=None, label="tab:deps-test"):
    treebanks = sorted(data)
    models = [m for m in model_order if any(m in data[tb] for tb in treebanks)]
    models += sorted({m for tb in treebanks for m in data[tb]} - set(models))

    # best value per (treebank, metric) for bolding
    best = {(tb, i): max(v[i] for v in data[tb].values())
            for tb in treebanks for i in (0, 1)}

    def cell(tb, m, i):
        if m not in data[tb]:
            return "--"
        v = data[tb][m][i]
        s = f"{100 * v:.{decimals}f}"
        return rf"\textbf{{{s}}}" if round(v, 10) == round(best[(tb, i)], 10) else s

    tb_disp = [TREEBANK_NAMES.get(tb, tex_escape(tb)) for tb in treebanks]
    if caption is None:
        caption = ("Test-set dependency parsing results (UAS and LAS, \\%) on "
                   + ", ".join(tb_disp)
                   + ". Scores are computed over all syntactic words including "
                     "punctuation with full dependency labels, using the learning "
                     "rate selected on the development set. Best results in bold.")

    ncols = 2 * len(treebanks)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "rr" * len(treebanks) + "}",
        r"\toprule",
    ]
    if len(treebanks) > 1:
        lines.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for t in tb_disp) + r" \\")
        lines.append(" ".join(rf"\cmidrule(lr){{{2 + 2*k}-{3 + 2*k}}}" for k in range(len(treebanks))))
    lines.append("Model & " + " & ".join(["UAS", "LAS"] * len(treebanks)) + r" \\")
    lines.append(r"\midrule")
    for m in models:
        row = [tex_escape(name_map.get(m, m))]
        for tb in treebanks:
            row += [cell(tb, m, 0), cell(tb, m, 1)]
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="folder containing spacy_deps_* directories")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--decimals", type=int, default=1)
    ap.add_argument("--label", default="tab:deps-test")
    ap.add_argument("--caption", default="Dependency parsing in Northern S{\'a}mi evaluation results.")
    args = ap.parse_args()

    data = collect(args.root)
    if not data:
        raise SystemExit(f"No results.json found under {args.root}")
    tex = make_table(data, args.decimals, args.caption, args.label)
    if args.output:
        args.output.write_text(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()