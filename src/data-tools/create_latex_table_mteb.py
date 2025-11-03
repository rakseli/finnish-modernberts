import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from model results")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output LaTeX file")
    return parser.parse_args()

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
                data.append(d)
            except json.decoder.JSONDecodeError as e:
                print(f"Couldn't read line {line}")
                continue
    return data


MODEL_CATEGORIES = {
    # Define monolingual models - adjust these based on your actual model names
    "monolingual": ["bert-base-finnish-cased-v1","bert-large-finnish-cased-v1",
     "roberta-large-1160k","deberta-v3-base","deberta-v3-large","GTE-en-MLM-large"
     ,"GTE-en-MLM-base","ModernBERT-large", "ModernBERT-base","GTE-en-MLM-base","RoBERTa-base"
    ]
    # All other models will be considered multilingual by default
}


def generate_latex_table(data, model_categories):
    # Identify all unique tasks from the data
    tasks = set()
    for entry in data:
        tasks.update(entry["tasks"].keys())
    tasks = sorted(list(tasks))
    
    # Find the best score for each task and the best average
    best_scores = defaultdict(float)
    best_average = 0.0
    
    for entry in data:
        for task, score in entry["tasks"].items():
            best_scores[task] = max(best_scores[task], score)
        best_average = max(best_average, entry["average"])
    
    # Start building the LaTeX table
    header_row = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{task}}}" for task in tasks])
    if len(tasks)>1:
        header_row = header_row + " & \\textbf{Average} \\\\"
    else:
        header_row = header_row + " \\\\"
    latex = [
        "\\begin{table*}",
        "\\centering",
        "\\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}",
        "\\toprule", header_row
        
    ]
    
    # Group models by category
    models_by_category = defaultdict(list)
    for entry in data:
        if entry["model"] in model_categories['monolingual']:
            category = 'monolingual'
        else:
            category = 'multilingual'
    
        models_by_category[category].append(entry)
    
    # Add each category and its models
    for category in ["monolingual", "multilingual"]:
        if category in models_by_category and models_by_category[category]:
            # Add category header
            if len(tasks)>1:
                addition = 2
            else:
                addition = 1
            latex.append("\\midrule")
            latex.append(f"\\multicolumn{{{len(tasks) + addition}}}{{l}}{{\\textbf{{{category.capitalize()} Models}}}} \\\\")
            latex.append("\\midrule")
            
            # Add models in this category
            for entry in sorted(models_by_category[category], key=lambda x: x["model"]):
                model_name = entry["model"]
                scores = []
                
                for task in tasks:
                    score = entry["tasks"].get(task, 0.0)
                    # Bold if this is the best score for this task
                    if abs(score - best_scores[task]) < 1e-5:  # Using a small epsilon for float comparison
                        scores.append(f"\\textbf{{{score:.3f}}}")
                    else:
                        scores.append(f"{score:.3f}")
                
                # Add the average score, bold if it's the best
                if len(tasks)>1:
                    if abs(entry["average"] - best_average) < 1e-5:
                        scores.append(f"\\textbf{{{entry['average']:.3f}}}")
                    else:
                        scores.append(f"{entry['average']:.3f}")
                
                latex.append(model_name + " & " + " & ".join(scores) + " \\\\")
    
    # Complete the table
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Model performance across different tasks. Best results for each task and best average are highlighted in bold.}",
        "\\label{tab:model-performance}",
        "\\end{table*}"
    ])
    
    return "\n".join(latex)

def main():
    args = parse_args()
    data = read_jsonl(args.input)
    latex_table = generate_latex_table(data, MODEL_CATEGORIES)
    
    with open(args.output, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table has been written to {args.output}")

if __name__ == "__main__":
    main()