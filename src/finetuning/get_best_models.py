import os
import json
import argparse
from pathlib import Path
import statistics

def parse_args():
    parser = argparse.ArgumentParser(description='Process model evaluation results and find best learning rates.')
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory containing model results')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSONL file')
    return parser.parse_args()

def extract_lr_from_path(path):
    """Extract learning rate from path name."""
    import re
    
    # Get the basename of the path
    basename = os.path.basename(path)
    
    # Look for common learning rate patterns
    # Pattern for scientific notation (e.g., 1e-05) or decimal (e.g., 0.0001)
    lr_match = re.search(r'((?:\d+\.\d+)|(?:\d+e-\d+))$', basename)
    if lr_match:
        return float(lr_match.group(1))
    
    # Alternative approach: find the last sequence of digits with optional decimal or scientific notation
    lr_match = re.search(r'[-](\d+(?:\.\d+)?(?:e-\d+)?)$', basename)
    if lr_match:
        return float(lr_match.group(1))
    
    # If we couldn't find a clear match, try to extract the last component and see if it converts
    try:
        # Last component after the last hyphen
        last_part = basename.split('-')[-1]
        return float(last_part)
    except ValueError:
        print(f"Warning: Could not parse learning rate from '{path}'")
        return None


def process_results(root_path):
    results = {}
    wanted_models = set(['finnish-modernbert-base',  
                         'finnish-modernbert-base-short',
                         'finnish-modernbert-base-short-cpt',
                         'finnish-modernbert-large',
                         'finnish-modernbert-large-short',
                         'finnish-modernbert-large-short-cpt',
                         'finnish-modernbert-tiny',
                         'finnish-modernbert-tiny-short',
                         'finnish-modernbert-tiny-short-cpt',
                         'mmBERT-base',
                         'xlm-roberta-large',
                         'finnish-modernbert-base-edu',  
                         'finnish-modernbert-large-edu', 
                         'finnish-modernbert-tiny-edu',
                         'finnish-modernbert-base-short-edu',  
                         'finnish-modernbert-large-short-edu', 
                         'finnish-modernbert-tiny-short-edu'])
    # List all model directories
    model_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d in wanted_models]
    
    for model in model_dirs:
        model_path = os.path.join(root_path, model)
        results[model] = {}
        
        # List all learning rate directories for this model
        lr_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for lr_dir in lr_dirs:
            lr = extract_lr_from_path(lr_dir)
            lr_path = os.path.join(model_path, lr_dir)
            
            # Path to the actual task result files
            result_path = os.path.join(lr_path, "no_model_name_available", "no_revision_available")
            
            if not os.path.exists(result_path):
                continue
                
            # Get all JSON files that are not model_meta.json
            task_files = [f for f in os.listdir(result_path) if f.endswith('.json') and f != "model_meta.json"]
            
            if not task_files:
                continue
                
            task_scores = {}
            for task_file in task_files:
                with open(os.path.join(result_path, task_file), 'r') as f:
                    data = json.load(f)
                    
                    # Extract task name and main score
                    task_name = data.get("task_name")
                    if not task_name:
                        # Use filename without extension as fallback
                        task_name = os.path.splitext(task_file)[0]
                        
                    # Extract main score from the first test item
                    main_score = data.get("scores", {}).get("test", [{}])[0].get("main_score")
                    
                    if main_score is not None:
                        task_scores[task_name] = main_score
            
            # Calculate average score if there are any tasks
            if task_scores:
                average_score = statistics.mean(task_scores.values())
                results[model][lr] = {
                    "tasks": task_scores,
                    "average": average_score
                }
    
    return results

def find_best_lr_for_models(results):
    best_configs = {}
    
    for model, lr_data in results.items():
        if not lr_data:
            continue
            
        # Find the learning rate with the highest average score
        best_lr = max(lr_data.items(), key=lambda x: x[1]["average"])[0]
        
        best_configs[model] = {
            "lr": best_lr,
            "tasks": lr_data[best_lr]["tasks"],
            "average": lr_data[best_lr]["average"]
        }
    
    return best_configs

def save_results_as_jsonl(results, output_path):
    with open(output_path, 'w') as f:
        for model, data in results.items():
            output_record = {
                "model": model,
                "lr": data["lr"],
                "tasks": data["tasks"],
                "average": data["average"]
            }
            f.write(json.dumps(output_record) + '\n')

def main():
    args = parse_args()
    
    # Process all results
    all_results = process_results(args.input_dir)
    # Find best learning rate for each model
    best_configs = find_best_lr_for_models(all_results)
    
    # Save results to JSONL file
    save_results_as_jsonl(best_configs, args.output_path)
    
    print(f"Results have been saved to {args.output_path}")

if __name__ == "__main__":
    main()
