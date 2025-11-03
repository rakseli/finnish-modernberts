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

def process_results(root_path):
    results = {}

    # List all model directories
    model_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    for model in model_dirs:
        model_path = os.path.join(root_path, model)
        results[model] = {}
        
        # List all result files
        res_files = [d for d in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, d)) and "model_meta.json" not in d]
        task_scores = {}
        for r_f in res_files:
            res_file_path = os.path.join(model_path, r_f)
            if not os.path.exists(res_file_path):
                continue
                

            with open(res_file_path, 'r') as f:
                data = json.load(f)
                
                # Extract task name and main score
                task_name = data.get("task_name")
                if not task_name:
                    print(f"Did not find task_name from path {res_file_path}")
                    continue
                    
                # Extract scores
                test_scores = data.get("scores", {}).get("test", [{}])
                if len(test_scores)>1:
                    print("Task has more than 1 entry meaning that we have to get the mean")
                    main_score = statistics.mean([s.get("main_score") for s in test_scores])
                else:
                    main_score = data.get("scores", {}).get("test", [{}])[0].get("main_score")
                
                if main_score is not None:
                    task_scores[task_name] = main_score
            
            # Calculate average score if there are any tasks
            if task_scores:
                average_score = statistics.mean(task_scores.values())
                results[model]= {
                    "tasks": task_scores,
                    "average": average_score
                }
    
    return results



def save_results_as_jsonl(results, output_path):
    with open(output_path, 'w') as f:
        for model, data in results.items():
            print(model)
            output_record = {
                "model": model,
                "tasks": data["tasks"],
                "average": data["average"]
            }
            f.write(json.dumps(output_record) + '\n')

def main():
    args = parse_args()
    # Process all results
    all_results = process_results(args.input_dir)
    print(all_results)
    # Save results to JSONL file
    save_results_as_jsonl(all_results, args.output_path)
    
    print(f"Results have been saved to {args.output_path}")

if __name__ == "__main__":
    main()