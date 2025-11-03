# Adapted from https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/evaluate_st.py
# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import mteb
import argparse
import os
from sentence_transformers import SentenceTransformer
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name',help="model name")
    ap.add_argument('--lr',help="lr for bookkeeping")
    ap.add_argument('--task_names', nargs='+',help="tasks to run",default= ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"])
    args = ap.parse_args()
    all_models_dir = "../../results/finetuned-models/sentence-transfromers"
    output_base_dir = "../../results/mteb"
    if "/" in args.model_name:
        model_shortname = args.model_name.split("/")[-1]
    else:
        model_shortname = args.model_name
        
    run_name = f"{model_shortname}-{args.lr}"
    base_models_dir = os.path.join(all_models_dir,model_shortname+"_msmarco")
    individual_run_dir = os.path.join(base_models_dir,run_name)
    final_model_dir = os.path.join(individual_run_dir,"final")
    model = SentenceTransformer(final_model_dir)
    if "COIRCodeSearchNetRetrieval" in args.task_names:
        tasks = mteb.get_tasks(tasks=args.task_names)
    else:
        tasks = mteb.get_tasks(tasks=args.task_names,languages=["eng"])
    
    evaluation = mteb.MTEB(tasks=tasks)
    if "MultiLongDocRetrieval" in args.task_names:
        output_folder = os.path.join(output_base_dir,"mldr-results",model_shortname)
        batch_size = 16
    elif "COIRCodeSearchNetRetrieval" in args.task_names:
        output_folder = os.path.join(output_base_dir,"code-results",model_shortname)
        batch_size = 32
    else:
        output_folder = os.path.join(output_base_dir,model_shortname,run_name)
        batch_size = 32
        
    results = evaluation.run(model, output_folder=output_folder,encode_kwargs={"batch_size": batch_size})
