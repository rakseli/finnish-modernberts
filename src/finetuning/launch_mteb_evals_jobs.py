import os
import time
import subprocess
import argparse
import json
from launch_st_finetuning_jobs import get_running_job_names

def create_slurm_scripts(lr,model_name,running_jobs,args):
    """Creates a slurm script in right string format

    Args:
        lr (float): learning rate 
        model_name (str): model to train in HF format
        args (argparse.Namespace): args
    Returns:
    - str: the script, will run code evals by default
    """
    if "/" in model_name:
        model_shortname = model_name.split("/")[-1]
    else:
        model_shortname = model_name
    print(model_shortname)
    if args.hp_search:
        job_name = f"eval-{model_shortname}-lr-{lr}-mteb"
        task_str =''
        final_file = "TRECCOVID.json"
        final_res = f"{args.output_dir}/{model_shortname}/{model_shortname}-{lr}/no_model_name_available/no_revision_available/{final_file}"

    elif args.mldr:
        job_name = f"eval-{model_shortname}-lr-{lr}-mldr"
        task_str = '--task_names MultiLongDocRetrieval'
        final_file = "MultiLongDocRetrieval.json"
        final_res = f"{args.output_dir}/mldr-results/{model_shortname}/{final_file}"

    else:
        job_name = f"eval-{model_shortname}-lr-{lr}-code-tasks"
        task_str = '--task_names COIRCodeSearchNetRetrieval StackOverflowQA'
        final_file = "StackOverflowQA.json"
        final_res = f"{args.output_dir}/code-results/{model_shortname}/{final_file}"

    if job_name in running_jobs:
        print(f"Job {job_name} is currently running, skipping...")
        return None
    script_content = f"""#!/bin/bash
#SBATCH --job-name {job_name}
#SBATCH --account=<project_number>
#SBATCH --partition={args.partition}         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --output=../../logs/mteb-logs/%x_%j.output
#SBATCH --error=../../logs/mteb-logs/%x_%j.error

echo "Start time: $(date)"
export EBU_USER_PREFIX=/scratch/<project_number>/<user_name>/EasyBuild
module purge
module load LUMI/24.03
module load PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250530
set -euo pipefail
export OMP_NUM_THREADS=1
rocm-smi
srun singularity exec $SIF python run_mteb.py --lr {lr} --model_name {model_name} {task_str}
echo "End time: $(date)"
""" 
    print(final_res)
    if os.path.exists(final_res):
        print(f"Model {model_shortname} lr {lr} results exists, skipping...")
        return None

    return script_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument('--time',help="time for processing",default="01:00:00")
    ap.add_argument('--partition',help="slurm partition",default="small-g")
    ap.add_argument('--mem',help="slurm mem",default="64G")
    ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
    ap.add_argument('--test', action='store_true', help="launch one test job")
    ap.add_argument("--output_dir",type=str,default="../../results/mteb")
    ap.add_argument("--hp_search",action='store_true',help="whether to run hp search evals")
    ap.add_argument("--mldr",action='store_true',help="whether to run mldr eval")

    running_jobs = get_running_job_names()    
    args = ap.parse_args()
    models = ["TurkuNLP/finnish-modernbert-large",
            "TurkuNLP/finnish-modernbert-base",
            "TurkuNLP/finnish-modernbert-tiny",
            "FacebookAI/xlm-roberta-large",
            "jhu-clsp/mmBERT-base",
            "TurkuNLP/finnish-modernbert-tiny-short",
            "TurkuNLP/finnish-modernbert-base-short",
            "TurkuNLP/finnish-modernbert-large-short",
            "TurkuNLP/finnish-modernbert-tiny-short-cpt",
            "TurkuNLP/finnish-modernbert-base-short-cpt",
            "TurkuNLP/finnish-modernbert-large-short-cpt"
            "TurkuNLP/finnish-modernbert-tiny-short-edu",
            "TurkuNLP/finnish-modernbert-large-short-edu",
            "TurkuNLP/finnish-modernbert-base-short-edu",
            "TurkuNLP/finnish-modernbert-tiny-edu",
            "TurkuNLP/finnish-modernbert-large-edu",
            "TurkuNLP/finnish-modernbert-base-edu"
            ]
    if not args.hp_search:
        lines = []
        with open(f"{args.output_dir}/hp-search-results/mteb_results.jsonl") as hp_search_results:
            for l in hp_search_results:
                lines.append(json.loads(l))
        best_lrs = dict([(l['model'],l['lr']) for l in lines])
        print(best_lrs)
    lrs = [1e-5, 2e-5, 3e-5, 5e-5, 8e-5,1e-4]
    should_break=False
    dry_run_jobs = 0
    job_count = 0
    for m in models:
        if should_break:
            break
        for l in lrs:
            if not args.hp_search:
                if l != best_lrs[m.split("/")[-1]]:
                    continue
            command = create_slurm_scripts(lr=l,model_name=m,running_jobs=running_jobs,args=args)
            if command is None:
                continue
            if args.dry_run:
                print(command)
                dry_run_jobs+=1
            else:
                temp_file_name = f"{os.getcwd()}/temp_slurm_job.sh"
                with open(temp_file_name,"w") as temp_file:
                    temp_file.write(command)
                    # Submit the SLURM job using sbatch with the temporary file
                result=subprocess.run(["sbatch", temp_file_name], text=True)
                print(result)
                time.sleep(1)
                os.remove(temp_file_name)
                job_count+=1
                running_jobs = get_running_job_names()
                if args.test:
                    should_break = True
                    break


    print(f"Launched {job_count} jobs")
    if args.dry_run:
        print(f"Would have launched {dry_run_jobs} jobs")

