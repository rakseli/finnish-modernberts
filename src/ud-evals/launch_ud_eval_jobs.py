import os
import time
import subprocess
import argparse

def get_running_job_names():
    try:
        # Run the squeue command for current user
        result = subprocess.run(["squeue", "--me", "--Format=Name:100", "--noheader"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        # Split the output into lines and strip whitespace
        job_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return job_names
    except subprocess.CalledProcessError as e:
        print(f"Error running squeue: {e.stderr}")
        return []


def create_slurm_scripts(model_name,running_jobs,args):
    """Creates a slurm script in right string format

    Args:
        model_name (str): model to train in HF format
        args (argparse.Namespace): args
    Returns:
    - str: the script
    """
    model_shortname = model_name.split("/")[-1]
    job_name = f"ud-train-and-eval-{model_shortname}"
    if job_name in running_jobs:
        print(f"Job {job_name} is currently running, skipping...")
        return None
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=<project_name>
#SBATCH --time=02:00:00
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpusmall
#SBATCH --output=/scratch/<project_name>/finnish-modernberts/logs/ud-evals/%x_%j.output
#SBATCH --error=/scratch/<project_name>/finnish-modernberts/logs/ud-evals/%x_%j.error
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
module purge
#set -euo pipefail
#Mahti
CONTAINER_PATH="/scratch/project_2017000/containers/pytorch_container_20260407_205334.sif"
export LOCAL_SCRATCH=/scratch/<project_name>/.cache
TMPDIR=$LOCAL_SCRATCH
#export APPTAINERENV_LD_PRELOAD=/appl/soft/ai/lib/fake_tcp_ulp.so
echo "Cache location: $TMPDIR"
#Ray and GF
export HF_HOME=/scratch/<project_name>/.cache/hf_cache
export TUNE_MAX_PENDING_TRIALS_PG=1
export RAY_DISABLE_RESOURCE_DETECTION=1
export OMP_NUM_THREADS=1
export RAY_OVERRIDE_RESOURCES='{{"CPU": 32, "GPU": 1}}'
export RAY_TRAIN_V2_ENABLED=0
#Python
export PYTHONWARNINGS=ignore

apptainer exec --bind="/users,/appl/soft/ai/lib/,/projappl,/scratch,$LOCAL_SCRATCH" --nv $CONTAINER_PATH python /scratch/<project_name>/finnish-modernberts/src/ud-evals/train_model_ud.py --model {model_name} --do_test
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
""" 
    final_res = f"/scratch/<project_name>/finnish-modernberts/results/best_checkpoint_{model_shortname}/results.json"
    if os.path.exists(final_res):
        print(f"Final results for {model_shortname} already exists, skipping...")
        return None
    return script_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
    ap.add_argument('--test', action='store_true', help="launch one test job")
    running_jobs = get_running_job_names()    
    args = ap.parse_args()
    models = ["TurkuNLP/finnish-modernbert-large",
            "FacebookAI/xlm-roberta-large",
            "TurkuNLP/finnish-modernbert-base",
            "TurkuNLP/finnish-modernbert-tiny",
            "jhu-clsp/mmBERT-base",
            "TurkuNLP/finnish-modernbert-tiny-short",
            "TurkuNLP/finnish-modernbert-base-short",
            "TurkuNLP/finnish-modernbert-large-short",
            "TurkuNLP/finnish-modernbert-tiny-short-edu",
            "TurkuNLP/finnish-modernbert-base-short-edu",
           "TurkuNLP/finnish-modernbert-large-short-edu",
            "TurkuNLP/finnish-modernbert-tiny-edu",
            "TurkuNLP/finnish-modernbert-base-edu",
             "TurkuNLP/finnish-modernbert-large-edu"
            ]
    should_break=False
    dry_run_jobs = 0
    job_count = 0
    for m in models:
        if should_break:
            break
        command = create_slurm_scripts(model_name=m,running_jobs=running_jobs,args=args)
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
            result=subprocess.run(["sbatch", temp_file_name], universal_newlines=True)
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

