import os
import time
import subprocess
import argparse

def get_running_job_names():
    try:
        # Run the squeue command for current user
        result = subprocess.run(["squeue", "--me", "--Format=Name:100", "--noheader"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # Split the output into lines and strip whitespace
        job_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return job_names
    except subprocess.CalledProcessError as e:
        print(f"Error running squeue: {e.stderr}")
        return []


def create_slurm_scripts(lr,model_name,running_jobs,args):
    """Creates a slurm script in right string format

    Args:
        lr (float): learning rate 
        model_name (str): model to train in HF format
        args (argparse.Namespace): args
    Returns:
    - str: the script
    """
    model_shortname = model_name.split("/")[-1]
    job_name = f"train-{model_shortname}-lr-{lr}-msmarco"
    if job_name in running_jobs:
        print(f"Job {job_name} is currently running, skipping...")
        return None
    script_content = f"""#!/bin/bash
#SBATCH --job-name {job_name}
#SBATCH --account=<project_number>
#SBATCH --partition={args.partition}         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=8     
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --output=../../logs/finetuning/%x_%j.output
#SBATCH --error=../../logs/finetuning/%x_%j.error

echo "Start time: $(date)"
export EBU_USER_PREFIX=/scratch/<project_number>/<user_name>/EasyBuild
module purge
module load LUMI/24.03
module load PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250619
set -euo pipefail
srun singularity exec $SIF /scratch/<project_number>/<user_name>/finnish-modernberts/scripts/launcher_csc_module.sh train_st.py --lr {lr} --model_name {model_name} --output_dir {args.output_dir}
echo "End time: $(date)"
""" 
    final_model = f"{args.output_dir}/{model_shortname+'_msmarco'}/{model_shortname}-{lr}/final/model.safetensors"
    if os.path.exists(final_model):
        print(f"Finetuned model for {model_shortname} lr {lr} already exists, skipping...")
        return None

    return script_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument('--time',help="time for processing",default="48:00:00")
    ap.add_argument('--partition',help="slurm partition",default="standard-g")
    ap.add_argument('--mem',help="slurm mem",default="480G")
    ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
    ap.add_argument('--test', action='store_true', help="launch one test job")
    ap.add_argument("--output_dir",type=str,default="../../results/finetuned-models/sentence-transfromers")
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
            "TurkuNLP/finnish-modernbert-tiny-short-cpt",
            "TurkuNLP/finnish-modernbert-base-short-cpt",
            "TurkuNLP/finnish-modernbert-large-short-cpt",
            "TurkuNLP/finnish-modernbert-tiny-short-edu",
            "TurkuNLP/finnish-modernbert-base-short-edu",
           "TurkuNLP/finnish-modernbert-large-short-edu",
            "TurkuNLP/finnish-modernbert-tiny-edu",
            "TurkuNLP/finnish-modernbert-base-edu",
             "TurkuNLP/finnish-modernbert-large-edu"
            ]
    lrs = [1e-5, 2e-5, 3e-5, 5e-5, 8e-5,1e-4]
    should_break=False
    dry_run_jobs = 0
    job_count = 0
    for m in models:
        if should_break:
            break
        for l in lrs:
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

