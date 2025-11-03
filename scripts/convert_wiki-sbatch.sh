#!/bin/bash
#SBATCH --job-name=convert-wiki
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G 
#SBATCH --partition=small
#SBATCH --time=03:00:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error


#* fail the script if:
#* -e exit if any command [1] has a non-zero exit status
#* -u a reference to any variable you haven't previously defined - with the exceptions of $* and $@ - is an error
#* -o pipefail  If any command in a pipeline fails, that return code will be used as the return code of the whole pipeline
set -euo pipefail
module purge
module load LUMI/24.03
module load cray-python
wikifolder=$1
output_dir=$(dirname "$wikifolder")
filename=$(basename "$wikifolder")
echo "wikifile $wikifolder"
echo "output_dir $output_dir"
echo "filename $filename"
python ../src/data-tools/convert_wikiextrator.py $wikifolder > "$output_dir/$filename.jsonl"
