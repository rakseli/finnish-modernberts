#!/bin/bash
#SBATCH --job-name=sample-to-target-size
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --account=<project_number>
#SBATCH --output=../logs/data-logs/%x_%j.output
#SBATCH --error=../logs/data-logs/%x_%j.error
declare -A factors
base_path=/scratch/<project_number>/<user_name>/finnish-modernberts/data/final-data/train
output_path=/scratch/<project_number>/<user_name>/finnish-modernberts/data/final-data/train/sampled-data
mkdir -p "$output_path"


FORCE=false
while getopts ":f" option; do
  case "$option" in
    f)
      FORCE=true
      ;;
    *)
      echo "Usage: $0 [-f] (force delete mode)"
      exit 1
      ;;
  esac
done


#code
factors["$base_path/code/code-smollmpythonedu-train.jsonl"]=30 
factors["$base_path/code/code-starcodergithubissues-train.jsonl"]=0.83
#english
factors["$base_path/en/en-britishlibrary-train.jsonl"]=1.0  
factors["$base_path/en/en-europarlen-train.jsonl"]=5.0
factors["$base_path/en/en-finewebedufortified-train.jsonl"]=0.5  
factors["$base_path/en/en-naturalinstructions-train.jsonl"]=1.0
factors["$base_path/en/en-pes2o-train.jsonl"]=0.13  
factors["$base_path/en/en-pubmedabstracts-train.jsonl"]=1.0
factors["$base_path/en/en-pubmedcentral-train.jsonl"]=0.1  
factors["$base_path/en/en-wikien-train.jsonl"]=9.0
##finnish
factors["$base_path/fi/fi-ccfi-train.jsonl"]=4.0  
factors["$base_path/fi/fi-culturaxfi-train.jsonl"]=3.7
factors["$base_path/fi/fi-europarlfi-train.jsonl"]=6.0  
factors["$base_path/fi/fi-hplt2fi-train.jsonl"]=3.7
factors["$base_path/fi/fi-nlfclfi-train.jsonl"]=6.0  
factors["$base_path/fi/fi-projectlonnrot-train.jsonl"]=6.0 
factors["$base_path/fi/fi-redditsuomi-train.jsonl"]=6.0
factors["$base_path/fi/fi-suomi24-train.jsonl"]=6.0  
factors["$base_path/fi/fi-wikifi-train.jsonl"]=30.0
factors["$base_path/fi/fi-ylenewsfi-train.jsonl"]=30.0 
factors["$base_path/fi/fi-ylilauta-train.jsonl"]=5.0
##la
factors["$base_path/la/la-culturaxla-train.jsonl"]=30.0
##sme
factors["$base_path/sme/sme-glot500-train.jsonl"]=30.0
factors["$base_path/sme/sme-saamiweb-train.jsonl"]=30.0
factors["$base_path/sme/sme-salt-train.jsonl"]=30.0
##sv
factors["$base_path/sv/sv-culturaxsv-train.jsonl"]=1.09
factors["$base_path/sv/sv-europarlsv-train.jsonl"]=5.0
factors["$base_path/sv/sv-fstc-train.jsonl"]=5.0
factors["$base_path/sv/sv-hplt2sv-train.jsonl"]=1.05
factors["$base_path/sv/sv-nlfcsv-train.jsonl"]=5.0
factors["$base_path/sv/sv-wikisv-train.jsonl"]=30.0
factors["$base_path/sv/sv-ylenewssv-train.jsonl"]=30.0
##xling
factors["$base_path/xling/xling-enfi-train.jsonl"]=0.62
factors["$base_path/xling/xling-ensme-train.jsonl"]=30.0
factors["$base_path/xling/xling-ensv-train.jsonl"]=0.57
factors["$base_path/xling/xling-fien-train.jsonl"]=0.62
factors["$base_path/xling/xling-fisme-train.jsonl"]=30.0
factors["$base_path/xling/xling-fisv-train.jsonl"]=5.7
factors["$base_path/xling/xling-smeen-train.jsonl"]=30.0
factors["$base_path/xling/xling-smefi-train.jsonl"]=30.0
factors["$base_path/xling/xling-smesv-train.jsonl"]=30.0
factors["$base_path/xling/xling-sven-train.jsonl"]=0.58
factors["$base_path/xling/xling-svfi-train.jsonl"]=5.7
factors["$base_path/xling/xling-svsme-train.jsonl"]=30.0


for file in "${!factors[@]}"; do
    f_name=$(basename $file .jsonl)
    factor=${factors[$file]}
    if [ -e "$output_path/$f_name-sampled.jsonl" ]; then
        if $FORCE; then
            echo "output file $output_path/$f_name-sampled.jsonl exists, force mode enabled. Deleting..."
            rm "$output_path/$f_name-sampled.jsonl"
            echo "File deleted."
        else
            echo "File $output_path/$f_name-sampled.jsonl exists," 
            echo "but force delete mode is not enabled. Use -f to delete."
            echo "Continuing to next file"
            continue
        fi
    else
    echo "Output file, for $file does not exist, continuing to create one..."
    fi
    if (( $(echo "$factor >= 1" | bc -l) )); then
        # Separate integer and fractional parts
        int_part=${factor%.*}
        frac_part=$(echo "$factor - $int_part" | bc)
        for ((i = 0; i < int_part; i++)); do
            cat "$file" >> "$output_path/$f_name-sampled.jsonl"
        done
        if [[ $(echo "$frac_part > 0" | bc) -eq 1 ]]; then
            awk -v frac="$frac_part" 'BEGIN {srand()} {if (rand() <= frac) print $0}' "$file" >> "$output_path/$f_name-sampled.jsonl"
        fi

    else
        # Downsample
        awk -v f="$factor" 'BEGIN {srand()} {if (rand() <= f) print $0}' "$file" > "$output_path/$f_name-sampled.jsonl"
    fi
done