import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--score",type=int,default=5,help="score to filter")

if __name__ == "__main__":
    args = parser.parse_args()
    sentence_dataset = load_dataset("json", data_files="/scratch/<project_number>/<user_name>/finnish-modernberts/data/code/smollm-python-edu-combined.jsonl", split='train')
    sentence_dataset = sentence_dataset.filter(lambda x:x['int_score']==args.score)
    sentence_dataset.to_json(f"/scratch/<project_number>/<user_name>/finnish-modernberts/data/code/smollm-python-edu-score-{args.score}-combined.jsonl",force_ascii=False)
    