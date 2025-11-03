import json
import argparse
from datasets import load_dataset,DownloadMode
from torch.utils.data import DataLoader
from datasets import disable_caching
from file_helpers import naive_data_collator, DateTimeEncoder,format_duration
from timer import Timer


disable_caching()

parser = argparse.ArgumentParser()
parser.add_argument("--output-base", type=str,default='/scratch/<project_number>/<user_name>/finnish-modernberts/data/code', help="path to dir of files")
parser.add_argument("--test",action='store_true')

# Optionally, print the first example to verify the data
def download_data(output_file):
    print(f"Starting to download",flush=True)
    result = {}
    t = Timer()
    with t("download"):
        ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train",streaming=True,download_mode=DownloadMode.FORCE_REDOWNLOAD)
        dataloader= DataLoader(ds, batch_size=10000,num_workers=2,collate_fn=naive_data_collator)
        with open(output_file, 'w') as jsonl_file:
            print("File opened",flush=True)
            for i,batch in enumerate(dataloader):
                if i == 0:
                    print("First batch loaded",flush=True)
                if i % 100 == 0 and i != 0:
                    print(f"Downloaded {i+1} batches",flush=True)
                for json_object in batch:
                    json_line = json.dumps(json_object,cls=DateTimeEncoder,ensure_ascii=False)
                    jsonl_file.write(json_line + '\n')
                if args.test:
                    break
    result['run_time']=format_duration(int(t.elapsed_times.get('download', 0)))
    print(result)
    return result


if __name__ == "__main__":
    args = parser.parse_args()
    res = download_data(f"{args.output_base}/smollm-python-edu-blobs.jsonl")
