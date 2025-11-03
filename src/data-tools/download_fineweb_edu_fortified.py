
import json
import concurrent.futures
import argparse
import os
import subprocess
import sys
from datasets import load_dataset, get_dataset_config_names,DownloadMode
from torch.utils.data import DataLoader
from file_helpers import naive_data_collator, DateTimeEncoder,format_duration
from timer import Timer
from datasets import disable_caching

disable_caching()

parser = argparse.ArgumentParser()
parser.add_argument("--output-base", type=str, help="path to dir of files")
parser.add_argument("--test",action='store_true')
parser.add_argument("--manual",action='store_true')
parser.add_argument("--split",default=0,type=int,help="int in 0<=split<=4")

hf_cache = os.getenv('HF_HOME')
print(f"HF cache would be saved here <{hf_cache}>if not disabled")

def sort_and_chunk(lst):
    # Step 1: Sort the list
    lst_sorted = sorted(lst)
    # Step 2: Split the list into chunks of the specified size
    chunks = [lst_sorted[i:i + 19] for i in range(0, len(lst_sorted), 19)]
    return chunks

def download_crawl(name,output_file):
    print(f"Starting to download crawl {name}",flush=True)
    result = {}
    result['crawl_id']=name
    t = Timer()
    with t("download"):
        try:
            fw = load_dataset("airtrain-ai/fineweb-edu-fortified", name=name, split="train", streaming=True,download_mode=DownloadMode.FORCE_REDOWNLOAD)
            fw = fw.remove_columns("embedding")
            dataloader= DataLoader(fw, batch_size=10000,num_workers=4,collate_fn=naive_data_collator)
            with open(output_file, 'w') as jsonl_file:
                for batch in dataloader:
                    for json_object in batch:
                        json_line = json.dumps(json_object,cls=DateTimeEncoder,ensure_ascii=False)
                        jsonl_file.write(json_line + '\n')
            result['success']=True

        except Exception as e:
            result['success']=False
            result['exception']=e
            pass
        if result['success']==True:
            c_res = subprocess.run(['/appl/lumi/SW/LUMI-24.03/C/EB/zstd/1.5.5-cpeGNU-24.03/bin/zstd','-3', '-T4','--rm', output_file],stdout=subprocess.PIPE)
            if c_res.returncode == 0:
               result['compression_success']=True
            else:
               result['compression_success']=False
    
    result['run_time']=format_duration(int(t.elapsed_times.get('download', 0)))
    print(result)
    return result

        
all_crawls = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")
def download_in_parallel(crawl_list,output_base):
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download_crawl, c,f"{output_base}/{c}.jsonl") for c in crawl_list]
        # Collect results from futures
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results


if __name__ == "__main__":
    args = parser.parse_args()
    all_crawl_chunks=sort_and_chunk(all_crawls)
    crawls=all_crawl_chunks[args.split]
    if args.test:
        crawls=crawls[:1]
    if args.manual:
        crawls = ['CC-MAIN-2017-34','CC-MAIN-2022-33','CC-MAIN-2022-49']
    res = download_in_parallel(crawls,args.output_base)
