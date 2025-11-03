
import json
import argparse
import os
import subprocess
from datasets import load_dataset,DownloadMode
from torch.utils.data import DataLoader
from file_helpers import naive_data_collator, DateTimeEncoder,format_duration
from timer import Timer
from datasets import disable_caching
from huggingface_hub import login

disable_caching()
hf_token_path=os.getenv('HF_TOKEN_PATH')
with open(hf_token_path,'r') as f: 
   hf_token = f.read()

login(token=hf_token,new_session=False)
parser = argparse.ArgumentParser()
parser.add_argument("--output-base", type=str,default='../culturax', help="path to dir of files")
parser.add_argument("--test",action='store_true')
parser.add_argument("--lang",type=str,default='fi',help="fi, la or sv")

hf_cache = os.getenv('HF_HOME')
print(f"HF cache would be saved here <{hf_cache}>if not disabled")


def download_language(lang,output_file):
    print(f"Starting to download lang {lang}",flush=True)
    result = {}
    result['lang']=lang
    t = Timer()
    with t("download"):
        try:
            culturax = load_dataset("uonlp/CulturaX",lang,split='train',token=hf_token,streaming=True,download_mode=DownloadMode.FORCE_REDOWNLOAD)
            dataloader= DataLoader(culturax, batch_size=10000,num_workers=4,collate_fn=naive_data_collator)
                
            with open(output_file, 'w') as jsonl_file:
                for batch in dataloader:
                    for json_object in batch:
                        json_line = json.dumps(json_object,cls=DateTimeEncoder,ensure_ascii=False)
                        jsonl_file.write(json_line + '\n')
                    if args.test:
                        break

            result['success']=True

        except Exception as e:
            result['success']=False
            result['exception']=e
            pass
        if result['success']==True:
            c_res = subprocess.run(['/appl/lumi/SW/LUMI-24.03/C/EB/zstd/1.5.5-cpeGNU-24.03/bin/zstd','-3', '-T4','--rm', output_file],stdout=subprocess.PIPE)
            print(c_res)
            if c_res.returncode == 0:
               result['compression_success']=True
            else:
               result['compression_success']=False
    
    result['run_time']=format_duration(int(t.elapsed_times.get('download', 0)))
    print(result)
    return result


if __name__ == "__main__":
    args = parser.parse_args()
    res = download_language(args.lang,f"{args.output_base}/culturax-{args.lang}.jsonl")
