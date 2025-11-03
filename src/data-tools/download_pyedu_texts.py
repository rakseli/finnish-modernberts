import boto3
import gzip
import json
import argparse
import concurrent.futures

from datasets import load_dataset,DownloadMode
from torch.utils.data import DataLoader
from botocore.exceptions import ClientError
from datasets import disable_caching
from file_helpers import naive_data_collator, DateTimeEncoder,format_duration
from timer import Timer


disable_caching()

parser = argparse.ArgumentParser()
parser.add_argument("--output-base", type=str,default='/scratch/<project_number>/<user_name>/finnish-modernberts/data/code', help="path to dir of files")
parser.add_argument("--test",action='store_true')


s3 = boto3.client('s3')
bucket_name = "softwareheritage"

def download_contents(blob_id):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True,"blob_id":blob_id}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            return {"text": "", "download_success": False,"blob_id":blob_id}
        else:
            raise


def download_data(output_file,base=True):
    print(f"Starting to download",flush=True)
    dl_result = {}
    t = Timer()
    with t("download"):
        ds = load_dataset('json', data_files='/scratch/<project_number>/<user_name>/finnish-modernberts/data/code/smollm-python-edu-blobs.jsonl', split='train')
        ds = ds['blob_id']
        ds_len = len(ds)
        dataloader= DataLoader(ds, batch_size=10000,num_workers=2,collate_fn=naive_data_collator)
        with open(output_file, 'w') as jsonl_file:
            print("File opened",flush=True)
            for i,batch in enumerate(dataloader):
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    future_to_blob_id = {executor.submit(download_contents, blob_id): blob_id for blob_id in batch}
                    # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_blob_id):
                    result = future.result()
                    results.append(result)
                if i % 10 == 0 and i != 0:
                    print(f"Downloaded {(i+1)*10000}/{ds_len} ({(((i+1)*10000)/ds_len)*100:.2f}/100 %) samples",flush=True)
                for json_object in results:
                    json_line = json.dumps(json_object,cls=DateTimeEncoder,ensure_ascii=False)
                    jsonl_file.write(json_line + '\n')
                if args.test:
                    break
    
    
    dl_result['run_time']=format_duration(int(t.elapsed_times.get('download', 0)))
    print(dl_result)
    return result


if __name__ == "__main__":
    args = parser.parse_args()
    res = download_data(f"{args.output_base}/smollm-python-edu-texts.jsonl")
