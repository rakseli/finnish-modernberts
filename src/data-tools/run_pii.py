import json
import os
import glob
from itertools import islice
from pii_manager import PiiEnum
from pii_manager.api import PiiManager
from pii_manager.lang import COUNTRY_ANY
from argparse import ArgumentParser
from datasets import load_dataset,disable_caching
from file_helpers import format_duration,DateTimeEncoder,naive_data_collator
from timer import Timer
from torch.utils.data import DataLoader

disable_caching()

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--input-file')
    ap.add_argument('--lang')
    ap.add_argument('--safe-mode',action='store_true',help='load jsons with json library instead of torch dataloader')
    ap.add_argument('--test',action='store_true')
    return ap

def yield_batches_from_jsonl(file_path, batch_size, pii_remover):
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                data = remove_pii(data,pii_remover)
                batch.append(data)
                # If the batch reaches the desired size, yield it
                if len(batch) == batch_size:
                    yield batch
                    batch = []  # Reset the batch
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} | Line: {line}")
        # Yield the last batch if it's not empty
        if batch:
            yield batch

def remove_pii(example,pii_remover):
    example['text']=pii_remover(example['text'])
    return example

def compare_docs(d_original,d_redacted):
    to_find = ["example@email.com", "+0 0000000", "0.0.0.0"]
    text_og = d_original['text']
    text_redacted = d_redacted['text']
    matches = [text_redacted.count(f) for f in to_find]
    found = False
    for m in matches:
        if m>0:
            found = True
            print(f"Original text: \n {text_og}",flush=True)
            print(f"Redacted text: \n {text_redacted}",flush=True)
            break
    return found

def process_jsonlprint_file(args):
    print(f"Starting processing {args.input_file}",flush=True)
    country=COUNTRY_ANY
    tasklist = (PiiEnum.IP_ADDRESS, PiiEnum.EMAIL_ADDRESS, PiiEnum.PHONE_NUMBER)
    pii_remover = PiiManager(args.lang, country, tasks=tasklist, mode="convert")
    base_directory = os.path.dirname(args.input_file)
    basename_without_suffix = os.path.splitext(os.path.basename(args.input_file))[0]
    t = Timer()
    with t(args.input_file):
        if not args.safe_mode:
            ds = load_dataset("json",data_files=args.input_file, split="train",streaming=True)
            if args.test:
                original_ds = ds.take(10000)
            ds = ds.map(remove_pii,fn_kwargs={"pii_remover":pii_remover})
            dataloader= DataLoader(ds, batch_size=10000,num_workers=1,collate_fn=naive_data_collator)
        else:
            dataloader = yield_batches_from_jsonl(args.input_file,10000,pii_remover)
        with open(f"{base_directory}/{basename_without_suffix}-pii-removed.jsonl", 'w') as output_file:
            for batch in dataloader:
                for json_object in batch:
                    json_line = json.dumps(json_object,ensure_ascii=False)
                    output_file.write(json_line + '\n')
                if args.test:
                    found_count = 0
                    for d_o, d_r in zip(original_ds,batch):
                        found = compare_docs(d_o,d_r)
                        if found:
                            found_count+=1
                        if found_count>100:
                            break
                    break
    result = f"Run time for file {args.input_file}: {format_duration(int(t.elapsed_times.get(args.input_file, 0)))}"
    return result

if __name__ == "__main__":
    args = argparser().parse_args()
    res = process_jsonlprint_file(args)
    print(res)
