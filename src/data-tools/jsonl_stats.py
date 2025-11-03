#!/usr/bin/env python3

# Provide basic corpus statistics for JSONL format with one document
# per line and document text as 'text'.

import sys
import json
import glob
import os
import concurrent.futures
import fcntl
from argparse import ArgumentParser
from collections import defaultdict
from timer import Timer
from file_helpers import format_duration
from basic_tokenizer import basic_tokenize


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--output_file',help="output file path for results")
    ap.add_argument('--input_root',help="root dir for jsonl files")
    ap.add_argument('--per_source',action="store_true",help="wheter to count stats per source")
    return ap

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

def sifmt(i):
    affix = iter(['', 'K', 'M', 'G', 'T', 'P', 'E'])
    while i > 1000:
        i /= 1000
        next(affix)
    return f'{i:.1f}{next(affix)}'


def write_output_file(d,output_file_path):
    # Exclusive lock (for writing)
    json_line = json.dumps(d,ensure_ascii=False)
    with open(output_file_path, 'a') as out_file:
        # Blocks until lock is acquired
        fcntl.flock(out_file, fcntl.LOCK_EX)  
        out_file.write(json_line + '\n')

def get_existing_stats(args):
    existing_stats = set()
    if os.path.exists(args.output_file):
        with open(args.output_file,"r") as f:
            for l in f:
                try:
                    e_f = json.loads(l).get("file_path")
                    existing_stats.add(e_f)
                except json.decoder.JSONDecodeError as e:
                    print(f"{e} in line {l}")
    if args.input_root in existing_stats:
        print(f"{args.input_root} root already in output file, exiting...")
        return None
         
    return existing_stats


def get_jsonl_stats(fn,args):
    docs, words, chars = 0, 0, 0
    timer = Timer()
    print(f"Starting {fn}",flush=True)
    if not args.per_source:
        with timer(fn):
            with open(fn) as f:
                for line in f:
                    data = json.loads(line)
                    text = data['text']
                    docs += 1
                    words += len(basic_tokenize(text))
                    chars += len(text)
    else:
        source_dict = defaultdict(lambda: defaultdict(int))
        with timer(fn):
            with open(fn) as f:
                for line in f:
                    data = json.loads(line)
                    text = data['text']
                    source = data['dataset_source']
                    source_dict[source]['docs'] += 1
                    source_dict[source]['words'] += len(basic_tokenize(text))
                    source_dict[source]['chars'] += len(text)
                          
    runtime = format_duration(int(timer.elapsed_times.get(fn, 0)))
    
    print(f"Done {fn}",flush=True)
    if not args.per_source:
        result_dict={'file_path':fn,
                 'docs':sifmt(docs),
                 'words':sifmt(words),
                 'chars':sifmt(chars),
                 'docs_raw':docs,
                 'words_raw':words,
                 'chars_raw':chars,
                 'run_time':runtime
                 }
    else:
        #transform to regular dict for writing
        source_dict = defaultdict_to_dict(source_dict)
        result_dict={'file_path':fn,
                     'dataset_source':source_dict,
                     'run_time':runtime
                    }
    
    #write result to file
    write_output_file(result_dict,args.output_file)
    return result_dict


def main():
    args = argparser().parse_args()
    if os.path.isfile(args.input_root):
        jsonl_files = [args.input_root]
    else:
        jsonl_files =  glob.glob(f"{args.input_root}/*.jsonl", recursive=True)

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK",1))
    
    #filter files where stats have been already counted
    existing_stats = get_existing_stats(args)
    if existing_stats is None:
        sys.exit(0)
        
    filterd_jsonl_files = [j for j in jsonl_files if j not in existing_stats]
    
    n_files = len(filterd_jsonl_files)
    results = []
    
    if n_cpus > n_files:
        n_cpus = n_files
    if len(filterd_jsonl_files)>0:
        print(f"Starting count jsonl stats from {len(filterd_jsonl_files)} files with {n_cpus} processors",flush=True)
        if n_cpus > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
                future_to_stats = {executor.submit(get_jsonl_stats, fn,args): fn for fn in filterd_jsonl_files}

            for future in concurrent.futures.as_completed(future_to_stats):
                result = future.result()
                results.append(result)

            executor.shutdown(wait=True)
        else:
            for fn in filterd_jsonl_files:
                get_jsonl_stats(fn,args)
    else:
        print("All individual stats have been counted")


    total_docs, total_words, total_chars = 0, 0, 0
    
    if len(results) != len(jsonl_files):
        with open(args.output_file,"r") as f:
            for l in f:
                try:
                    d = json.loads(l)
                    results.append(d)
                except json.decoder.JSONDecodeError as e:
                    print(f"{e} in line {l}")
        assert len(results) == len(jsonl_files)
    if not args.per_source:
        for d in results:
            total_docs += d['docs_raw']
            total_words += d['words_raw']
            total_chars += d['chars_raw']
        total = {'file_path':f'{args.input_root}','total_docs':sifmt(total_docs),'total_words':sifmt(total_words),'total_chars':sifmt(total_chars)}
        write_output_file(total,args.output_file)
    else:
        source_dict = defaultdict(lambda: defaultdict(int))
        for d in results:
            for k,v in d['dataset_source'].items():
                source_dict[k]['docs']+= v['docs']
                source_dict[k]['words'] += v['words']
                source_dict[k]['chars'] += v['chars']
        source_dict = defaultdict_to_dict(source_dict)
        # Method 1: Iterate and apply function to each inner value
        for outer_key, inner_dict in source_dict.items():
            for inner_key in ['docs', 'words', 'chars']:
                if inner_key in inner_dict:
                    inner_dict[f"rounded_{inner_key}"] = sifmt(inner_dict[inner_key])
                    
        source_dict['file_path']=args.input_root
        write_output_file(source_dict,args.output_file)

if __name__ == '__main__':
    main()