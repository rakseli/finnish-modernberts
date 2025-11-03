import json
import random
import argparse
import zstandard as zstd
import io
import pickle
import os
import concurrent.futures

parser = argparse.ArgumentParser()

parser.add_argument("--output-file",type=str,help="output filename")
parser.add_argument("--index-file",type=str,help="output filename")
parser.add_argument("--input-file",type=str,help="input filename")

# Define sampling ratios
sampling_ratios = {
        "<1000": 0.01,
        "1000-10000": 0.4,
        "10000-16000": 0.4,
        ">16000": 0.15
    }

def write_buffer(buffer):
    with open(args.output_file, "a") as out_f:
        for l in buffer:
            json_line = json.dumps(l,ensure_ascii=False)
            out_f.write(json_line + "\n")

if __name__ == "__main__":
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    args = parser.parse_args()
    bins = {
        "<1000": [],
        "1000-10000": [],
        "10000-16000": [],
        ">16000": []
    }
    print(f"Starting to count bins",flush=True)
    row_number = 0
    if os.path.exists(args.index_file):
        print("Index file exists, loading it from file...",flush=True)
        with open(args.index_file, "rb") as f:
            bins,row_number = pickle.load(f)
    else:
        print("Index file does not exist, creating one...",flush=True)
        if "jsonl.zst" in args.input_file:
            mode = "rb"
        with open(args.input_file, mode) as f:
            if mode == "rb":
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(f)
                text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8',errors='ignore')
            else:
                text_stream = f            
            for line in text_stream:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    continue
                
                tokens = len(data["text"].split())
                # Categorize row numbers instead of storing full samples
                if tokens < 1000:
                    bins["<1000"].append(row_number)
                elif tokens < 10000:
                    bins["1000-10000"].append(row_number)
                elif tokens <= 16000:
                    bins["10000-16000"].append(row_number)
                else:
                    bins[">16000"].append(row_number)
                row_number += 1

        with open(args.index_file, "wb") as i_f:
            pickle.dump((bins,row_number), i_f)
            
    total_samples = row_number
    print(f"Total {total_samples} in the data",flush=True)
    selected_indices = []
    sample_sizes = {}
    for bin_name, ratio in sampling_ratios.items():
        bin_indices = bins[bin_name]
        bin_size = len(bin_indices)
        sample_size = int(ratio * total_samples)
        print(f"Bin ({bin_name}) size {bin_size}/{total_samples} = {bin_size/total_samples * 100:4f}%")
        if bin_size < sample_size:
            print(f"Warning: Not enough samples in {bin_name}, taking all available ({bin_size})")
            selected_indices.extend(bin_indices)
            sample_sizes[bin_name] = bin_size
        else:
            selected_indices.extend(random.sample(bin_indices, sample_size))
            sample_sizes[bin_name] = sample_size

            
    selected_indices = set(selected_indices)
    print(f"Selected {len(selected_indices)} samples",flush=True)
    print(f"Final sample distribution will be:",flush=True)
    for bin_name,s_size in sample_sizes.items():
        print(f"{bin_name}: {s_size/len(selected_indices)*100:4f} %",flush=True)

    if "jsonl.zst" in args.input_file:
        mode = "rb"
    else:
        mode = "r"
    
    with open(args.input_file, mode) as f:
        if mode == "rb":
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8',errors='ignore')
        else:
            text_stream = f   

        buffer = []
        row_number = 0

        for line in text_stream:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                continue
            
            if row_number in selected_indices:
                buffer.append(line)  # Store line in buffer instead of writing immediately

            row_number += 1

            if row_number % 10000 == 0:
                print(f"Gone through {row_number} samples...", flush=True)
                if len(buffer)>=10000:
                    executor.submit(write_buffer, buffer)
                    buffer = []  

        if buffer:
            write_buffer(buffer)
            
    executor.shutdown(wait=True)
    print(f"Final sampled dataset written to {args.output_file}", flush=True)

