import argparse
import os
import glob
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC
from pathlib import Path

def naive_data_collator(batch):
    """Does nothing, only for dataloader to batch samples 
    and not to convert them to tensors
    
    batch (list): list of dicts 
    Returns:
        list: list of dicts
    """    
    return batch


def calculate_total_size_gb(file_paths):
    total_size_bytes = 0
    for path in file_paths:
        try:
            p = Path(path)
            if p.is_file():
                total_size_bytes += p.stat().st_size
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return total_size_bytes / (1024 ** 3)



def get_training_corpus(dataset):
    for batch in dataset:
        yield batch['text']

whitespace_token = " "
whitespace_tokens = [whitespace_token * i for i in range(2, 25)]
special_tokens = ["[MASK]","[CLS]","[PAD]","[UNK]","[SEP]"]
special_tokens.extend(whitespace_tokens)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default='base', help="model size, tiny base or large")
    parser.add_argument("--input_path", type=str, default='base', help="path to input data")
    parser.add_argument("--output_path", type=str,default='/scratch/<project_number>/<user_name>/finnish-modernberts/results/tokenizers', help="output path")
    parser.add_argument("--make_vocab_divisible_by",type=int, default=64, help="make vocab divisible by. defaults to 128 for accelerated NVIDIA matmul")
    parser.add_argument("--replication",action="store_true")
    args = parser.parse_args()
    tokenizer_name = f"modernbert_{args.model_size}_bpe_hf"
    if not os.path.exists(f"{args.output_path}/{tokenizer_name}"):
        os.mkdir(f"{args.output_path}/{tokenizer_name}")        
    total_n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK"))
    if args.model_size == 'base':
        vocab_size = 42200
    elif args.model_size == 'tiny':
        vocab_size = 27224
    elif args.model_size == 'large':
        vocab_size = 55571
    elif args.model_size == 'experimental':
        vocab_size = 128000
    else:
        raise ValueError(f"Model size should be base, tiny, large or experimental. {args.model_size} given")
    
    if args.make_vocab_divisible_by:
        original_vocab_size = vocab_size
        not_divisible = True
        while not_divisible:
            if vocab_size % args.make_vocab_divisible_by != 0:
                vocab_size+=1
            else:
                not_divisible = False
                
        print(f"Original vocab size: {original_vocab_size}",flush=True)
        print(f"New vocab size: {vocab_size}",flush=True)

    if os.path.isdir(args.input_path):
        data_files = glob.glob(f"{args.input_path}/*.jsonl")
        if len(data_files)>10:
            data_files = data_files[:30]
            
    else:
        data_files = [args.input_path]
    
    training_dataset_size = calculate_total_size_gb(data_files)
    print(f"Using {training_dataset_size}GiB of data for training")
        
    dataset = load_dataset("json", data_files=data_files,split='train',streaming=True)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=naive_data_collator,shuffle=False)

    args = parser.parse_args()
    if not args.replication:
        tokenizer = Tokenizer(BPE(byte_fallback=False))
        tokenizer.normalizer = NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(fuse_unk=False,add_prefix_space=False)
     
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        print("Starting training tokenizer",flush=True)
        tokenizer.train_from_iterator(get_training_corpus(dataset),trainer=trainer)
        print("Training done",flush=True)
        tokenizer.post_processor = processors.TemplateProcessing(
                                      single="[CLS] $0 [SEP]",
                                      pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                      special_tokens=[("[CLS]", 1), ("[SEP]", 4)])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save(f"{args.output_path}/{tokenizer_name}/tokenizer.json")

    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        print("Starting training tokenizer",flush=True)
        tokenizer.train_new_from_iterator(get_training_corpus(dataset),vocab_size=vocab_size,new_special_tokens=special_tokens)
        tokenizer_name = tokenizer_name + "_pretrained"
        print("Training done",flush=True)
        tokenizer.save_pretrained(f"{args.output_path}/{tokenizer_name}")
        print(f"Tokenizer saved to: {args.output_path}/{tokenizer_name}")