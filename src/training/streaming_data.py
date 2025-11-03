import argparse
import json
import glob
import math
import numpy as np
import torch
import os
import logging
import sys
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator,PreTrainedTokenizerFast
from dataclasses import dataclass,field
from collections import deque
from itertools import cycle,islice
from torch.utils.data import IterableDataset
from pyrsmi import rocml

from distributed_utils import get_rank, get_local_rank, get_world_size
from utils import seed_everything
# Parse arguments
parser = argparse.ArgumentParser(description=" Straming dataset")
parser.add_argument("--tokenizer", type=str, default="answerdotai/ModernBERT-base", help="Model id or path to tokenizer")
parser.add_argument("--input-root", type=str, help="input to be tokenized")
parser.add_argument("--output-root", type=str,default="../data/final-data/sampled-data/tokenized", help="root dir for output")
parser.add_argument("--test", type=bool,default=False, help="root dir for output")
parser.add_argument("--long-sequence-probability",type=float,default=None)
parser.add_argument("--higly-quality-only",action="store_true")

'''
TODO
FIX INTERLEAVE STREAMS AS IT DO NOT YIELD DATA FROM ALL DOCUMENTS
'''
@dataclass
class CustomDataCollatorForMLM(DataCollatorForLanguageModeling):
    mlm: bool = True
    mlm_probability: float = 0.30
    def torch_call(self, examples):
        batch = default_data_collator(examples)
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        return batch
    
@dataclass
class StreamingMLMDatasetConfig:
    input_files: list
    tokenizer: PreTrainedTokenizerFast
    max_sequence_lenght: int
    collate_fn: CustomDataCollatorForMLM
    seed: int = field(default=42)
    epoch: int = field(default=0)
    high_quality_only: bool = field(default=False)
    english_only: bool = field(default=False)
    batch_size: int = field(default=16)
    high_quality_sources: set = field(default_factory=lambda:set(["britishlibrary",
                                               "europarlen",
                                               "europarlfi",
                                               "europarlsv",
                                               "finewebedufortified",
                                               "fstc",
                                               "nlfclfi",
                                               "nlfcsv",
                                               "pes2o",
                                               "projectlonnrot",
                                               "pubmedabstracts",
                                               "pubmedcentral",
                                               "saamiweb",
                                               "smollmpythonedu",
                                               "wikien",
                                               "wikifi",
                                               "wikisv",
                                               "ylenewsfi",
                                               "ylenewssv",
                                               "hpltv2-edu-swe",
                                               "hpltv2-edu-fi"
                                                ]))

    english_only_sources: set = field(default_factory=lambda:set(["britishlibrary",
                                               "europarlen",
                                               "finewebedufortified",
                                               "pes2o",
                                               "pubmedabstracts",
                                               "pubmedcentral",
                                               "wikien"]
    ))

    
class StreamingMLMDataset(IterableDataset):
    def __init__(self, config):
        self.input_files = sorted(config.input_files)
        self.epoch = config.epoch
        self.tokenizer = config.tokenizer
        self.max_sequence_lenght = config.max_sequence_lenght
        self.current_max_length = self.max_sequence_lenght
        self.high_quality_only = config.high_quality_only
        self.high_quality_sources = config.high_quality_sources
        self.english_only = config.english_only
        self.english_only_sources = config.english_only_sources
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.total_tokens = 0
        self.buffer_size= 64
        self.batch_size = config.batch_size
        self.mlm_probability = 0.30
        self.collate_fn = config.collate_fn
        self.seed = config.seed
        seed_everything(config.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        
    def _read_file(self,file_path):
        with open(file_path, "r") as input_reader:
            for line in input_reader:
                try:
                    yield json.loads(line)
                except Exception as e:
                    logging.info(f"Error {e} in file {file_path} while trying to read json")

                
    def _chunk_document(self,document,file_path):
        chunks = []
        base_tokens = self.tokenizer(document['text'],truncation=False,add_special_tokens=False,return_length=True,return_tensors='np')
        if base_tokens.length[0] > 0:
            if base_tokens.length[0] > self.current_max_length-2:
                splitted_tokens = self._split_to_max_len(base_tokens)
                for s in splitted_tokens:
                    self.total_tokens +=len(s['input_ids'])
                    processed_sample = self._pad_and_create_attention_mask(s)
                    processed_sample['source_file']=file_path
                    chunks.append(processed_sample)
            else:
                self.total_tokens += base_tokens.length[0]
                b_t = {"input_ids":base_tokens['input_ids'][0]}
                processed_sample = self._pad_and_create_attention_mask(b_t)
                processed_sample['source_file']=file_path
                chunks.append(processed_sample)
        logging.debug(f"Document produced {len(chunks)} chucks")
        return chunks
    
    def _stream_tokens_from_file(self,file_path):
        for doc in self._read_file(file_path):
            yield self.tokenizer(doc['text'],truncation=False,add_special_tokens=False,return_length=True,return_tensors='np')
            
    def _stream_chunks_from_file(self,file_path):
        """Stream and chunk documents from a gzipped JSONL file with padding."""
        for i,doc in enumerate(self._read_file(file_path),start=1):
            logging.debug(f"Reading doc {i} from file {file_path}")
            yield_doc = True
            if self.english_only:
                yield_doc = False
                if doc['dataset_source'] in self.english_only_sources:
                    yield_doc = True
                else:
                    continue
            if self.high_quality_only:
                yield_doc = False
                if doc['dataset_source'] in self.high_quality_sources:
                    yield_doc = True
                else:
                    continue
            if yield_doc:
                chunks = self._chunk_document(doc,file_path)
                if chunks:
                    for j,c in enumerate(chunks,start=1):
                        logging.debug(f"yielding chunk {j}")
                        yield c
            else:
                logging.debug(f"No chunks in doc {doc}")
                continue

    def _interleave_streams(self, streams):
        """
        Interleave samples from multiple generators using a fixed-size buffer.
        Stops when all streams are exhausted.
        """
        buffer = deque(maxlen=self.buffer_size)
        active_streams = list(enumerate(streams))
        stream_cycle = cycle(active_streams)

        while active_streams or buffer:
            # Yield from buffer if it has data
            if buffer:
                yield buffer.popleft()

            # Try to refill the buffer
            if len(buffer) < self.buffer_size and active_streams:
                for _ in range(len(active_streams)):
                    i, stream = next(stream_cycle)
                    try:
                        buffer.append(next(stream))
                        break  # refill succeeded, try to yield next
                    except StopIteration:
                        logging.debug(f"Stream {i} exhausted")
                        # Remove this stream from active list
                        active_streams = [(j, s) for j, s in active_streams if j != i]
                        stream_cycle = cycle(active_streams)  # Rebuild the cycle
                        break  # Avoid using stale cycle
    
    def _split_to_max_len(self,b_tokens):
        """Split to max sequence lenght-2 to take account [CLS] and [SEP]. If sequence lenght is 1% larger than max, just truncate it. 
        Args:
            b_tokens (dict): tokenized sample

        Returns:
            list: list of splitted tokens 
        """
        splitted_sample = []
        chunk_size = self.current_max_length - 2
        if ((self.current_max_length - 2)/ b_tokens.length[0])>=0.99:
            chunk = {}
            split_tokens = b_tokens['input_ids'][0][:chunk_size]
            chunk['input_ids'] = split_tokens
            splitted_sample.append(chunk)
        else:
            for i in range(0, int(b_tokens.length[0]), chunk_size):
                chunk = {}
                split_tokens = b_tokens['input_ids'][0][i:i + chunk_size]
                chunk['input_ids'] = split_tokens
                splitted_sample.append(chunk)

        return splitted_sample
    
    def _pad_and_create_attention_mask(self,tokens,target_length=None):
        """Pad to max sequence lenght and add [CLS] and [SEP] tokens

        Args:
            tokens (dict): _description_
        """
        if target_length is None:
            target_length = self.current_max_length
        else:
            special_token_ids = set(self.tokenizer.all_special_ids)
            filtered_ids = [id for id in tokens['input_ids'] if id not in special_token_ids]
            tokens['input_ids'] = filtered_ids
        
        sequence_length = len(tokens['input_ids'])
        padding_length = target_length - sequence_length - 2
        tokens["input_ids"] = [self.cls_token_id] + list(tokens["input_ids"]) + [self.sep_token_id] + [self.pad_token_id] * padding_length
        tokens["attention_mask"] = [1] * (sequence_length + 2)  + [0] * padding_length
        assert len(tokens['attention_mask'])==target_length
        assert len(tokens['input_ids'])==target_length
        return tokens
    
    def _batchify(self,iterable):
        """Generate batches of size batch_size from iterable."""
        while batch := list(islice(iterable, self.batch_size)):
            seq_lens =[len(sample["input_ids"]) for sample in batch]
            uniq_seq_lens = set(seq_lens)
            if len(uniq_seq_lens) == 1:
                yield self.collate_fn(batch)
            else:
                max_len_in_batch = max(seq_lens)
                batch = [self._pad_and_create_attention_mask(sample, max_len_in_batch) for sample in batch]
                yield self.collate_fn(batch)
    
    def __iter__(self):
        streams = [self._stream_chunks_from_file(fp) for fp in self.input_files]
        mixed_streams = self._interleave_streams(streams)
        batched_stream = self._batchify(mixed_streams)
        return batched_stream

    def _iter_single(self):
        streams = [self._stream_chunks_from_file(fp) for fp in self.input_files]
        mixed_streams = self._interleave_streams(streams)
        return mixed_streams
    
    def _iter_chunks(self):
        streams = [self._stream_chunks_from_file(fp) for fp in self.input_files]
        for s in streams:
            for c in s:
                yield c
                
    def set_max_length(self, new_max_length):
        self.current_max_length = new_max_length
    
    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
    
    def set_high_quality(self, high_quality):
        self.high_quality_only = high_quality
    
    def shuffle_files(self,seed):
        rng = np.random.default_rng(seed)
        rng.shuffle(self.input_files)
        
    

#  IterableDataset for Sharded Data
class DistributedStreamingMLMDataset(StreamingMLMDataset):
    
    def __init__(self,config,world_size=None,rank=None,local_rank=None):
        super().__init__(config) 
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]"
            )

        self.world_size = world_size
        self.rank = rank
        self.all_files = self.input_files
        self.get_device_shards()

            
    def get_device_shards(self):
        num_files_per_gpu = math.ceil(len(self.all_files) / self.world_size)
        total_size = num_files_per_gpu * self.world_size
        even_shards = self.all_files[:total_size]
        rank_specific_shards = even_shards[self.rank: total_size : self.world_size]
        self.input_files = rank_specific_shards
        

def test_dynamic_changes(dataset):
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []

    for i,b_d in enumerate(dataset):
        data_1.append(b_d)
        if i == 0:
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 5:
            print("Testing dynamic changes, changing batch size larger...")
            dataset.set_batch_size(32)
        if i == 6:
            print(f"bs should be 32")
            print(f"Shape: {b_d['input_ids'].shape}")
        if i >= 7 and i<15:
            print(f"bs should be 32")
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 20:
            print(f"bs should be 32")
            print(f"Shape: {b_d['input_ids'].shape}")
            break
    for i,b_d in enumerate(dataset):
        data_2.append(b_d)
        if i == 0:
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 5:
            print("Testing dynamic changes, changing max seq leng...")
            dataset.set_max_length(128000)
        if i == 6:
            print(f"sequence lenght should be 128000")
            print(f"Shape: {b_d['input_ids'].shape}")
        if i >= 7 and i<15:
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 20:
            print(f"sequence lenght should be 128000")
            print(f"Shape: {b_d['input_ids'].shape}")
            break
    for i,b_d in enumerate(dataset):
        data_3.append(b_d)
        if i == 0:
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 5:
            print("Testing dynamic changes, changing max seq leng and batch size larger...")
            dataset.set_max_length(1024)
            dataset.set_batch_size(32)
        if i == 6:
            print(f"sequence lenght should be 128000, bs 32")
            print(f"Shape: {b_d['input_ids'].shape}")
        if i >= 7 and i<15:
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 20:
            print("Testing dynamic changes, changing max seq leng larger and batch size smaller...")
            dataset.set_max_length(128000)
            dataset.set_batch_size(3)
            break

    for i,b_d in enumerate(dataset):
        data_4.append(b_d)
        if i == 0:
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 5:
            print("Testing dynamic changes, changing max seq leng and batch size smaller...")
            dataset.set_max_length(128000)
            dataset.set_batch_size(8)
        if i == 6:
            print(f"sequence lenght should be 128000, bs 8")
            print(f"Shape: {b_d['input_ids'].shape}")
        if i >= 7 and i<15:
            print(f"Shape: {b_d['input_ids'].shape}")
        if i == 20:
            print(f"sequence lenght should be 128000, bs 8")
            print(f"Shape: {b_d['input_ids'].shape}")
            break
    return [data_1,data_2,data_3,data_4]



def test_determenistic(config):
    if get_world_size()>1:
        dataset_1 = DistributedStreamingMLMDataset(config=config)
        dataset_2 = DistributedStreamingMLMDataset(config=config)
        print(f"Using files: {dataset_1.input_files}")
    else:
        dataset_1 = StreamingMLMDataset(config=config)
        dataset_2 = StreamingMLMDataset(config=config)

    batches_1 = test_dynamic_changes(dataset_1)
    batches_2 = test_dynamic_changes(dataset_2)
    for test_case_1, test_case_2 in zip(batches_1,batches_2):
        for b_1,b_2 in zip(test_case_1, test_case_2):
            for s_1,s_2 in zip(b_1['input_ids'],b_2['input_ids']):
                assert torch.equal(s_1,s_2)
    print("All tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,format="%(asctime)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stderr)])   

    args = parser.parse_args()
    print("Loading tokenizer",flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("Done",flush=True)
    files = glob.glob(f"{args.input_root}/*.jsonl")
    config = StreamingMLMDatasetConfig(input_files=files,max_sequence_lenght=1024,tokenizer=tokenizer,collate_fn=CustomDataCollatorForMLM(tokenizer),high_quality_only=False,batch_size=10)
    try:
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError as e:
        print(f"{e}, falling back to get_world_size()")
        world_size = get_world_size()
        

    if world_size>1:
        rocml.smi_initialize()
        print(f"World size: {world_size}")
        node_id = os.environ["SLURM_NODEID"]
        rank = int(os.environ["RANK"])
        assert torch.cuda.is_available()
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        print(f"Device count on node: {torch.cuda.device_count()}")
        assert gpus_per_node == torch.cuda.device_count()
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        if rank == 0:
            print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)
        dataset = DistributedStreamingMLMDataset(config=config)
        # Set correct GPU device
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Node {node_id} | Global Rank {rank} | Local Rank {local_rank}, Device id {rocml.smi_get_device_uuid(rank)}, Device {device}| Files {dataset.input_files}")
    else:
        print("Creating dataset iterator",flush=True)
        dataset = StreamingMLMDataset(config=config)
        print("Done",flush=True)

    if args.test and world_size == 1:
        lengths = []
        texts = []
        logging.info("Starting to iterate over the data")
        logging.info(f"In total data contains 1000 documents (10 shards x 100 docs) so at least {1000/dataset.batch_size} batches should be outputted")
        total_tokens=0
        for i,tokens in enumerate(dataset,start=1):
            logging.info(f"Batch size: {len(tokens['input_ids'])}")
            input_ids = tokens["input_ids"].flatten()
            logging.info(f"Batch contain following text: {tokenizer.decode(input_ids)}")
            num_tokens_in_batch = (input_ids != dataset.pad_token_id).sum()
            total_tokens += num_tokens_in_batch
            logging.info(f"Processed {i} batches, current token count: {total_tokens}")
            logging.info(f"Total seen chunks {i*len(tokens['input_ids'])}")


        logging.info(f"All batches from the files were processed")
    if world_size>1:      
        torch.distributed.destroy_process_group()
   