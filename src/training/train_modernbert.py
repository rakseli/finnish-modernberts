import time
_TRAIN_START_TIME = time.time()
import torch
import argparse
import os
import glob
import yaml
import contextlib
import logging
import sys
import numpy as np
from datetime import timedelta
from statistics import mean
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from socket import gethostname
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler, AdamW
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
from tqdm import tqdm
from distributed_utils import is_main_process, get_rank, get_world_size, get_local_rank
try:
    from stableadamw_optimizer import StableAdamW
except ModuleNotFoundError as e:
    logging.debug("Env do not contain optimi, install it to use StableAdamW")

from sheduler import cosine_schedule_with_warmup,TrapezoidalLRSheduler,one_minus_sqrt_schedule
from itertools import count
from streaming_data import StreamingMLMDatasetConfig, DistributedStreamingMLMDataset, CustomDataCollatorForMLM
from utils import seed_everything, pretty_json
from hardware_utils import flush, get_gpu_utilization

# from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
# Copyright 2019 Ross Wightman, Apache-2.0 License
def param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]

def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--checkpoint_path", default=None, help="Path to checkpoint")
    parser.add_argument("--annealing", action="store_true", help="whether to perform annealing")


    args = parser.parse_args()
    config = load_config(args.config)
    config["annealing"]=args.annealing
    
    return config, args.checkpoint_path


def load_checkpoint_and_update_config(checkpoint_path, config):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    keys_to_ignore=set(["per_device_constant_batch_size",
                        "exit_duration_in_mins",
                        "per_device_anneling_batch_size",
                        "n_gradient_accumulation_steps",
                        "n_gradient_accumulation_steps_annealing",
                        "run_name",
                        "logging_level"
                        "final_seq_length",
                        "per_device_eval_batch_size",
                        "save_steps"])
    
    if not config['annealing']:
        for key in checkpoint.get("config", {}):
            if key not in keys_to_ignore:
                config[key] = checkpoint["config"][key]
    else:
        config["current_max_length"]=checkpoint["config"]["current_max_length"]
        config["current_global_rope_theta"]=checkpoint["config"]["current_global_rope_theta"]
        
    config['checkpoint_path']=checkpoint_path
    return config, checkpoint


def setup_training(config):
    assert torch.cuda.is_available()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    logging.info(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.")

    seed_everything(config['seed']+ rank)
    
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size,timeout=timedelta(minutes=90))
    if rank == 0:
        logging.info(f"Group initialized? {torch.distributed.is_initialized()}")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.info(f"RCCL started on device {device}")
    logging.info(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if is_main_process():
        os.system(f"mkdir -p {config['output_dir']}")
    
    config["per_device_max_steps"] = config["max_steps"] // world_size

    if is_main_process():
        logging.info(f"Training for {config['max_steps']:,} steps with {get_world_size()} GPUs")
        logging.info(f"In total, the model will be trained on ({config['per_device_max_steps']:,}) x 'GPUs'({world_size}) = {config['per_device_max_steps'] * world_size:,} steps")
        
    global GLOBAL_TENSORBOARD_WRITER
    log_dir = f"{config['tensorboard_dir']}/{config['run_name']}/{config['run_name']}-{os.getenv('SLURM_JOBID')}"
    if is_main_process():
        os.makedirs(f"{log_dir}", exist_ok=True)
    
    GLOBAL_TENSORBOARD_WRITER = SummaryWriter(log_dir=log_dir)
    if is_main_process():
        GLOBAL_TENSORBOARD_WRITER.add_text("config",pretty_json(config),global_step=0)
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME]).to(device)
    torch.distributed.all_reduce(start_time_tensor,op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    
    if is_main_process():
        logging.info('Time to initialize distributed setup (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
    return device, local_rank


def prepare_model_tokenizer_and_optimizer(config, device, local_rank, checkpoint):
    def get_loss_function(self):
        def chunked_cross_entropy(logits, labels,vocab_size,ignore_index: int = -100,num_output_chunks=8):
            #Adapted from https://docs.pytorch.org/torchtune/0.3/_modules/torchtune/modules/loss/ce_chunked_output_loss.html#CEWithChunkedOutputLoss
            # Copyright (c) Meta Platforms, Inc. and affiliates.
            # All rights reserved.
            # This source code is licensed under the BSD-style license
            # reshape logits [(num_tokens/num_chunks)]
            labels = [target_chunk for target_chunk in labels.chunk(num_output_chunks, dim=0)]
            # reshape logits [(num_tokens/num_chunks, vocab)]
            logits = [logit_chunk for logit_chunk in logits.chunk(num_output_chunks,dim=0)]
            total_loss = 0.0
            for logits_chunk, labels_chunk in zip(logits, labels):
                labels_chunk = labels_chunk.to(logits_chunk.device)
                total_loss += torch.nn.functional.cross_entropy(logits_chunk.float(), labels_chunk, ignore_index=ignore_index, reduction="sum")
                
            return total_loss

        return chunked_cross_entropy
    if config['model_size'] == 'base' or config['model_size'] == 'tiny':
        model_id = "answerdotai/ModernBERT-base"
    else:
        model_id = "answerdotai/ModernBERT-large"
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    if is_main_process():
        logging.debug(tokenizer)
    vocab_size = len(tokenizer)
    model_config = AutoConfig.from_pretrained(model_id)
    model_config.local_rope_theta = config["local_rope_theta"]
    model_config.local_attention = config["window_size"]
    model_config.vocab_size = vocab_size
    model_config.attn_implementation = "flash_attention_2"
    model_config.max_position_embeddings=config["final_seq_length"]
    model_config.deterministic_flash_attn=True
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.eos_token_id = tokenizer.sep_token_id
    model_config.sep_token_id = tokenizer.sep_token_id
    model_config.bos_token_id = tokenizer.cls_token_id
    model_config.cls_token_id = tokenizer.cls_token_id
    
    if checkpoint is not None:
        model_config.max_position_embeddings = checkpoint["config"]["current_max_length"]
        model_config.global_rope_theta = checkpoint["config"]["current_global_rope_theta"]
    else:
        model_config.max_position_embeddings = config["seq_length"]
        config["current_max_length"] = config["seq_length"]
        model_config.global_rope_theta = config["global_rope_theta"]

    if config['model_size'] == 'tiny':
        model_config.num_hidden_layers = 6
        
    if is_main_process():
        logging.debug(f"Model config {model_config}")

    model = AutoModelForMaskedLM.from_config(model_config)
    model.__class__.loss_function = property(get_loss_function)
    if is_main_process():
        logging.debug(f"Loss func :{model.loss_function}")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    
    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Using model size: {config['model_size']}")
        logging.info(f"Model: \n {model}")
        logging.info(f"Number of parameters: {n_params}\n")
        logging.info(f"Model config: {config}\n")

    model.to(device)
    
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )
    params = param_groups_weight_decay(model, weight_decay=config["weight_decay"])
    if config['use_zero']:
        if is_main_process():
            if config["optimizer"] != 'adamw':
                logging.info(f"Because ZeRO is used AdamW optimizer is used and config optimizer ({config['optimizer']}) is overridden and WSD schedule is used")
        optimizer = ZeroRedundancyOptimizer(
            params,
            parameters_as_bucket_view=True,
            optimizer_class=AdamW,
            lr=config["learning_rate"],
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=config["weight_decay"],
            )
    else:
        if config["optimizer"] == "stableadamw":
            optimizer = StableAdamW(
                params,
                lr=config["learning_rate"],
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=config["weight_decay"],
                decouple_lr=True,
            )
        else:
            if is_main_process():
                if config["optimizer"] != "adamw":
                    logging.info(f"Defaulting to AdamW optimizer ({config['optimizer']}) is overridden")
            optimizer = AdamW(
            params,
            lr=config["learning_rate"],
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=config["weight_decay"],
            )

    num_warmup_steps = int(config["per_device_max_steps"] * config["lr_warmup_proportion"]) if int(config["per_device_max_steps"] * config["lr_warmup_proportion"]) > 0 else 1

    if config["lr_scheduler"] == "cosine":
        scheduler = cosine_schedule_with_warmup(optimizer, num_warmup_steps, config["per_device_max_steps"], 0.1)

    elif config["lr_scheduler"] == "linear":
        scheduler = lr_scheduler.ChainedScheduler([
            lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=num_warmup_steps),
            lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=config["per_device_max_steps"])
        ])
    elif config["lr_scheduler"] == "wsd":
        num_constant_steps = int(config["per_device_max_steps"] * config["wsd_second_constant_proportion"]) if int(config["per_device_max_steps"] * config["wsd_second_constant_proportion"]) > 0 else 1
        num_constant_steps_2 = int(config['per_device_max_steps'] * config["wsd_annealing_proportion"]) if int(config['per_device_max_steps'] * config["wsd_annealing_proportion"]) > 0 else 1
        if is_main_process():    
            logging.info(f"Using {num_warmup_steps} warmup steps")
            logging.info(f"Training constant lr for {num_constant_steps} steps")
            logging.info(f"Before annealing LR is dropped and trained constant lr for {num_constant_steps_2} steps")
        scheduler = TrapezoidalLRSheduler(optimizer=optimizer,num_warmup_steps=num_warmup_steps,
                                          num_constant_steps=num_constant_steps,num_training_steps=config["per_device_max_steps"],
                                          min_lr=1e-8,seq_len_lr=config['seq_len_lr'],num_constant_steps_2=num_constant_steps_2)
    elif config['annealing']:
        scheduler = one_minus_sqrt_schedule(optimizer,config["per_device_max_steps"],1e-8)

    if checkpoint is not None:
        if config["use_zero"]:
            if is_main_process():
                logging.debug("Zero optimizer is used so optimizer and scheduler states are loaded to GPU")
            optimizer_checkpoint = torch.load(config["checkpoint_path"], map_location=device)
            model.module.load_state_dict(optimizer_checkpoint["model"])
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
            if not config["annealing"]:
                scheduler = TrapezoidalLRSheduler(optimizer=optimizer,num_warmup_steps=num_warmup_steps,
                                          num_constant_steps=num_constant_steps,num_training_steps=config["per_device_max_steps"],
                                          min_lr=1e-8,seq_len_lr=config['seq_len_lr'],num_constant_steps_2=num_constant_steps_2)
                scheduler.load_state_dict(optimizer_checkpoint["scheduler"])

            else:
                scheduler = one_minus_sqrt_schedule(optimizer,config["per_device_max_steps"],1e-8)

        else:
            optimizer.load_state_dict(checkpoint["optimizer"])    
            scheduler.load_state_dict(checkpoint["scheduler"])
    #set gradient accumulation steps
    if config["long_sequences"]:
        config["n_gradient_accumulation_steps"]=config["n_gradient_accumulation_steps_annealing"]
    
    if is_main_process():
        model_config_dict = model_config.to_dict()
        GLOBAL_TENSORBOARD_WRITER.add_text("model_config",pretty_json(model_config_dict),global_step=0)
    if is_main_process():
        logging.info('Time to initialize model, optimizer, scheduler and tokenizer (seconds): {:.3f}'.format(time.time()-_TRAIN_START_TIME)) 
    return model, optimizer, scheduler, tokenizer


def save(model,dataset, optimizer, scheduler, global_step, epoch, epoch_steps,config,total_tokens,custom_message=''):
    checkpoint_dir = f"{config['output_dir']}/{config['run_name']}/checkpoint-{custom_message}{global_step}"
    checkpoint_path = f"{checkpoint_dir}/model.bin"
    os.system(f"mkdir -p {checkpoint_dir}")
    config["checkpoint_path"] = checkpoint_path
    config["current_max_length"] = dataset.current_max_length
    thetas = get_rope_thetas(model)
    config['current_global_rope_theta']=thetas['global_attention']
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
    torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "total_tokens":total_tokens.item(),
                "epoch": epoch,
                "epoch_steps":epoch_steps,
                "config": config,
            },
            checkpoint_path
        )
    
    return checkpoint_path

def seq_length_scheduler(config, global_step, num_parts=6):
    """
    Linearly increases the sequence length from `config['seq_length']` to `config['final_seq_length']`
    in `num_parts` equal steps after a warmup period. Ensures the final length is trained for
    the same amount of steps as each intermediate stage.
    """
    warmup_steps = int(config["per_device_max_steps"] * config["seq_length_warmup_proportion"])
    warmup_steps = max(warmup_steps, 1)

    total_steps = config["per_device_max_steps"]
    expansion_steps = total_steps - warmup_steps

    # Ensure num_parts + 1 phases total: one for each increment + final fixed length phase
    steps_per_phase = expansion_steps // (num_parts + 1)
    step_idx = (global_step - warmup_steps) // steps_per_phase if global_step >= warmup_steps else -1

    # Clamp to max phase index
    step_idx = max(-1, min(step_idx, num_parts))

    if step_idx == -1:
        return config['seq_length']

    # Calculate linearly increasing lengths including final length
    length_range = config['final_seq_length'] - config['seq_length']
    length_per_step = length_range / num_parts
    current_length = int(config['seq_length'] + (step_idx + 1) * length_per_step)

    return min(current_length, config['final_seq_length'])

def batch_size_scheduler(config,global_step):
    warmup_steps = int(config['bs_warmup_proportion'] * config["per_device_max_steps"]) if int(config['bs_warmup_proportion'] * config["per_device_max_steps"]) > 0 else 1
    constant_steps = int(config["seq_length_warmup_proportion"]*config['per_device_max_steps'])
    if global_step < warmup_steps:
        return int(config["per_device_batch_size"] + (config["per_device_constant_batch_size"] - config["per_device_batch_size"]) * (global_step / warmup_steps))
    if global_step < constant_steps:
        return config["per_device_constant_batch_size"]
    
    return config["per_device_anneling_batch_size"]

def high_quality_scheduler(config,global_step):
    warmup_steps = int(config["high_quality_warmup_proportion"] * config["per_device_max_steps"]) if int(config["high_quality_warmup_proportion"] * config["per_device_max_steps"]) > 0 else 1
    if global_step < warmup_steps:
        return False
    return True


def rope_theta_scheduler(config,global_step,model,final=1000000):
    warmup_steps = int(config["per_device_max_steps"] * config["seq_length_warmup_proportion"]) if int(config["per_device_max_steps"] * config["seq_length_warmup_proportion"]) > 0 else 1
    if global_step > warmup_steps:
        theta_tensor = torch.tensor(final, dtype=torch.float32).cuda()    
        torch.distributed.broadcast(theta_tensor, src=0)
        for module in model.modules():
            if hasattr(module, "rotary_emb") and hasattr(module, "local_attention"):
                #update theta only to global layers
                if module.local_attention == (-1, -1):
                    if module.rotary_emb.base != theta_tensor:
                        module.rotary_emb.base=theta_tensor.item()

def get_rope_thetas(model):
    thetas = {}
    for module in model.modules():
        if hasattr(module, "rotary_emb") and hasattr(module, "local_attention"):
            #update theta only to global layers
            if module.local_attention == (-1, -1):
               thetas['global_attention']=module.rotary_emb.base
            else:
                thetas['local_attention']=module.rotary_emb.base
        if ("global_attention" in thetas.keys()) and ("local_attention" in thetas.keys()):
            break
    return thetas

            
def load_train_dataset(config,global_step,epoch,tokenizer):
    if config["high_quality_warmup"]:
        high_quality = high_quality_scheduler(config,global_step)
        config['high_quality']=high_quality
        if is_main_process():
            logging.info(f"High quality: {config['high_quality']}")
    else:
        high_quality = config["high_quality"]
        
    if config['long_sequences']:
        root_dir = config['long_dir']
    else:
        root_dir = config["train_dir"]
        
    if config['annealing'] or config['high_quality']:
        root_dir = config['annealing_dir']


    if config['seq_length_warmup']:
        seq_len = seq_length_scheduler(config,global_step)
    else:
        seq_len = config['seq_length']
    
    if config["batch_size_warmup"]:
        batch_size = batch_size_scheduler(config,global_step)
    else:
        batch_size = config["per_device_batch_size"]
    

            
    files = glob.glob(f"{root_dir}/*.jsonl")
    if is_main_process():
        logging.info(f"Using root dir {root_dir}")
    dataset_config = StreamingMLMDatasetConfig(input_files=files,batch_size=batch_size,max_sequence_lenght=seq_len,tokenizer=tokenizer,
                                               collate_fn=CustomDataCollatorForMLM(tokenizer),high_quality_only=high_quality,
                                               english_only=config['english_only'])
    dataset = DistributedStreamingMLMDataset(config=dataset_config)
    dataset.shuffle_files(seed=config["seed"] + get_rank() + epoch)
    logging.info(f"Loaded dataset on | Global Rank {get_rank()} | Local Rank {get_local_rank()} | Files {dataset.input_files}")
    return dataset

def load_eval_dataset(config,tokenizer,dataset_type):
    def tokenize(examples):
        batch = tokenizer(examples["text"], truncation=True,max_length=config["current_max_length"])
        return batch
    if dataset_type == "val":
        data_path = config["val_file"]
    elif dataset_type == "test":
        data_path = config["test_file"]
    else:
        raise ValueError(f"eval dataset should be val or test not {dataset_type}")
    logging.debug(f"Reading dataset on rank: {get_rank()}")
    dataset = load_dataset("json", data_files=data_path,split='train')
    dataset = split_dataset_by_node(dataset, rank=get_rank(), world_size=get_world_size())
    dataset = dataset.remove_columns("dataset_source") 
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns("text")
    dataset = dataset.with_format("torch")
    logging.debug(f"Dataset processed on rank: {get_rank()}")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True,mlm_probability=0.3,return_tensors='pt')
    dataloader = DataLoader(dataset, batch_size=config["per_device_eval_batch_size"], collate_fn=data_collator,drop_last=True)
    return dataloader

def training_epoch(model, optimizer, scheduler, global_step, config, tokenizer,device,epoch,epoch_steps,total_tokens):
    logging.info(f"Starting epoch {epoch}")
    train_dataset = load_train_dataset(config,global_step,epoch,tokenizer)
    val_dataset = load_eval_dataset(config,tokenizer,'val')
    model = model.train()
    optimizer.zero_grad(set_to_none=True)
    if total_tokens > 0 and is_main_process():
        total_tokens_processed = torch.tensor([total_tokens], dtype=torch.long, device=device)
    else:
        total_tokens_processed = torch.tensor([0], dtype=torch.long, device=device)
        
    torch.distributed.broadcast(total_tokens_processed, src=0)

    total_updates = range(config["per_device_max_steps"])
    
    if is_main_process():
        train_iter = tqdm(total_updates, desc="Train iteration", initial=global_step, total=config["per_device_max_steps"],file=sys.stdout)
    else:
        train_iter = total_updates
        
    local_data_done = torch.tensor([0], device=device)        
    training_data_iterator = iter(train_dataset)
    if epoch_steps>0 and not (config['high_quality'] or config['long_sequences']):
        logging.info(f"Epoch step is {epoch_steps}>0, so training is continued from checkpoint. skipping already seen batches")
        logging.info(f"Note that skipping only done for regular data subset as other subsets are shuffled using global step as seed")
        for _ in range(int(global_step*config['n_gradient_accumulation_steps'])):
            try:
                next(training_data_iterator)
            except Exception as e:
                logging.info(f"Failed to continue iteration from epoch_step variable with error {e} on rank {get_rank()}")
                local_data_done[0] = 1            
                break
        
    torch.distributed.all_reduce(local_data_done, op=torch.distributed.ReduceOp.MAX)
    done_data = local_data_done.item()
    if done_data>=1:
        logging.info(f"Local dataloader exhausted in some rank, starting new epoch, rank {get_rank()}")
        epoch_steps = 0
        return global_step, total_tokens_processed.item(), epoch_steps


        
    total_loss, total_accuracy, total_grad_norm = 0.0, 0.0, 0.0
    for _ in train_iter:
        epoch_steps+=1
        model_saved_current_iteration = False
        if is_main_process():
            iteration_start_time = time.time()
        #for correct loss calculation gather number of non-paddded tokens, pre load full batch used in gradient accumulation
        batch_samples = []
        for _ in range(config['n_gradient_accumulation_steps']):
            try:
                batch_samples += [next(training_data_iterator)]
            except Exception as e:
                logging.info(f"Fetching new batch failed with error {e} on rank {get_rank()}")
                local_data_done[0] = 1            
                break
    
        torch.distributed.all_reduce(local_data_done, op=torch.distributed.ReduceOp.MAX)
        done_data = local_data_done.item()
        if done_data>=1:
            logging.info(f"Local dataloader exhausted in some rank, starting new epoch, rank {get_rank()}")
            epoch_steps = 0
            return global_step, total_tokens_processed.item(), epoch_steps
        
        #get local num labels in full local batch
        local_num_labels_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
        logging.debug(f"N labels in local batch: {local_num_labels_in_batch.item()}")
        #get sum of all local labels
        total_label_tokens = torch.tensor(local_num_labels_in_batch.detach().item(), device=device, dtype=torch.long)
        torch.distributed.all_reduce(total_label_tokens, op=torch.distributed.ReduceOp.SUM)
        for i,batch in enumerate(batch_samples):
            num_operation_tokens = batch["input_ids"].detach().numel()
            input_ids = batch["input_ids"].detach().flatten()
            num_tokens_in_batch = (input_ids != train_dataset.pad_token_id).sum().to(device)
            torch.distributed.all_reduce(num_tokens_in_batch, op=torch.distributed.ReduceOp.SUM)
            total_tokens_processed += num_tokens_in_batch
            del input_ids
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
                masked_model_output = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=inputs['labels'])
                # multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by total_tokens already takes into account all devices
                # Same reason for gradient_accumulation_steps
                weight = get_world_size() * config["n_gradient_accumulation_steps"] / total_label_tokens
                logging.debug(f"loss will be: {masked_model_output.loss} * {get_world_size()} * {config['n_gradient_accumulation_steps']} / {total_label_tokens} = {masked_model_output.loss * weight}")
                masked_model_output.loss =  masked_model_output.loss * weight

            with torch.no_grad():
                accuracy = (masked_model_output.logits.argmax(-1) == inputs['labels']).float().mean()
            if i < len(batch_samples):
                ctx = model.no_sync
            else:
                ctx = contextlib.nullcontext
            flush()
            with ctx():
                masked_model_output.loss.backward()
                
            total_loss += masked_model_output.loss.detach()
            total_accuracy += accuracy * weight
        logging.debug(f"Starting optimizer step on rank: {get_rank()}")
        total_grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) * weight
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        logging.debug(f"Optimizer step done on rank: {get_rank()}")

        flush()
        with torch.no_grad():
            metrics = torch.stack([total_loss, total_accuracy])
            torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        total_loss, total_accuracy = metrics.tolist()
        total_loss = total_loss / config['n_gradient_accumulation_steps']
        total_accuracy = total_accuracy / config['n_gradient_accumulation_steps']

        if is_main_process():
            iteration_elapsed_time = time.time() - iteration_start_time
            tokens_per_second_per_iteration = num_operation_tokens / iteration_elapsed_time if iteration_elapsed_time > 0 else 0
            train_iter.set_postfix_str(f"loss: {total_loss:.2f}, accuracy: {total_accuracy * 100.0:.4f}, lr: {optimizer.param_groups[0]['lr']:.5f}")
            logging.info(f"step: {global_step} loss: {total_loss:.2f}, accuracy: {total_accuracy * 100.0:.4f}, lr: {optimizer.param_groups[0]['lr']:.5f}")
            GLOBAL_TENSORBOARD_WRITER.add_scalar('performance/tokens_per_second_per_iteration_rank_0', tokens_per_second_per_iteration, global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('loss/train', total_loss, global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('perplexity/train', np.exp(total_loss), global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar("loss/tokens", total_loss,total_tokens_processed.item())
            GLOBAL_TENSORBOARD_WRITER.add_scalar('accuracy/train', total_accuracy * 100.0, global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('optimizer params/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar(f'optimizer params/grad_norm',total_grad_norm, global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('data params/device bs', int(train_dataset.batch_size * config["n_gradient_accumulation_steps"]), global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('data params/global bs', int(train_dataset.batch_size * get_world_size() * config["n_gradient_accumulation_steps"]) , global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('data params/sequence length', train_dataset.current_max_length , global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('data params/high quality data', int(train_dataset.high_quality_only) , global_step=global_step)
            GLOBAL_TENSORBOARD_WRITER.add_scalar('data params/total tokens', total_tokens_processed.item(), global_step=global_step)
            thetas = get_rope_thetas(model)
            for k,v in thetas.items():
                GLOBAL_TENSORBOARD_WRITER.add_scalar(f'rope params/{k}',v, global_step=global_step)
                
        if config["batch_size_warmup"]:
            new_batch_size = batch_size_scheduler(config,global_step)
            if new_batch_size != train_dataset.batch_size:
                if new_batch_size < train_dataset.batch_size:
                    config["per_device_eval_batch_size"] = 4

                train_dataset.set_batch_size(new_batch_size)
        
        if config['seq_length_warmup']:
            new_seq_len = seq_length_scheduler(config,global_step)
            if new_seq_len != train_dataset.current_max_length:
                previous_max_len = train_dataset.current_max_length
                if config['use_zero']:
                    torch.distributed.barrier()
                    logging.debug(f"Moving states into rank 0 from rank {get_rank()}")
                    optimizer.consolidate_state_dict(to=0)
                    logging.debug(f"Done moving on rank {get_rank()}")
                if is_main_process():
                    logging.info(f"Model is saved before training with new theta or seq leng")
                    checkpoint_path = save(model,train_dataset, optimizer, scheduler, global_step, epoch, epoch_steps, config, total_tokens_processed,f"seq-len-{previous_max_len}-")
                    logging.info(f"Saved model to: {checkpoint_path}")
                torch.distributed.barrier()
                model_saved_current_iteration = True
                logging.debug(get_gpu_utilization())
                logging.info(f"Sequence length is changed from {train_dataset.current_max_length} to {new_seq_len} and new dataloader is created")
                previous_max_len = train_dataset.current_max_length
                config['long_sequences']=True
                config["current_max_length"]=new_seq_len
                config["n_gradient_accumulation_steps"]=config['n_gradient_accumulation_steps_annealing']
                train_dataset = load_train_dataset(config,global_step,global_step,tokenizer)
                logging.debug(f"Using batch size {train_dataset.batch_size}")
                val_dataset = load_eval_dataset(config,tokenizer,'val')
                training_data_iterator = iter(train_dataset)

                
    

        if config["high_quality_warmup"]:
            new_high_quality_warmup = high_quality_scheduler(config,global_step)
            if new_high_quality_warmup is not train_dataset.high_quality_only:
                logging.info(f"High quality data is loaded and new dataloader is created")
                config["high_quality"]=True
                train_dataset = load_train_dataset(config,global_step,global_step,tokenizer)
                training_data_iterator = iter(train_dataset)
                if config['use_zero']:
                    torch.distributed.barrier()
                    logging.debug(f"Moving states into rank 0 from rank {get_rank()}")
                    optimizer.consolidate_state_dict(to=0)
                    logging.debug(f"Done moving on rank {get_rank()}")
                if is_main_process():
                    logging.info(f"Model is saved before training with high quality data")
                    checkpoint_path = save(model,train_dataset, optimizer, scheduler, global_step, epoch, epoch_steps, config, total_tokens_processed,f"seq-len-{train_dataset.current_max_length}-last-checkpoint-before-annealing")
                    logging.info(f"Saved model to: {checkpoint_path}")
                torch.distributed.barrier()
                model_saved_current_iteration = True



        if config["rope_theta_warmup"]:
            rope_theta_scheduler(config,global_step,model)

        #validation step 
        if (global_step % config["eval_steps"] == 0 and global_step !=0):
            logging.debug(f"Starting validation epoch on rank {get_rank()}")
            validation_epoch(model,global_step,device,val_dataset,eval_type='val')
            flush()
            model.train()


        #saving step
        if (global_step % config["save_steps"] == 0 and global_step !=0 and not model_saved_current_iteration):
            if config['use_zero']:
                logging.debug(f"Moving states into rank 0 from rank {get_rank()}")
                optimizer.consolidate_state_dict(to=0)
                logging.debug(f"Done moving on rank {get_rank()}")

            if is_main_process():
                logging.debug(f"Starting saving on rank 0")
                checkpoint_path = save(model,train_dataset, optimizer, scheduler, global_step, epoch, epoch_steps, config, total_tokens_processed)
                logging.info(f"Saved model to: {checkpoint_path}")
            torch.distributed.barrier()
    
        total_loss, total_accuracy, total_grad_norm = 0.0, 0.0, 0.0
    
    
        # Exiting the training due to hitting max steps.
        # Using -1 as iterable is made from range() and global step starts from 0
        if global_step >= config["per_device_max_steps"]-1:
            if config['use_zero']:
                torch.distributed.barrier()
                logging.debug(f"Moving states into rank 0 from rank {get_rank()}")
                optimizer.consolidate_state_dict(to=0)
                logging.debug(f"Done moving on rank {get_rank()}")
            if is_main_process():
                logging.info(f"Max training steps {config['per_device_max_steps']} for device is achieved")
                checkpoint_path = save(model,train_dataset, optimizer, scheduler, global_step, epoch, epoch_steps,config, total_tokens_processed,f"final-")
                logging.info(f"Saved model to: {checkpoint_path}")
            torch.distributed.barrier()
            logging.info(f"Starting validation epoch")
            validation_epoch(model,global_step,device,val_dataset,eval_type='val')
            global_step += 1
            return global_step, total_tokens_processed.item(), epoch_steps

        
        flush()
        if config["exit_duration_in_mins"]:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.tensor([train_time > config["exit_duration_in_mins"]]).to(model.device)
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if config['use_zero']:
                    torch.distributed.barrier()
                    logging.debug(f"Moving states into rank 0 from rank {get_rank()}")
                    optimizer.consolidate_state_dict(to=0)
                    logging.debug(f"Done moving on rank {get_rank()}")
                if is_main_process():
                    logging.info(f"Exit duration achieved, saving and exiting gracefylly...")
                    checkpoint_path = save(model,train_dataset, optimizer, scheduler, global_step, epoch, epoch_steps,config, total_tokens_processed)
                    logging.info(f"Saved model to: {checkpoint_path}")
                    GLOBAL_TENSORBOARD_WRITER.flush()
                    GLOBAL_TENSORBOARD_WRITER.close()
                torch.distributed.barrier()
                logging.info('exiting program gracefully after {} minutes'.format(train_time))
                torch.distributed.destroy_process_group()
                exit()
        global_step += 1

    epoch_steps = 0
    return global_step, total_tokens_processed.item(), epoch_steps

@torch.no_grad()
def validation_epoch(model, global_step, device, eval_dataset,eval_type):
    model = model.eval()
    losses, accuracies = [], []
    logging.info(f"Starting to iterate validation data on rank {get_rank()}")
    for batch in eval_dataset:
        flush()
        inputs = {k: v.to(device) for k, v in batch.items()}
        num_labels_in_batch = (inputs['labels'] != -100).sum().to(device)
        total_tokens_val = torch.tensor(num_labels_in_batch.detach().item(), device=device, dtype=torch.long)
        torch.distributed.all_reduce(total_tokens_val, op=torch.distributed.ReduceOp.SUM)
        weight = get_world_size() / total_tokens_val
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
            masked_model_output = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=inputs['labels'])
            masked_model_output.loss =  masked_model_output.loss.detach() * weight
            accuracy = (masked_model_output.logits.detach().argmax(-1) == inputs['labels']).float().mean()
            accuracy = accuracy * weight
        metrics = torch.stack([masked_model_output.loss, accuracy])
        logging.debug(f"Starting all reduce on rank {get_rank()}")
        torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        logging.debug(f"All reduce on rank {get_rank()} finnished")
        loss, accuracy = metrics.tolist()
        losses.append(loss)
        accuracies.append(accuracy)
        
    logging.info(f"Validation done on rank: {get_rank()}")

    if is_main_process():
        GLOBAL_TENSORBOARD_WRITER.add_scalar(f'loss/{eval_type}',mean(losses), global_step=global_step)
        GLOBAL_TENSORBOARD_WRITER.add_scalar(f'accuracy/{eval_type}',mean(accuracies), global_step=global_step)
        GLOBAL_TENSORBOARD_WRITER.add_scalar(f'perplexity/{eval_type}', np.exp(mean(losses)), global_step=global_step)



if __name__ == "__main__":
    training_config,checkpoint_path = parse_arguments()
    
    if training_config['logging_level'] == 'debug':
        level = logging.DEBUG
    else:
        level = logging.INFO
     
    logging.basicConfig(level=level,format="%(asctime)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stderr)])   
    
    if checkpoint_path is not None:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        training_config,checkpoint = load_checkpoint_and_update_config(checkpoint_path,training_config)
        total_tokens = checkpoint['total_tokens']
        if training_config['annealing']:
            initial_epoch=0
            global_step=0
            epoch_steps=0
            total_tokens=0
        else:
            initial_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step'] + 1
            epoch_steps = checkpoint.get('epoch_steps',0)
    else:
        checkpoint, initial_epoch, global_step, total_tokens, epoch_steps = None, 0, 0, 0, 0

    device, local_rank = setup_training(training_config)
    model, optimizer, scheduler,tokenizer = prepare_model_tokenizer_and_optimizer(training_config, device, local_rank, checkpoint)
    for epoch in count(initial_epoch):
        global_step, total_tokens, epoch_steps = training_epoch(model=model, optimizer=optimizer, scheduler=scheduler, global_step=global_step,epoch=epoch,epoch_steps=epoch_steps,config=training_config,device=device,tokenizer=tokenizer,total_tokens=total_tokens)
        if global_step >= training_config["per_device_max_steps"]:
            break
    
    if training_config["do_test"]:
        test_dataset = load_eval_dataset(training_config,tokenizer,'test')
        validation_epoch(model,global_step,device,test_dataset,'test')
        
    
    if is_main_process():
        logging.info(f"Training complete and last checkpoint saved to {training_config['checkpoint_path']}")
        GLOBAL_TENSORBOARD_WRITER.flush()
        GLOBAL_TENSORBOARD_WRITER.close()
    
    torch.distributed.destroy_process_group()
