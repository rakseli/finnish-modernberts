
from pyrsmi import rocml
import gc
import torch

def get_gpu_utilization():
  rocml.smi_initialize()
  ngpus = rocml.smi_get_device_count()
  device_names = [rocml.smi_get_device_name(d) for d in range(ngpus)]
  mem_total = [rocml.smi_get_device_memory_total(d) * 1e-9 for d in range(ngpus)]
  mem_used = [rocml.smi_get_device_memory_used(d) * 1e-6 for d in range(ngpus)]  
  device_message = f'no. of devices = {ngpus}\n'
  message_header = f'device id\tdevice name\ttotal memory(GB)  used memory(GB) \n'
  message_delimiter = '-' * 65 + '\n'
  log_message = device_message + message_header + message_delimiter
  for i, d in enumerate(range(ngpus)):
      log_message = log_message + f'{i:6}    {device_names[i]}\t\t{mem_total[i]:.2f}\t\t{mem_used[i]/1024:.2f}\n'
  rocml.smi_shutdown()
  return log_message
    
    
def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

def mb_to_giga_bytes(bytes):
  return bytes / 1024


def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()