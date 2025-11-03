import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_local_rank():
    if not dist.is_available():
        raise RuntimeError("Distributed should be available to get local rank!"
    )
    if not dist.is_initialized():
        raise RuntimeError("Distributed should be initialized to get local rank!")
    return dist.get_node_local_rank()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0