import math
import torch
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step

#from https://github.com/ltgoslo/ltg-bert/blob/main/training/train.py
def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
    def lr_lambda_cosine(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        factor = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return factor
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine)

def one_minus_sqrt_schedule(optimizer, num_training_steps: int, min_factor: float):
    def lr_lamda_one_minus_sqrt(current_step):
        progress = float(current_step) / float(max(1,num_training_steps))
        coeff = 1 - math.sqrt(progress)
        factor = max(min_factor,coeff)
        return factor
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lamda_one_minus_sqrt)

    
class TrapezoidalLRSheduler(LRScheduler):
    '''Trapezoidal LR scheduler that increases LR to lr and keep it constant until decay'''
    def __init__(self,optimizer,
                 num_warmup_steps,
                 num_constant_steps,
                 num_constant_steps_2,
                 num_training_steps,
                 seq_len_lr,
                 min_lr,
                 verbose="deprecated",last_epoch: int = -1):  # noqa: D107
        self.num_warmup_steps = num_warmup_steps
        self.num_constant_steps = num_constant_steps
        self.num_training_steps = num_training_steps
        self.num_constant_steps_2 = num_constant_steps_2
        self.seq_len_lr = seq_len_lr
        self.min_lr = min_lr
        self.start_factor = 1e-9
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]

        # Warmup phase
        if self.last_epoch <= self.num_warmup_steps:
            gamma = float(self.last_epoch) / float(max(1, self.num_warmup_steps))
            return [base_lr * gamma for base_lr in self.base_lrs]

        # Constant LR after warmup (phase 1)
        if self.last_epoch <= self.num_constant_steps:
            return [base_lr for base_lr in self.base_lrs]

        if self.last_epoch <= self.num_constant_steps_2:
            return [self.seq_len_lr for _ in self.base_lrs]

        # Decay phase
        progress = float(self.last_epoch - self.num_constant_steps) / float(
            max(1, self.num_training_steps - self.num_constant_steps)
        )
        coeff = 1 - math.sqrt(progress)

        return [
            max(self.min_lr, self.min_lr + (base_lr - self.min_lr) * coeff)
            for base_lr in self.base_lrs
        ]
            
