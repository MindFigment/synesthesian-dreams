import math
import torch.optim.lr_scheduler as schedulers


class RLScheduleCosine(object):
    def __init__(self, optimizer, epoch=0, epoch_start=0, lr_max=0.05, lr_min=0.001, t_mul=10, verbose=False):
        self.optimizer = optimizer
        self.epoch = epoch
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.t_start = epoch_start
        self.t_mul = t_mul
        self.lr = lr_max
        self.verbose = verbose

    
    def step(self, value):
        self.epoch += 1
        self.lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1. + math.cos(math.pi * (self.epoch - self.t_start) / self.t_mul))

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        
        if self.epoch == self.t_start + self.t_mul:
            self.t_start += self.t_mul
            self.t_mul *= 2

        if self.verbose:
            print(f"Set new learning rate {self.lr}")
        
        return self.lr