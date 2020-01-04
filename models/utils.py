from models.schedulers.cosine import RLScheduleCosine
import torch
import torch.nn as nn
import numpy as np


################################
## Schedulers helper function ##
################################

def get_scheduler(scheduler_name, optimizer, **kwargs):
    
    schedulers = ["Cosine", "ReduceRLOnPlateau", "MultiStepLR", "StepLR"]

    if scheduler_name not in schedulers:
        raise AttributeError(f"No scheduler named: {scheduler_name}, choose on of: {schedulers}")
    
    if scheduler_name == "Cosine":
        return RLScheduleCosine(optimizer, **kwargs)

    if scheduler_name == "ReduceRLOnPlateau":
        # kwargs: mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    if scheduler_name == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)

    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)



############################
## Losses helper function ##
############################

def get_loss_criterion(loss):
    if loss == "BCE":
        return nn.BCELoss()

    elif loss == "BCELogits":
        return nn.BCEWithLogitsLoss()

        